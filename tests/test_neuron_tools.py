"""Tests for tools/neuron_tools.py — discover_neurons and analyze_neuron."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.tools.neuron_tools import (
    _analyze_neuron_impl,
    _cosine_sim,
    _discover_neurons_impl,
    _neuron_trace_impl,
    analyze_neuron,
    discover_neurons,
    neuron_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN_DIM = 64


def _make_deterministic_vector(seed: int) -> list[float]:
    """Return a deterministic 64-dim float vector from a given seed."""
    rng = np.random.RandomState(seed)
    return rng.randn(HIDDEN_DIM).astype(np.float32).tolist()


def _make_extraction_side_effect(vectors: dict[str, list[float]]):
    """
    Build a side_effect callable for extract_activation_at_layer.
    `vectors` maps prompt text -> activation vector.
    """

    def _side_effect(model, config, tokenizer, prompt, layer, token_position=-1):
        return vectors[prompt]

    return _side_effect


# ---------------------------------------------------------------------------
# TestDiscoverNeurons — async tool-level tests
# ---------------------------------------------------------------------------


class TestDiscoverNeurons:
    """Tests for the discover_neurons async tool wrapper."""

    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state) -> None:
        result = await discover_neurons(
            layer=0,
            positive_prompts=["hello"],
            negative_prompts=["goodbye"],
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"
        assert result["tool"] == "discover_neurons"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state) -> None:
        result = await discover_neurons(
            layer=10,
            positive_prompts=["hello"],
            negative_prompts=["goodbye"],
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state) -> None:
        result = await discover_neurons(
            layer=-1,
            positive_prompts=["hello"],
            negative_prompts=["goodbye"],
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_positive_prompts(self, loaded_model_state) -> None:
        result = await discover_neurons(
            layer=0,
            positive_prompts=[],
            negative_prompts=["goodbye"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "positive" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_empty_negative_prompts(self, loaded_model_state) -> None:
        result = await discover_neurons(
            layer=0,
            positive_prompts=["hello"],
            negative_prompts=[],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "negative" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_top_k_zero(self, loaded_model_state) -> None:
        result = await discover_neurons(
            layer=0,
            positive_prompts=["hello"],
            negative_prompts=["goodbye"],
            top_k=0,
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "top_k" in result["message"]

    @pytest.mark.asyncio
    async def test_invalid_top_k_too_high(self, loaded_model_state) -> None:
        result = await discover_neurons(
            layer=0,
            positive_prompts=["hello"],
            negative_prompts=["goodbye"],
            top_k=1001,
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "top_k" in result["message"]

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state) -> None:
        """Mock _discover_neurons_impl and verify the tool returns its output."""
        fake_result = {
            "layer": 0,
            "top_k": 5,
            "num_prompts": 4,
            "num_labels": 2,
            "labels": ["negative", "positive"],
            "neurons": [],
        }
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._discover_neurons_impl",
            return_value=fake_result,
        ):
            result = await discover_neurons(
                layer=0,
                positive_prompts=["a", "b"],
                negative_prompts=["c", "d"],
                top_k=5,
            )
        assert result == fake_result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._discover_neurons_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await discover_neurons(
                layer=0,
                positive_prompts=["a"],
                negative_prompts=["b"],
            )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# TestDiscoverNeuronsImpl — direct _impl tests
# ---------------------------------------------------------------------------


class TestDiscoverNeuronsImpl:
    """Tests for _discover_neurons_impl logic."""

    def test_returns_expected_keys(self) -> None:
        """Result dict must have the expected top-level keys."""
        vec = _make_deterministic_vector(0)
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            return_value=vec,
        ):
            result = _discover_neurons_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                positive_prompts=["pos1"],
                negative_prompts=["neg1"],
                positive_label="positive",
                negative_label="negative",
                top_k=5,
                token_position=-1,
            )
        assert "layer" in result
        assert "top_k" in result
        assert "num_prompts" in result
        assert "num_labels" in result
        assert "labels" in result
        assert "neurons" in result

    def test_known_vectors_correct_ranking(self) -> None:
        """
        Use multiple samples per group so pooled_std (not overall_std) is used.
        Neuron 0: large mean diff, small within-group variance -> high Cohen's d.
        Neuron 1: small mean diff, large within-group variance -> low Cohen's d.
        Neuron 0 should rank higher than neuron 1.

        Cohen's d = |mean_diff| / pooled_std
        where pooled_std = sqrt((std_pos^2 + std_neg^2) / 2)
        """
        # 2 positive samples, 2 negative samples
        pos1 = [0.0] * HIDDEN_DIM
        pos1[0] = 11.0  # neuron 0: pos group centered at ~10, small spread
        pos1[1] = 20.0  # neuron 1: pos group centered at 5, huge spread

        pos2 = [0.0] * HIDDEN_DIM
        pos2[0] = 9.0  # neuron 0: pos group mean=10, std=1
        pos2[1] = -10.0  # neuron 1: pos group mean=5, std=15

        neg1 = [0.0] * HIDDEN_DIM
        neg1[0] = -9.0  # neuron 0: neg group centered at ~-10, small spread
        neg1[1] = 6.0  # neuron 1: neg group centered at 3, small spread

        neg2 = [0.0] * HIDDEN_DIM
        neg2[0] = -11.0  # neuron 0: neg group mean=-10, std=1
        neg2[1] = 0.0  # neuron 1: neg group mean=3, std=3

        # Neuron 0: mean_diff=20, pooled_std=sqrt((1^2+1^2)/2)=1 -> Cohen's d=20
        # Neuron 1: mean_diff=2, pooled_std=sqrt((15^2+3^2)/2)~=10.8 -> Cohen's d~0.18

        vectors = {"p1": pos1, "p2": pos2, "n1": neg1, "n2": neg2}

        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            side_effect=_make_extraction_side_effect(vectors),
        ):
            result = _discover_neurons_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                positive_prompts=["p1", "p2"],
                negative_prompts=["n1", "n2"],
                positive_label="positive",
                negative_label="negative",
                top_k=5,
                token_position=-1,
            )

        neurons = result["neurons"]
        assert len(neurons) == 5
        # Neuron 0 should be the top-ranked neuron (high Cohen's d)
        assert neurons[0]["neuron_idx"] == 0
        assert neurons[0]["separation"] > 10.0  # ~20.0
        # Neuron 1 should have much lower separation
        neuron_1_entry = next(n for n in neurons if n["neuron_idx"] == 1)
        assert neurons[0]["separation"] > neuron_1_entry["separation"]

    def test_single_sample_per_group(self) -> None:
        """
        With 1 sample per group, fallback to overall_std for Cohen's d.
        Should still produce valid separation scores.
        """
        pos_vec = [5.0] * HIDDEN_DIM
        neg_vec = [-5.0] * HIDDEN_DIM

        vectors = {"p": pos_vec, "n": neg_vec}

        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            side_effect=_make_extraction_side_effect(vectors),
        ):
            result = _discover_neurons_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                positive_prompts=["p"],
                negative_prompts=["n"],
                positive_label="pos",
                negative_label="neg",
                top_k=3,
                token_position=-1,
            )

        neurons = result["neurons"]
        assert len(neurons) == 3
        # All neurons have the same separation since all dims separate equally
        for n in neurons:
            assert n["separation"] > 0

    def test_top_k_limits_output(self) -> None:
        """top_k=2 should return exactly 2 neurons even if hidden_dim is 64."""
        vec = _make_deterministic_vector(42)
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            return_value=vec,
        ):
            result = _discover_neurons_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                positive_prompts=["a"],
                negative_prompts=["b"],
                positive_label="p",
                negative_label="n",
                top_k=2,
                token_position=-1,
            )

        assert len(result["neurons"]) == 2

    def test_group_means_correct(self) -> None:
        """Verify group_means reflect the actual activation values."""
        pos_vec = [1.0] * HIDDEN_DIM
        neg_vec = [3.0] * HIDDEN_DIM

        vectors = {"pos": pos_vec, "neg": neg_vec}

        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            side_effect=_make_extraction_side_effect(vectors),
        ):
            result = _discover_neurons_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=2,
                positive_prompts=["pos"],
                negative_prompts=["neg"],
                positive_label="A",
                negative_label="B",
                top_k=HIDDEN_DIM,
                token_position=-1,
            )

        # Every neuron should have mean 1.0 for group A and 3.0 for group B
        for n in result["neurons"]:
            assert n["group_means"]["A"] == pytest.approx(1.0, abs=1e-4)
            assert n["group_means"]["B"] == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# TestAnalyzeNeuron — async tool-level tests
# ---------------------------------------------------------------------------


class TestAnalyzeNeuron:
    """Tests for the analyze_neuron async tool wrapper."""

    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state) -> None:
        result = await analyze_neuron(layer=0, neuron_indices=[0], prompts=["hello"])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"
        assert result["tool"] == "analyze_neuron"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state) -> None:
        result = await analyze_neuron(layer=99, neuron_indices=[0], prompts=["hello"])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_no_prompts(self, loaded_model_state) -> None:
        result = await analyze_neuron(layer=0, neuron_indices=[0], prompts=[])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "prompt" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_no_neuron_indices(self, loaded_model_state) -> None:
        result = await analyze_neuron(layer=0, neuron_indices=[], prompts=["hello"])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "neuron" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_neuron_index_out_of_range(self, loaded_model_state) -> None:
        """Neuron index >= hidden_dim (64) should be rejected."""
        result = await analyze_neuron(layer=0, neuron_indices=[64], prompts=["hello"])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "out of range" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_too_many_neurons(self, loaded_model_state) -> None:
        """More than 100 neuron indices should be rejected."""
        result = await analyze_neuron(layer=0, neuron_indices=list(range(101)), prompts=["hello"])
        # Note: indices go up to 100 which is >= hidden_dim=64, but the
        # length check (>100) comes before the range check. 101 items > 100.
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "100" in result["message"]

    @pytest.mark.asyncio
    async def test_too_many_prompts(self, loaded_model_state) -> None:
        """More than 50 prompts should be rejected."""
        result = await analyze_neuron(
            layer=0,
            neuron_indices=[0],
            prompts=[f"prompt_{i}" for i in range(51)],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "50" in result["message"]

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state) -> None:
        """Mock _analyze_neuron_impl and verify the tool returns its output."""
        fake_result = {
            "layer": 0,
            "num_prompts": 2,
            "neurons": [],
            "prompts": ["a", "b"],
        }
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._analyze_neuron_impl",
            return_value=fake_result,
        ):
            result = await analyze_neuron(layer=0, neuron_indices=[0, 1], prompts=["a", "b"])
        assert result == fake_result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_extraction_failed(self, loaded_model_state) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._analyze_neuron_impl",
            side_effect=ValueError("extraction kaboom"),
        ):
            result = await analyze_neuron(layer=0, neuron_indices=[0], prompts=["hello"])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"
        assert "kaboom" in result["message"]


# ---------------------------------------------------------------------------
# TestAnalyzeNeuronImpl — direct _impl tests
# ---------------------------------------------------------------------------


class TestAnalyzeNeuronImpl:
    """Tests for _analyze_neuron_impl logic."""

    def test_returns_expected_keys(self) -> None:
        """Result dict must have the expected top-level keys."""
        vec = _make_deterministic_vector(0)
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            return_value=vec,
        ):
            result = _analyze_neuron_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                neuron_indices=[0, 1],
                prompts=["test prompt"],
                token_position=-1,
                detailed=False,
            )
        assert "layer" in result
        assert "num_prompts" in result
        assert "neurons" in result
        assert "prompts" in result

    def test_known_vectors_correct_stats(self) -> None:
        """
        Use known activation vectors and verify min/max/mean/std.
        Prompt A: neuron 0 = 2.0, neuron 1 = 4.0
        Prompt B: neuron 0 = 6.0, neuron 1 = 8.0
        """
        vec_a = [0.0] * HIDDEN_DIM
        vec_a[0] = 2.0
        vec_a[1] = 4.0

        vec_b = [0.0] * HIDDEN_DIM
        vec_b[0] = 6.0
        vec_b[1] = 8.0

        vectors = {"prompt_a": vec_a, "prompt_b": vec_b}

        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            side_effect=_make_extraction_side_effect(vectors),
        ):
            result = _analyze_neuron_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                neuron_indices=[0, 1],
                prompts=["prompt_a", "prompt_b"],
                token_position=-1,
                detailed=False,
            )

        neurons = result["neurons"]
        assert len(neurons) == 2

        # Neuron 0: values [2.0, 6.0] => min=2, max=6, mean=4, std=2
        n0 = neurons[0]
        assert n0["neuron_idx"] == 0
        assert n0["min_val"] == pytest.approx(2.0, abs=1e-4)
        assert n0["max_val"] == pytest.approx(6.0, abs=1e-4)
        assert n0["mean_val"] == pytest.approx(4.0, abs=1e-4)
        assert n0["std_val"] == pytest.approx(2.0, abs=1e-4)

        # Neuron 1: values [4.0, 8.0] => min=4, max=8, mean=6, std=2
        n1 = neurons[1]
        assert n1["neuron_idx"] == 1
        assert n1["min_val"] == pytest.approx(4.0, abs=1e-4)
        assert n1["max_val"] == pytest.approx(8.0, abs=1e-4)
        assert n1["mean_val"] == pytest.approx(6.0, abs=1e-4)
        assert n1["std_val"] == pytest.approx(2.0, abs=1e-4)

    def test_detailed_includes_per_prompt(self) -> None:
        """When detailed=True, per_prompt_activations should be present."""
        vec_a = [1.0] * HIDDEN_DIM
        vec_b = [2.0] * HIDDEN_DIM

        vectors = {"p1": vec_a, "p2": vec_b}

        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            side_effect=_make_extraction_side_effect(vectors),
        ):
            result = _analyze_neuron_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                neuron_indices=[0, 5],
                prompts=["p1", "p2"],
                token_position=-1,
                detailed=True,
            )

        assert "per_prompt_activations" in result
        per_prompt = result["per_prompt_activations"]
        assert len(per_prompt) == 2

        # First prompt
        assert per_prompt[0]["prompt"] == "p1"
        assert per_prompt[0]["neuron_0"] == pytest.approx(1.0, abs=1e-4)
        assert per_prompt[0]["neuron_5"] == pytest.approx(1.0, abs=1e-4)

        # Second prompt
        assert per_prompt[1]["prompt"] == "p2"
        assert per_prompt[1]["neuron_0"] == pytest.approx(2.0, abs=1e-4)
        assert per_prompt[1]["neuron_5"] == pytest.approx(2.0, abs=1e-4)

    def test_not_detailed_excludes_per_prompt(self) -> None:
        """When detailed=False, per_prompt_activations should be excluded (exclude_none)."""
        vec = _make_deterministic_vector(7)
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools.extract_activation_at_layer",
            return_value=vec,
        ):
            result = _analyze_neuron_impl(
                model=MagicMock(),
                config=MagicMock(),
                tokenizer=MagicMock(),
                layer=0,
                neuron_indices=[0],
                prompts=["hello"],
                token_position=-1,
                detailed=False,
            )

        # model_dump(exclude_none=True) means the key should be absent
        assert "per_prompt_activations" not in result


# ---------------------------------------------------------------------------
# TestNeuronTrace (async tool)
# ---------------------------------------------------------------------------


class TestNeuronTrace:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await neuron_trace(prompt="hello", layer=0, neuron_index=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await neuron_trace(prompt="hello", layer=99, neuron_index=0)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_neuron_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await neuron_trace(prompt="hello", layer=0, neuron_index=9999)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_target_before_source(self, loaded_model_state: MagicMock) -> None:
        result = await neuron_trace(prompt="hello", layer=2, neuron_index=0, target_layers=[1])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_target_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await neuron_trace(prompt="hello", layer=0, neuron_index=0, target_layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "neuron": {
                "layer": 0,
                "neuron_index": 5,
                "activation": 0.5,
                "output_direction_norm": 1.0,
                "top_token": "tok",
            },
            "num_trace_layers": 2,
            "trace": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._neuron_trace_impl",
            return_value=mock_result,
        ):
            result = await neuron_trace(
                prompt="hello", layer=0, neuron_index=5, target_layers=[1, 2]
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_targets(self, loaded_model_state: MagicMock) -> None:
        """When target_layers=None, auto-generates range(layer+1, ...)."""
        mock_result = {
            "prompt": "hello",
            "token_position": -1,
            "token_text": "hello",
            "neuron": {},
            "num_trace_layers": 3,
            "trace": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._neuron_trace_impl",
            return_value=mock_result,
        ):
            result = await neuron_trace(prompt="hello", layer=0, neuron_index=5)
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.neuron.tools._neuron_trace_impl",
            side_effect=RuntimeError("trace failed"),
        ):
            result = await neuron_trace(prompt="hello", layer=0, neuron_index=5, target_layers=[1])
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    @pytest.mark.asyncio
    async def test_no_target_layers_for_last_layer(self, loaded_model_state: MagicMock) -> None:
        """Source at last layer (3) → no valid target layers → error."""
        result = await neuron_trace(prompt="hello", layer=3, neuron_index=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"


# ---------------------------------------------------------------------------
# TestNeuronTraceImpl (sync helper)
# ---------------------------------------------------------------------------


class TestNeuronTraceImpl:
    """Test _neuron_trace_impl with mocked decomposition forward."""

    def _run(
        self,
        layer: int = 0,
        neuron_index: int = 5,
        target_layers: list[int] | None = None,
    ) -> dict:
        import mlx.core as mx

        target_layers = target_layers or [1, 2]
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.side_effect = lambda ids, **kw: " ".join(f"tok{i}" for i in ids)

        metadata = MagicMock()
        metadata.num_layers = 4
        metadata.num_attention_heads = 4
        metadata.num_kv_heads = 4
        metadata.head_dim = 16
        metadata.hidden_dim = 64
        metadata.intermediate_size = 256

        # Build fake decomposition result
        all_layers = sorted(set([layer] + target_layers))
        hidden_states = {}
        prev_hidden = {}
        attn_outputs = {}
        ffn_outputs = {}

        for li in all_layers:
            hidden_states[li] = mx.array(np.random.randn(1, 5, 64).astype(np.float32))
            prev_hidden[li] = mx.array(np.random.randn(1, 5, 64).astype(np.float32))
            attn_outputs[li] = mx.array(np.random.randn(1, 5, 64).astype(np.float32))
            ffn_outputs[li] = mx.array(np.random.randn(1, 5, 64).astype(np.float32))

        decomp_result = {
            "embeddings": mx.array(np.random.randn(1, 5, 64).astype(np.float32)),
            "hidden_states": hidden_states,
            "prev_hidden": prev_hidden,
            "attn_outputs": attn_outputs,
            "ffn_outputs": ffn_outputs,
        }

        # Mock MLP components with proper shapes
        mock_gate = mx.array(np.random.randn(1, 256).astype(np.float32))
        mock_up = mx.array(np.random.randn(1, 256).astype(np.float32))
        mock_down_weight = mx.array(np.random.randn(64, 256).astype(np.float32))

        # Build fake model layers with working MLP components
        fake_mlp = MagicMock()
        fake_mlp.gate_proj = lambda x: mock_gate
        fake_mlp.up_proj = lambda x: mock_up
        fake_mlp.down_proj.weight = mock_down_weight

        fake_layers = []
        for _ in range(4):
            fl = MagicMock()
            fl.mlp = fake_mlp
            fl.post_attention_layernorm = lambda x: x
            fake_layers.append(fl)

        model.model.layers = fake_layers

        with (
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                return_value=decomp_result,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=MagicMock(),
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._project_to_logits",
                return_value=mx.array(np.random.randn(100).astype(np.float32)),
            ),
        ):
            return _neuron_trace_impl(
                model,
                config,
                tokenizer,
                metadata,
                prompt="hello",
                layer=layer,
                neuron_index=neuron_index,
                target_layers=target_layers,
                token_position=-1,
                top_k_heads=5,
            )

    def test_output_structure(self) -> None:
        result = self._run()
        assert isinstance(result, dict)
        assert result["prompt"] == "hello"
        assert "neuron" in result
        assert "trace" in result
        assert "summary" in result
        assert "num_trace_layers" in result
        assert result["token_position"] == -1

    def test_neuron_info_fields(self) -> None:
        result = self._run()
        neuron = result["neuron"]
        assert "layer" in neuron
        assert "neuron_index" in neuron
        assert "activation" in neuron
        assert "output_direction_norm" in neuron
        assert "top_token" in neuron
        assert neuron["layer"] == 0
        assert neuron["neuron_index"] == 5

    def test_trace_length(self) -> None:
        result = self._run(target_layers=[1, 2, 3])
        assert len(result["trace"]) == 3

    def test_trace_entry_fields(self) -> None:
        result = self._run()
        for entry in result["trace"]:
            assert "layer" in entry
            assert "residual_alignment" in entry
            assert "residual_projection" in entry

    def test_alignment_range(self) -> None:
        result = self._run()
        for entry in result["trace"]:
            assert -1.0 <= entry["residual_alignment"] <= 1.0

    def test_attention_alignment_range(self) -> None:
        result = self._run()
        for entry in result["trace"]:
            if "attention_alignment" in entry:
                assert -1.0 <= entry["attention_alignment"] <= 1.0

    def test_ffn_alignment_range(self) -> None:
        result = self._run()
        for entry in result["trace"]:
            if "ffn_alignment" in entry:
                assert -1.0 <= entry["ffn_alignment"] <= 1.0

    def test_summary(self) -> None:
        result = self._run()
        summary = result["summary"]
        assert "max_residual_alignment" in summary
        assert "max_alignment_layer" in summary
        assert "num_trace_layers" in summary
        assert "top_token" in summary


# ---------------------------------------------------------------------------
# TestCosineSim
# ---------------------------------------------------------------------------


class TestCosineSim:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(v, v) - 1.0) < 1e-6

    def test_opposite_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_sim(v, -v) - (-1.0)) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_sim(a, b) == 0.0
