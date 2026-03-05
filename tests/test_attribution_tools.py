"""Tests for tools/attribution_tools.py — attribution_sweep."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.attribution_tools import (
    _attribution_sweep_impl,
    attribution_sweep,
)


def _fake_logit_attribution_impl(
    model,
    config,
    tokenizer,
    prompt,
    layers,
    position,
    target_token,
    normalized,
    num_layers,
):
    """Return a fake logit_attribution result dict."""
    layer_entries = []
    for layer_idx in layers:
        layer_entries.append(
            {
                "layer": layer_idx,
                "attention_logit": 0.5 + layer_idx * 0.1,
                "ffn_logit": 0.3 + layer_idx * 0.05,
                "total_logit": 0.8 + layer_idx * 0.15,
                "cumulative_logit": 1.0 + layer_idx * 0.2,
                "attention_top_token": "tok",
                "ffn_top_token": "tok",
            }
        )

    return {
        "prompt": prompt,
        "token_position": position,
        "token_text": "test",
        "target_token": target_token or "tok",
        "target_token_id": 42,
        "model_logit": 5.0,
        "model_probability": 0.9,
        "embedding_logit": 0.1,
        "layers": layer_entries,
        "attribution_sum": layer_entries[-1]["cumulative_logit"],
        "summary": {
            "mode": "normalized" if normalized else "raw_dla",
            "dominant_component": "ffn",
            "top_positive_layer": layers[-1] if layers else 0,
            "top_negative_layer": layers[0] if layers else 0,
        },
    }


# ---------------------------------------------------------------------------
# TestAttributionSweep (async tool)
# ---------------------------------------------------------------------------


class TestAttributionSweep:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await attribution_sweep(prompts=["hello"])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_empty_prompts(self, loaded_model_state: MagicMock) -> None:
        result = await attribution_sweep(prompts=[])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_prompts(self, loaded_model_state: MagicMock) -> None:
        result = await attribution_sweep(prompts=["x"] * 51)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_labels_length_mismatch(self, loaded_model_state: MagicMock) -> None:
        result = await attribution_sweep(
            prompts=["a", "b"],
            labels=["only_one"],
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await attribution_sweep(
            prompts=["hello"],
            layers=[99],
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.attribution_tools._attribution_sweep_impl",
            return_value={
                "num_prompts": 2,
                "num_layers": 4,
                "target_token": None,
                "normalized": True,
                "per_prompt": [{}, {}],
                "prompt_summary": [],
                "layer_summary": [],
                "dominant_layer": 0,
                "dominant_component": "ffn",
            },
        ):
            result = await attribution_sweep(
                prompts=["hello", "world"],
                layers=[0, 1, 2, 3],
            )
        assert "error" not in result
        assert result["num_prompts"] == 2
        assert "prompt_summary" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.attribution_tools._attribution_sweep_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await attribution_sweep(
                prompts=["hello"],
                layers=[0, 1, 2, 3],
            )
        assert result["error"] is True
        assert result["error_type"] == "ExtractionFailed"

    @pytest.mark.asyncio
    async def test_value_error_returns_invalid_input(self, loaded_model_state: MagicMock) -> None:
        with patch(
            "chuk_mcp_lazarus.tools.attribution_tools._attribution_sweep_impl",
            side_effect=ValueError("bad token"),
        ):
            result = await attribution_sweep(
                prompts=["hello"],
                layers=[0, 1, 2, 3],
            )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: MagicMock) -> None:
        """When layers=None, auto-sampling should work."""
        with patch(
            "chuk_mcp_lazarus.tools.attribution_tools._attribution_sweep_impl",
            return_value={
                "num_prompts": 1,
                "num_layers": 4,
                "target_token": None,
                "normalized": True,
                "per_prompt": [{}],
                "layer_summary": [],
                "dominant_layer": 0,
                "dominant_component": "ffn",
            },
        ):
            result = await attribution_sweep(prompts=["hello"])
        assert "error" not in result


# ---------------------------------------------------------------------------
# TestAttributionSweepImpl (sync helper)
# ---------------------------------------------------------------------------


class TestAttributionSweepImpl:
    def test_aggregation(self) -> None:
        """Test that aggregation computes correct mean/std."""
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._logit_attribution_impl",
            side_effect=_fake_logit_attribution_impl,
        ):
            result = _attribution_sweep_impl(
                model,
                config,
                tokenizer,
                prompts=["a", "b"],
                layers=[0, 1, 2],
                position=-1,
                target_token=None,
                normalized=True,
                num_layers=4,
                labels=["pos", "neg"],
            )

        assert result["num_prompts"] == 2
        assert result["num_layers"] == 3
        assert len(result["per_prompt"]) == 2
        assert len(result["layer_summary"]) == 3
        assert result["labels"] == ["pos", "neg"]

        # All prompts get same values in our mock → std should be 0
        for ls in result["layer_summary"]:
            assert ls["std_attention_logit"] == 0.0
            assert ls["std_ffn_logit"] == 0.0

    def test_dominant_layer_selection(self) -> None:
        """Dominant layer should be the one with highest mean |total_logit|."""
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._logit_attribution_impl",
            side_effect=_fake_logit_attribution_impl,
        ):
            result = _attribution_sweep_impl(
                model,
                config,
                tokenizer,
                prompts=["a"],
                layers=[0, 1, 2],
                position=-1,
                target_token=None,
                normalized=True,
                num_layers=4,
                labels=None,
            )

        # Layer 2 has highest total_logit (0.8 + 2*0.15 = 1.1)
        assert result["dominant_layer"] == 2

    def test_dominant_component(self) -> None:
        """dominant_component should be based on sum of mean contributions."""
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._logit_attribution_impl",
            side_effect=_fake_logit_attribution_impl,
        ):
            result = _attribution_sweep_impl(
                model,
                config,
                tokenizer,
                prompts=["a"],
                layers=[0, 1],
                position=-1,
                target_token=None,
                normalized=True,
                num_layers=4,
                labels=None,
            )

        # attention = (0.5 + 0.6) = 1.1, ffn = (0.3 + 0.35) = 0.65
        assert result["dominant_component"] == "attention"


# ---------------------------------------------------------------------------
# TestAttributionSweepPromptSummary
# ---------------------------------------------------------------------------


class TestAttributionSweepPromptSummary:
    """Tests for the prompt_summary field in attribution_sweep results."""

    def _run(
        self,
        prompts: list[str],
        layers: list[int],
        labels: list[str] | None = None,
    ) -> dict:
        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.residual_tools._logit_attribution_impl",
            side_effect=_fake_logit_attribution_impl,
        ):
            return _attribution_sweep_impl(
                model,
                config,
                tokenizer,
                prompts=prompts,
                layers=layers,
                position=-1,
                target_token=None,
                normalized=True,
                num_layers=4,
                labels=labels,
            )

    def test_prompt_summary_present(self) -> None:
        result = self._run(["a"], [0, 1])
        assert "prompt_summary" in result
        assert isinstance(result["prompt_summary"], list)

    def test_length_matches_prompts(self) -> None:
        result = self._run(["a", "b", "c"], [0, 1])
        assert len(result["prompt_summary"]) == 3

    def test_label_from_labels(self) -> None:
        result = self._run(["a", "b"], [0], labels=["alpha", "beta"])
        assert result["prompt_summary"][0]["label"] == "alpha"
        assert result["prompt_summary"][1]["label"] == "beta"

    def test_label_none_when_no_labels(self) -> None:
        result = self._run(["a"], [0])
        # exclude_none=True strips None label from model_dump output
        assert "label" not in result["prompt_summary"][0]

    def test_net_attention_correct(self) -> None:
        """net_attention should be sum of attention_logit across layers."""
        result = self._run(["a"], [0, 1, 2])
        ps = result["prompt_summary"][0]
        # Mock: attention_logit = 0.5 + layer * 0.1
        # Layers 0,1,2: 0.5 + 0.6 + 0.7 = 1.8
        expected = round(0.5 + 0.6 + 0.7, 6)
        assert ps["net_attention"] == expected

    def test_net_ffn_correct(self) -> None:
        """net_ffn should be sum of ffn_logit across layers."""
        result = self._run(["a"], [0, 1, 2])
        ps = result["prompt_summary"][0]
        # Mock: ffn_logit = 0.3 + layer * 0.05
        # Layers 0,1,2: 0.3 + 0.35 + 0.4 = 1.05
        expected = round(0.3 + 0.35 + 0.4, 6)
        assert ps["net_ffn"] == expected

    def test_dominant_component(self) -> None:
        """dominant_component should be based on net attention vs net FFN."""
        result = self._run(["a"], [0, 1])
        ps = result["prompt_summary"][0]
        # attention = 0.5 + 0.6 = 1.1, ffn = 0.3 + 0.35 = 0.65
        assert ps["dominant_component"] == "attention"

    def test_top_layers_from_per_prompt(self) -> None:
        """top_positive_layer and top_negative_layer from summary."""
        result = self._run(["a"], [0, 1, 2])
        ps = result["prompt_summary"][0]
        # Mock: top_positive_layer = layers[-1] = 2, top_negative_layer = layers[0] = 0
        assert ps["top_positive_layer"] == 2
        assert ps["top_negative_layer"] == 0

    def test_final_logit_and_probability(self) -> None:
        """final_logit and probability from model_logit and model_probability."""
        result = self._run(["a"], [0])
        ps = result["prompt_summary"][0]
        assert ps["final_logit"] == 5.0
        assert ps["probability"] == 0.9
        assert ps["top_prediction"] == "tok"
