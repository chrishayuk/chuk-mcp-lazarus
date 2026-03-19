"""Tests for tools/ablation_tools.py — ablate_layers, patch_activations."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_mcp_lazarus.tools.ablation_tools import (
    _ablate_layers_impl,
    _patch_activations_impl,
    _word_overlap_similarity,
    ablate_layers,
    patch_activations,
)


class TestWordOverlapSimilarity:
    """Pure function — no mocking needed."""

    def test_identical(self) -> None:
        assert _word_overlap_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert _word_overlap_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _word_overlap_similarity("hello world foo", "hello bar baz")
        assert 0.0 < sim < 1.0

    def test_both_empty(self) -> None:
        assert _word_overlap_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert _word_overlap_similarity("hello", "") == 0.0
        assert _word_overlap_similarity("", "hello") == 0.0

    def test_case_insensitive(self) -> None:
        assert _word_overlap_similarity("Hello World", "hello world") == 1.0


class TestAblateLayers:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_empty_layers(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[])
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], max_new_tokens=0)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_max_tokens_too_high(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], max_new_tokens=1001)
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[99])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_invalid_ablation_type(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], ablation_type="random")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_component(self, loaded_model_state: MagicMock) -> None:
        result = await ablate_layers(prompt="hello", layers=[0], component="invalid")
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "prompt": "hello",
            "ablated_layers": [0],
            "ablation_type": "zero",
            "component": "mlp",
            "ablated_output": "output",
            "baseline_output": "output",
            "output_similarity": 1.0,
            "disruption_score": 0.0,
        }
        with patch(
            "chuk_mcp_lazarus.tools.causal.tools._ablate_layers_impl",
            return_value=mock_result,
        ):
            result = await ablate_layers(prompt="hello", layers=[0])
        assert "error" not in result
        assert result["prompt"] == "hello"

    @pytest.mark.asyncio
    async def test_exception_path(self, loaded_model_state: MagicMock) -> None:
        """When _ablate_layers_impl raises, the tool returns an AblationFailed error."""
        with patch(
            "chuk_mcp_lazarus.tools.causal.tools._ablate_layers_impl",
            side_effect=RuntimeError("internal ablation error"),
        ):
            result = await ablate_layers(prompt="hello", layers=[0])
        assert result["error"] is True
        assert result["error_type"] == "AblationFailed"
        assert "internal ablation error" in result["message"]

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: MagicMock) -> None:
        """Negative layer indices should be rejected as out of range."""
        result = await ablate_layers(prompt="hello", layers=[-1])
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"
        assert "-1" in result["message"]


class TestPatchActivations:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: MagicMock) -> None:
        result = await patch_activations(source_prompt="a", target_prompt="b", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_invalid_max_tokens(self, loaded_model_state: MagicMock) -> None:
        result = await patch_activations(
            source_prompt="a", target_prompt="b", layer=0, max_new_tokens=0
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: MagicMock) -> None:
        result = await patch_activations(source_prompt="a", target_prompt="b", layer=99)
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: MagicMock) -> None:
        mock_result = {
            "source_prompt": "a",
            "target_prompt": "b",
            "patched_layer": 0,
            "patched_output": "out",
            "baseline_output": "base",
            "source_output": "src",
            "recovery_rate": 0.7,
            "effect_size": 0.2,
        }
        with patch(
            "chuk_mcp_lazarus.tools.causal.tools._patch_activations_impl",
            return_value=mock_result,
        ):
            result = await patch_activations(source_prompt="a", target_prompt="b", layer=0)
        assert "error" not in result
        assert result["patched_layer"] == 0

    @pytest.mark.asyncio
    async def test_exception_path(self, loaded_model_state: MagicMock) -> None:
        """When _patch_activations_impl raises, the tool returns an AblationFailed error."""
        with patch(
            "chuk_mcp_lazarus.tools.causal.tools._patch_activations_impl",
            side_effect=RuntimeError("internal patching error"),
        ):
            result = await patch_activations(source_prompt="a", target_prompt="b", layer=0)
        assert result["error"] is True
        assert result["error_type"] == "AblationFailed"
        assert "internal patching error" in result["message"]

    @pytest.mark.asyncio
    async def test_max_tokens_too_high(self, loaded_model_state: MagicMock) -> None:
        """max_new_tokens=1001 should be rejected as InvalidInput."""
        result = await patch_activations(
            source_prompt="a", target_prompt="b", layer=0, max_new_tokens=1001
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"
        assert "1001" in result["message"]


# ---------------------------------------------------------------------------
# Tests for _ablate_layers_impl (lines 184-230)
# ---------------------------------------------------------------------------


class TestAblateLayersImpl:
    """Direct tests for the sync _ablate_layers_impl function."""

    def test_basic_mlp_ablation(self) -> None:
        """Basic MLP ablation returns expected keys and values."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.causal.tools.generate_text",
            return_value=("baseline hello world", 3),
        ):
            result = _ablate_layers_impl(
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt="test prompt",
                layers=[0, 1],
                max_new_tokens=50,
                ablation_type="zero",
                component="mlp",
            )

        assert result["prompt"] == "test prompt"
        assert result["ablated_layers"] == [0, 1]
        assert result["ablation_type"] == "zero"
        assert result["component"] == "mlp"
        assert isinstance(result["ablated_output"], str)
        assert result["baseline_output"] == "baseline hello world"
        assert 0.0 <= result["output_similarity"] <= 1.0
        assert 0.0 <= result["disruption_score"] <= 1.0
        assert result["output_similarity"] + result["disruption_score"] == pytest.approx(1.0)

    def test_attention_component(self) -> None:
        """Ablation with component='attention' works correctly."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.causal.tools.generate_text",
            return_value=("attention baseline", 2),
        ):
            result = _ablate_layers_impl(
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt="attention test",
                layers=[2],
                max_new_tokens=30,
                ablation_type="zero",
                component="attention",
            )

        assert result["component"] == "attention"
        assert result["ablated_layers"] == [2]

    def test_both_component(self) -> None:
        """Ablation with component='both' works correctly."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.causal.tools.generate_text",
            return_value=("both baseline", 2),
        ):
            result = _ablate_layers_impl(
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt="both test",
                layers=[0],
                max_new_tokens=20,
                ablation_type="zero",
                component="both",
            )

        assert result["component"] == "both"

    def test_identical_outputs_give_zero_disruption(self) -> None:
        """When ablated and baseline outputs match, disruption should be 0."""
        from chuk_lazarus.introspection.ablation.study import AblationStudy

        # Monkey-patch the stub so ablated output matches baseline
        original_method = AblationStudy.ablate_and_generate
        AblationStudy.ablate_and_generate = lambda self, **kw: "same output"

        try:
            model = MagicMock()
            tokenizer = MagicMock()
            config = MagicMock()

            with patch(
                "chuk_mcp_lazarus.tools.causal.tools.generate_text",
                return_value=("same output", 2),
            ):
                result = _ablate_layers_impl(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    prompt="identity",
                    layers=[0],
                    max_new_tokens=10,
                    ablation_type="zero",
                    component="mlp",
                )

            assert result["output_similarity"] == 1.0
            assert result["disruption_score"] == 0.0
        finally:
            AblationStudy.ablate_and_generate = original_method

    def test_completely_different_outputs_give_high_disruption(self) -> None:
        """When outputs are completely different, disruption should be 1.0."""
        from chuk_lazarus.introspection.ablation.study import AblationStudy

        original_method = AblationStudy.ablate_and_generate
        AblationStudy.ablate_and_generate = lambda self, **kw: "xyz uvw"

        try:
            model = MagicMock()
            tokenizer = MagicMock()
            config = MagicMock()

            with patch(
                "chuk_mcp_lazarus.tools.causal.tools.generate_text",
                return_value=("abc def", 2),
            ):
                result = _ablate_layers_impl(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    prompt="divergent",
                    layers=[0],
                    max_new_tokens=10,
                    ablation_type="zero",
                    component="mlp",
                )

            assert result["output_similarity"] == 0.0
            assert result["disruption_score"] == 1.0
        finally:
            AblationStudy.ablate_and_generate = original_method

    def test_result_is_pydantic_model_dump(self) -> None:
        """Result must be a plain dict (model_dump output), not a Pydantic model."""
        model = MagicMock()
        tokenizer = MagicMock()
        config = MagicMock()

        with patch(
            "chuk_mcp_lazarus.tools.causal.tools.generate_text",
            return_value=("base", 1),
        ):
            result = _ablate_layers_impl(
                model=model,
                tokenizer=tokenizer,
                config=config,
                prompt="p",
                layers=[0],
                max_new_tokens=5,
                ablation_type="zero",
                component="mlp",
            )

        assert isinstance(result, dict)
        # Must contain all AblateLayersResult keys
        expected_keys = {
            "prompt",
            "ablated_layers",
            "ablation_type",
            "component",
            "ablated_output",
            "baseline_output",
            "output_similarity",
            "disruption_score",
        }
        assert set(result.keys()) == expected_keys

    def test_similarity_is_rounded(self) -> None:
        """output_similarity and disruption_score should be rounded to 4 decimals."""
        from chuk_lazarus.introspection.ablation.study import AblationStudy

        # Force partial overlap for a non-trivial similarity
        original_method = AblationStudy.ablate_and_generate
        AblationStudy.ablate_and_generate = lambda self, **kw: "hello bar baz qux"

        try:
            model = MagicMock()
            tokenizer = MagicMock()
            config = MagicMock()

            with patch(
                "chuk_mcp_lazarus.tools.causal.tools.generate_text",
                return_value=("hello world foo qux", 4),
            ):
                result = _ablate_layers_impl(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    prompt="rounding test",
                    layers=[0],
                    max_new_tokens=10,
                    ablation_type="zero",
                    component="mlp",
                )

            # Check values are properly rounded (at most 4 decimal places)
            sim_str = str(result["output_similarity"])
            if "." in sim_str:
                assert len(sim_str.split(".")[1]) <= 4
            dis_str = str(result["disruption_score"])
            if "." in dis_str:
                assert len(dis_str.split(".")[1]) <= 4
        finally:
            AblationStudy.ablate_and_generate = original_method


# ---------------------------------------------------------------------------
# Tests for _patch_activations_impl (lines 293-324)
# ---------------------------------------------------------------------------


class TestPatchActivationsImpl:
    """Direct tests for the sync _patch_activations_impl function."""

    def test_basic_patch(self) -> None:
        """Basic activation patching returns expected keys and values."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = _patch_activations_impl(
            model=model,
            tokenizer=tokenizer,
            source_prompt="the cat sat",
            target_prompt="the dog sat",
            layer=2,
        )

        assert result["source_prompt"] == "the cat sat"
        assert result["target_prompt"] == "the dog sat"
        assert result["patched_layer"] == 2
        assert isinstance(result["patched_output"], str)
        assert isinstance(result["baseline_output"], str)
        assert isinstance(result["source_output"], str)
        assert isinstance(result["recovery_rate"], float)
        assert isinstance(result["effect_size"], float)

    def test_result_is_dict(self) -> None:
        """Result must be a plain dict from model_dump()."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = _patch_activations_impl(
            model=model,
            tokenizer=tokenizer,
            source_prompt="a",
            target_prompt="b",
            layer=0,
        )

        assert isinstance(result, dict)
        expected_keys = {
            "source_prompt",
            "target_prompt",
            "patched_layer",
            "patched_output",
            "baseline_output",
            "source_output",
            "recovery_rate",
            "effect_size",
        }
        assert set(result.keys()) == expected_keys

    def test_recovery_rate_rounded(self) -> None:
        """recovery_rate and effect_size should be rounded to 4 decimals."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = _patch_activations_impl(
            model=model,
            tokenizer=tokenizer,
            source_prompt="src",
            target_prompt="tgt",
            layer=1,
        )

        rr_str = str(result["recovery_rate"])
        if "." in rr_str:
            assert len(rr_str.split(".")[1]) <= 4

        es_str = str(result["effect_size"])
        if "." in es_str:
            assert len(es_str.split(".")[1]) <= 4

    def test_uses_stub_values(self) -> None:
        """Verify the stub's default values propagate through correctly."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = _patch_activations_impl(
            model=model,
            tokenizer=tokenizer,
            source_prompt="clean",
            target_prompt="corrupt",
            layer=0,
        )

        # The conftest stub returns these defaults
        assert result["patched_output"] == "patched output stub"
        assert result["baseline_output"] == "corrupt output stub"
        assert result["source_output"] == "clean output stub"
        assert result["recovery_rate"] == 0.75
        assert result["effect_size"] == 0.25

    def test_layer_zero(self) -> None:
        """Patching at layer 0 should work."""
        model = MagicMock()
        tokenizer = MagicMock()

        result = _patch_activations_impl(
            model=model,
            tokenizer=tokenizer,
            source_prompt="x",
            target_prompt="y",
            layer=0,
        )

        assert result["patched_layer"] == 0

    def test_custom_patch_result(self) -> None:
        """Verify custom CounterfactualIntervention results propagate."""
        from chuk_lazarus.introspection.interventions import CounterfactualIntervention

        class CustomResult:
            patched_output = "custom patched"
            corrupt_output = "custom corrupt"
            clean_output = "custom clean"
            recovery_rate = 0.9123
            effect_size = 0.4123

        original_method = CounterfactualIntervention.patch_run
        CounterfactualIntervention.patch_run = lambda self, **kw: CustomResult()

        try:
            model = MagicMock()
            tokenizer = MagicMock()

            result = _patch_activations_impl(
                model=model,
                tokenizer=tokenizer,
                source_prompt="src",
                target_prompt="tgt",
                layer=3,
            )

            assert result["patched_output"] == "custom patched"
            assert result["baseline_output"] == "custom corrupt"
            assert result["source_output"] == "custom clean"
            assert result["recovery_rate"] == 0.9123
            assert result["effect_size"] == 0.4123
        finally:
            CounterfactualIntervention.patch_run = original_method
