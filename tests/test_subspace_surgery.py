"""Tests for subspace_surgery tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chuk_mcp_lazarus.dark_table_registry import (
    DarkTableMetadata,
    DarkTableRegistry,
)
from chuk_mcp_lazarus.subspace_registry import SubspaceMetadata, SubspaceRegistry
from chuk_mcp_lazarus.tools.geometry.subspace_surgery import (
    SubspaceEnergyAnalysis,
    SubspaceSurgeryResult,
    SurgeryVerification,
    _subspace_surgery_impl,
    subspace_surgery,
)
from chuk_mcp_lazarus.tools.geometry.inject_residual import TokenPrediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 8
VOCAB_SIZE = 5
RANK = 2

# Subspace basis: first 2 standard basis vectors [rank=2, hidden_dim=8]
MOCK_BASIS = np.eye(RANK, DIM, dtype=np.float32)

# Full hidden states [seq=3, hidden_dim=8]
RECIP_HIDDEN = np.array(
    [
        [3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 3.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

DONOR_HIDDEN = np.array(
    [
        [0.0, 0.0, 0.1, 3.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.5, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

RECIP_BASELINE_LOGITS = [2.0, 4.0, 1.0, 5.0, 3.0]  # top-1 = token 3
SURGICAL_LOGITS = [5.0, 3.0, 1.0, 2.0, 4.0]  # top-1 = token 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store_subspace(name: str = "test_sub") -> np.ndarray:
    """Pre-store a subspace and return the basis."""
    meta = SubspaceMetadata(
        name=name,
        layer=5,
        rank=RANK,
        num_prompts=10,
        hidden_dim=DIM,
        variance_explained=[0.5, 0.3],
        total_variance_explained=0.8,
        computed_at="2026-01-01T00:00:00+00:00",
    )
    SubspaceRegistry.get().store(name, MOCK_BASIS, meta)
    return MOCK_BASIS


def _store_dark_table(
    table_name: str = "test_dt",
    subspace_name: str = "test_sub",
) -> None:
    """Pre-store a dark table with a known entry."""
    coords = {
        "7": np.array([1.0, 2.0], dtype=np.float32),
        "12": np.array([3.0, 4.0], dtype=np.float32),
    }
    meta = DarkTableMetadata(
        table_name=table_name,
        subspace_name=subspace_name,
        layer=5,
        rank=RANK,
        num_entries=2,
        token_position=-1,
        computed_at="2026-01-01T00:00:00+00:00",
    )
    DarkTableRegistry.get().store(table_name, coords, meta)


# ---------------------------------------------------------------------------
# Async validation tests
# ---------------------------------------------------------------------------


class TestSubspaceSurgery:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="s",
            mode="donor",
            donor_prompt="d",
        )
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=99,
            subspace_name="test_sub",
            mode="donor",
            donor_prompt="d",
        )
        assert result["error"] is True
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_subspace_not_found(self, loaded_model_state: Any) -> None:
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="missing",
            mode="donor",
            donor_prompt="d",
        )
        assert result["error"] is True
        assert result["error_type"] == "VectorNotFound"

    @pytest.mark.asyncio
    async def test_invalid_mode(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="invalid",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_donor_mode_no_prompt(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="donor",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_coordinates_mode_no_coords(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="coordinates",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_coordinates_mode_wrong_length(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="coordinates",
            coordinates=[1.0, 2.0, 3.0],  # rank=2, so 3 is wrong
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_lookup_mode_no_key(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="lookup",
            table_name="dt",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_lookup_mode_no_table(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="lookup",
            lookup_key="7",
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_lookup_table_not_found(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="lookup",
            lookup_key="7",
            table_name="missing",
        )
        assert result["error"] is True
        assert result["error_type"] == "VectorNotFound"

    @pytest.mark.asyncio
    async def test_lookup_key_not_found(self, loaded_model_state: Any) -> None:
        _store_subspace()
        _store_dark_table()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="lookup",
            lookup_key="missing_key",
            table_name="test_dt",
        )
        assert result["error"] is True
        assert result["error_type"] == "VectorNotFound"

    @pytest.mark.asyncio
    async def test_top_k_out_of_range(self, loaded_model_state: Any) -> None:
        _store_subspace()
        result = await subspace_surgery(
            recipient_prompt="test",
            layer=0,
            subspace_name="test_sub",
            mode="donor",
            donor_prompt="d",
            top_k=0,
        )
        assert result["error"] is True
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success_returns_dict(self, loaded_model_state: Any) -> None:
        _store_subspace()
        fake_result = {"mode": "donor", "layer": 0}
        with patch(
            "chuk_mcp_lazarus.tools.geometry.subspace_surgery._subspace_surgery_impl",
            return_value=fake_result,
        ):
            result = await subspace_surgery(
                recipient_prompt="test",
                layer=0,
                subspace_name="test_sub",
                mode="donor",
                donor_prompt="d",
            )
        assert result["mode"] == "donor"

    @pytest.mark.asyncio
    async def test_exception_handling(self, loaded_model_state: Any) -> None:
        _store_subspace()
        with patch(
            "chuk_mcp_lazarus.tools.geometry.subspace_surgery._subspace_surgery_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await subspace_surgery(
                recipient_prompt="test",
                layer=0,
                subspace_name="test_sub",
                mode="donor",
                donor_prompt="d",
            )
        assert result["error"] is True
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# Impl tests
# ---------------------------------------------------------------------------


class TestSubspaceSurgeryImpl:
    def _run(
        self,
        mode: str = "donor",
        donor_prompt: str | None = "donor prompt",
        donor_layer: int | None = None,
        coordinates: list[float] | None = None,
        lookup_key: str | None = None,
        table_name: str | None = None,
        top_k: int = 5,
    ) -> dict:
        _store_subspace()

        if mode == "lookup":
            _store_dark_table()
            if lookup_key is None:
                lookup_key = "7"
            if table_name is None:
                table_name = "test_dt"

        if mode == "coordinates" and coordinates is None:
            coordinates = [1.0, 2.0]

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.num_layers = 4
        meta.hidden_dim = DIM

        # Build mock decomposition results
        # Recipient needs layer 0 and layer 3 (last_layer)
        # Donor needs effective_donor_layer

        def make_hidden_tensor(data: np.ndarray) -> MagicMock:
            """Create a mock that behaves like mx.array [1, seq, hidden_dim]."""
            mock = MagicMock()
            mock.__getitem__ = lambda self, idx: MagicMock(tolist=lambda: data.tolist())
            mock.ndim = 3
            mock.shape = (1, data.shape[0], data.shape[1])
            return mock

        recip_hidden_at_layer = make_hidden_tensor(RECIP_HIDDEN)
        recip_hidden_at_last = make_hidden_tensor(RECIP_HIDDEN)
        donor_hidden = make_hidden_tensor(DONOR_HIDDEN)

        decomp_calls = [0]

        def fake_decomp(model: Any, config: Any, input_ids: Any, layers: list[int]) -> dict:
            idx = decomp_calls[0]
            decomp_calls[0] += 1
            if idx == 0:
                # Recipient decomp
                return {
                    "hidden_states": {
                        0: recip_hidden_at_layer,
                        3: recip_hidden_at_last,
                    },
                    "prev_hidden": {},
                    "attn_outputs": {},
                    "ffn_outputs": {},
                }
            else:
                # Donor decomp
                layer_key = layers[0] if layers else 0
                return {
                    "hidden_states": {layer_key: donor_hidden},
                    "prev_hidden": {},
                    "attn_outputs": {},
                    "ffn_outputs": {},
                }

        # _extract_position returns last position vector
        def fake_extract_position(tensor: Any, position: int) -> MagicMock:
            vec = MagicMock()
            vec.tolist.return_value = RECIP_HIDDEN[-1].tolist()
            vec.reshape.return_value = vec
            return vec

        # _norm_project cycling: first = baseline, second = surgical
        norm_calls = [0]

        def fake_norm_project(final_norm: Any, lm_head: Any, vec: Any) -> MagicMock:
            idx = norm_calls[0]
            norm_calls[0] += 1
            if idx == 0:
                data = RECIP_BASELINE_LOGITS
            else:
                data = SURGICAL_LOGITS
            result = MagicMock()
            result.tolist.return_value = list(data)
            return result

        mock_lm_head = MagicMock()
        mock_helper = MagicMock()
        mock_helper._get_final_norm.return_value = MagicMock()

        mock_injected_hidden = MagicMock()
        mock_injected_hidden.ndim = 3
        mock_injected_hidden.shape = (1, 3, DIM)

        with (
            patch(
                "chuk_mcp_lazarus._residual_helpers._run_decomposition_forward",
                side_effect=fake_decomp,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._extract_position",
                side_effect=fake_extract_position,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._norm_project",
                side_effect=fake_norm_project,
            ),
            patch(
                "chuk_mcp_lazarus._residual_helpers._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch(
                "chuk_lazarus.introspection.hooks.ModelHooks",
                return_value=mock_helper,
            ),
            patch("mlx.core.eval"),
            patch("mlx.core.array", side_effect=lambda x: MagicMock(ndim=3, shape=(1, 3, DIM))),
            patch(
                "chuk_mcp_lazarus.tools.geometry.subspace_surgery._run_forward_with_injection",
                return_value=mock_injected_hidden,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.subspace_surgery._generate_from_hidden",
                return_value=("surgical text", 3),
            ),
        ):
            return _subspace_surgery_impl(
                model,
                config,
                tokenizer,
                meta,
                "The capital is",
                0,  # layer
                "test_sub",
                mode,
                donor_prompt,
                donor_layer,
                coordinates,
                lookup_key,
                table_name,
                50,  # max_new_tokens
                0.0,  # temperature
                top_k,
            )

    # -- Donor mode tests --

    def test_donor_mode_output_keys(self) -> None:
        r = self._run(mode="donor")
        for key in [
            "recipient_prompt",
            "mode",
            "layer",
            "subspace_name",
            "subspace_rank",
            "recipient_baseline",
            "surgical_output",
            "generated_text",
            "num_generated_tokens",
            "verification",
            "energy_analysis",
            "summary",
        ]:
            assert key in r, f"Missing key: {key}"

    def test_donor_mode_verification_clean(self) -> None:
        r = self._run(mode="donor")
        v = r["verification"]
        assert v["orthogonal_cosine"] >= 0.999
        assert v["surgery_clean"] is True

    def test_donor_mode_has_donor_prompt(self) -> None:
        r = self._run(mode="donor")
        assert r["donor_prompt"] == "donor prompt"

    # -- Coordinates mode tests --

    def test_coordinates_mode_output_keys(self) -> None:
        r = self._run(mode="coordinates", coordinates=[1.0, 2.0])
        assert r["mode"] == "coordinates"
        assert "verification" in r
        assert "energy_analysis" in r

    def test_coordinates_mode_verification(self) -> None:
        r = self._run(mode="coordinates", coordinates=[1.0, 2.0])
        v = r["verification"]
        assert v["orthogonal_cosine"] >= 0.999

    # -- Lookup mode tests --

    def test_lookup_mode_output_keys(self) -> None:
        r = self._run(mode="lookup")
        assert r["mode"] == "lookup"
        assert "verification" in r

    def test_lookup_mode_has_table_info(self) -> None:
        r = self._run(mode="lookup")
        assert r["lookup_key"] == "7"
        assert r["table_name"] == "test_dt"

    # -- General output tests --

    def test_baseline_top_k_present(self) -> None:
        r = self._run()
        assert len(r["recipient_baseline"]) > 0
        assert "token" in r["recipient_baseline"][0]

    def test_surgical_top_k_present(self) -> None:
        r = self._run()
        assert len(r["surgical_output"]) > 0
        assert "token" in r["surgical_output"][0]

    def test_generated_text_present(self) -> None:
        r = self._run()
        assert r["generated_text"] == "surgical text"
        assert r["num_generated_tokens"] == 3

    def test_energy_fractions_range(self) -> None:
        r = self._run()
        ea = r["energy_analysis"]
        assert 0.0 <= ea["recipient_subspace_energy_fraction"] <= 1.0
        assert 0.0 <= ea["new_content_energy_fraction"] <= 1.0

    def test_verification_norm_ratio_near_one(self) -> None:
        r = self._run()
        v = r["verification"]
        assert 0.99 <= v["orthogonal_norm_ratio"] <= 1.01

    def test_summary_keys(self) -> None:
        r = self._run()
        s = r["summary"]
        for key in [
            "recipient_baseline_token",
            "surgical_token",
            "prediction_changed",
            "orthogonal_cosine",
            "surgery_clean",
            "subspace_rank",
            "mode",
        ]:
            assert key in s, f"Missing summary key: {key}"

    def test_subspace_rank_in_output(self) -> None:
        r = self._run()
        assert r["subspace_rank"] == RANK

    def test_donor_layer_passthrough(self) -> None:
        r = self._run(mode="donor", donor_layer=2)
        assert r["donor_layer"] == 2

    def test_donor_layer_default(self) -> None:
        r = self._run(mode="donor", donor_layer=None)
        assert r["donor_layer"] is None

    def test_mode_in_output(self) -> None:
        r = self._run(mode="donor")
        assert r["mode"] == "donor"

    def test_donor_prompt_in_output(self) -> None:
        r = self._run(mode="donor", donor_prompt="my donor")
        assert r["donor_prompt"] == "my donor"

    def test_layer_in_output(self) -> None:
        r = self._run()
        assert r["layer"] == 0

    def test_subspace_name_in_output(self) -> None:
        r = self._run()
        assert r["subspace_name"] == "test_sub"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestSubspaceSurgeryModels:
    def test_surgery_verification(self) -> None:
        v = SurgeryVerification(
            orthogonal_cosine=0.9999,
            orthogonal_norm_ratio=1.0001,
            surgery_clean=True,
        )
        d = v.model_dump()
        assert d["orthogonal_cosine"] == 0.9999
        assert d["surgery_clean"] is True

    def test_subspace_energy_analysis(self) -> None:
        e = SubspaceEnergyAnalysis(
            recipient_subspace_energy_fraction=0.3,
            new_content_energy_fraction=0.4,
            recipient_subspace_norm=1.5,
            recipient_orthogonal_norm=2.5,
            new_content_norm=1.8,
        )
        d = e.model_dump()
        assert d["recipient_subspace_energy_fraction"] == 0.3
        assert d["new_content_norm"] == 1.8

    def test_subspace_surgery_result(self) -> None:
        r = SubspaceSurgeryResult(
            recipient_prompt="test",
            mode="donor",
            layer=5,
            subspace_name="s",
            subspace_rank=3,
            donor_prompt="d",
            recipient_baseline=[TokenPrediction(token="a", token_id=0, probability=0.5, rank=1)],
            surgical_output=[TokenPrediction(token="b", token_id=1, probability=0.6, rank=1)],
            generated_text="output",
            num_generated_tokens=5,
            verification=SurgeryVerification(
                orthogonal_cosine=1.0,
                orthogonal_norm_ratio=1.0,
                surgery_clean=True,
            ),
            energy_analysis=SubspaceEnergyAnalysis(
                recipient_subspace_energy_fraction=0.3,
                new_content_energy_fraction=0.4,
                recipient_subspace_norm=1.5,
                recipient_orthogonal_norm=2.5,
                new_content_norm=1.8,
            ),
            summary={"mode": "donor"},
        )
        d = r.model_dump()
        assert d["mode"] == "donor"
        assert d["subspace_rank"] == 3
        assert len(d["recipient_baseline"]) == 1
        assert d["verification"]["surgery_clean"] is True
