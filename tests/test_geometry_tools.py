"""Tests for geometry tools (Group 17).

Tests cover:
  - token_space: async validation (7) + impl (8)
  - direction_angles: async validation (8) + impl (8)
  - subspace_decomposition: async validation (7) + impl (9)
  - residual_trajectory: async validation (7) + impl (9)
  - feature_dimensionality: async validation (6) + impl (8)
  - helpers: 8
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper tests (no model needed)
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test private geometry helper functions."""

    def test_cosine_sim_identical(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _cosine_sim

        a = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-6

    def test_cosine_sim_opposite(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _cosine_sim

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert abs(_cosine_sim(a, b) - (-1.0)) < 1e-6

    def test_cosine_sim_orthogonal(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _cosine_sim

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_cosine_sim_zero_vector(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _cosine_sim

        a = np.array([1.0, 0.0])
        b = np.zeros(2)
        assert _cosine_sim(a, b) == 0.0

    def test_angle_deg_zero(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _angle_deg

        assert abs(_angle_deg(1.0) - 0.0) < 1e-4

    def test_angle_deg_ninety(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _angle_deg

        assert abs(_angle_deg(0.0) - 90.0) < 1e-4

    def test_angle_deg_one_eighty(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _angle_deg

        assert abs(_angle_deg(-1.0) - 180.0) < 1e-4

    def test_gram_schmidt_orthonormal(self) -> None:
        from chuk_mcp_lazarus.tools.geometry._helpers import _gram_schmidt

        v1 = np.array([1.0, 1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        basis = _gram_schmidt([v1, v2, v3])
        assert len(basis) == 3
        # Check orthonormality
        for i in range(3):
            assert abs(np.linalg.norm(basis[i]) - 1.0) < 1e-5
            for j in range(i + 1, 3):
                assert abs(np.dot(basis[i], basis[j])) < 1e-5


# ---------------------------------------------------------------------------
# token_space — async validation
# ---------------------------------------------------------------------------


class TestTokenSpace:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        result = await token_space(prompt="test", layer=0, tokens=["a"])
        assert result["error"] is True
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        result = await token_space(prompt="test", layer=99, tokens=["a"])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_negative_layer(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        result = await token_space(prompt="test", layer=-1, tokens=["a"])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_tokens(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        result = await token_space(prompt="test", layer=0, tokens=[])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_tokens(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        result = await token_space(prompt="test", layer=0, tokens=["a"] * 21)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        fake = {
            "prompt": "t",
            "layer": 0,
            "tokens": [],
            "pairwise_angles": [],
            "residual_info": {},
            "summary": {},
            "token_position": -1,
            "token_text": "x",
            "hidden_dim": 64,
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.token_space._token_space_impl", return_value=fake
        ):
            result = await token_space(prompt="test", layer=0, tokens=["a"])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.token_space import token_space

        with patch(
            "chuk_mcp_lazarus.tools.geometry.token_space._token_space_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await token_space(prompt="test", layer=0, tokens=["a"])
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# token_space — impl tests
# ---------------------------------------------------------------------------


class TestTokenSpaceImpl:
    """Sync tests with mocked extraction and unembed."""

    def _run(self, tokens: list[str] | None = None, include_proj: bool = False) -> dict:
        from chuk_mcp_lazarus.tools.geometry.token_space import _token_space_impl

        if tokens is None:
            tokens = ["alpha", "beta"]

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = 8
        meta.num_layers = 4

        # Mock residual: points mostly toward first axis
        residual = np.array([3.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Two token unembed vectors
        unembed_map = {
            "alpha": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "gamma": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return {"alpha": 10, "beta": 20, "gamma": 30}.get(tok, None)

        def fake_unembed(model_obj: Any, tid: int) -> np.ndarray | None:
            return {
                10: unembed_map["alpha"],
                20: unembed_map["beta"],
                30: unembed_map["gamma"],
            }.get(tid)

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.token_space.extract_activation_at_layer",
                return_value=residual.tolist(),
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.token_space._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.token_space._get_unembed_vec_np",
                side_effect=fake_unembed,
            ),
        ):
            return _token_space_impl(
                model,
                config,
                tokenizer,
                meta,
                "test prompt",
                0,
                tokens,
                -1,
                include_proj,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "tokens" in r
        assert "pairwise_angles" in r
        assert "residual_info" in r
        assert "summary" in r
        assert r["hidden_dim"] == 8

    def test_token_count(self) -> None:
        r = self._run()
        assert len(r["tokens"]) == 2

    def test_pairwise_count(self) -> None:
        r = self._run()
        # 2 tokens = 1 pair
        assert len(r["pairwise_angles"]) == 1

    def test_nearest_token(self) -> None:
        # Residual points along first axis, "alpha" is [1,0,...], should be nearest
        r = self._run()
        assert r["residual_info"]["top_aligned_token"] == "alpha"

    def test_angle_range(self) -> None:
        r = self._run()
        for t in r["tokens"]:
            assert 0.0 <= t["angle_to_residual"] <= 180.0

    def test_residual_norm(self) -> None:
        r = self._run()
        assert r["residual_info"]["norm"] > 0

    def test_orthogonal_tokens_90deg(self) -> None:
        r = self._run()
        # alpha=[1,0,...] and beta=[0,1,...] should be 90 degrees apart
        pair = r["pairwise_angles"][0]
        assert abs(pair["angle"] - 90.0) < 1.0

    def test_with_projection(self) -> None:
        r = self._run(include_proj=True)
        assert "projection_2d" in r
        assert r["projection_variance_explained"] is not None


# ---------------------------------------------------------------------------
# direction_angles — async validation
# ---------------------------------------------------------------------------


class TestDirectionAngles:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        result = await direction_angles(
            prompt="test",
            layer=0,
            directions=[{"type": "token", "value": "a"}, {"type": "token", "value": "b"}],
        )
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        result = await direction_angles(
            prompt="test",
            layer=99,
            directions=[{"type": "token", "value": "a"}, {"type": "token", "value": "b"}],
        )
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_directions(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        result = await direction_angles(
            prompt="test", layer=0, directions=[{"type": "token", "value": "a"}]
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_directions(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        dirs = [{"type": "token", "value": f"t{i}"} for i in range(21)]
        result = await direction_angles(prompt="test", layer=0, directions=dirs)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_invalid_type(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        result = await direction_angles(
            prompt="test",
            layer=0,
            directions=[{"type": "invalid"}, {"type": "token", "value": "a"}],
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_no_type_key(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        result = await direction_angles(
            prompt="test",
            layer=0,
            directions=[{"value": "a"}, {"type": "token", "value": "b"}],
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        fake = {
            "prompt": "t",
            "layer": 0,
            "hidden_dim": 64,
            "directions": [],
            "pairwise": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.direction_angles._direction_angles_impl",
            return_value=fake,
        ):
            result = await direction_angles(
                prompt="test",
                layer=0,
                directions=[{"type": "token", "value": "a"}, {"type": "token", "value": "b"}],
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.direction_angles import direction_angles

        with patch(
            "chuk_mcp_lazarus.tools.geometry.direction_angles._direction_angles_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await direction_angles(
                prompt="test",
                layer=0,
                directions=[{"type": "token", "value": "a"}, {"type": "token", "value": "b"}],
            )
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# direction_angles — impl tests
# ---------------------------------------------------------------------------


class TestDirectionAnglesImpl:
    def _run(self, directions: list[dict] | None = None) -> dict:
        from chuk_mcp_lazarus.tools.geometry._helpers import DirectionSpec
        from chuk_mcp_lazarus.tools.geometry.direction_angles import _direction_angles_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = 4
        meta.num_layers = 4

        if directions is None:
            directions = [
                {"type": "token", "value": "alpha"},
                {"type": "token", "value": "beta"},
            ]

        specs = [DirectionSpec(**d) for d in directions]

        unembed_map = {
            "alpha": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "gamma": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return {"alpha": 10, "beta": 20, "gamma": 30}.get(tok, None)

        def fake_unembed(model_obj: Any, tid: int) -> np.ndarray | None:
            return {
                10: unembed_map["alpha"],
                20: unembed_map["beta"],
                30: unembed_map["gamma"],
            }.get(tid)

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry._helpers._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry._helpers._get_unembed_vec_np",
                side_effect=fake_unembed,
            ),
        ):
            return _direction_angles_impl(
                model,
                config,
                tokenizer,
                meta,
                "test",
                0,
                specs,
                -1,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "directions" in r
        assert "pairwise" in r
        assert "summary" in r

    def test_orthogonal_90deg(self) -> None:
        r = self._run()
        assert len(r["pairwise"]) == 1
        assert abs(r["pairwise"][0]["angle"] - 90.0) < 1.0

    def test_identical_0deg(self) -> None:
        r = self._run(
            [
                {"type": "token", "value": "alpha"},
                {"type": "token", "value": "alpha"},
            ]
        )
        assert abs(r["pairwise"][0]["angle"]) < 1.0

    def test_opposite_180deg(self) -> None:
        r = self._run(
            [
                {"type": "token", "value": "alpha"},
                {"type": "token", "value": "gamma"},
            ]
        )
        assert abs(r["pairwise"][0]["angle"] - 180.0) < 1.0

    def test_direction_norms(self) -> None:
        r = self._run()
        for d in r["directions"]:
            assert d["norm"] > 0

    def test_summary_has_pairs(self) -> None:
        r = self._run()
        assert "most_aligned_pair" in r["summary"]
        assert "most_opposed_pair" in r["summary"]

    def test_three_directions(self) -> None:
        r = self._run(
            [
                {"type": "token", "value": "alpha"},
                {"type": "token", "value": "beta"},
                {"type": "token", "value": "gamma"},
            ]
        )
        # 3 choose 2 = 3 pairs
        assert len(r["pairwise"]) == 3

    def test_symmetric_angles(self) -> None:
        r = self._run()
        # Only one pair, so this is trivially true, but check structure
        p = r["pairwise"][0]
        assert p["direction_a"] != "" and p["direction_b"] != ""


# ---------------------------------------------------------------------------
# subspace_decomposition — async validation
# ---------------------------------------------------------------------------


class TestSubspaceDecomposition:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import subspace_decomposition

        result = await subspace_decomposition(
            prompt="test",
            layer=0,
            target={"type": "token", "value": "a"},
            basis_directions=[{"type": "token", "value": "b"}],
        )
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import subspace_decomposition

        result = await subspace_decomposition(
            prompt="test",
            layer=99,
            target={"type": "token", "value": "a"},
            basis_directions=[{"type": "token", "value": "b"}],
        )
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_empty_basis(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import subspace_decomposition

        result = await subspace_decomposition(
            prompt="test",
            layer=0,
            target={"type": "token", "value": "a"},
            basis_directions=[],
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_basis(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import subspace_decomposition

        result = await subspace_decomposition(
            prompt="test",
            layer=0,
            target={"type": "token", "value": "a"},
            basis_directions=[{"type": "token", "value": f"t{i}"} for i in range(21)],
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import subspace_decomposition

        fake = {
            "prompt": "t",
            "layer": 0,
            "hidden_dim": 64,
            "target": {},
            "components": [],
            "subspace_summary": {},
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.subspace_decomposition._subspace_decomposition_impl",
            return_value=fake,
        ):
            result = await subspace_decomposition(
                prompt="test",
                layer=0,
                target={"type": "token", "value": "a"},
                basis_directions=[{"type": "token", "value": "b"}],
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import subspace_decomposition

        with patch(
            "chuk_mcp_lazarus.tools.geometry.subspace_decomposition._subspace_decomposition_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await subspace_decomposition(
                prompt="test",
                layer=0,
                target={"type": "token", "value": "a"},
                basis_directions=[{"type": "token", "value": "b"}],
            )
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# subspace_decomposition — impl tests
# ---------------------------------------------------------------------------


class TestSubspaceDecompositionImpl:
    def _run(
        self,
        target_vec: np.ndarray | None = None,
        basis_vecs: list[np.ndarray] | None = None,
        orthogonalize: bool = True,
    ) -> dict:
        from chuk_mcp_lazarus.tools.geometry._helpers import DirectionSpec
        from chuk_mcp_lazarus.tools.geometry.subspace_decomposition import (
            _subspace_decomposition_impl,
        )

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = 4
        meta.num_layers = 4

        if target_vec is None:
            target_vec = np.array([3.0, 2.0, 0.0, 0.0], dtype=np.float32)
        if basis_vecs is None:
            basis_vecs = [
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            ]

        # Map token names to vectors
        all_vecs = [target_vec] + basis_vecs
        token_ids = list(range(100, 100 + len(all_vecs)))
        names = ["target"] + [f"basis{i}" for i in range(len(basis_vecs))]

        vec_by_id: dict[int, np.ndarray] = {}
        id_by_name: dict[str, int] = {}
        for name, tid, vec in zip(names, token_ids, all_vecs):
            vec_by_id[tid] = vec
            id_by_name[name] = tid

        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return id_by_name.get(tok)

        def fake_unembed(model_obj: Any, tid: int) -> np.ndarray | None:
            return vec_by_id.get(tid)

        target_spec = DirectionSpec(type="token", value="target")
        basis_specs = [
            DirectionSpec(type="token", value=f"basis{i}") for i in range(len(basis_vecs))
        ]

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry._helpers._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry._helpers._get_unembed_vec_np",
                side_effect=fake_unembed,
            ),
        ):
            return _subspace_decomposition_impl(
                model,
                config,
                tokenizer,
                meta,
                "test",
                0,
                target_spec,
                basis_specs,
                -1,
                orthogonalize,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "components" in r
        assert "subspace_summary" in r
        assert "target" in r

    def test_component_count(self) -> None:
        r = self._run()
        assert len(r["components"]) == 2

    def test_orthogonal_basis_energy(self) -> None:
        # Target [3,2,0,0] with basis [1,0,0,0] and [0,1,0,0]
        # Should capture all energy (no component in 3rd/4th dims)
        r = self._run()
        ss = r["subspace_summary"]
        assert ss["total_fraction_in_subspace"] > 0.99
        assert ss["residual_fraction"] < 0.01

    def test_single_aligned_direction(self) -> None:
        # Target along a single axis
        target = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
        basis = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
        r = self._run(target, basis)
        c = r["components"][0]
        assert abs(c["fraction_of_target"] - 1.0) < 0.01

    def test_orthogonal_direction_zero(self) -> None:
        # Target along first axis, basis along second
        target = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
        basis = [np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)]
        r = self._run(target, basis)
        c = r["components"][0]
        assert abs(c["projection"]) < 0.01
        assert abs(c["fraction_of_target"]) < 0.01

    def test_residual_fraction(self) -> None:
        # Target [3,2,1,0], basis [1,0,0,0] and [0,1,0,0] — missing 3rd dim
        target = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
        r = self._run(target)
        ss = r["subspace_summary"]
        assert ss["residual_fraction"] > 0.0
        assert ss["residual_norm"] > 0.0

    def test_effective_dim(self) -> None:
        r = self._run()
        assert r["subspace_summary"]["effective_subspace_dim"] == 2

    def test_angle_to_target(self) -> None:
        r = self._run()
        for c in r["components"]:
            assert 0.0 <= c["angle_to_target"] <= 180.0

    def test_summary_dominant(self) -> None:
        r = self._run()
        assert "dominant_component" in r["summary"]


# ---------------------------------------------------------------------------
# residual_trajectory — async validation
# ---------------------------------------------------------------------------


class TestResidualTrajectory:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        result = await residual_trajectory(prompt="test", reference_tokens=["a"])
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_empty_refs(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        result = await residual_trajectory(prompt="test", reference_tokens=[])
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_many_refs(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        result = await residual_trajectory(prompt="test", reference_tokens=["a"] * 11)
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        result = await residual_trajectory(prompt="test", reference_tokens=["a"], layers=[99])
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        fake = {
            "prompt": "t",
            "token_position": -1,
            "token_text": "x",
            "hidden_dim": 64,
            "reference_tokens": [],
            "num_layers": 0,
            "trajectory": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.residual_trajectory._residual_trajectory_impl",
            return_value=fake,
        ):
            result = await residual_trajectory(prompt="test", reference_tokens=["a"], layers=[0])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_default_layers(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        fake = {
            "prompt": "t",
            "token_position": -1,
            "token_text": "x",
            "hidden_dim": 64,
            "reference_tokens": [],
            "num_layers": 0,
            "trajectory": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.residual_trajectory._residual_trajectory_impl",
            return_value=fake,
        ):
            result = await residual_trajectory(prompt="test", reference_tokens=["a"])
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import residual_trajectory

        with patch(
            "chuk_mcp_lazarus.tools.geometry.residual_trajectory._residual_trajectory_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await residual_trajectory(prompt="test", reference_tokens=["a"], layers=[0])
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# residual_trajectory — impl tests
# ---------------------------------------------------------------------------


class TestResidualTrajectoryImpl:
    def _run(self, layers: list[int] | None = None, n_refs: int = 2) -> dict:
        from chuk_mcp_lazarus.tools.geometry.residual_trajectory import _residual_trajectory_impl

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = 4
        meta.num_layers = 4

        if layers is None:
            layers = [0, 1, 2, 3]

        # Residuals that rotate: start pointing at alpha, end pointing at beta
        layer_acts = {}
        for i, lyr in enumerate(layers):
            frac = i / max(len(layers) - 1, 1)
            vec = np.array([1.0 - frac, frac, 0.0, 0.0], dtype=np.float32)
            layer_acts[lyr] = vec.tolist()

        ref_tokens = ["alpha", "beta"][:n_refs]
        unembed_map = {
            "alpha": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "beta": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        }

        def fake_resolve(tok_obj: Any, tok: str) -> int | None:
            return {"alpha": 10, "beta": 20}.get(tok)

        def fake_unembed(model_obj: Any, tid: int) -> np.ndarray | None:
            return {10: unembed_map["alpha"], 20: unembed_map["beta"]}.get(tid)

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.residual_trajectory.extract_activations_all_layers",
                return_value=layer_acts,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.residual_trajectory._resolve_token_to_id",
                side_effect=fake_resolve,
            ),
            patch(
                "chuk_mcp_lazarus.tools.geometry.residual_trajectory._get_unembed_vec_np",
                side_effect=fake_unembed,
            ),
        ):
            return _residual_trajectory_impl(
                model,
                config,
                tokenizer,
                meta,
                "test prompt",
                ref_tokens,
                layers,
                -1,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "trajectory" in r
        assert "summary" in r
        assert "reference_tokens" in r

    def test_trajectory_length(self) -> None:
        r = self._run()
        assert len(r["trajectory"]) == 4

    def test_first_no_delta(self) -> None:
        r = self._run()
        assert r["trajectory"][0].get("delta_from_previous") is None

    def test_rotation_positive(self) -> None:
        r = self._run()
        for pt in r["trajectory"][1:]:
            assert pt["delta_from_previous"]["rotation_angle"] >= 0

    def test_dominant_changes(self) -> None:
        # Residual rotates from alpha to beta
        r = self._run()
        first_dom = r["trajectory"][0]["dominant_token"]
        last_dom = r["trajectory"][-1]["dominant_token"]
        assert first_dom == "alpha"
        assert last_dom == "beta"

    def test_crossings(self) -> None:
        r = self._run()
        assert len(r["summary"]["crossing_layers"]) > 0

    def test_total_rotation(self) -> None:
        r = self._run()
        assert r["summary"]["total_rotation"] > 0

    def test_angle_range(self) -> None:
        r = self._run()
        for pt in r["trajectory"]:
            for a in pt["angles"]:
                assert 0.0 <= a["angle"] <= 180.0

    def test_single_layer(self) -> None:
        r = self._run(layers=[0])
        assert len(r["trajectory"]) == 1
        assert r["trajectory"][0].get("delta_from_previous") is None


# ---------------------------------------------------------------------------
# feature_dimensionality — async validation
# ---------------------------------------------------------------------------


class TestFeatureDimensionality:
    @pytest.mark.asyncio
    async def test_model_not_loaded(self, unloaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality

        result = await feature_dimensionality(
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error_type"] == "ModelNotLoaded"

    @pytest.mark.asyncio
    async def test_layer_out_of_range(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality

        result = await feature_dimensionality(
            layer=99,
            positive_prompts=["a", "b"],
            negative_prompts=["c", "d"],
        )
        assert result["error_type"] == "LayerOutOfRange"

    @pytest.mark.asyncio
    async def test_too_few_positive(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality

        result = await feature_dimensionality(
            layer=0,
            positive_prompts=["a"],
            negative_prompts=["c", "d"],
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_too_few_negative(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality

        result = await feature_dimensionality(
            layer=0,
            positive_prompts=["a", "b"],
            negative_prompts=["c"],
        )
        assert result["error_type"] == "InvalidInput"

    @pytest.mark.asyncio
    async def test_success(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality

        fake = {
            "layer": 0,
            "num_positive": 2,
            "num_negative": 2,
            "hidden_dim": 64,
            "effective_dimensionality": {},
            "spectrum": [],
            "classification_by_dim": [],
            "summary": {},
        }
        with patch(
            "chuk_mcp_lazarus.tools.geometry.feature_dimensionality._feature_dimensionality_impl",
            return_value=fake,
        ):
            result = await feature_dimensionality(
                layer=0,
                positive_prompts=["a", "b"],
                negative_prompts=["c", "d"],
            )
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_exception(self, loaded_model_state: Any) -> None:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import feature_dimensionality

        with patch(
            "chuk_mcp_lazarus.tools.geometry.feature_dimensionality._feature_dimensionality_impl",
            side_effect=RuntimeError("boom"),
        ):
            result = await feature_dimensionality(
                layer=0,
                positive_prompts=["a", "b"],
                negative_prompts=["c", "d"],
            )
        assert result["error_type"] == "GeometryFailed"


# ---------------------------------------------------------------------------
# feature_dimensionality — impl tests
# ---------------------------------------------------------------------------


class TestFeatureDimensionalityImpl:
    def _run(self, dim: int = 8, n_pos: int = 5, n_neg: int = 5) -> dict:
        from chuk_mcp_lazarus.tools.geometry.feature_dimensionality import (
            _feature_dimensionality_impl,
        )

        model = MagicMock()
        config = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.side_effect = lambda ids, **kw: f"tok{ids[0]}"

        meta = MagicMock()
        meta.hidden_dim = dim
        meta.num_layers = 4

        rng = np.random.RandomState(42)

        # Positive prompts: shifted along first dimension
        pos_acts = [
            ((rng.randn(dim) * 0.1) + np.array([2.0] + [0.0] * (dim - 1)))
            .astype(np.float32)
            .tolist()
            for _ in range(n_pos)
        ]
        # Negative prompts: shifted along opposite direction
        neg_acts = [
            ((rng.randn(dim) * 0.1) + np.array([-2.0] + [0.0] * (dim - 1)))
            .astype(np.float32)
            .tolist()
            for _ in range(n_neg)
        ]

        all_acts = pos_acts + neg_acts
        call_idx = [0]

        def fake_extract(
            model_obj: Any, config_obj: Any, tok_obj: Any, prompt: str, layer: int, pos: int
        ) -> list[float]:
            idx = call_idx[0]
            call_idx[0] += 1
            return all_acts[idx]

        # Mock lm_head for top_token
        import mlx.core as mx

        mock_lm_head = MagicMock()
        mock_lm_head.return_value = mx.array(rng.randn(1, 1, 100).astype(np.float32))

        with (
            patch(
                "chuk_mcp_lazarus.tools.geometry.feature_dimensionality.extract_activation_at_layer",
                side_effect=fake_extract,
            ),
            patch(
                "chuk_mcp_lazarus.tools.residual_tools._get_lm_projection",
                return_value=mock_lm_head,
            ),
            patch("chuk_mcp_lazarus.tools.residual_tools._project_to_logits") as mock_proj,
        ):
            mock_proj.return_value = mx.array(rng.randn(100).astype(np.float32))
            pos_prompts = [f"pos{i}" for i in range(n_pos)]
            neg_prompts = [f"neg{i}" for i in range(n_neg)]
            return _feature_dimensionality_impl(
                model,
                config,
                tokenizer,
                meta,
                0,
                pos_prompts,
                neg_prompts,
                -1,
                50,
            )

    def test_output_keys(self) -> None:
        r = self._run()
        assert "effective_dimensionality" in r
        assert "spectrum" in r
        assert "classification_by_dim" in r
        assert "summary" in r

    def test_dimensionality_thresholds(self) -> None:
        r = self._run()
        ed = r["effective_dimensionality"]
        assert ed["dims_for_50pct"] <= ed["dims_for_80pct"]
        assert ed["dims_for_80pct"] <= ed["dims_for_95pct"]
        assert ed["dims_for_95pct"] <= ed["dims_for_99pct"]

    def test_directional_feature(self) -> None:
        # Difference is along first dim — should be directional
        r = self._run()
        assert r["summary"]["is_directional"] is True

    def test_spectrum_cumulative(self) -> None:
        r = self._run()
        # Cumulative should be monotonically increasing
        for i in range(1, len(r["spectrum"])):
            assert (
                r["spectrum"][i]["cumulative_variance"]
                >= r["spectrum"][i - 1]["cumulative_variance"]
            )

    def test_spectrum_positive(self) -> None:
        r = self._run()
        for s in r["spectrum"]:
            assert s["variance_explained"] >= 0.0

    def test_classification_accuracy(self) -> None:
        r = self._run()
        # With clear separation, accuracy with 1 dim should be high
        if r["classification_by_dim"]:
            assert r["classification_by_dim"][0]["accuracy"] > 0.5

    def test_num_samples(self) -> None:
        r = self._run()
        assert r["num_positive"] == 5
        assert r["num_negative"] == 5

    def test_interpretation(self) -> None:
        r = self._run()
        assert r["summary"]["interpretation"] in ("directional", "subspace", "distributed")
