"""Tests for _serialize.py — serialization helpers."""

import numpy as np
import pytest

from chuk_mcp_lazarus._serialize import (
    cosine_similarity_matrix,
    hidden_state_to_list,
    mx_to_list,
    np_to_python,
    pca_2d,
    serialize_config,
    to_pylist,
)


class TestMxToList:
    """mx_to_list converts arrays to Python lists."""

    def test_mx_array(self) -> None:
        import mlx.core as mx

        arr = mx.array([1.0, 2.0, 3.0])
        result = mx_to_list(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_plain_list(self) -> None:
        result = mx_to_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_nested(self) -> None:
        import mlx.core as mx

        arr = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = mx_to_list(arr)
        assert result == [[1.0, 2.0], [3.0, 4.0]]


class TestToPylist:
    """to_pylist wraps .tolist()."""

    def test_mx_array(self) -> None:
        import mlx.core as mx

        arr = mx.array([1.0, 2.0])
        result = to_pylist(arr)
        assert result == [1.0, 2.0]

    def test_numpy_array(self) -> None:
        arr = np.array([1, 2, 3])
        result = to_pylist(arr)
        assert result == [1, 2, 3]


class TestNpToPython:
    """np_to_python converts numpy scalars to Python types."""

    def test_integer(self) -> None:
        assert np_to_python(np.int64(42)) == 42
        assert isinstance(np_to_python(np.int64(42)), int)

    def test_float(self) -> None:
        assert np_to_python(np.float32(3.14)) == pytest.approx(3.14, rel=1e-5)
        assert isinstance(np_to_python(np.float32(3.14)), float)

    def test_bool(self) -> None:
        assert np_to_python(np.bool_(True)) is True
        assert isinstance(np_to_python(np.bool_(True)), bool)

    def test_0d_array(self) -> None:
        val = np.array(7)
        result = np_to_python(val)
        assert result == 7

    def test_passthrough(self) -> None:
        assert np_to_python("hello") == "hello"
        assert np_to_python(None) is None
        assert np_to_python(42) == 42


class TestSerializeConfig:
    """serialize_config extracts standard fields."""

    def test_full_config(self) -> None:
        class FakeConfig:
            model_type = "test"
            vocab_size = 100
            hidden_size = 64
            num_hidden_layers = 4
            num_attention_heads = 4
            num_key_value_heads = 4
            intermediate_size = 256
            max_position_embeddings = 512
            head_dim = 16

        result = serialize_config(FakeConfig())
        assert result["model_type"] == "test"
        assert result["vocab_size"] == 100
        assert result["num_hidden_layers"] == 4
        assert len(result) == 9

    def test_partial_config(self) -> None:
        class PartialConfig:
            model_type = "partial"

        result = serialize_config(PartialConfig())
        assert result == {"model_type": "partial"}

    def test_empty_config(self) -> None:
        result = serialize_config(object())
        assert result == {}

    def test_numpy_values(self) -> None:
        class NpConfig:
            vocab_size = np.int64(1000)
            hidden_size = np.float32(256.0)

        result = serialize_config(NpConfig())
        assert result["vocab_size"] == 1000
        assert isinstance(result["vocab_size"], int)


class TestHiddenStateToList:
    """hidden_state_to_list handles 1D/2D/3D arrays."""

    def test_1d(self) -> None:
        import mlx.core as mx

        arr = mx.array([1.0, 2.0, 3.0])
        result = hidden_state_to_list(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_2d_last(self) -> None:
        import mlx.core as mx

        arr = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = hidden_state_to_list(arr, position=-1)
        assert result == [5.0, 6.0]

    def test_2d_first(self) -> None:
        import mlx.core as mx

        arr = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = hidden_state_to_list(arr, position=0)
        assert result == [1.0, 2.0]

    def test_3d(self) -> None:
        import mlx.core as mx

        arr = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
        result = hidden_state_to_list(arr, position=-1)
        assert result == [3.0, 4.0]

    def test_plain_list(self) -> None:
        result = hidden_state_to_list([1, 2, 3])
        assert result == [1, 2, 3]


class TestCosineSimilarityMatrix:
    """cosine_similarity_matrix returns correct pairwise similarities."""

    def test_identical_vectors(self) -> None:
        vecs = [[1.0, 0.0], [1.0, 0.0]]
        sim = cosine_similarity_matrix(vecs)
        assert sim[0][0] == pytest.approx(1.0, abs=1e-5)
        assert sim[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        sim = cosine_similarity_matrix(vecs)
        assert sim[0][1] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self) -> None:
        vecs = [[1.0, 0.0], [-1.0, 0.0]]
        sim = cosine_similarity_matrix(vecs)
        assert sim[0][1] == pytest.approx(-1.0, abs=1e-5)

    def test_shape(self) -> None:
        vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        sim = cosine_similarity_matrix(vecs)
        assert len(sim) == 3
        assert all(len(row) == 3 for row in sim)


class TestPca2d:
    """pca_2d projects to 2D."""

    def test_shape(self) -> None:
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = pca_2d(vecs)
        assert len(result) == 3
        assert all(len(point) == 2 for point in result)

    def test_single_vector(self) -> None:
        result = pca_2d([[1.0, 2.0, 3.0]])
        assert result == [[0.0, 0.0]]

    def test_two_vectors(self) -> None:
        result = pca_2d([[1.0, 0.0], [0.0, 1.0]])
        assert len(result) == 2
        assert all(len(point) == 2 for point in result)
