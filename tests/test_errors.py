"""Tests for errors.py — ToolError enum and make_error()."""

import pytest

from chuk_mcp_lazarus.errors import ToolError, ToolErrorResult, make_error


class TestToolError:
    """ToolError enum has all expected members."""

    def test_all_members(self) -> None:
        expected = {
            "MODEL_NOT_LOADED",
            "LAYER_OUT_OF_RANGE",
            "PROBE_NOT_FOUND",
            "VECTOR_NOT_FOUND",
            "INVALID_INPUT",
            "EXTRACTION_FAILED",
            "TRAINING_FAILED",
            "EVALUATION_FAILED",
            "GENERATION_FAILED",
            "ABLATION_FAILED",
            "COMPARISON_FAILED",
            "COMPARISON_INCOMPATIBLE",
            "EXPERIMENT_NOT_FOUND",
            "EXPERIMENT_STORE_ERROR",
            "INTERVENTION_FAILED",
            "LOAD_FAILED",
        }
        assert set(ToolError.__members__.keys()) == expected

    def test_count(self) -> None:
        assert len(ToolError) == 16

    @pytest.mark.parametrize("member", list(ToolError))
    def test_string_value(self, member: ToolError) -> None:
        assert isinstance(member.value, str)
        assert len(member.value) > 0


class TestToolErrorResult:
    """ToolErrorResult Pydantic model."""

    def test_defaults(self) -> None:
        r = ToolErrorResult(
            error_type=ToolError.MODEL_NOT_LOADED,
            message="test",
            tool="test_tool",
        )
        assert r.error is True

    def test_model_dump(self) -> None:
        r = ToolErrorResult(
            error=True,
            error_type=ToolError.INVALID_INPUT,
            message="bad input",
            tool="my_tool",
        )
        d = r.model_dump()
        assert d["error"] is True
        assert d["error_type"] == "InvalidInput"
        assert d["message"] == "bad input"
        assert d["tool"] == "my_tool"


class TestMakeError:
    """make_error() returns a well-formed dict."""

    @pytest.mark.parametrize("error_type", list(ToolError))
    def test_all_error_types(self, error_type: ToolError) -> None:
        result = make_error(error_type, "msg", "tool_name")
        assert isinstance(result, dict)
        assert result["error"] is True
        assert result["error_type"] == error_type.value
        assert result["message"] == "msg"
        assert result["tool"] == "tool_name"

    def test_keys(self) -> None:
        result = make_error(ToolError.MODEL_NOT_LOADED, "x", "y")
        assert set(result.keys()) == {"error", "error_type", "message", "tool"}
