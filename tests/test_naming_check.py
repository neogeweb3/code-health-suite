"""Tests for the naming convention checker engine."""
import os
import textwrap
import pytest

from code_health_suite.engines import naming_check


# --- Helper ---

def _write_py(tmp_path, code: str, name: str = "sample.py") -> str:
    """Write a Python source file and return its path."""
    filepath = tmp_path / name
    filepath.write_text(textwrap.dedent(code))
    return str(filepath)


# --- is_snake_case ---

class TestIsSnakeCase:
    def test_valid_snake(self):
        assert naming_check.is_snake_case("my_function")
        assert naming_check.is_snake_case("_private")
        assert naming_check.is_snake_case("__dunder__")
        assert naming_check.is_snake_case("x")
        assert naming_check.is_snake_case("_")
        assert naming_check.is_snake_case("__")
        assert naming_check.is_snake_case("my_func_2")
        assert naming_check.is_snake_case("_leading")

    def test_invalid_snake(self):
        assert not naming_check.is_snake_case("myFunction")
        assert not naming_check.is_snake_case("MyFunction")
        assert not naming_check.is_snake_case("myFunc")
        assert not naming_check.is_snake_case("HTTPClient")


# --- is_camel_case ---

class TestIsCamelCase:
    def test_valid_camel(self):
        assert naming_check.is_camel_case("MyClass")
        assert naming_check.is_camel_case("HTTPClient")
        assert naming_check.is_camel_case("A")
        assert naming_check.is_camel_case("_Private")

    def test_invalid_camel(self):
        assert not naming_check.is_camel_case("my_class")
        assert not naming_check.is_camel_case("myClass")


# --- is_upper_snake_case ---

class TestIsUpperSnakeCase:
    def test_valid_upper(self):
        assert naming_check.is_upper_snake_case("MY_CONSTANT")
        assert naming_check.is_upper_snake_case("MAX_SIZE")
        assert naming_check.is_upper_snake_case("X")
        assert naming_check.is_upper_snake_case("_PRIVATE_CONST")

    def test_invalid_upper(self):
        assert not naming_check.is_upper_snake_case("My_Constant")
        assert not naming_check.is_upper_snake_case("myConstant")


# --- to_snake_case ---

class TestToSnakeCase:
    def test_camel_to_snake(self):
        assert naming_check.to_snake_case("myFunction") == "my_function"
        assert naming_check.to_snake_case("MyClass") == "my_class"
        assert naming_check.to_snake_case("HTTPClient") == "http_client"
        assert naming_check.to_snake_case("getHTTPResponse") == "get_http_response"

    def test_preserves_leading_underscore(self):
        assert naming_check.to_snake_case("_MyPrivate") == "_my_private"
        assert naming_check.to_snake_case("__internal") == "__internal"

    def test_already_snake(self):
        assert naming_check.to_snake_case("already_snake") == "already_snake"


# --- to_camel_case ---

class TestToCamelCase:
    def test_snake_to_camel(self):
        assert naming_check.to_camel_case("my_class") == "MyClass"
        assert naming_check.to_camel_case("http_client") == "HttpClient"

    def test_preserves_leading_underscore(self):
        assert naming_check.to_camel_case("_my_class") == "_MyClass"


# --- analyze_file ---

class TestAnalyzeFile:
    def test_clean_file(self, tmp_path):
        code = """\
        def my_function():
            pass

        class MyClass:
            def my_method(self, my_param):
                pass

        MY_CONSTANT = 42
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []
        assert result.names_checked > 0

    def test_bad_function_name(self, tmp_path):
        code = """\
        def myFunction():
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v.name == "myFunction"
        assert v.kind == "function"
        assert v.convention == "snake_case"
        assert "my_function" in v.suggestion

    def test_bad_class_name(self, tmp_path):
        code = """\
        class my_class:
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v.name == "my_class"
        assert v.kind == "class"
        assert v.convention == "CamelCase"

    def test_bad_method_name(self, tmp_path):
        code = """\
        class MyClass:
            def badMethod(self):
                pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v.name == "badMethod"
        assert v.kind == "method"

    def test_bad_parameter_name(self, tmp_path):
        code = """\
        def func(goodParam):
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        # function name is fine, parameter is bad
        param_violations = [v for v in result.violations if v.kind == "parameter"]
        assert len(param_violations) == 1
        assert param_violations[0].name == "goodParam"

    def test_bad_variable_name(self, tmp_path):
        code = """\
        myVariable = 42
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        var_violations = [v for v in result.violations if v.kind == "variable"]
        assert len(var_violations) == 1
        assert var_violations[0].name == "myVariable"

    def test_dunder_skipped(self, tmp_path):
        code = """\
        class MyClass:
            def __init__(self):
                pass

            def __repr__(self):
                pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_single_char_allowed(self, tmp_path):
        code = """\
        def f():
            pass

        x = 1
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_self_cls_skipped(self, tmp_path):
        code = """\
        class MyClass:
            def method(self):
                pass

            @classmethod
            def factory(cls):
                pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_module_dunder_skipped(self, tmp_path):
        code = """\
        __version__ = "1.0.0"
        __all__ = ["MyClass"]
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_async_function(self, tmp_path):
        code = """\
        async def badAsyncFunc():
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert len(result.violations) == 1
        assert result.violations[0].name == "badAsyncFunc"

    def test_annotated_assignment(self, tmp_path):
        code = """\
        myVar: int = 42
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        var_violations = [v for v in result.violations if v.kind == "variable"]
        assert len(var_violations) == 1
        assert var_violations[0].name == "myVar"

    def test_constant_valid(self, tmp_path):
        code = """\
        MAX_RETRIES = 3
        DEFAULT_TIMEOUT = 30
        _INTERNAL_LIMIT = 100
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_syntax_error_handled(self, tmp_path):
        path = _write_py(tmp_path, "def broken(:\n")
        result = naming_check.analyze_file(path)
        assert result.error
        assert "SyntaxError" in result.error

    def test_multiple_violations(self, tmp_path):
        code = """\
        def badFunc():
            pass

        class bad_class:
            def badMethod(self, badParam):
                pass

        myVar = 1
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert len(result.violations) >= 4  # func, class, method, param (var may or may not)
        kinds = {v.kind for v in result.violations}
        assert "function" in kinds
        assert "class" in kinds
        assert "method" in kinds
        assert "parameter" in kinds


# --- scan ---

class TestScan:
    def test_scan_directory(self, tmp_path):
        _write_py(tmp_path, "def goodFunc():\n    pass\n", "a.py")
        _write_py(tmp_path, "def good_func():\n    pass\n", "b.py")
        result = naming_check.scan(str(tmp_path))
        assert result.files_scanned == 2
        assert result.total_violations == 1  # goodFunc in a.py

    def test_scan_single_file(self, tmp_path):
        path = _write_py(tmp_path, "class my_bad:\n    pass\n")
        result = naming_check.scan(path)
        assert result.files_scanned == 1
        assert result.total_violations == 1

    def test_skips_pycache(self, tmp_path):
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        _write_py(tmp_path, "def badFunc():\n    pass\n", "__pycache__/cached.py")
        _write_py(tmp_path, "def good_func():\n    pass\n", "main.py")
        result = naming_check.scan(str(tmp_path))
        assert result.files_scanned == 1
        assert result.total_violations == 0

    def test_by_kind_aggregation(self, tmp_path):
        code = """\
        def badFunc():
            pass

        class bad_class:
            pass
        """
        _write_py(tmp_path, code)
        result = naming_check.scan(str(tmp_path))
        assert "function" in result.by_kind
        assert "class" in result.by_kind


# --- compute_score ---

class TestComputeScore:
    def test_perfect_score(self, tmp_path):
        _write_py(tmp_path, "def good_func():\n    pass\n\nclass GoodClass:\n    pass\n")
        result = naming_check.scan(str(tmp_path))
        score = naming_check.compute_score(result)
        assert score.score == 100
        assert score.grade == "A"
        assert score.total_violations == 0

    def test_bad_score(self, tmp_path):
        code = """\
        def badFunc():
            pass

        def anotherBad():
            pass

        class yet_another:
            pass
        """
        _write_py(tmp_path, code)
        result = naming_check.scan(str(tmp_path))
        score = naming_check.compute_score(result)
        assert score.score < 100
        assert score.total_violations > 0
        assert score.violation_rate > 0

    def test_empty_dir(self, tmp_path):
        result = naming_check.scan(str(tmp_path))
        score = naming_check.compute_score(result)
        assert score.score == 100
        assert score.grade == "A"

    def test_grade_boundaries(self):
        # Test the grade function directly
        assert naming_check._score_to_grade(95) == "A"
        assert naming_check._score_to_grade(90) == "A"
        assert naming_check._score_to_grade(80) == "B"
        assert naming_check._score_to_grade(65) == "C"
        assert naming_check._score_to_grade(45) == "D"
        assert naming_check._score_to_grade(30) == "F"


# --- MCP integration ---

class TestMCPIntegration:
    def test_check_naming_tool(self, tmp_path):
        _write_py(tmp_path, "def badFunc():\n    pass\n")
        from code_health_suite.server import handle_check_naming
        result = handle_check_naming({"path": str(tmp_path)})
        assert result["total_violations"] == 1
        assert result["violations"][0]["name"] == "badFunc"
        assert result["violations"][0]["kind"] == "function"

    def test_get_naming_score_tool(self, tmp_path):
        _write_py(tmp_path, "def good_func():\n    pass\n")
        from code_health_suite.server import handle_get_naming_score
        result = handle_get_naming_score({"path": str(tmp_path)})
        assert result["score"] == 100
        assert result["grade"] == "A"

    def test_check_naming_bad_path(self):
        from code_health_suite.server import handle_check_naming
        result = handle_check_naming({"path": "/nonexistent/path"})
        assert "error" in result

    def test_full_health_check_includes_naming(self, tmp_path):
        _write_py(tmp_path, "def good_func():\n    pass\n")
        from code_health_suite.server import handle_full_health_check
        result = handle_full_health_check({"path": str(tmp_path)})
        assert "naming" in result
        assert "score" in result["naming"]
        assert "grade" in result["naming"]


# --- Edge cases ---

class TestEdgeCases:
    def test_decorator_names_not_checked(self, tmp_path):
        code = """\
        import functools

        @functools.lru_cache
        def cached_func():
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        # Only the function name is checked, not the decorator
        assert all(v.name != "lru_cache" for v in result.violations)

    def test_lambda_not_checked(self, tmp_path):
        code = """\
        my_func = lambda x: x + 1
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_nested_class(self, tmp_path):
        code = """\
        class Outer:
            class inner_bad:
                pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        class_violations = [v for v in result.violations if v.kind == "class"]
        assert len(class_violations) == 1
        assert class_violations[0].name == "inner_bad"

    def test_mixed_case_constant(self, tmp_path):
        """Module-level assignment with mixed case caught as variable."""
        code = """\
        badVariable = "hello"
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert any(v.name == "badVariable" for v in result.violations)

    def test_private_function_snake(self, tmp_path):
        code = """\
        def _private_func():
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        assert result.violations == []

    def test_kwonly_params(self, tmp_path):
        code = """\
        def func(*, badKwarg=None):
            pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        param_violations = [v for v in result.violations if v.kind == "parameter"]
        assert len(param_violations) == 1
        assert param_violations[0].name == "badKwarg"

    def test_empty_file(self, tmp_path):
        path = _write_py(tmp_path, "")
        result = naming_check.analyze_file(path)
        assert result.violations == []
        assert result.names_checked == 0

    def test_ast_visitor_methods_allowed(self, tmp_path):
        """visit_* methods from ast.NodeVisitor should not be flagged."""
        code = """\
        import ast

        class MyVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                pass

            def visit_Name(self, node):
                pass

            def generic_visit(self, node):
                pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        # Only the class name should be checked (and it's fine)
        method_violations = [v for v in result.violations if v.kind in ("method", "function")]
        assert method_violations == []

    def test_unittest_methods_allowed(self, tmp_path):
        """setUp/tearDown methods from unittest should not be flagged."""
        code = """\
        import unittest

        class TestSomething(unittest.TestCase):
            def setUp(self):
                pass

            def tearDown(self):
                pass

            def test_something(self):
                pass
        """
        path = _write_py(tmp_path, code)
        result = naming_check.analyze_file(path)
        method_violations = [v for v in result.violations if v.kind in ("method", "function")]
        assert method_violations == []
