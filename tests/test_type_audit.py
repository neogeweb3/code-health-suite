"""Tests for the type_audit engine — Python type annotation coverage analyzer."""
from __future__ import annotations

import ast
import json
import os
import textwrap

import pytest

from code_health_suite.engines.type_audit import (
    AnyUsage,
    AuditResult,
    FileResult,
    FunctionInfo,
    TypeAnnotationVisitor,
    TypeIgnore,
    aggregate_results,
    analyze_file,
    classify_profile,
    compute_score,
    find_python_files,
    find_type_ignores,
    format_json,
    format_score,
    format_text,
    main,
    score_to_grade,
    SKIP_DIRS,
    SKIP_FUNCTION_NAMES,
)


# --- Helpers ---

def _visit(code: str, file_path: str = "test.py") -> TypeAnnotationVisitor:
    """Parse code and run the TypeAnnotationVisitor on it."""
    source = textwrap.dedent(code)
    tree = ast.parse(source, filename=file_path)
    lines = source.splitlines()
    visitor = TypeAnnotationVisitor(file_path, lines)
    visitor.visit(tree)
    return visitor


def _make_result(**kwargs) -> AuditResult:
    """Create an AuditResult with specified values."""
    defaults = dict(
        target=".",
        files_scanned=1,
        total_functions=10,
        fully_typed_functions=5,
        partially_typed_functions=3,
        untyped_functions=2,
        total_params=20,
        annotated_params=15,
        functions_with_return=7,
        any_count=0,
        type_ignore_count=0,
    )
    defaults.update(kwargs)
    return AuditResult(**defaults)


# ============================================================
# FunctionInfo data model
# ============================================================

class TestFunctionInfo:
    def test_fully_typed_all_annotated(self):
        fi = FunctionInfo("foo", "f.py", 1, has_return_annotation=True,
                          total_params=3, annotated_params=3)
        assert fi.fully_typed is True

    def test_fully_typed_no_params(self):
        fi = FunctionInfo("foo", "f.py", 1, has_return_annotation=True,
                          total_params=0, annotated_params=0)
        assert fi.fully_typed is True

    def test_not_fully_typed_missing_return(self):
        fi = FunctionInfo("foo", "f.py", 1, has_return_annotation=False,
                          total_params=2, annotated_params=2)
        assert fi.fully_typed is False

    def test_not_fully_typed_missing_params(self):
        fi = FunctionInfo("foo", "f.py", 1, has_return_annotation=True,
                          total_params=3, annotated_params=1)
        assert fi.fully_typed is False

    def test_param_coverage_all(self):
        fi = FunctionInfo("foo", "f.py", 1, total_params=4, annotated_params=4)
        assert fi.param_coverage == 1.0

    def test_param_coverage_partial(self):
        fi = FunctionInfo("foo", "f.py", 1, total_params=4, annotated_params=2)
        assert fi.param_coverage == 0.5

    def test_param_coverage_zero_params(self):
        fi = FunctionInfo("foo", "f.py", 1, total_params=0, annotated_params=0)
        assert fi.param_coverage == 1.0

    def test_param_coverage_none_annotated(self):
        fi = FunctionInfo("foo", "f.py", 1, total_params=5, annotated_params=0)
        assert fi.param_coverage == 0.0


# ============================================================
# AuditResult coverage properties
# ============================================================

class TestAuditResultProperties:
    def test_function_coverage(self):
        r = _make_result(total_functions=10, fully_typed_functions=7)
        assert r.function_coverage == 0.7

    def test_function_coverage_zero(self):
        r = _make_result(total_functions=0, fully_typed_functions=0)
        assert r.function_coverage == 1.0

    def test_param_coverage(self):
        r = _make_result(total_params=20, annotated_params=15)
        assert r.param_coverage == 0.75

    def test_param_coverage_zero(self):
        r = _make_result(total_params=0, annotated_params=0)
        assert r.param_coverage == 1.0

    def test_return_coverage(self):
        r = _make_result(total_functions=10, functions_with_return=6)
        assert r.return_coverage == 0.6

    def test_return_coverage_zero(self):
        r = _make_result(total_functions=0, functions_with_return=0)
        assert r.return_coverage == 1.0


# ============================================================
# TypeAnnotationVisitor — function detection
# ============================================================

class TestVisitorFunctions:
    def test_simple_function_untyped(self):
        v = _visit("def foo(x, y): pass")
        assert len(v.functions) == 1
        f = v.functions[0]
        assert f.name == "foo"
        assert f.total_params == 2
        assert f.annotated_params == 0
        assert f.has_return_annotation is False
        assert f.is_method is False
        assert f.unannotated_param_names == ("x", "y")

    def test_fully_typed_function(self):
        v = _visit("def foo(x: int, y: str) -> bool: pass")
        f = v.functions[0]
        assert f.total_params == 2
        assert f.annotated_params == 2
        assert f.has_return_annotation is True
        assert f.fully_typed is True
        assert f.unannotated_param_names == ()

    def test_partially_typed_function(self):
        v = _visit("def foo(x: int, y) -> None: pass")
        f = v.functions[0]
        assert f.annotated_params == 1
        assert f.total_params == 2
        assert f.unannotated_param_names == ("y",)

    def test_no_params_no_return(self):
        v = _visit("def foo(): pass")
        f = v.functions[0]
        assert f.total_params == 0
        assert f.has_return_annotation is False

    def test_no_params_with_return(self):
        v = _visit("def foo() -> int: pass")
        f = v.functions[0]
        assert f.total_params == 0
        assert f.has_return_annotation is True
        assert f.fully_typed is True


class TestVisitorMethods:
    def test_method_skips_self(self):
        v = _visit("""\
        class Foo:
            def bar(self, x: int) -> None: pass
        """)
        f = v.functions[0]
        assert f.is_method is True
        assert f.total_params == 1  # x only, self skipped
        assert f.annotated_params == 1
        assert f.name == "Foo.bar"

    def test_classmethod_skips_cls(self):
        v = _visit("""\
        class Foo:
            @classmethod
            def create(cls, name: str) -> None: pass
        """)
        f = v.functions[0]
        assert f.total_params == 1  # name only, cls skipped
        assert f.annotated_params == 1

    def test_method_no_params_besides_self(self):
        v = _visit("""\
        class Foo:
            def bar(self) -> int: pass
        """)
        f = v.functions[0]
        assert f.total_params == 0
        assert f.fully_typed is True

    def test_nested_class(self):
        v = _visit("""\
        class Outer:
            class Inner:
                def method(self, x) -> None: pass
        """)
        f = v.functions[0]
        assert f.name == "Inner.method"
        assert f.is_method is True


class TestVisitorDunders:
    def test_skip_repr(self):
        v = _visit("""\
        class Foo:
            def __repr__(self): return "Foo"
        """)
        assert len(v.functions) == 0

    def test_skip_str(self):
        v = _visit("""\
        class Foo:
            def __str__(self): return ""
        """)
        assert len(v.functions) == 0

    def test_skip_eq(self):
        v = _visit("""\
        class Foo:
            def __eq__(self, other): pass
        """)
        assert len(v.functions) == 0

    def test_keep_init(self):
        v = _visit("""\
        class Foo:
            def __init__(self, x: int) -> None: pass
        """)
        assert len(v.functions) == 1
        assert v.functions[0].name == "Foo.__init__"

    def test_keep_call(self):
        v = _visit("""\
        class Foo:
            def __call__(self, x) -> None: pass
        """)
        assert len(v.functions) == 1

    def test_all_skip_names_are_dunders(self):
        for name in SKIP_FUNCTION_NAMES:
            assert name.startswith("__") and name.endswith("__")


class TestVisitorAsyncFunctions:
    def test_async_function(self):
        v = _visit("async def fetch(url: str) -> bytes: pass")
        assert len(v.functions) == 1
        f = v.functions[0]
        assert f.name == "fetch"
        assert f.fully_typed is True

    def test_async_method(self):
        v = _visit("""\
        class Client:
            async def request(self, url: str) -> bytes: pass
        """)
        f = v.functions[0]
        assert f.is_method is True
        assert f.name == "Client.request"
        assert f.total_params == 1


class TestVisitorArgTypes:
    def test_vararg(self):
        v = _visit("def foo(*args: int) -> None: pass")
        f = v.functions[0]
        assert f.total_params == 1
        assert f.annotated_params == 1

    def test_kwarg(self):
        v = _visit("def foo(**kwargs: str) -> None: pass")
        f = v.functions[0]
        assert f.total_params == 1
        assert f.annotated_params == 1

    def test_vararg_unannotated(self):
        v = _visit("def foo(*args) -> None: pass")
        f = v.functions[0]
        assert f.total_params == 1
        assert f.annotated_params == 0
        assert f.unannotated_param_names == ("args",)

    def test_kwonly_args(self):
        v = _visit("def foo(*, key: str, value) -> None: pass")
        f = v.functions[0]
        assert f.total_params == 2
        assert f.annotated_params == 1
        assert f.unannotated_param_names == ("value",)

    def test_posonly_args(self):
        v = _visit("def foo(x: int, y: int, /) -> None: pass")
        f = v.functions[0]
        assert f.total_params == 2
        assert f.annotated_params == 2

    def test_mixed_arg_types(self):
        v = _visit("def foo(a: int, /, b, *args: str, key: bool, **kw) -> None: pass")
        f = v.functions[0]
        # a (posonly) + b (regular) + args (vararg) + key (kwonly) + kw (kwarg) = 5
        assert f.total_params == 5
        assert f.annotated_params == 3  # a, args, key
        assert set(f.unannotated_param_names) == {"b", "kw"}


class TestVisitorNestedFunctions:
    def test_nested_function_detected(self):
        v = _visit("""\
        def outer(x: int) -> None:
            def inner(y) -> str:
                pass
        """)
        assert len(v.functions) == 2
        names = [f.name for f in v.functions]
        assert "outer" in names
        assert "inner" in names

    def test_multiple_functions(self):
        v = _visit("""\
        def foo(x: int) -> int: pass
        def bar(y) -> None: pass
        def baz(z: str) -> str: pass
        """)
        assert len(v.functions) == 3


# ============================================================
# Any detection
# ============================================================

class TestAnyDetection:
    def test_any_in_param(self):
        v = _visit("""\
        from typing import Any
        def foo(x: Any) -> None: pass
        """)
        assert len(v.any_usages) == 1
        assert v.any_usages[0].line_number > 0

    def test_any_in_return(self):
        v = _visit("""\
        from typing import Any
        def foo() -> Any: pass
        """)
        assert len(v.any_usages) == 1

    def test_any_in_multiple_places(self):
        v = _visit("""\
        from typing import Any
        def foo(x: Any, y: Any) -> Any: pass
        """)
        assert len(v.any_usages) == 3

    def test_no_any(self):
        v = _visit("def foo(x: int) -> str: pass")
        assert len(v.any_usages) == 0

    def test_any_in_complex_type(self):
        v = _visit("""\
        from typing import Any, Dict
        def foo(x: Dict[str, Any]) -> None: pass
        """)
        assert len(v.any_usages) >= 1

    def test_any_in_vararg(self):
        v = _visit("""\
        from typing import Any
        def foo(*args: Any) -> None: pass
        """)
        assert len(v.any_usages) == 1

    def test_any_in_kwarg(self):
        v = _visit("""\
        from typing import Any
        def foo(**kwargs: Any) -> None: pass
        """)
        assert len(v.any_usages) == 1


# ============================================================
# find_type_ignores
# ============================================================

class TestFindTypeIgnores:
    def test_basic_type_ignore(self):
        result = find_type_ignores("x = 1  # type: ignore", "test.py")
        assert len(result) == 1
        assert result[0].line_number == 1
        assert result[0].codes == ""

    def test_type_ignore_with_code(self):
        result = find_type_ignores("x = 1  # type: ignore[assignment]", "test.py")
        assert len(result) == 1
        assert result[0].codes == "[assignment]"

    def test_type_ignore_with_multiple_codes(self):
        result = find_type_ignores("x = 1  # type: ignore[assignment, override]", "test.py")
        assert len(result) == 1
        assert "assignment" in result[0].codes

    def test_no_type_ignore(self):
        result = find_type_ignores("x = 1  # normal comment", "test.py")
        assert len(result) == 0

    def test_multiple_type_ignores(self):
        source = "x = 1  # type: ignore\ny = 2  # type: ignore[attr-defined]\nz = 3"
        result = find_type_ignores(source, "test.py")
        assert len(result) == 2
        assert result[0].line_number == 1
        assert result[1].line_number == 2

    def test_type_ignore_preserves_context(self):
        result = find_type_ignores("    x: int = foo()  # type: ignore", "test.py")
        assert "x: int = foo()" in result[0].context

    def test_type_ignore_with_extra_spaces(self):
        result = find_type_ignores("x = 1  #  type:  ignore", "test.py")
        assert len(result) == 1

    def test_type_ignore_file_path(self):
        result = find_type_ignores("x = 1  # type: ignore", "my/file.py")
        assert result[0].file_path == "my/file.py"


# ============================================================
# analyze_file
# ============================================================

class TestAnalyzeFile:
    def test_analyze_valid_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo(x: int) -> str: pass\n")
        result = analyze_file(str(f))
        assert result.parse_error == ""
        assert len(result.functions) == 1
        assert result.functions[0].name == "foo"

    def test_analyze_syntax_error(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(:\n")
        result = analyze_file(str(f))
        assert "SyntaxError" in result.parse_error
        assert len(result.functions) == 0

    def test_analyze_nonexistent_file(self):
        result = analyze_file("/nonexistent/file.py")
        assert result.parse_error != ""

    def test_analyze_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = analyze_file(str(f))
        assert result.parse_error == ""
        assert len(result.functions) == 0

    def test_analyze_with_type_ignores(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1  # type: ignore\ndef foo(): pass\n")
        result = analyze_file(str(f))
        assert len(result.type_ignores) == 1
        assert len(result.functions) == 1

    def test_analyze_with_any(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from typing import Any\ndef foo(x: Any) -> Any: pass\n")
        result = analyze_file(str(f))
        assert len(result.any_usages) == 2


# ============================================================
# find_python_files
# ============================================================

class TestFindPythonFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("pass")
        files = find_python_files(str(f))
        assert len(files) == 1
        assert files[0].endswith("test.py")

    def test_non_python_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        files = find_python_files(str(f))
        assert len(files) == 0

    def test_directory_scan(self, tmp_path):
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.py").write_text("pass")
        (tmp_path / "c.txt").write_text("not python")
        files = find_python_files(str(tmp_path))
        assert len(files) == 2

    def test_skip_dirs(self, tmp_path):
        sub = tmp_path / "__pycache__"
        sub.mkdir()
        (sub / "cached.py").write_text("pass")
        (tmp_path / "real.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("real.py")

    def test_skip_venv(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("pass")
        (tmp_path / "app.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_skip_egg_info(self, tmp_path):
        egg = tmp_path / "pkg.egg-info"
        egg.mkdir()
        (egg / "setup.py").write_text("pass")
        (tmp_path / "main.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        (sub / "core.py").write_text("pass")
        (tmp_path / "main.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert len(files) == 3

    def test_sorted_output(self, tmp_path):
        (tmp_path / "z.py").write_text("pass")
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "m.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert basenames == sorted(basenames)


# ============================================================
# aggregate_results
# ============================================================

class TestAggregateResults:
    def test_single_file_fully_typed(self):
        fr = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 1, has_return_annotation=True,
                         total_params=2, annotated_params=2),
        ])
        result = aggregate_results(".", [fr])
        assert result.files_scanned == 1
        assert result.total_functions == 1
        assert result.fully_typed_functions == 1
        assert result.untyped_functions == 0

    def test_single_file_untyped(self):
        fr = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 1, has_return_annotation=False,
                         total_params=2, annotated_params=0),
        ])
        result = aggregate_results(".", [fr])
        assert result.untyped_functions == 1
        assert result.fully_typed_functions == 0
        assert result.files_with_issues == 1

    def test_partially_typed(self):
        fr = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 1, has_return_annotation=True,
                         total_params=3, annotated_params=1),
        ])
        result = aggregate_results(".", [fr])
        assert result.partially_typed_functions == 1

    def test_parse_error_counted(self):
        fr = FileResult(path="bad.py", parse_error="SyntaxError")
        result = aggregate_results(".", [fr])
        assert result.parse_errors == 1
        assert result.files_scanned == 1

    def test_any_and_ignores_counted(self):
        fr = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 1, has_return_annotation=True,
                         total_params=0, annotated_params=0),
        ], any_usages=[
            AnyUsage("a.py", 1, "x: Any"),
            AnyUsage("a.py", 2, "y: Any"),
        ], type_ignores=[
            TypeIgnore("a.py", 3, "# type: ignore"),
        ])
        result = aggregate_results(".", [fr])
        assert result.any_count == 2
        assert result.type_ignore_count == 1
        assert result.files_with_issues == 1

    def test_multiple_files(self):
        fr1 = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 1, has_return_annotation=True,
                         total_params=2, annotated_params=2),
        ])
        fr2 = FileResult(path="b.py", functions=[
            FunctionInfo("bar", "b.py", 1, has_return_annotation=False,
                         total_params=1, annotated_params=0),
        ])
        result = aggregate_results(".", [fr1, fr2])
        assert result.files_scanned == 2
        assert result.total_functions == 2
        assert result.fully_typed_functions == 1
        assert result.untyped_functions == 1
        assert result.files_with_issues == 1  # only b.py

    def test_param_aggregation(self):
        fr = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 1, has_return_annotation=True,
                         total_params=3, annotated_params=2),
            FunctionInfo("bar", "a.py", 5, has_return_annotation=True,
                         total_params=2, annotated_params=2),
        ])
        result = aggregate_results(".", [fr])
        assert result.total_params == 5
        assert result.annotated_params == 4

    def test_empty_file_list(self):
        result = aggregate_results(".", [])
        assert result.files_scanned == 0
        assert result.total_functions == 0


# ============================================================
# Scoring
# ============================================================

class TestComputeScore:
    def test_perfect_score(self):
        r = _make_result(total_functions=10, fully_typed_functions=10,
                         total_params=20, annotated_params=20,
                         functions_with_return=10, any_count=0,
                         type_ignore_count=0)
        assert compute_score(r) == 100

    def test_no_functions(self):
        r = _make_result(total_functions=0)
        assert compute_score(r) == 100

    def test_zero_coverage(self):
        r = _make_result(total_functions=10, fully_typed_functions=0,
                         total_params=10, annotated_params=0,
                         functions_with_return=0, any_count=0,
                         type_ignore_count=0)
        assert compute_score(r) == 10  # only base points

    def test_any_penalty(self):
        perfect = _make_result(total_functions=10, fully_typed_functions=10,
                               total_params=20, annotated_params=20,
                               functions_with_return=10, any_count=0)
        with_any = _make_result(total_functions=10, fully_typed_functions=10,
                                total_params=20, annotated_params=20,
                                functions_with_return=10, any_count=3)
        assert compute_score(perfect) > compute_score(with_any)

    def test_any_penalty_capped(self):
        r3 = _make_result(total_functions=10, fully_typed_functions=10,
                          total_params=20, annotated_params=20,
                          functions_with_return=10, any_count=3)
        r100 = _make_result(total_functions=10, fully_typed_functions=10,
                            total_params=20, annotated_params=20,
                            functions_with_return=10, any_count=100)
        # Cap at 5, so r3 (penalty 3) and r100 (penalty 5) differ by 2
        assert compute_score(r3) - compute_score(r100) == 2

    def test_type_ignore_penalty(self):
        clean = _make_result(total_functions=10, fully_typed_functions=10,
                             total_params=20, annotated_params=20,
                             functions_with_return=10, type_ignore_count=0)
        dirty = _make_result(total_functions=10, fully_typed_functions=10,
                             total_params=20, annotated_params=20,
                             functions_with_return=10, type_ignore_count=4)
        assert compute_score(clean) > compute_score(dirty)

    def test_score_clamped_at_zero(self):
        # Even with extreme penalties, should not go below 0
        r = _make_result(total_functions=10, fully_typed_functions=0,
                         total_params=10, annotated_params=0,
                         functions_with_return=0, any_count=100,
                         type_ignore_count=100)
        assert compute_score(r) >= 0

    def test_score_clamped_at_100(self):
        r = _make_result(total_functions=10, fully_typed_functions=10,
                         total_params=20, annotated_params=20,
                         functions_with_return=10)
        assert compute_score(r) <= 100


class TestScoreToGrade:
    def test_grade_a(self):
        assert score_to_grade(100) == "A"
        assert score_to_grade(90) == "A"

    def test_grade_b(self):
        assert score_to_grade(89) == "B"
        assert score_to_grade(80) == "B"

    def test_grade_c(self):
        assert score_to_grade(79) == "C"
        assert score_to_grade(70) == "C"

    def test_grade_d(self):
        assert score_to_grade(69) == "D"
        assert score_to_grade(60) == "D"

    def test_grade_f(self):
        assert score_to_grade(59) == "F"
        assert score_to_grade(0) == "F"


class TestClassifyProfile:
    def test_no_functions(self):
        r = _make_result(total_functions=0)
        assert classify_profile(r) == "no_functions"

    def test_fully_typed(self):
        r = _make_result(total_functions=100, fully_typed_functions=96,
                         any_count=2)
        assert classify_profile(r) == "fully_typed"

    def test_well_typed(self):
        r = _make_result(total_functions=100, fully_typed_functions=85,
                         any_count=2)
        assert classify_profile(r) == "well_typed"

    def test_partially_typed(self):
        r = _make_result(total_functions=100, fully_typed_functions=55,
                         any_count=2)
        assert classify_profile(r) == "partially_typed"

    def test_any_heavy(self):
        r = _make_result(total_functions=10, fully_typed_functions=3,
                         any_count=5)
        assert classify_profile(r) == "any_heavy"

    def test_ignore_heavy(self):
        r = _make_result(total_functions=10, fully_typed_functions=3,
                         any_count=1, type_ignore_count=5)
        assert classify_profile(r) == "ignore_heavy"

    def test_untyped(self):
        r = _make_result(total_functions=100, fully_typed_functions=5,
                         any_count=2, type_ignore_count=0)
        assert classify_profile(r) == "untyped"

    def test_mixed(self):
        r = _make_result(total_functions=10, fully_typed_functions=2,
                         any_count=2, type_ignore_count=1)
        assert classify_profile(r) == "mixed"


# ============================================================
# Output formatting
# ============================================================

class TestFormatText:
    def test_no_functions(self):
        r = _make_result(total_functions=0, files_scanned=3)
        text = format_text(r)
        assert "No Python functions found" in text
        assert "Files scanned: 3" in text

    def test_summary_lines(self):
        r = _make_result()
        text = format_text(r)
        assert "Functions:" in text
        assert "Parameters:" in text
        assert "Return types:" in text
        assert "Any usages:" in text

    def test_parse_errors_shown(self):
        r = _make_result(parse_errors=2)
        text = format_text(r)
        assert "Parse errors: 2" in text

    def test_untyped_functions_listed(self):
        fr = FileResult(path="/proj/a.py", functions=[
            FunctionInfo("foo", "/proj/a.py", 10, has_return_annotation=False,
                         total_params=1, annotated_params=0,
                         unannotated_param_names=("x",)),
        ])
        r = aggregate_results("/proj", [fr])
        text = format_text(r)
        assert "foo" in text
        assert "no return type" in text
        assert "untyped params: x" in text


class TestFormatScore:
    def test_contains_score_and_grade(self):
        r = _make_result(total_functions=10, fully_typed_functions=10,
                         total_params=20, annotated_params=20,
                         functions_with_return=10)
        text = format_score(r)
        assert "Score:" in text
        assert "Grade:" in text
        assert "Profile:" in text

    def test_contains_coverage_breakdown(self):
        r = _make_result()
        text = format_score(r)
        assert "Function coverage:" in text
        assert "Parameter coverage:" in text
        assert "Return type coverage:" in text


class TestFormatJson:
    def test_valid_json(self):
        r = _make_result()
        output = format_json(r)
        data = json.loads(output)
        assert "summary" in data
        assert "untyped_functions" in data
        assert "any_usages" in data
        assert "type_ignores" in data

    def test_summary_fields(self):
        r = _make_result(total_functions=5, fully_typed_functions=3,
                         total_params=10, annotated_params=8)
        data = json.loads(format_json(r))
        assert data["summary"]["total_functions"] == 5
        assert data["summary"]["fully_typed"] == 3
        assert data["summary"]["total_params"] == 10
        assert data["summary"]["annotated_params"] == 8

    def test_includes_score_when_requested(self):
        r = _make_result()
        data = json.loads(format_json(r, include_score=True))
        assert "score" in data
        assert "value" in data["score"]
        assert "grade" in data["score"]
        assert "profile" in data["score"]

    def test_no_score_by_default(self):
        r = _make_result()
        data = json.loads(format_json(r))
        assert "score" not in data

    def test_untyped_functions_in_output(self):
        fr = FileResult(path="a.py", functions=[
            FunctionInfo("foo", "a.py", 5, has_return_annotation=False,
                         total_params=2, annotated_params=0,
                         unannotated_param_names=("x", "y")),
        ])
        r = aggregate_results(".", [fr])
        data = json.loads(format_json(r))
        assert len(data["untyped_functions"]) == 1
        assert data["untyped_functions"][0]["name"] == "foo"
        assert data["untyped_functions"][0]["unannotated"] == ["x", "y"]


# ============================================================
# CLI / main
# ============================================================

class TestMain:
    def test_nonexistent_target(self, capsys):
        ret = main(["/nonexistent/path"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_no_python_files(self, tmp_path, capsys):
        ret = main([str(tmp_path)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "No Python files" in captured.err

    def test_scan_directory(self, tmp_path, capsys):
        (tmp_path / "test.py").write_text("def foo(x: int) -> str: pass\n")
        ret = main([str(tmp_path)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Functions:" in captured.out

    def test_json_output(self, tmp_path, capsys):
        (tmp_path / "test.py").write_text("def foo(x): pass\n")
        ret = main([str(tmp_path), "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert "summary" in data

    def test_score_output(self, tmp_path, capsys):
        (tmp_path / "test.py").write_text("def foo(x: int) -> str: pass\n")
        ret = main([str(tmp_path), "--score"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Grade:" in captured.out

    def test_json_with_score(self, tmp_path, capsys):
        (tmp_path / "test.py").write_text("def foo() -> None: pass\n")
        ret = main([str(tmp_path), "--json", "--score"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert "score" in data

    def test_single_file_target(self, tmp_path, capsys):
        f = tmp_path / "test.py"
        f.write_text("def foo(x) -> None: pass\n")
        ret = main([str(f)])
        assert ret == 0


# ============================================================
# Integration: full pipeline
# ============================================================

class TestIntegration:
    def test_fully_typed_project(self, tmp_path):
        (tmp_path / "typed.py").write_text(textwrap.dedent("""\
            def add(x: int, y: int) -> int:
                return x + y

            def greet(name: str) -> str:
                return f"Hello {name}"
        """))
        files = find_python_files(str(tmp_path))
        results = [analyze_file(f) for f in files]
        audit = aggregate_results(str(tmp_path), results)
        assert audit.total_functions == 2
        assert audit.fully_typed_functions == 2
        assert audit.function_coverage == 1.0
        assert compute_score(audit) == 100
        assert classify_profile(audit) == "fully_typed"

    def test_untyped_project(self, tmp_path):
        (tmp_path / "untyped.py").write_text(textwrap.dedent("""\
            def add(x, y):
                return x + y

            def greet(name):
                return f"Hello {name}"
        """))
        files = find_python_files(str(tmp_path))
        results = [analyze_file(f) for f in files]
        audit = aggregate_results(str(tmp_path), results)
        assert audit.total_functions == 2
        assert audit.untyped_functions == 2
        assert audit.function_coverage == 0.0
        assert compute_score(audit) == 10
        assert classify_profile(audit) == "untyped"

    def test_mixed_project(self, tmp_path):
        (tmp_path / "mixed.py").write_text(textwrap.dedent("""\
            from typing import Any

            def typed(x: int) -> str:
                return str(x)

            def untyped(x, y):
                return x + y

            def any_typed(x: Any) -> Any:  # type: ignore
                return x
        """))
        files = find_python_files(str(tmp_path))
        results = [analyze_file(f) for f in files]
        audit = aggregate_results(str(tmp_path), results)
        assert audit.total_functions == 3
        assert audit.fully_typed_functions == 2  # typed + any_typed
        assert audit.untyped_functions == 1  # untyped
        assert audit.any_count == 2  # param + return in any_typed
        assert audit.type_ignore_count == 1

    def test_class_with_methods(self, tmp_path):
        (tmp_path / "cls.py").write_text(textwrap.dedent("""\
            class Calculator:
                def __init__(self, value: int) -> None:
                    self.value = value

                def __repr__(self) -> str:
                    return f"Calc({self.value})"

                def add(self, x: int) -> int:
                    return self.value + x

                def multiply(self, x):
                    return self.value * x
        """))
        files = find_python_files(str(tmp_path))
        results = [analyze_file(f) for f in files]
        audit = aggregate_results(str(tmp_path), results)
        # __repr__ is skipped, so 3 functions
        assert audit.total_functions == 3
        assert audit.fully_typed_functions == 2  # __init__ + add
        # multiply has no return and untyped param
        assert audit.untyped_functions == 1
