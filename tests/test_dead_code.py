"""Tests for the dead_code engine — AST-based Python dead code detector.

Covers: data models, scoring, AST analysis (imports, functions, variables,
arguments, unreachable code), cross-module analysis, file discovery, and CLI.
"""
from __future__ import annotations

import ast
import json
import os
import textwrap
from pathlib import Path

import pytest

from code_health_suite.engines.dead_code import (
    # Constants
    CATEGORIES, SEVERITY_MAP, SEVERITY_ORDER, IGNORED_NAMES,
    DUNDER_METHODS, EXTERNAL_USE_DECORATORS,
    # Data models
    Finding, FileReport, ScanResult, DeadCodeProjectStats,
    # Scoring
    _score_to_grade, classify_dead_code_profile, compute_project_stats,
    format_score_text, format_score_json, _SEVERITY_WEIGHTS,
    # AST helpers
    NameCollector, _get_decorator_names, _has_external_decorator,
    # Detectors
    find_unused_imports, find_unused_functions,
    find_unused_variables, find_unused_arguments, find_unreachable_code,
    # Cross-module
    _path_to_module, _build_module_file_map, _resolve_import_target,
    build_cross_module_refs,
    # File discovery
    find_python_files, _is_test_file, _should_auto_cross_module,
    EXCLUDED_DIRS,
    # Integration
    analyze_file, scan, main,
    # Formatting
    format_severity, format_finding, print_summary,
)


# ============================================================
# Helper: parse code snippet into AST module
# ============================================================

def _parse(code: str) -> ast.Module:
    """Parse a dedented code string into an AST module."""
    return ast.parse(textwrap.dedent(code))


def _write_py(tmp_path: Path, code: str, name: str = "sample.py") -> str:
    """Write a Python source file and return its path."""
    filepath = tmp_path / name
    filepath.write_text(textwrap.dedent(code))
    return str(filepath)


# ============================================================
# 1. Data Models
# ============================================================

class TestFinding:
    def test_basic_creation(self):
        f = Finding(file="foo.py", line=10, category="unused-import",
                    name="os", message="'os' imported but never used")
        assert f.file == "foo.py"
        assert f.line == 10
        assert f.category == "unused-import"
        assert f.name == "os"

    def test_auto_severity_from_category(self):
        f = Finding(file="", line=1, category="unused-import", name="x", message="m")
        assert f.severity == "medium"

        f2 = Finding(file="", line=1, category="unused-function", name="x", message="m")
        assert f2.severity == "low"

        f3 = Finding(file="", line=1, category="unused-argument", name="x", message="m")
        assert f3.severity == "info"

    def test_explicit_severity_overrides(self):
        f = Finding(file="", line=1, category="unused-import", name="x",
                    message="m", severity="high")
        assert f.severity == "high"

    def test_unknown_category_defaults_to_low(self):
        f = Finding(file="", line=1, category="unknown-cat", name="x", message="m")
        assert f.severity == "low"

    def test_end_line(self):
        f = Finding(file="", line=5, category="unreachable-code", name="x",
                    message="m", end_line=10)
        assert f.end_line == 10

    def test_end_line_defaults_none(self):
        f = Finding(file="", line=5, category="unreachable-code", name="x", message="m")
        assert f.end_line is None


class TestFileReport:
    def test_defaults(self):
        r = FileReport(file="test.py")
        assert r.file == "test.py"
        assert r.findings == []
        assert r.error is None

    def test_with_error(self):
        r = FileReport(file="bad.py", error="SyntaxError")
        assert r.error == "SyntaxError"

    def test_with_findings(self):
        f = Finding(file="a.py", line=1, category="unused-import", name="os", message="m")
        r = FileReport(file="a.py", findings=[f])
        assert len(r.findings) == 1


class TestScanResult:
    def test_defaults(self):
        r = ScanResult()
        assert r.files_scanned == 0
        assert r.total_findings == 0
        assert r.by_category == {}
        assert r.by_severity == {}
        assert r.reports == []


class TestDeadCodeProjectStats:
    def test_creation(self):
        stats = DeadCodeProjectStats(
            score=85, grade="B", profile="import_heavy",
            files_scanned=10, files_with_findings=3, clean_file_pct=70.0,
            total_findings=5, density=0.5,
            by_category={"unused-import": 5},
            by_severity={"medium": 5},
            dominant_category="unused-import", dominant_pct=100.0,
        )
        assert stats.score == 85
        assert stats.grade == "B"
        assert stats.profile == "import_heavy"


# ============================================================
# 2. Scoring Functions
# ============================================================

class TestScoreToGrade:
    @pytest.mark.parametrize("score,expected", [
        (100, "A"), (95, "A"), (90, "A"),
        (89, "B"), (75, "B"),
        (74, "C"), (60, "C"),
        (59, "D"), (40, "D"),
        (39, "F"), (0, "F"),
    ])
    def test_grade_boundaries(self, score, expected):
        assert _score_to_grade(score) == expected


class TestClassifyDeadCodeProfile:
    def test_clean_project(self):
        result = ScanResult(total_findings=0, by_category={})
        profile, dominant, pct = classify_dead_code_profile(result)
        assert profile == "clean"

    def test_clean_with_few_findings(self):
        result = ScanResult(total_findings=2, by_category={"unused-import": 2})
        profile, dominant, pct = classify_dead_code_profile(result)
        assert profile == "clean"

    def test_import_heavy(self):
        result = ScanResult(total_findings=10,
                            by_category={"unused-import": 8, "unused-variable": 2})
        profile, dominant, pct = classify_dead_code_profile(result)
        assert profile == "import_heavy"
        assert dominant == "unused-import"
        assert pct == 80.0

    def test_function_heavy(self):
        result = ScanResult(total_findings=10,
                            by_category={"unused-function": 6, "unused-import": 4})
        profile, _, _ = classify_dead_code_profile(result)
        assert profile == "function_heavy"

    def test_variable_heavy(self):
        result = ScanResult(total_findings=10,
                            by_category={"unused-variable": 7, "unused-import": 3})
        profile, _, _ = classify_dead_code_profile(result)
        assert profile == "variable_heavy"

    def test_argument_heavy(self):
        result = ScanResult(total_findings=10,
                            by_category={"unused-argument": 8, "unused-import": 2})
        profile, _, _ = classify_dead_code_profile(result)
        assert profile == "argument_heavy"

    def test_mixed_no_dominant(self):
        result = ScanResult(total_findings=10,
                            by_category={"unused-import": 3, "unused-function": 3,
                                         "unused-variable": 4})
        profile, _, pct = classify_dead_code_profile(result)
        assert profile == "mixed"
        assert pct <= 50.0

    def test_unreachable_code_dominant_is_mixed(self):
        """unreachable-code maps to 'mixed' even when >50%."""
        result = ScanResult(total_findings=10,
                            by_category={"unreachable-code": 8, "unused-import": 2})
        profile, _, _ = classify_dead_code_profile(result)
        assert profile == "mixed"


class TestComputeProjectStats:
    def test_perfect_score(self):
        result = ScanResult(files_scanned=10, total_findings=0,
                            by_category={}, by_severity={}, reports=[])
        stats = compute_project_stats(result)
        assert stats.score == 100
        assert stats.grade == "A"
        assert stats.clean_file_pct == 100.0
        assert stats.density == 0.0

    def test_score_decreases_with_findings(self):
        reports = [FileReport(file="a.py",
                              findings=[Finding(file="a.py", line=1,
                                               category="unused-import",
                                               name="x", message="m")])]
        result = ScanResult(files_scanned=1, total_findings=1,
                            by_category={"unused-import": 1},
                            by_severity={"medium": 1},
                            reports=reports)
        stats = compute_project_stats(result)
        assert stats.score < 100
        assert stats.files_with_findings == 1
        assert stats.clean_file_pct == 0.0

    def test_many_findings_low_score(self):
        findings = [Finding(file="a.py", line=i, category="unused-import",
                           name=f"x{i}", message="m") for i in range(50)]
        reports = [FileReport(file="a.py", findings=findings)]
        result = ScanResult(files_scanned=1, total_findings=50,
                            by_category={"unused-import": 50},
                            by_severity={"medium": 50},
                            reports=reports)
        stats = compute_project_stats(result)
        assert stats.score == 0
        assert stats.grade == "F"

    def test_zero_files_no_division_error(self):
        result = ScanResult(files_scanned=0, total_findings=0)
        stats = compute_project_stats(result)
        assert stats.score == 100

    def test_severity_weights_affect_score(self):
        """High severity findings reduce score more than info."""
        r_high = ScanResult(files_scanned=10, total_findings=1,
                            by_category={"unused-import": 1},
                            by_severity={"high": 1}, reports=[])
        r_info = ScanResult(files_scanned=10, total_findings=1,
                            by_category={"unused-argument": 1},
                            by_severity={"info": 1}, reports=[])
        s_high = compute_project_stats(r_high)
        s_info = compute_project_stats(r_info)
        assert s_high.score < s_info.score


class TestFormatScoreText:
    def test_output_contains_key_fields(self):
        stats = DeadCodeProjectStats(
            score=85, grade="B", profile="import_heavy",
            files_scanned=10, files_with_findings=3, clean_file_pct=70.0,
            total_findings=5, density=0.5,
            by_category={"unused-import": 5},
            by_severity={"medium": 5},
            dominant_category="unused-import", dominant_pct=100.0,
        )
        text = format_score_text(stats)
        assert "85/100" in text
        assert "(B)" in text
        assert "import_heavy" in text
        assert "unused-import" in text

    def test_empty_categories(self):
        stats = DeadCodeProjectStats(
            score=100, grade="A", profile="clean",
            files_scanned=5, files_with_findings=0, clean_file_pct=100.0,
            total_findings=0, density=0.0,
            by_category={}, by_severity={},
            dominant_category="none", dominant_pct=0.0,
        )
        text = format_score_text(stats)
        assert "100/100" in text
        assert "By category" not in text


class TestFormatScoreJson:
    def test_valid_json(self):
        stats = DeadCodeProjectStats(
            score=90, grade="A", profile="clean",
            files_scanned=5, files_with_findings=0, clean_file_pct=100.0,
            total_findings=0, density=0.0,
            by_category={}, by_severity={},
            dominant_category="none", dominant_pct=0.0,
        )
        j = json.loads(format_score_json(stats))
        assert j["score"] == 90
        assert j["grade"] == "A"


# ============================================================
# 3. AST Helpers
# ============================================================

class TestNameCollector:
    def test_collects_simple_names(self):
        tree = _parse("x = 1\nprint(x)")
        c = NameCollector()
        c.visit(tree)
        assert "x" in c.names
        assert "print" in c.names

    def test_collects_attribute_names(self):
        tree = _parse("obj.method()")
        c = NameCollector()
        c.visit(tree)
        assert "method" in c.attr_names
        assert "obj" in c.names

    def test_collects_dotted_names(self):
        tree = _parse("os.path.join('a', 'b')")
        c = NameCollector()
        c.visit(tree)
        assert "os.path.join" in c.dotted_names
        assert "os.path" in c.dotted_names or "os" in c.names

    def test_nested_attribute(self):
        tree = _parse("a.b.c.d")
        c = NameCollector()
        c.visit(tree)
        assert "a.b.c.d" in c.dotted_names


class TestGetDecoratorNames:
    def test_simple_decorator(self):
        tree = _parse("""
        @property
        def x(self):
            return 1
        """)
        func = tree.body[0]
        names = _get_decorator_names(func.decorator_list)
        assert "property" in names

    def test_dotted_decorator(self):
        tree = _parse("""
        @app.route("/")
        def index():
            pass
        """)
        func = tree.body[0]
        names = _get_decorator_names(func.decorator_list)
        assert "app.route" in names

    def test_call_decorator(self):
        tree = _parse("""
        @pytest.mark.parametrize("x", [1, 2])
        def test_x(x):
            pass
        """)
        func = tree.body[0]
        names = _get_decorator_names(func.decorator_list)
        assert "pytest.mark.parametrize" in names

    def test_no_decorators(self):
        tree = _parse("""
        def plain():
            pass
        """)
        func = tree.body[0]
        names = _get_decorator_names(func.decorator_list)
        assert names == set()


class TestHasExternalDecorator:
    def test_property_is_external(self):
        tree = _parse("""
        @property
        def x(self):
            return 1
        """)
        assert _has_external_decorator(tree.body[0].decorator_list) is True

    def test_no_decorator(self):
        tree = _parse("""
        def x():
            pass
        """)
        assert _has_external_decorator(tree.body[0].decorator_list) is False

    def test_suffix_match(self):
        tree = _parse("""
        @flask_app.route("/")
        def index():
            pass
        """)
        assert _has_external_decorator(tree.body[0].decorator_list) is True

    def test_fixture_decorator(self):
        tree = _parse("""
        @pytest.fixture
        def db():
            pass
        """)
        assert _has_external_decorator(tree.body[0].decorator_list) is True


# ============================================================
# 4. find_unused_imports
# ============================================================

class TestFindUnusedImports:
    def test_unused_import(self):
        tree = _parse("""
        import os
        x = 1
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 1
        assert findings[0].name == "os"
        assert findings[0].category == "unused-import"

    def test_used_import_not_flagged(self):
        tree = _parse("""
        import os
        os.path.join("a", "b")
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_from_import_unused(self):
        tree = _parse("""
        from os.path import join
        x = 1
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 1
        assert findings[0].name == "join"

    def test_from_import_used(self):
        tree = _parse("""
        from os.path import join
        join("a", "b")
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_aliased_import_unused(self):
        tree = _parse("""
        import numpy as np
        x = 1
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 1
        assert findings[0].name == "np"

    def test_aliased_import_used(self):
        tree = _parse("""
        import numpy as np
        np.array([1])
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_future_import_ignored(self):
        tree = _parse("""
        from __future__ import annotations
        x: int = 1
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_star_import_ignored(self):
        tree = _parse("""
        from os import *
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_ignored_names_not_flagged(self):
        tree = _parse("""
        import __all__
        import __version__
        import _
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_all_list_makes_import_used(self):
        tree = _parse("""
        from os import getcwd
        __all__ = ["getcwd"]
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_multiple_unused(self):
        tree = _parse("""
        import os
        import sys
        import json
        x = 1
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 3
        names = {f.name for f in findings}
        assert names == {"os", "sys", "json"}

    def test_mixed_used_unused(self):
        tree = _parse("""
        import os
        import sys
        print(os.getcwd())
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 1
        assert findings[0].name == "sys"

    def test_import_os_path_base_name_used(self):
        """import os.path — usable name is 'os'."""
        tree = _parse("""
        import os.path
        os.path.join("a", "b")
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 0

    def test_import_os_path_unused(self):
        tree = _parse("""
        import os.path
        x = 1
        """)
        findings = find_unused_imports(tree)
        assert len(findings) == 1


# ============================================================
# 5. find_unused_functions
# ============================================================

class TestFindUnusedFunctions:
    def test_unused_function(self):
        tree = _parse("""
        def foo():
            pass
        x = 1
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 1
        assert findings[0].name == "foo"

    def test_used_function_not_flagged(self):
        tree = _parse("""
        def foo():
            return 1
        x = foo()
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 0

    def test_dunder_methods_ignored(self):
        tree = _parse("""
        class MyClass:
            def __init__(self):
                pass
            def __repr__(self):
                return "MyClass"
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 0

    def test_ignored_names_skipped(self):
        tree = _parse("""
        def _():
            pass
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 0

    def test_external_decorator_skipped(self):
        tree = _parse("""
        @property
        def value(self):
            return 1
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 0

    def test_wholesale_import_suppresses_all(self):
        tree = _parse("""
        def foo():
            pass
        def bar():
            pass
        """)
        findings = find_unused_functions(tree, is_wholesale_imported=True)
        assert len(findings) == 0

    def test_cross_module_names_suppress(self):
        tree = _parse("""
        def foo():
            pass
        def bar():
            pass
        """)
        findings = find_unused_functions(tree, cross_module_names={"foo"})
        assert len(findings) == 1
        assert findings[0].name == "bar"

    def test_test_file_skips_test_functions(self):
        tree = _parse("""
        def test_something():
            assert True
        def setup_module():
            pass
        def teardown_module():
            pass
        def helper():
            pass
        """)
        findings = find_unused_functions(tree, is_test_file=True)
        assert len(findings) == 1
        assert findings[0].name == "helper"

    def test_test_class_methods_skipped_in_test_file(self):
        tree = _parse("""
        class TestFoo:
            def test_bar(self):
                pass
            def setup_method(self):
                pass
        """)
        findings = find_unused_functions(tree, is_test_file=True)
        assert len(findings) == 0

    def test_unused_class_method(self):
        tree = _parse("""
        class MyClass:
            def public_method(self):
                pass
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 1
        assert "MyClass.public_method" in findings[0].name

    def test_class_method_used_via_attr(self):
        tree = _parse("""
        class MyClass:
            def do_thing(self):
                pass
        obj = MyClass()
        obj.do_thing()
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 0

    def test_visitor_pattern_class_skip_visit_methods(self):
        tree = _parse("""
        class MyVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                pass
            def generic_visit(self, node):
                pass
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 0

    def test_async_function_unused(self):
        tree = _parse("""
        async def afoo():
            pass
        x = 1
        """)
        findings = find_unused_functions(tree)
        assert len(findings) == 1
        assert findings[0].name == "afoo"

    def test_cross_module_class_name_suppresses_methods(self):
        tree = _parse("""
        class MyClass:
            def some_method(self):
                pass
        """)
        findings = find_unused_functions(tree, cross_module_names={"MyClass"})
        assert len(findings) == 0


# ============================================================
# 6. find_unused_variables
# ============================================================

class TestFindUnusedVariables:
    def test_unused_variable(self):
        tree = _parse("""
        def foo():
            x = 1
            return None
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 1
        assert findings[0].name == "x"

    def test_used_variable_not_flagged(self):
        tree = _parse("""
        def foo():
            x = 1
            return x
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0

    def test_underscore_prefix_ignored(self):
        tree = _parse("""
        def foo():
            _temp = 1
            return None
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0

    def test_ignored_names_skipped(self):
        tree = _parse("""
        def foo():
            _ = 1
            return None
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0

    def test_global_variable_not_flagged(self):
        tree = _parse("""
        counter = 0
        def foo():
            global counter
            counter = 1
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0

    def test_nonlocal_variable_not_flagged(self):
        tree = _parse("""
        def outer():
            x = 0
            def inner():
                nonlocal x
                x = 1
            inner()
            return x
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0

    def test_tuple_unpack_unused(self):
        tree = _parse("""
        def foo():
            x, y = 1, 2
            return x
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 1
        assert findings[0].name == "y"

    def test_annotated_assign_unused(self):
        tree = _parse("""
        def foo():
            x: int = 1
            return None
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 1
        assert findings[0].name == "x"

    def test_closure_read_counts_as_usage(self):
        tree = _parse("""
        def outer():
            x = 1
            def inner():
                return x
            return inner
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0

    def test_multiple_assignments_last_unused(self):
        tree = _parse("""
        def foo():
            result = compute()
            result = transform(result)
            return None
        """)
        findings = find_unused_variables(tree)
        # 'result' is assigned but the function reads it in transform(result)
        assert len(findings) == 0

    def test_module_level_not_flagged(self):
        """Module-level variables are not analyzed (only function bodies)."""
        tree = _parse("""
        x = 1
        """)
        findings = find_unused_variables(tree)
        assert len(findings) == 0


# ============================================================
# 7. find_unused_arguments
# ============================================================

class TestFindUnusedArguments:
    def test_unused_argument(self):
        tree = _parse("""
        def foo(x):
            return 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 1
        assert findings[0].name == "x"

    def test_used_argument_not_flagged(self):
        tree = _parse("""
        def foo(x):
            return x + 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_self_ignored(self):
        tree = _parse("""
        class C:
            def method(self):
                return 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_cls_ignored(self):
        tree = _parse("""
        class C:
            @classmethod
            def method(cls):
                return 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_underscore_prefix_ignored(self):
        tree = _parse("""
        def foo(_unused):
            return 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_pass_only_body_skipped(self):
        tree = _parse("""
        def foo(x):
            pass
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_ellipsis_body_skipped(self):
        tree = _parse("""
        def foo(x):
            ...
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_docstring_only_body_skipped(self):
        tree = _parse("""
        def foo(x):
            "A docstring."
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_docstring_plus_pass_skipped(self):
        tree = _parse("""
        def foo(x):
            "A docstring."
            pass
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_docstring_plus_ellipsis_skipped(self):
        tree = _parse("""
        def foo(x):
            "A docstring."
            ...
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_external_decorator_skipped(self):
        tree = _parse("""
        @staticmethod
        def foo(x):
            return 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_varargs_suppresses(self):
        tree = _parse("""
        def foo(x, *args):
            return sum(args)
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_kwargs_suppresses(self):
        tree = _parse("""
        def foo(x, **kwargs):
            return kwargs
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 0

    def test_kwonly_argument_unused(self):
        tree = _parse("""
        def foo(*, key):
            return 1
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 1
        assert findings[0].name == "key"

    def test_posonly_argument_unused(self):
        tree = _parse("""
        def foo(x, /, y):
            return y
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 1
        assert findings[0].name == "x"

    def test_async_function_unused_arg(self):
        tree = _parse("""
        async def handler(request):
            return "ok"
        """)
        findings = find_unused_arguments(tree)
        assert len(findings) == 1
        assert findings[0].name == "request"


# ============================================================
# 8. find_unreachable_code
# ============================================================

class TestFindUnreachableCode:
    def test_after_return(self):
        tree = _parse("""
        def foo():
            return 1
            x = 2
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1
        assert "return" in findings[0].message

    def test_after_raise(self):
        tree = _parse("""
        def foo():
            raise ValueError("oops")
            cleanup()
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1
        assert "raise" in findings[0].message

    def test_after_break(self):
        tree = _parse("""
        def foo():
            for i in range(10):
                break
                print(i)
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1
        assert "break" in findings[0].message

    def test_after_continue(self):
        tree = _parse("""
        def foo():
            for i in range(10):
                continue
                print(i)
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1
        assert "continue" in findings[0].message

    def test_no_unreachable(self):
        tree = _parse("""
        def foo():
            x = 1
            return x
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 0

    def test_string_after_return_not_flagged(self):
        """String constants after return might be docs, skip them."""
        tree = _parse("""
        def foo():
            return 1
            "This is documentation"
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 0

    def test_if_block_unreachable(self):
        tree = _parse("""
        def foo():
            if True:
                return 1
                x = 2
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_else_block_unreachable(self):
        tree = _parse("""
        def foo():
            if True:
                pass
            else:
                return 1
                x = 2
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_for_loop_unreachable(self):
        tree = _parse("""
        def foo():
            for i in range(10):
                return i
                x = 2
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_while_loop_unreachable(self):
        tree = _parse("""
        def foo():
            while True:
                return 1
                x = 2
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_with_block_unreachable(self):
        tree = _parse("""
        def foo():
            with open("f") as f:
                return f.read()
                extra()
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_except_block_unreachable(self):
        tree = _parse("""
        def foo():
            try:
                x = 1
            except Exception:
                raise
                cleanup()
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_try_block_unreachable(self):
        tree = _parse("""
        def foo():
            try:
                return 1
                x = 2
            except Exception:
                pass
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_finally_block_unreachable(self):
        tree = _parse("""
        def foo():
            try:
                pass
            finally:
                return 1
                x = 2
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_async_with_unreachable(self):
        """BUG-59 fix: async with blocks should be checked."""
        tree = _parse("""
        async def foo():
            async with something() as s:
                return s
                extra()
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1

    def test_end_line_set(self):
        tree = _parse("""
        def foo():
            return 1
            x = 2
            y = 3
        """)
        findings = find_unreachable_code(tree)
        assert len(findings) == 1
        assert findings[0].end_line is not None
        assert findings[0].end_line >= findings[0].line


# ============================================================
# 9. _is_test_file
# ============================================================

class TestIsTestFile:
    def test_test_prefix(self):
        assert _is_test_file("test_foo.py") is True

    def test_test_suffix(self):
        assert _is_test_file("foo_test.py") is True

    def test_tests_directory(self):
        assert _is_test_file("/project/tests/test_bar.py") is True

    def test_test_directory(self):
        assert _is_test_file("/project/test/helper.py") is True

    def test_conftest(self):
        assert _is_test_file("conftest.py") is True

    def test_normal_file(self):
        assert _is_test_file("main.py") is False

    def test_utils_file(self):
        assert _is_test_file("utils.py") is False


# ============================================================
# 10. Cross-Module Analysis
# ============================================================

class TestPathToModule:
    def test_simple_file(self, tmp_path):
        f = tmp_path / "mymodule.py"
        f.write_text("x = 1")
        mod = _path_to_module(str(f), str(tmp_path))
        assert mod == "mymodule"

    def test_package_init(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        f = pkg / "__init__.py"
        f.write_text("")
        mod = _path_to_module(str(f), str(tmp_path))
        assert mod == "mypkg"

    def test_nested_module(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        f = pkg / "sub.py"
        f.write_text("")
        mod = _path_to_module(str(f), str(tmp_path))
        assert mod == "mypkg.sub"

    def test_src_layout_strips_src(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        # No __init__.py in src = namespace dir, should be stripped
        pkg = src / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        f = pkg / "core.py"
        f.write_text("")
        mod = _path_to_module(str(f), str(tmp_path))
        assert mod == "mypkg.core"

    def test_src_layout_with_init_keeps_src(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "__init__.py").write_text("")  # src IS a package
        pkg = src / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        f = pkg / "core.py"
        f.write_text("")
        mod = _path_to_module(str(f), str(tmp_path))
        assert mod == "src.mypkg.core"

    def test_outside_root(self):
        mod = _path_to_module("/other/path/file.py", "/root")
        assert mod == "file"


class TestBuildModuleFileMap:
    def test_basic(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("")
        f2 = tmp_path / "b.py"
        f2.write_text("")
        m = _build_module_file_map([str(f1), str(f2)], str(tmp_path))
        assert "a" in m
        assert "b" in m


class TestResolveImportTarget:
    def test_direct_match(self):
        module_map = {"mymodule": "/path/mymodule.py"}
        result = _resolve_import_target("caller", "mymodule", 0, module_map)
        assert result == "/path/mymodule.py"

    def test_external_module(self):
        module_map = {"mymodule": "/path/mymodule.py"}
        result = _resolve_import_target("caller", "numpy", 0, module_map)
        assert result is None

    def test_relative_import(self):
        module_map = {"mypkg.sub": "/path/mypkg/sub.py"}
        result = _resolve_import_target("mypkg.main", "sub", 1, module_map)
        assert result == "/path/mypkg/sub.py"

    def test_package_prefix_match(self):
        module_map = {"mypkg.sub": "/path/mypkg/sub.py"}
        result = _resolve_import_target("caller", "mypkg", 0, module_map)
        assert result == "/path/mypkg/sub.py"

    def test_partial_resolution(self):
        module_map = {"foo.bar": "/path/foo/bar.py"}
        result = _resolve_import_target("caller", "foo.bar.baz", 0, module_map)
        assert result == "/path/foo/bar.py"


class TestBuildCrossModuleRefs:
    def test_basic_cross_ref(self, tmp_path):
        # a.py imports foo from b.py
        a = tmp_path / "a.py"
        a.write_text("from b import foo\nfoo()")
        b = tmp_path / "b.py"
        b.write_text("def foo(): pass")
        files = [str(a), str(b)]
        imported_names, wholesale = build_cross_module_refs(files, str(tmp_path))
        assert "foo" in imported_names.get(str(b), set())

    def test_star_import_marks_wholesale(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("from b import *")
        b = tmp_path / "b.py"
        b.write_text("def foo(): pass")
        files = [str(a), str(b)]
        imported_names, wholesale = build_cross_module_refs(files, str(tmp_path))
        assert str(b) in wholesale

    def test_import_module_marks_wholesale(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("import b\nb.foo()")
        b = tmp_path / "b.py"
        b.write_text("def foo(): pass")
        files = [str(a), str(b)]
        _, wholesale = build_cross_module_refs(files, str(tmp_path))
        assert str(b) in wholesale

    def test_syntax_error_file_skipped(self, tmp_path):
        a = tmp_path / "a.py"
        a.write_text("from b import foo\nfoo()")
        b = tmp_path / "b.py"
        b.write_text("def foo() pass")  # syntax error
        files = [str(a), str(b)]
        # Should not raise
        imported_names, _ = build_cross_module_refs(files, str(tmp_path))


# ============================================================
# 11. File Discovery
# ============================================================

class TestFindPythonFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        files = find_python_files(str(tmp_path))
        assert len(files) == 2

    def test_single_file(self, tmp_path):
        f = tmp_path / "single.py"
        f.write_text("")
        files = find_python_files(str(f))
        assert len(files) == 1

    def test_non_py_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("")
        files = find_python_files(str(f))
        assert len(files) == 0

    def test_excludes_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.py").write_text("")
        (tmp_path / "real.py").write_text("")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_excludes_venv(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "activate.py").write_text("")
        (tmp_path / "main.py").write_text("")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("")
        (tmp_path / "main.py").write_text("")
        files = find_python_files(str(tmp_path))
        assert len(files) == 2

    def test_sorted_output(self, tmp_path):
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        files = find_python_files(str(tmp_path))
        assert files == sorted(files)


class TestShouldAutoCrossModule:
    def test_single_file(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        assert _should_auto_cross_module(str(tmp_path)) is False

    def test_two_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        assert _should_auto_cross_module(str(tmp_path)) is True

    def test_not_directory(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("")
        assert _should_auto_cross_module(str(f)) is False


# ============================================================
# 12. analyze_file
# ============================================================

class TestAnalyzeFile:
    def test_clean_file(self, tmp_path):
        f = _write_py(tmp_path, """
        def foo(x):
            return x + 1
        result = foo(1)
        """)
        report = analyze_file(f)
        assert report.error is None
        assert len(report.findings) == 0

    def test_file_with_findings(self, tmp_path):
        f = _write_py(tmp_path, """
        import os
        def unused():
            pass
        x = 1
        """)
        report = analyze_file(f)
        assert len(report.findings) >= 2  # unused import + unused function

    def test_nonexistent_file(self):
        report = analyze_file("/nonexistent/path.py")
        assert report.error is not None

    def test_syntax_error_file(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo( pass")
        report = analyze_file(str(f))
        assert report.error is not None
        assert "SyntaxError" in report.error

    def test_file_path_set_on_findings(self, tmp_path):
        f = _write_py(tmp_path, """
        import json
        x = 1
        """)
        report = analyze_file(f)
        for finding in report.findings:
            assert finding.file == f

    def test_cross_module_names_param(self, tmp_path):
        f = _write_py(tmp_path, """
        def exported():
            pass
        """)
        report = analyze_file(f, cross_module_names={"exported"})
        # exported should not be flagged as unused
        names = {finding.name for finding in report.findings}
        assert "exported" not in names

    def test_wholesale_imported_param(self, tmp_path):
        f = _write_py(tmp_path, """
        def foo():
            pass
        def bar():
            pass
        """)
        report = analyze_file(f, is_wholesale_imported=True)
        func_findings = [f for f in report.findings
                        if f.category == "unused-function"]
        assert len(func_findings) == 0


# ============================================================
# 13. scan (integration)
# ============================================================

class TestScan:
    def test_clean_project(self, tmp_path):
        _write_py(tmp_path, """
        def add(a, b):
            return a + b
        result = add(1, 2)
        """)
        result = scan(str(tmp_path))
        assert result.files_scanned == 1
        assert result.total_findings == 0

    def test_project_with_issues(self, tmp_path):
        _write_py(tmp_path, """
        import os
        import sys
        def foo():
            x = 1
            return None
        """)
        result = scan(str(tmp_path))
        assert result.total_findings >= 2

    def test_category_filter(self, tmp_path):
        _write_py(tmp_path, """
        import os
        def unused():
            pass
        x = 1
        """)
        result = scan(str(tmp_path), category="unused-import")
        cats = set(result.by_category.keys())
        assert cats == {"unused-import"} or cats == set()

    def test_severity_filter(self, tmp_path):
        _write_py(tmp_path, """
        import os
        def unused():
            pass
        def foo(arg):
            return 1
        x = 1
        """)
        result = scan(str(tmp_path), min_severity="medium")
        # Only medium+ findings (unused-import is medium, unused-function is low)
        for sev in result.by_severity:
            assert SEVERITY_ORDER[sev] >= SEVERITY_ORDER["medium"]

    def test_ignore_patterns(self, tmp_path):
        _write_py(tmp_path, """
        import os
        """, name="ignored.py")
        _write_py(tmp_path, """
        import sys
        """, name="kept.py")
        result = scan(str(tmp_path), ignore_patterns=["ignored"])
        # ignored.py should not be analyzed
        files_analyzed = {r.file for r in result.reports}
        assert not any("ignored" in f for f in files_analyzed)

    def test_cross_module_auto(self, tmp_path):
        """Cross-module analysis auto-enables for 2+ files."""
        a = _write_py(tmp_path, """
        from b import helper
        helper()
        """, name="a.py")
        b = _write_py(tmp_path, """
        def helper():
            return 1
        """, name="b.py")
        result = scan(str(tmp_path))
        # helper should NOT be flagged as unused because a.py imports it
        all_names = {f.name for r in result.reports for f in r.findings}
        assert "helper" not in all_names

    def test_cross_module_disabled(self, tmp_path):
        a = _write_py(tmp_path, """
        from b import helper
        helper()
        """, name="a.py")
        b = _write_py(tmp_path, """
        def helper():
            return 1
        """, name="b.py")
        result = scan(str(tmp_path), cross_module=False)
        # Without cross-module, helper may be flagged
        names_in_b = {f.name for r in result.reports
                     for f in r.findings if r.file.endswith("b.py")}
        assert "helper" in names_in_b

    def test_single_file_scan(self, tmp_path):
        f = _write_py(tmp_path, """
        import json
        x = 1
        """)
        result = scan(f)
        assert result.files_scanned == 1

    def test_by_category_populated(self, tmp_path):
        _write_py(tmp_path, """
        import os
        def unused():
            pass
        x = 1
        """)
        result = scan(str(tmp_path))
        assert len(result.by_category) > 0

    def test_by_severity_populated(self, tmp_path):
        _write_py(tmp_path, """
        import os
        x = 1
        """)
        result = scan(str(tmp_path))
        assert len(result.by_severity) > 0


# ============================================================
# 14. CLI (main)
# ============================================================

class TestMain:
    def test_clean_project_exit_0(self, tmp_path):
        _write_py(tmp_path, """
        def add(a, b):
            return a + b
        result = add(1, 2)
        """)
        exit_code = main([str(tmp_path)])
        assert exit_code == 0

    def test_findings_exit_1(self, tmp_path):
        _write_py(tmp_path, """
        import os
        x = 1
        """)
        exit_code = main([str(tmp_path)])
        assert exit_code == 1  # medium+ findings

    def test_json_output(self, tmp_path, capsys):
        _write_py(tmp_path, """
        import os
        x = 1
        """)
        main([str(tmp_path), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "findings" in data
        assert "version" in data

    def test_category_filter_cli(self, tmp_path, capsys):
        _write_py(tmp_path, """
        import os
        def unused():
            pass
        x = 1
        """)
        main([str(tmp_path), "--category", "unused-import", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        for f in data["findings"]:
            assert f["category"] == "unused-import"

    def test_severity_filter_cli(self, tmp_path, capsys):
        _write_py(tmp_path, """
        import os
        def unused():
            pass
        x = 1
        """)
        main([str(tmp_path), "--severity", "medium", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        for f in data["findings"]:
            assert SEVERITY_ORDER[f["severity"]] >= SEVERITY_ORDER["medium"]

    def test_file_flag(self, tmp_path, capsys):
        f = _write_py(tmp_path, """
        import os
        x = 1
        """)
        main(["-f", f, "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["files_scanned"] == 1

    def test_score_mode(self, tmp_path, capsys):
        _write_py(tmp_path, """
        def add(a, b):
            return a + b
        result = add(1, 2)
        """)
        exit_code = main([str(tmp_path), "--score"])
        captured = capsys.readouterr()
        assert "Dead Code Health Score" in captured.out
        assert exit_code == 0

    def test_score_mode_json(self, tmp_path, capsys):
        _write_py(tmp_path, """
        def add(a, b):
            return a + b
        result = add(1, 2)
        """)
        main([str(tmp_path), "--score", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "score" in data
        assert "grade" in data

    def test_no_cross_module_flag(self, tmp_path):
        _write_py(tmp_path, """
        def foo():
            return 1
        """, name="a.py")
        _write_py(tmp_path, """
        def bar():
            return 1
        """, name="b.py")
        exit_code = main([str(tmp_path), "--no-cross-module"])
        assert isinstance(exit_code, int)

    def test_cross_module_flag(self, tmp_path):
        _write_py(tmp_path, """
        def foo():
            return 1
        """, name="a.py")
        _write_py(tmp_path, """
        def bar():
            return 1
        """, name="b.py")
        exit_code = main([str(tmp_path), "--cross-module"])
        assert isinstance(exit_code, int)

    def test_ignore_flag(self, tmp_path, capsys):
        _write_py(tmp_path, """
        import os
        """, name="skip_me.py")
        _write_py(tmp_path, """
        def add(a, b):
            return a + b
        result = add(1, 2)
        """, name="keep.py")
        main([str(tmp_path), "--ignore", "skip_me", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        files = {f["file"] for f in data.get("findings", [])}
        assert not any("skip_me" in f for f in files)


# ============================================================
# 15. Formatting Functions
# ============================================================

class TestFormatting:
    def test_format_severity(self):
        result = format_severity("medium")
        assert "MEDIUM" in result

    def test_format_finding(self):
        f = Finding(file="foo.py", line=10, category="unused-import",
                    name="os", message="'os' imported but never used")
        result = format_finding(f)
        assert "foo.py:10" in result
        assert "unused-import" in result

    def test_print_summary(self, capsys):
        result = ScanResult(
            files_scanned=5, total_findings=3,
            by_category={"unused-import": 2, "unused-function": 1},
            by_severity={"medium": 2, "low": 1},
        )
        print_summary(result)
        captured = capsys.readouterr()
        assert "Dead Code Summary" in captured.out
        assert "Files scanned: 5" in captured.out


# ============================================================
# 16. Constants Sanity
# ============================================================

class TestConstants:
    def test_categories_all_have_severity(self):
        for cat in CATEGORIES:
            assert cat in SEVERITY_MAP

    def test_severity_order_complete(self):
        for sev in set(SEVERITY_MAP.values()):
            assert sev in SEVERITY_ORDER

    def test_severity_weights_complete(self):
        for sev in SEVERITY_ORDER:
            assert sev in _SEVERITY_WEIGHTS

    def test_ignored_names_are_strings(self):
        for name in IGNORED_NAMES:
            assert isinstance(name, str)

    def test_dunder_methods_start_end_with_double_under(self):
        for name in DUNDER_METHODS:
            assert name.startswith("__") and name.endswith("__")

    def test_excluded_dirs_are_strings(self):
        for d in EXCLUDED_DIRS:
            assert isinstance(d, str)


# ============================================================
# 17. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        f = _write_py(tmp_path, "")
        report = analyze_file(f)
        assert report.error is None
        assert len(report.findings) == 0

    def test_only_comments(self, tmp_path):
        f = _write_py(tmp_path, "# Just a comment\n# Another one\n")
        report = analyze_file(f)
        assert len(report.findings) == 0

    def test_encoding_issues(self, tmp_path):
        f = tmp_path / "utf8.py"
        f.write_text("# -*- coding: utf-8 -*-\nx = '你好'\n", encoding="utf-8")
        report = analyze_file(str(f))
        assert report.error is None

    def test_very_nested_code(self, tmp_path):
        f = _write_py(tmp_path, """
        def outer():
            def middle():
                def inner():
                    return 1
                return inner
            return middle
        result = outer()
        """)
        report = analyze_file(f)
        # Should not crash on deeply nested code
        assert report.error is None

    def test_lambda_not_flagged(self, tmp_path):
        f = _write_py(tmp_path, """
        fn = lambda x: x + 1
        result = fn(1)
        """)
        report = analyze_file(f)
        func_findings = [ff for ff in report.findings
                        if ff.category == "unused-function"]
        assert len(func_findings) == 0

    def test_class_with_all_dunders(self, tmp_path):
        f = _write_py(tmp_path, """
        class MyNum:
            def __init__(self, val):
                self.val = val
            def __add__(self, other):
                return MyNum(self.val + other.val)
            def __str__(self):
                return str(self.val)
        n = MyNum(1)
        """)
        report = analyze_file(f)
        func_findings = [ff for ff in report.findings
                        if ff.category == "unused-function"]
        assert len(func_findings) == 0

    def test_decorator_with_arguments(self, tmp_path):
        f = _write_py(tmp_path, """
        import pytest
        @pytest.mark.parametrize("x", [1, 2, 3])
        def test_values(x):
            assert x > 0
        """)
        report = analyze_file(f)
        # pytest.mark.parametrize should suppress unused-argument for x
        # and test_values should not be flagged
        func_findings = [ff for ff in report.findings
                        if ff.category in ("unused-function", "unused-argument")]
        assert len(func_findings) == 0

    def test_multiple_findings_same_file(self, tmp_path):
        f = _write_py(tmp_path, """
        import os
        import sys
        import json
        def unused_func():
            pass
        def another_unused():
            x = 1
            return None
        """)
        report = analyze_file(f)
        assert len(report.findings) >= 4  # 3 imports + 2 funcs + 1 var
