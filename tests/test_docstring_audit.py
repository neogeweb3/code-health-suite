#!/usr/bin/env python3
"""Tests for docstring_audit engine."""
from __future__ import annotations

import os
import textwrap
import tempfile
import pytest

from code_health_suite.engines import docstring_audit


# --- Helpers ---

def _write_temp(code: str, suffix: str = ".py") -> str:
    """Write code to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(textwrap.dedent(code))
    f.close()
    return f.name


def _write_temp_dir(files: dict[str, str]) -> str:
    """Write multiple files to a temp dir, return dir path."""
    d = tempfile.mkdtemp()
    for name, code in files.items():
        path = os.path.join(d, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(textwrap.dedent(code))
    return d


# =============================================================================
# EntityInfo basics
# =============================================================================

class TestEntityInfo:
    def test_dataclass_fields(self):
        e = docstring_audit.EntityInfo(
            name="foo", kind="function", line_number=1,
            has_docstring=True, docstring_lines=3, param_count=2,
        )
        assert e.name == "foo"
        assert e.kind == "function"
        assert e.has_docstring is True
        assert e.docstring_lines == 3
        assert e.param_count == 2
        assert e.is_public is True

    def test_default_values(self):
        e = docstring_audit.EntityInfo(
            name="bar", kind="class", line_number=10, has_docstring=False,
        )
        assert e.docstring_lines == 0
        assert e.has_params_doc is False
        assert e.has_returns_doc is False
        assert e.has_raises_doc is False
        assert e.param_count == 0


# =============================================================================
# _is_public
# =============================================================================

class TestIsPublic:
    def test_public_name(self):
        assert docstring_audit._is_public("foo") is True

    def test_private_name(self):
        assert docstring_audit._is_public("_foo") is False

    def test_dunder_is_public(self):
        assert docstring_audit._is_public("__init__") is True
        assert docstring_audit._is_public("__str__") is True

    def test_name_mangled(self):
        assert docstring_audit._is_public("__foo") is False


# =============================================================================
# _count_params
# =============================================================================

class TestCountParams:
    def test_no_params(self):
        code = "def foo(): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._count_params(func) == 0

    def test_simple_params(self):
        code = "def foo(a, b, c): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._count_params(func) == 3

    def test_self_excluded(self):
        code = "def foo(self, a, b): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._count_params(func) == 2

    def test_cls_excluded(self):
        code = "def foo(cls, a): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._count_params(func) == 1

    def test_args_kwargs(self):
        code = "def foo(*args, **kwargs): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._count_params(func) == 2

    def test_kwonly(self):
        code = "def foo(a, *, b, c): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._count_params(func) == 3


# =============================================================================
# _get_docstring
# =============================================================================

class TestGetDocstring:
    def test_function_with_docstring(self):
        code = '''
def foo():
    """This is a docstring."""
    pass
'''
        tree = __import__("ast").parse(textwrap.dedent(code))
        func = tree.body[0]
        assert docstring_audit._get_docstring(func) == "This is a docstring."

    def test_function_without_docstring(self):
        code = "def foo(): pass"
        tree = __import__("ast").parse(code)
        func = tree.body[0]
        assert docstring_audit._get_docstring(func) is None

    def test_class_with_docstring(self):
        code = '''
class Foo:
    """Foo class."""
    pass
'''
        tree = __import__("ast").parse(textwrap.dedent(code))
        cls = tree.body[0]
        assert docstring_audit._get_docstring(cls) == "Foo class."

    def test_module_docstring(self):
        code = '"""Module doc."""\nx = 1'
        tree = __import__("ast").parse(code)
        assert docstring_audit._get_docstring(tree) == "Module doc."


# =============================================================================
# _check_docstring_quality
# =============================================================================

class TestDocstringQuality:
    def test_good_docstring(self):
        doc = "This is a good docstring with enough content."
        issues = docstring_audit._check_docstring_quality(doc, 0)
        assert issues == []

    def test_too_short(self):
        doc = "Short."
        issues = docstring_audit._check_docstring_quality(doc, 0)
        assert "too_short" in issues

    def test_missing_params_google_style(self):
        doc = "Does something important with the value."
        issues = docstring_audit._check_docstring_quality(doc, 2)
        assert "no_params" in issues

    def test_has_params_sphinx_style(self):
        doc = "Do something.\n\n:param x: The value.\n:param y: Another."
        issues = docstring_audit._check_docstring_quality(doc, 2)
        assert "no_params" not in issues

    def test_has_params_google_style(self):
        doc = "Do something.\n\nArgs:\n    x: The value.\n    y: Another."
        issues = docstring_audit._check_docstring_quality(doc, 2)
        assert "no_params" not in issues

    def test_no_params_needed(self):
        doc = "Does something with no parameters needed."
        issues = docstring_audit._check_docstring_quality(doc, 0)
        assert "no_params" not in issues


# =============================================================================
# analyze_file
# =============================================================================

class TestAnalyzeFile:
    def test_fully_documented(self):
        code = '''
"""Module docstring."""

def foo():
    """Foo function."""
    pass

class Bar:
    """Bar class."""
    def method(self):
        """Method docstring."""
        pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            assert result.error == ""
            assert result.total_public == 4  # module + foo + Bar + method
            assert result.documented == 4
            assert result.coverage == 1.0
            assert result.issues == []  # all documented, no quality issues
        finally:
            os.unlink(path)

    def test_missing_docstrings(self):
        code = '''
def foo():
    pass

class Bar:
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            assert result.total_public >= 2  # foo + Bar (+ module)
            assert result.coverage < 1.0
            missing = [i for i in result.issues if i.issue_type == "missing"]
            assert len(missing) >= 2  # foo and Bar
        finally:
            os.unlink(path)

    def test_private_functions_excluded(self):
        code = '''
"""Module doc."""

def _private():
    pass

def public():
    """Public func."""
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            # Module + public = 2 public, both documented
            assert result.total_public == 2
            assert result.documented == 2
        finally:
            os.unlink(path)

    def test_dunder_methods_are_public(self):
        code = '''
"""Module doc."""

class Foo:
    """Foo class."""
    def __init__(self):
        pass
    def __str__(self):
        """String repr."""
        return "Foo"
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            # __init__ is public (dunder) but missing docstring
            missing = [i for i in result.issues if i.issue_type == "missing"]
            names = [i.name for i in missing]
            assert "__init__" in names
        finally:
            os.unlink(path)

    def test_syntax_error(self):
        path = _write_temp("def foo(:")
        try:
            result = docstring_audit.analyze_file(path)
            assert "SyntaxError" in result.error
        finally:
            os.unlink(path)

    def test_non_python_file(self):
        path = _write_temp("hello", suffix=".txt")
        try:
            result = docstring_audit.find_python_files(path)
            assert result == []
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = _write_temp("")
        try:
            result = docstring_audit.analyze_file(path)
            assert result.error == ""
        finally:
            os.unlink(path)

    def test_async_function(self):
        code = '''
"""Module doc."""

async def fetch():
    """Fetch data."""
    pass

async def process():
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            missing = [i for i in result.issues if i.issue_type == "missing"]
            names = [i.name for i in missing]
            assert "process" in names
            assert "fetch" not in names
        finally:
            os.unlink(path)

    def test_quality_issues_detected(self):
        code = '''
"""Module doc."""

def compute(x, y, z):
    """Do it."""
    return x + y + z
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            quality = [i for i in result.issues if i.issue_type != "missing"]
            assert len(quality) >= 1  # too_short or no_params
        finally:
            os.unlink(path)


# =============================================================================
# scan (directory)
# =============================================================================

class TestScan:
    def test_single_file(self):
        code = '''
"""Module."""

def foo():
    """Foo."""
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.scan(path)
            assert result.files_scanned == 1
            assert result.total_public >= 1
        finally:
            os.unlink(path)

    def test_directory_scan(self):
        d = _write_temp_dir({
            "a.py": '"""A."""\ndef foo():\n    """Foo."""\n    pass\n',
            "b.py": 'def bar(): pass\n',
        })
        try:
            result = docstring_audit.scan(d)
            assert result.files_scanned == 2
            assert result.total_documented >= 2  # a.py module + foo
        finally:
            import shutil
            shutil.rmtree(d)

    def test_skip_venv(self):
        d = _write_temp_dir({
            "main.py": '"""Main."""\ndef run():\n    """Run."""\n    pass\n',
            "venv/lib.py": 'def internal(): pass\n',
        })
        try:
            result = docstring_audit.scan(d)
            assert result.files_scanned == 1  # venv skipped
        finally:
            import shutil
            shutil.rmtree(d)

    def test_by_kind_aggregation(self):
        code = '''
"""Module."""

class Foo:
    """A class."""
    def method(self):
        """A method."""
        pass

def standalone():
    """Standalone func."""
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.scan(path)
            assert "module" in result.by_kind
            assert "class" in result.by_kind
            assert "function" in result.by_kind
        finally:
            os.unlink(path)

    def test_errors_collected(self):
        d = _write_temp_dir({
            "good.py": '"""Good."""\n',
            "bad.py": 'def foo(:\n',
        })
        try:
            result = docstring_audit.scan(d)
            assert len(result.errors) == 1
        finally:
            import shutil
            shutil.rmtree(d)

    def test_empty_directory(self):
        d = tempfile.mkdtemp()
        try:
            result = docstring_audit.scan(d)
            assert result.files_scanned == 0
            assert result.coverage == 0.0
        finally:
            os.rmdir(d)

    def test_nonexistent_path(self):
        result = docstring_audit.scan("/nonexistent/path/xyz")
        assert result.files_scanned == 0


# =============================================================================
# compute_score
# =============================================================================

class TestComputeScore:
    def test_perfect_score(self):
        code = '''
"""Module."""

def foo():
    """Foo function does something useful."""
    pass

class Bar:
    """Bar class for testing purposes."""
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.scan(path)
            score = docstring_audit.compute_score(result)
            assert score.score >= 90
            assert score.grade == "A"
        finally:
            os.unlink(path)

    def test_zero_score(self):
        code = '''
def foo():
    pass

def bar():
    pass

class Baz:
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.scan(path)
            score = docstring_audit.compute_score(result)
            assert score.score < 50
            assert score.grade in ("D", "F")
        finally:
            os.unlink(path)

    def test_empty_project(self):
        d = tempfile.mkdtemp()
        try:
            result = docstring_audit.scan(d)
            score = docstring_audit.compute_score(result)
            assert score.score == 100
            assert score.grade == "A"
        finally:
            os.rmdir(d)

    def test_grade_boundaries(self):
        assert docstring_audit._score_to_grade(95) == "A"
        assert docstring_audit._score_to_grade(90) == "A"
        assert docstring_audit._score_to_grade(89) == "B"
        assert docstring_audit._score_to_grade(80) == "B"
        assert docstring_audit._score_to_grade(79) == "C"
        assert docstring_audit._score_to_grade(70) == "C"
        assert docstring_audit._score_to_grade(69) == "D"
        assert docstring_audit._score_to_grade(60) == "D"
        assert docstring_audit._score_to_grade(59) == "F"
        assert docstring_audit._score_to_grade(0) == "F"

    def test_to_dict(self):
        code = '"""Module."""\ndef foo():\n    """Foo."""\n    pass\n'
        path = _write_temp(code)
        try:
            result = docstring_audit.scan(path)
            score = docstring_audit.compute_score(result)
            d = score.to_dict()
            assert isinstance(d, dict)
            assert "score" in d
            assert "grade" in d
            assert "coverage" in d
            assert "total_public" in d
        finally:
            os.unlink(path)

    def test_worst_files(self):
        d = _write_temp_dir({
            "good.py": '"""Good."""\ndef foo():\n    """Foo."""\n    pass\n',
            "bad.py": 'def bar(): pass\ndef baz(): pass\n',
        })
        try:
            result = docstring_audit.scan(d)
            score = docstring_audit.compute_score(result)
            assert len(score.worst_files) >= 1
            # Worst file should have lower coverage
            assert score.worst_files[0]["coverage"] < 1.0
        finally:
            import shutil
            shutil.rmtree(d)

    def test_quality_penalty(self):
        # Many functions with short docstrings + missing params
        funcs = "\n".join(
            f'def func{i}(x, y):\n    """Do."""\n    pass\n'
            for i in range(15)
        )
        code = f'"""Module."""\n{funcs}'
        path = _write_temp(code)
        try:
            result = docstring_audit.scan(path)
            score = docstring_audit.compute_score(result)
            # Should have quality penalty
            assert score.score < 100
        finally:
            os.unlink(path)


# =============================================================================
# find_python_files
# =============================================================================

class TestFindPythonFiles:
    def test_single_file(self):
        path = _write_temp("x = 1")
        try:
            files = docstring_audit.find_python_files(path)
            assert len(files) == 1
        finally:
            os.unlink(path)

    def test_non_python_ignored(self):
        path = _write_temp("text", suffix=".txt")
        try:
            files = docstring_audit.find_python_files(path)
            assert files == []
        finally:
            os.unlink(path)

    def test_directory(self):
        d = _write_temp_dir({
            "a.py": "x = 1",
            "b.py": "y = 2",
            "c.txt": "text",
        })
        try:
            files = docstring_audit.find_python_files(d)
            assert len(files) == 2
        finally:
            import shutil
            shutil.rmtree(d)

    def test_skip_dirs(self):
        d = _write_temp_dir({
            "main.py": "x = 1",
            "__pycache__/cache.py": "y = 2",
            ".git/hook.py": "z = 3",
        })
        try:
            files = docstring_audit.find_python_files(d)
            assert len(files) == 1
        finally:
            import shutil
            shutil.rmtree(d)


# =============================================================================
# DocstringIssue
# =============================================================================

class TestDocstringIssue:
    def test_missing_issue(self):
        issue = docstring_audit.DocstringIssue(
            file_path="test.py", line_number=1, name="foo",
            kind="function", issue_type="missing",
            message="Public function 'foo' has no docstring",
        )
        assert issue.severity == "medium"

    def test_custom_severity(self):
        issue = docstring_audit.DocstringIssue(
            file_path="test.py", line_number=1, name="Foo",
            kind="class", issue_type="missing",
            message="Public class 'Foo' has no docstring",
            severity="high",
        )
        assert issue.severity == "high"


# =============================================================================
# CLI
# =============================================================================

class TestCLI:
    def test_help(self):
        assert docstring_audit.main(["--help"]) == 0

    def test_version(self):
        assert docstring_audit.main(["--version"]) == 0

    def test_nonexistent_path(self):
        assert docstring_audit.main(["/nonexistent/xyz"]) == 1

    def test_default_scan(self, tmp_path):
        (tmp_path / "test.py").write_text('"""Module."""\ndef foo():\n    """Foo."""\n    pass\n')
        assert docstring_audit.main([str(tmp_path)]) == 0

    def test_json_mode(self, tmp_path):
        (tmp_path / "test.py").write_text('"""Module."""\n')
        assert docstring_audit.main([str(tmp_path), "--json"]) == 0

    def test_score_mode(self, tmp_path):
        (tmp_path / "test.py").write_text('"""Module."""\n')
        assert docstring_audit.main([str(tmp_path), "--score"]) == 0

    def test_score_json_mode(self, tmp_path):
        (tmp_path / "test.py").write_text('"""Module."""\n')
        assert docstring_audit.main([str(tmp_path), "--score", "--json"]) == 0


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_class_with_only_init(self):
        code = '''
"""Module."""

class Foo:
    def __init__(self, x):
        self.x = x
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            missing = [i for i in result.issues if i.issue_type == "missing"]
            names = [i.name for i in missing]
            assert "Foo" in names
        finally:
            os.unlink(path)

    def test_decorated_function(self):
        code = '''
"""Module."""

import functools

@functools.lru_cache
def cached():
    """Cached function."""
    return 42
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            # cached should be documented
            entities = [e for e in result.entities if e.name == "cached"]
            assert len(entities) == 1
            assert entities[0].has_docstring is True
        finally:
            os.unlink(path)

    def test_nested_class(self):
        code = '''
"""Module."""

class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""
        pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            classes = [e for e in result.entities if e.kind == "class"]
            assert len(classes) == 2
        finally:
            os.unlink(path)

    def test_property_method(self):
        code = '''
"""Module."""

class Foo:
    """Foo class."""

    @property
    def value(self):
        """Get value."""
        return 42
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            entities = [e for e in result.entities if e.name == "value"]
            assert len(entities) == 1
            assert entities[0].has_docstring is True
        finally:
            os.unlink(path)

    def test_multiline_docstring(self):
        code = '''
"""Module docstring."""

def foo(x, y):
    """Compute something.

    Args:
        x: First value.
        y: Second value.

    Returns:
        The sum of x and y.
    """
    return x + y
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            entities = [e for e in result.entities if e.name == "foo"]
            assert len(entities) == 1
            assert entities[0].has_docstring is True
            assert entities[0].has_params_doc is True
            assert entities[0].has_returns_doc is True
            # No quality issues for well-documented function
            foo_issues = [i for i in result.issues if i.name == "foo"]
            assert len(foo_issues) == 0
        finally:
            os.unlink(path)

    def test_sphinx_style_params(self):
        code = '''
"""Module."""

def bar(a, b):
    """Do bar.

    :param a: First.
    :param b: Second.
    :returns: Result.
    """
    return a + b
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            bar_issues = [i for i in result.issues if i.name == "bar" and i.issue_type == "no_params"]
            assert len(bar_issues) == 0
        finally:
            os.unlink(path)

    def test_module_without_docstring(self):
        code = "x = 1\ny = 2\n"
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            # Module-level entity should show as undocumented
            mod = [e for e in result.entities if e.kind == "module"]
            assert len(mod) == 1
            assert mod[0].has_docstring is False
        finally:
            os.unlink(path)

    def test_file_with_all_private(self):
        code = '''
"""Module."""

def _private1():
    pass

def _private2():
    pass
'''
        path = _write_temp(code)
        try:
            result = docstring_audit.analyze_file(path)
            # Only module is public
            assert result.total_public == 1
            assert result.documented == 1
            assert result.coverage == 1.0
        finally:
            os.unlink(path)
