"""Tests for the test_quality engine."""
from __future__ import annotations

import ast
import json
import os
import textwrap
from pathlib import Path

import pytest

from code_health_suite.engines.test_quality import (
    TestIssue,
    TestFunctionInfo,
    FileReport,
    SuiteReport,
    DEFAULT_MAX_TEST_LENGTH,
    DEFAULT_MAX_ASSERTIONS,
    SKIP_DIRS,
    VENV_PATTERNS,
    GENERIC_NAME_RE,
    SEVERITY_WEIGHTS,
    _count_assertions,
    _get_call_name,
    _is_empty_body,
    _is_bare_assert,
    _has_broad_except,
    _has_sleep_call,
    _is_test_function,
    _is_test_class,
    analyze_test_function,
    analyze_file,
    _track_duplicate,
    _is_venv,
    _is_test_file,
    discover_test_files,
    compute_score,
    analyze_suite,
    format_text,
    format_json,
    _relative_path,
    build_parser,
    main,
)


# --- Helpers ---

def _parse_func(code: str) -> ast.FunctionDef:
    """Parse code snippet and return the first function node."""
    tree = ast.parse(textwrap.dedent(code))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("No function found in code")


def _parse_body(code: str) -> list[ast.stmt]:
    """Parse code and return the body statements."""
    tree = ast.parse(textwrap.dedent(code))
    return tree.body


def _make_issue(
    check: str = "no_assertions",
    severity: str = "high",
    message: str = "test issue",
    line: int = 1,
    function: str = "test_foo",
    file: str = "test.py",
) -> TestIssue:
    """Create a TestIssue for testing."""
    return TestIssue(
        check=check, severity=severity, message=message,
        line=line, function=function, file=file,
    )


# Sample test files as strings
GOOD_TEST_FILE = textwrap.dedent("""\
    def test_addition():
        result = 1 + 2
        assert result == 3

    def test_subtraction():
        result = 5 - 3
        assert result == 2
""")

EMPTY_TEST_FILE = textwrap.dedent("""\
    def test_empty():
        pass
""")

NO_ASSERTION_TEST = textwrap.dedent("""\
    def test_no_assert():
        x = 1 + 2
        print(x)
""")

LONG_TEST_FILE = textwrap.dedent("""\
    def test_very_long():
        a = 1
        b = 2
        c = 3
        d = 4
        e = 5
        f = 6
        g = 7
        h = 8
        i = 9
        j = 10
        k = 11
        l = 12
        m = 13
        n = 14
        o = 15
        p = 16
        q = 17
        r = 18
        s = 19
        t = 20
        u = 21
        v = 22
        w = 23
        x = 24
        y = 25
        z = 26
        aa = 27
        bb = 28
        cc = 29
        dd = 30
        ee = 31
        ff = 32
        gg = 33
        hh = 34
        ii = 35
        jj = 36
        kk = 37
        ll = 38
        mm = 39
        nn = 40
        oo = 41
        pp = 42
        qq = 43
        rr = 44
        ss = 45
        tt = 46
        uu = 47
        vv = 48
        ww = 49
        xx = 50
        yy = 51
        assert yy == 51
""")


# =============================================================================
# Data Models
# =============================================================================


class TestDataModels:
    def test_test_issue_fields(self):
        issue = TestIssue(
            check="no_assertions", severity="high",
            message="No assertions", line=5, function="test_foo", file="t.py",
        )
        assert issue.check == "no_assertions"
        assert issue.severity == "high"
        assert issue.message == "No assertions"
        assert issue.line == 5
        assert issue.function == "test_foo"
        assert issue.file == "t.py"

    def test_test_function_info_defaults(self):
        info = TestFunctionInfo(
            name="test_x", qualified_name="test_x",
            file="t.py", line=1, end_line=5, length=5, assertion_count=2,
        )
        assert info.issues == []

    def test_test_function_info_with_issues(self):
        issue = _make_issue()
        info = TestFunctionInfo(
            name="test_x", qualified_name="test_x",
            file="t.py", line=1, end_line=5, length=5, assertion_count=0,
            issues=[issue],
        )
        assert len(info.issues) == 1

    def test_file_report_fields(self):
        report = FileReport(
            file="test.py", test_count=3,
            total_assertions=5, assertion_density=1.7,
        )
        assert report.file == "test.py"
        assert report.test_count == 3
        assert report.total_assertions == 5
        assert report.assertion_density == 1.7
        assert report.issues == []
        assert report.tests == []

    def test_suite_report_defaults(self):
        report = SuiteReport(
            files_analyzed=2, total_tests=10,
            total_assertions=20, total_issues=0,
        )
        assert report.score == 100
        assert report.grade == "A"
        assert report.issues_by_severity == {}
        assert report.issues_by_check == {}
        assert report.files == []


# =============================================================================
# Constants
# =============================================================================


class TestConstants:
    def test_default_max_test_length(self):
        assert DEFAULT_MAX_TEST_LENGTH == 50

    def test_default_max_assertions(self):
        assert DEFAULT_MAX_ASSERTIONS == 15

    def test_skip_dirs_contains_pycache(self):
        assert "__pycache__" in SKIP_DIRS

    def test_skip_dirs_contains_git(self):
        assert ".git" in SKIP_DIRS

    def test_venv_patterns(self):
        assert ".venv" in VENV_PATTERNS
        assert "venv" in VENV_PATTERNS

    def test_severity_weights(self):
        assert SEVERITY_WEIGHTS["high"] == 5
        assert SEVERITY_WEIGHTS["medium"] == 3
        assert SEVERITY_WEIGHTS["low"] == 1

    @pytest.mark.parametrize("name", [
        "test", "test_", "test_1", "test_42", "test_it",
        "test_func", "test_method", "test_case", "test_something",
    ])
    def test_generic_name_re_matches(self, name):
        assert GENERIC_NAME_RE.match(name)

    @pytest.mark.parametrize("name", [
        "test_addition", "test_user_login", "test_parse_json",
        "test_empty_string", "test_method_returns_none",
    ])
    def test_generic_name_re_rejects_descriptive(self, name):
        assert not GENERIC_NAME_RE.match(name)


# =============================================================================
# _count_assertions
# =============================================================================


class TestCountAssertions:
    def test_single_assert_statement(self):
        func = _parse_func("def test_x():\n    assert True")
        assert _count_assertions(func) == 1

    def test_multiple_assert_statements(self):
        func = _parse_func("""\
            def test_x():
                assert 1 == 1
                assert 2 == 2
                assert 3 == 3
        """)
        assert _count_assertions(func) == 3

    def test_no_assertions(self):
        func = _parse_func("def test_x():\n    x = 1")
        assert _count_assertions(func) == 0

    def test_self_assert_calls(self):
        func = _parse_func("""\
            def test_x(self):
                self.assertEqual(1, 1)
                self.assertTrue(True)
                self.assertIn(1, [1, 2])
        """)
        assert _count_assertions(func) == 3

    def test_pytest_assert_raises(self):
        func = _parse_func("""\
            def test_x():
                assert 1 == 1
        """)
        assert _count_assertions(func) == 1

    def test_mixed_assert_styles(self):
        func = _parse_func("""\
            def test_x(self):
                assert 1 == 1
                self.assertEqual(2, 2)
        """)
        assert _count_assertions(func) == 2

    def test_nested_assert_in_if(self):
        func = _parse_func("""\
            def test_x():
                if True:
                    assert 1 == 1
        """)
        assert _count_assertions(func) == 1

    def test_assert_in_loop(self):
        func = _parse_func("""\
            def test_x():
                for i in range(3):
                    assert i >= 0
        """)
        assert _count_assertions(func) == 1


# =============================================================================
# _get_call_name
# =============================================================================


class TestGetCallName:
    def test_simple_name(self):
        tree = ast.parse("foo()")
        call = tree.body[0].value
        assert _get_call_name(call) == "foo"

    def test_attribute_call(self):
        tree = ast.parse("obj.method()")
        call = tree.body[0].value
        assert _get_call_name(call) == "method"

    def test_chained_attribute(self):
        tree = ast.parse("a.b.c()")
        call = tree.body[0].value
        assert _get_call_name(call) == "c"

    def test_subscript_call_returns_none(self):
        tree = ast.parse("a[0]()")
        call = tree.body[0].value
        assert _get_call_name(call) is None


# =============================================================================
# _is_empty_body
# =============================================================================


class TestIsEmptyBody:
    def test_pass_only(self):
        func = _parse_func("def test_x():\n    pass")
        assert _is_empty_body(func.body) is True

    def test_ellipsis_only(self):
        func = _parse_func("def test_x():\n    ...")
        assert _is_empty_body(func.body) is True

    def test_docstring_only(self):
        func = _parse_func('def test_x():\n    """A test."""')
        assert _is_empty_body(func.body) is True

    def test_docstring_plus_pass(self):
        func = _parse_func('def test_x():\n    """A test."""\n    pass')
        assert _is_empty_body(func.body) is True

    def test_real_body(self):
        func = _parse_func("def test_x():\n    x = 1\n    assert x == 1")
        assert _is_empty_body(func.body) is False

    def test_single_statement(self):
        func = _parse_func("def test_x():\n    return 42")
        assert _is_empty_body(func.body) is False


# =============================================================================
# _is_bare_assert
# =============================================================================


class TestIsBareAssert:
    def test_assert_true(self):
        tree = ast.parse("assert True")
        node = tree.body[0]
        assert _is_bare_assert(node) is True

    def test_assert_one(self):
        tree = ast.parse("assert 1")
        node = tree.body[0]
        assert _is_bare_assert(node) is True

    def test_assert_string(self):
        tree = ast.parse('assert "hello"')
        node = tree.body[0]
        assert _is_bare_assert(node) is True

    def test_assert_false_is_not_bare(self):
        tree = ast.parse("assert False")
        node = tree.body[0]
        assert _is_bare_assert(node) is False

    def test_assert_zero_is_not_bare(self):
        tree = ast.parse("assert 0")
        node = tree.body[0]
        assert _is_bare_assert(node) is False

    def test_assert_empty_string_is_not_bare(self):
        tree = ast.parse('assert ""')
        node = tree.body[0]
        assert _is_bare_assert(node) is False

    def test_assert_comparison_is_not_bare(self):
        tree = ast.parse("assert x == 1")
        node = tree.body[0]
        assert _is_bare_assert(node) is False

    def test_assert_call_is_not_bare(self):
        tree = ast.parse("assert foo()")
        node = tree.body[0]
        assert _is_bare_assert(node) is False


# =============================================================================
# _has_broad_except
# =============================================================================


class TestHasBroadExcept:
    def test_bare_except(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except:
                    pass
        """)
        lines = _has_broad_except(func)
        assert len(lines) == 1

    def test_except_exception(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except Exception:
                    pass
        """)
        lines = _has_broad_except(func)
        assert len(lines) == 1

    def test_specific_exception_ok(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except ValueError:
                    pass
        """)
        lines = _has_broad_except(func)
        assert len(lines) == 0

    def test_no_try_except(self):
        func = _parse_func("def test_x():\n    assert True")
        lines = _has_broad_except(func)
        assert lines == []

    def test_multiple_broad_excepts(self):
        func = _parse_func("""\
            def test_x():
                try:
                    a()
                except:
                    pass
                try:
                    b()
                except Exception:
                    pass
        """)
        lines = _has_broad_except(func)
        assert len(lines) == 2


# =============================================================================
# _has_sleep_call
# =============================================================================


class TestHasSleepCall:
    def test_time_sleep(self):
        func = _parse_func("""\
            def test_x():
                import time
                time.sleep(1)
        """)
        lines = _has_sleep_call(func)
        assert len(lines) == 1

    def test_no_sleep(self):
        func = _parse_func("def test_x():\n    assert True")
        lines = _has_sleep_call(func)
        assert lines == []

    def test_multiple_sleep_calls(self):
        func = _parse_func("""\
            def test_x():
                time.sleep(1)
                do_something()
                time.sleep(2)
        """)
        lines = _has_sleep_call(func)
        assert len(lines) == 2

    def test_sleep_function_not_time_sleep(self):
        # sleep() called as a plain function name also matches
        func = _parse_func("""\
            def test_x():
                sleep(1)
        """)
        lines = _has_sleep_call(func)
        assert len(lines) == 1


# =============================================================================
# _is_test_function
# =============================================================================


class TestIsTestFunction:
    def test_test_prefix(self):
        tree = ast.parse("def test_foo(): pass")
        node = tree.body[0]
        assert _is_test_function(node) is True

    def test_bare_test_name(self):
        tree = ast.parse("def test(): pass")
        node = tree.body[0]
        assert _is_test_function(node) is True

    def test_non_test_function(self):
        tree = ast.parse("def helper(): pass")
        node = tree.body[0]
        assert _is_test_function(node) is False

    def test_async_test_function(self):
        tree = ast.parse("async def test_async(): pass")
        node = tree.body[0]
        assert _is_test_function(node) is True

    def test_class_node_is_not_test_function(self):
        tree = ast.parse("class TestFoo: pass")
        node = tree.body[0]
        assert _is_test_function(node) is False

    def test_function_named_testing(self):
        tree = ast.parse("def testing_foo(): pass")
        node = tree.body[0]
        assert _is_test_function(node) is False


# =============================================================================
# _is_test_class
# =============================================================================


class TestIsTestClass:
    def test_test_class(self):
        tree = ast.parse("class TestFoo: pass")
        node = tree.body[0]
        assert _is_test_class(node) is True

    def test_non_test_class(self):
        tree = ast.parse("class FooHelper: pass")
        node = tree.body[0]
        assert _is_test_class(node) is False

    def test_function_not_a_class(self):
        tree = ast.parse("def TestFoo(): pass")
        node = tree.body[0]
        assert _is_test_class(node) is False

    def test_class_named_test(self):
        tree = ast.parse("class Test: pass")
        node = tree.body[0]
        assert _is_test_class(node) is True


# =============================================================================
# analyze_test_function — all 9 check types
# =============================================================================


class TestAnalyzeTestFunction:
    """Test all 9 check types detected by analyze_test_function."""

    def test_clean_function_no_issues(self):
        func = _parse_func("""\
            def test_addition():
                result = 1 + 2
                assert result == 3
        """)
        info = analyze_test_function(func, "t.py")
        assert info.issues == []
        assert info.assertion_count == 1
        assert info.name == "test_addition"
        assert info.qualified_name == "test_addition"

    def test_with_class_name(self):
        func = _parse_func("""\
            def test_foo():
                assert True
        """)
        info = analyze_test_function(func, "t.py", class_name="TestBar")
        assert info.qualified_name == "TestBar.test_foo"

    # Check 1: empty_test
    def test_check_empty_test_pass(self):
        func = _parse_func("def test_x():\n    pass")
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "empty_test" in checks

    def test_check_empty_test_ellipsis(self):
        func = _parse_func("def test_x():\n    ...")
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "empty_test" in checks

    def test_check_empty_test_docstring_only(self):
        func = _parse_func('def test_x():\n    """placeholder"""')
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "empty_test" in checks

    def test_empty_test_severity_is_high(self):
        func = _parse_func("def test_x():\n    pass")
        info = analyze_test_function(func, "t.py")
        empty_issues = [i for i in info.issues if i.check == "empty_test"]
        assert empty_issues[0].severity == "high"

    # Check 2: no_assertions
    def test_check_no_assertions(self):
        func = _parse_func("""\
            def test_x():
                x = 1 + 2
                print(x)
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "no_assertions" in checks

    def test_no_assertions_severity_is_high(self):
        func = _parse_func("def test_x():\n    x = 1")
        info = analyze_test_function(func, "t.py")
        no_assert = [i for i in info.issues if i.check == "no_assertions"]
        assert no_assert[0].severity == "high"

    def test_empty_test_suppresses_no_assertions(self):
        """empty_test and no_assertions should not both fire for pass-only body."""
        func = _parse_func("def test_x():\n    pass")
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "empty_test" in checks
        assert "no_assertions" not in checks

    # Check 3: too_long
    def test_check_too_long(self):
        # Build a function longer than default (50 lines)
        lines = ["def test_long():"]
        for i in range(55):
            lines.append(f"    x{i} = {i}")
        lines.append("    assert True")
        code = "\n".join(lines)
        func = _parse_func(code)
        info = analyze_test_function(func, "t.py", max_length=50)
        checks = [i.check for i in info.issues]
        assert "too_long" in checks

    def test_check_too_long_severity_medium(self):
        lines = ["def test_long():"]
        for i in range(55):
            lines.append(f"    x{i} = {i}")
        lines.append("    assert True")
        code = "\n".join(lines)
        func = _parse_func(code)
        info = analyze_test_function(func, "t.py", max_length=50)
        long_issues = [i for i in info.issues if i.check == "too_long"]
        assert long_issues[0].severity == "medium"

    def test_exactly_at_limit_is_ok(self):
        # Build a function of exactly max_length lines
        lines = ["def test_exact():"]
        for i in range(8):
            lines.append(f"    x{i} = {i}")
        lines.append("    assert True")
        code = "\n".join(lines)
        func = _parse_func(code)
        info = analyze_test_function(func, "t.py", max_length=10)
        checks = [i.check for i in info.issues]
        assert "too_long" not in checks

    def test_custom_max_length(self):
        func = _parse_func("""\
            def test_x():
                a = 1
                b = 2
                c = 3
                d = 4
                e = 5
                assert a == 1
        """)
        info = analyze_test_function(func, "t.py", max_length=5)
        checks = [i.check for i in info.issues]
        assert "too_long" in checks

    # Check 4: too_many_asserts
    def test_check_too_many_asserts(self):
        lines = ["def test_many():"]
        for i in range(20):
            lines.append(f"    assert {i} == {i}")
        code = "\n".join(lines)
        func = _parse_func(code)
        info = analyze_test_function(func, "t.py", max_assertions=15)
        checks = [i.check for i in info.issues]
        assert "too_many_asserts" in checks

    def test_check_too_many_asserts_severity_medium(self):
        lines = ["def test_many():"]
        for i in range(20):
            lines.append(f"    assert {i} == {i}")
        code = "\n".join(lines)
        func = _parse_func(code)
        info = analyze_test_function(func, "t.py", max_assertions=15)
        many_issues = [i for i in info.issues if i.check == "too_many_asserts"]
        assert many_issues[0].severity == "medium"

    def test_exactly_at_assertion_limit_is_ok(self):
        lines = ["def test_exact():"]
        for i in range(5):
            lines.append(f"    assert {i} == {i}")
        code = "\n".join(lines)
        func = _parse_func(code)
        info = analyze_test_function(func, "t.py", max_assertions=5)
        checks = [i.check for i in info.issues]
        assert "too_many_asserts" not in checks

    def test_custom_max_assertions(self):
        func = _parse_func("""\
            def test_x():
                assert 1
                assert 2
                assert 3
        """)
        info = analyze_test_function(func, "t.py", max_assertions=2)
        checks = [i.check for i in info.issues]
        assert "too_many_asserts" in checks

    # Check 5: bare_assert
    def test_check_bare_assert_true(self):
        func = _parse_func("""\
            def test_x():
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "bare_assert" in checks

    def test_check_bare_assert_one(self):
        func = _parse_func("""\
            def test_x():
                assert 1
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "bare_assert" in checks

    def test_check_bare_assert_string(self):
        func = _parse_func("""\
            def test_x():
                assert "hello"
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "bare_assert" in checks

    def test_bare_assert_severity_is_high(self):
        func = _parse_func("def test_x():\n    assert True")
        info = analyze_test_function(func, "t.py")
        bare_issues = [i for i in info.issues if i.check == "bare_assert"]
        assert bare_issues[0].severity == "high"

    def test_bare_assert_only_reported_once(self):
        func = _parse_func("""\
            def test_x():
                assert True
                assert 1
                assert "x"
        """)
        info = analyze_test_function(func, "t.py")
        bare_issues = [i for i in info.issues if i.check == "bare_assert"]
        assert len(bare_issues) == 1  # break after first

    def test_assert_false_is_not_bare(self):
        func = _parse_func("def test_x():\n    assert False")
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "bare_assert" not in checks

    # Check 6: broad_except
    def test_check_broad_except_bare(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except:
                    pass
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "broad_except" in checks

    def test_check_broad_except_exception(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except Exception:
                    pass
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "broad_except" in checks

    def test_broad_except_severity_medium(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except:
                    pass
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        except_issues = [i for i in info.issues if i.check == "broad_except"]
        assert except_issues[0].severity == "medium"

    def test_specific_except_no_issue(self):
        func = _parse_func("""\
            def test_x():
                try:
                    risky()
                except ValueError:
                    pass
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "broad_except" not in checks

    # Check 7: sleep_in_test
    def test_check_sleep_in_test(self):
        func = _parse_func("""\
            def test_x():
                time.sleep(1)
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "sleep_in_test" in checks

    def test_sleep_severity_low(self):
        func = _parse_func("""\
            def test_x():
                time.sleep(1)
                assert True
        """)
        info = analyze_test_function(func, "t.py")
        sleep_issues = [i for i in info.issues if i.check == "sleep_in_test"]
        assert sleep_issues[0].severity == "low"

    # Check 8: no_description (generic names)
    @pytest.mark.parametrize("name", [
        "test_", "test_1", "test_42", "test_it",
        "test_func", "test_method", "test_case", "test_something",
    ])
    def test_check_no_description(self, name):
        func = _parse_func(f"def {name}():\n    assert True")
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "no_description" in checks

    def test_no_description_severity_low(self):
        func = _parse_func("def test_1():\n    assert True")
        info = analyze_test_function(func, "t.py")
        desc_issues = [i for i in info.issues if i.check == "no_description"]
        assert desc_issues[0].severity == "low"

    def test_descriptive_name_ok(self):
        func = _parse_func("def test_user_can_login():\n    assert True")
        info = analyze_test_function(func, "t.py")
        checks = [i.check for i in info.issues]
        assert "no_description" not in checks

    # Check 9: duplicate_name (tested via _track_duplicate and analyze_file)
    # — see TestTrackDuplicate and TestAnalyzeFile sections

    # --- Combined checks ---
    def test_multiple_issues_on_one_function(self):
        """A function can have multiple issues at once."""
        func = _parse_func("""\
            def test_1():
                assert True
                time.sleep(1)
        """)
        info = analyze_test_function(func, "t.py")
        checks = {i.check for i in info.issues}
        # bare_assert + sleep_in_test + no_description
        assert "bare_assert" in checks
        assert "sleep_in_test" in checks
        assert "no_description" in checks

    def test_function_length_computation(self):
        func = _parse_func("""\
            def test_x():
                a = 1
                b = 2
                assert a + b == 3
        """)
        info = analyze_test_function(func, "t.py")
        assert info.length == 4  # line 1 to line 4

    def test_issue_file_field(self):
        func = _parse_func("def test_x():\n    pass")
        info = analyze_test_function(func, "/some/path/test_foo.py")
        assert info.issues[0].file == "/some/path/test_foo.py"

    def test_async_test_function(self):
        func = _parse_func("async def test_async():\n    assert True")
        info = analyze_test_function(func, "t.py")
        assert info.name == "test_async"
        assert info.assertion_count == 1


# =============================================================================
# _track_duplicate
# =============================================================================


class TestTrackDuplicate:
    def test_first_occurrence_no_issue(self):
        seen = {}
        info = TestFunctionInfo(
            name="test_a", qualified_name="test_a",
            file="t.py", line=1, end_line=3, length=3, assertion_count=1,
        )
        _track_duplicate(seen, "test_a", 1, info, "t.py")
        assert len(info.issues) == 0
        assert seen["test_a"] == 1

    def test_duplicate_creates_issue(self):
        seen = {"test_a": 5}
        info = TestFunctionInfo(
            name="test_a", qualified_name="test_a",
            file="t.py", line=20, end_line=22, length=3, assertion_count=1,
        )
        _track_duplicate(seen, "test_a", 20, info, "t.py")
        assert len(info.issues) == 1
        assert info.issues[0].check == "duplicate_name"
        assert info.issues[0].severity == "medium"
        assert "first at line 5" in info.issues[0].message


# =============================================================================
# analyze_file
# =============================================================================


class TestAnalyzeFile:
    def test_good_test_file(self, tmp_path):
        f = tmp_path / "test_good.py"
        f.write_text(GOOD_TEST_FILE)
        report = analyze_file(str(f))
        assert report is not None
        assert report.test_count == 2
        assert report.total_assertions == 2
        assert report.assertion_density == 1.0
        assert report.issues == []

    def test_empty_test_detected(self, tmp_path):
        f = tmp_path / "test_empty.py"
        f.write_text(EMPTY_TEST_FILE)
        report = analyze_file(str(f))
        assert report is not None
        checks = [i.check for i in report.issues]
        assert "empty_test" in checks

    def test_no_assertion_detected(self, tmp_path):
        f = tmp_path / "test_noassert.py"
        f.write_text(NO_ASSERTION_TEST)
        report = analyze_file(str(f))
        assert report is not None
        checks = [i.check for i in report.issues]
        assert "no_assertions" in checks

    def test_syntax_error_returns_none(self, tmp_path):
        f = tmp_path / "test_bad.py"
        f.write_text("def test_broken(:\n    pass")
        report = analyze_file(str(f))
        assert report is None

    def test_non_test_file_returns_none(self, tmp_path):
        f = tmp_path / "test_nontests.py"
        f.write_text("def helper():\n    return 42\n")
        report = analyze_file(str(f))
        assert report is None

    def test_class_methods_analyzed(self, tmp_path):
        f = tmp_path / "test_cls.py"
        f.write_text(textwrap.dedent("""\
            class TestMath:
                def test_add(self):
                    assert 1 + 1 == 2

                def test_sub(self):
                    assert 3 - 1 == 2
        """))
        report = analyze_file(str(f))
        assert report is not None
        assert report.test_count == 2
        names = [t.qualified_name for t in report.tests]
        assert "TestMath.test_add" in names
        assert "TestMath.test_sub" in names

    def test_duplicate_name_in_file(self, tmp_path):
        f = tmp_path / "test_dup.py"
        f.write_text(textwrap.dedent("""\
            def test_foo():
                assert True

            def test_foo():
                assert False
        """))
        report = analyze_file(str(f))
        assert report is not None
        checks = [i.check for i in report.issues]
        assert "duplicate_name" in checks

    def test_custom_thresholds(self, tmp_path):
        f = tmp_path / "test_thresholds.py"
        f.write_text(textwrap.dedent("""\
            def test_x():
                a = 1
                b = 2
                c = 3
                d = 4
                assert a + b + c + d == 10
        """))
        # With max_length=3, this should be too long
        report = analyze_file(str(f), max_length=3)
        assert report is not None
        checks = [i.check for i in report.issues]
        assert "too_long" in checks

    def test_assertion_density_calculation(self, tmp_path):
        f = tmp_path / "test_density.py"
        f.write_text(textwrap.dedent("""\
            def test_a():
                assert 1 == 1
                assert 2 == 2
                assert 3 == 3

            def test_b():
                assert 4 == 4
        """))
        report = analyze_file(str(f))
        assert report is not None
        # 4 assertions / 2 tests = 2.0
        assert report.assertion_density == 2.0

    def test_unicode_file(self, tmp_path):
        f = tmp_path / "test_unicode.py"
        f.write_text('def test_emoji():\n    assert "\U0001f600" == "\U0001f600"\n', encoding="utf-8")
        report = analyze_file(str(f))
        assert report is not None
        assert report.test_count == 1


# =============================================================================
# _is_venv
# =============================================================================


class TestIsVenv:
    def test_venv_with_pyvenv_cfg(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr")
        assert _is_venv(venv) is True

    def test_venv_without_cfg(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        assert _is_venv(venv) is False

    def test_non_venv_name(self, tmp_path):
        d = tmp_path / "src"
        d.mkdir()
        (d / "pyvenv.cfg").write_text("home = /usr")
        assert _is_venv(d) is False

    @pytest.mark.parametrize("name", list(VENV_PATTERNS))
    def test_all_venv_patterns(self, tmp_path, name):
        d = tmp_path / name
        d.mkdir()
        (d / "pyvenv.cfg").write_text("home = /usr")
        assert _is_venv(d) is True


# =============================================================================
# _is_test_file
# =============================================================================


class TestIsTestFile:
    def test_test_prefix(self, tmp_path):
        f = tmp_path / "test_foo.py"
        f.touch()
        assert _is_test_file(f) is True

    def test_test_suffix(self, tmp_path):
        f = tmp_path / "foo_test.py"
        f.touch()
        assert _is_test_file(f) is True

    def test_conftest(self, tmp_path):
        f = tmp_path / "conftest.py"
        f.touch()
        assert _is_test_file(f) is True

    def test_regular_py_file(self, tmp_path):
        f = tmp_path / "utils.py"
        f.touch()
        assert _is_test_file(f) is False

    def test_non_py_file(self, tmp_path):
        f = tmp_path / "test_foo.txt"
        f.touch()
        assert _is_test_file(f) is False

    def test_test_in_middle(self, tmp_path):
        f = tmp_path / "my_test_helper.py"
        f.touch()
        assert _is_test_file(f) is False


# =============================================================================
# discover_test_files
# =============================================================================


class TestDiscoverTestFiles:
    def test_find_test_files(self, tmp_path):
        (tmp_path / "test_a.py").write_text("def test_a(): assert True")
        (tmp_path / "test_b.py").write_text("def test_b(): assert True")
        (tmp_path / "utils.py").write_text("x = 1")
        files = discover_test_files(str(tmp_path))
        assert len(files) == 2

    def test_find_in_subdirectories(self, tmp_path):
        sub = tmp_path / "tests"
        sub.mkdir()
        (sub / "test_sub.py").write_text("def test_s(): assert True")
        (tmp_path / "test_top.py").write_text("def test_t(): assert True")
        files = discover_test_files(str(tmp_path))
        assert len(files) == 2

    def test_skip_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "test_cached.py").write_text("def test_c(): pass")
        (tmp_path / "test_real.py").write_text("def test_r(): assert True")
        files = discover_test_files(str(tmp_path))
        assert len(files) == 1

    def test_skip_venv(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr")
        (venv / "test_venv.py").write_text("def test_v(): pass")
        (tmp_path / "test_real.py").write_text("def test_r(): assert True")
        files = discover_test_files(str(tmp_path))
        assert len(files) == 1

    def test_single_file_input(self, tmp_path):
        f = tmp_path / "test_single.py"
        f.write_text("def test_s(): assert True")
        files = discover_test_files(str(f))
        assert len(files) == 1
        assert files[0] == str(f)

    def test_single_non_py_file(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        files = discover_test_files(str(f))
        assert files == []

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "test_z.py").write_text("def test_z(): assert True")
        (tmp_path / "test_a.py").write_text("def test_a(): assert True")
        (tmp_path / "test_m.py").write_text("def test_m(): assert True")
        files = discover_test_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in files]
        assert basenames == sorted(basenames)

    def test_finds_test_suffix_files(self, tmp_path):
        (tmp_path / "foo_test.py").write_text("def test_foo(): assert True")
        files = discover_test_files(str(tmp_path))
        assert len(files) == 1

    def test_finds_conftest(self, tmp_path):
        (tmp_path / "conftest.py").write_text("import pytest")
        files = discover_test_files(str(tmp_path))
        assert len(files) == 1

    def test_empty_directory(self, tmp_path):
        files = discover_test_files(str(tmp_path))
        assert files == []

    def test_skip_git_directory(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "test_gitfile.py").write_text("def test_g(): pass")
        files = discover_test_files(str(tmp_path))
        assert files == []


# =============================================================================
# compute_score — boundary conditions
# =============================================================================


class TestComputeScore:
    def test_no_tests_perfect_score(self):
        score, grade = compute_score(0, [])
        assert score == 100
        assert grade == "A"

    def test_no_issues_perfect_score(self):
        score, grade = compute_score(10, [])
        assert score == 100
        assert grade == "A"

    def test_single_high_issue(self):
        issues = [_make_issue(severity="high")]
        # deduction = 5, max_deduction = 1*3=3, score = max(0, 100 - 100*5/3) = 0
        score, grade = compute_score(1, issues)
        assert score == 0
        assert grade == "F"

    def test_single_low_issue_many_tests(self):
        issues = [_make_issue(severity="low")]
        # deduction = 1, max_deduction = 100*3=300, score = 100 - int(100*1/300) = 100
        score, grade = compute_score(100, issues)
        assert score == 100
        assert grade == "A"

    def test_score_floors_at_zero(self):
        issues = [_make_issue(severity="high") for _ in range(100)]
        score, grade = compute_score(1, issues)
        assert score == 0
        assert grade == "F"

    # Grade boundary tests
    def test_grade_a_at_90(self):
        # Need score exactly 90
        # score = max(0, 100 - int(100 * deduction / max_deduction))
        # For 10 tests: max_deduction = 30
        # To get score=90: 100 - int(100*d/30) = 90 => d = 3
        # 1 medium issue = weight 3
        issues = [_make_issue(severity="medium")]
        score, grade = compute_score(10, issues)
        assert score == 90
        assert grade == "A"

    def test_grade_b_boundary(self):
        # score = 80..89 -> B
        # For 10 tests: max_deduction = 30
        # 100 - int(100*d/30) for d=6 => 100 - 20 = 80 -> B
        issues = [_make_issue(severity="medium"), _make_issue(severity="medium")]
        score, grade = compute_score(10, issues)
        assert score == 80
        assert grade == "B"

    def test_grade_c_boundary(self):
        # For 10 tests: max_deduction = 30
        # 100 - int(100*d/30) for d=9 => 100 - 30 = 70 -> C
        issues = [_make_issue(severity="medium") for _ in range(3)]
        score, grade = compute_score(10, issues)
        assert score == 70
        assert grade == "C"

    def test_grade_d_boundary(self):
        # For 10 tests: max_deduction = 30
        # 100 - int(100*d/30) for d=15 => 100 - 50 = 50 -> D
        issues = [_make_issue(severity="high") for _ in range(3)]
        score, grade = compute_score(10, issues)
        assert score == 50
        assert grade == "D"

    def test_grade_f_below_50(self):
        # For 10 tests: max_deduction = 30
        # d=16 => 100 - int(100*16/30) = 100 - 53 = 47 -> F
        issues = [_make_issue(severity="high") for _ in range(3)]
        issues.append(_make_issue(severity="low"))
        score, grade = compute_score(10, issues)
        assert score < 50
        assert grade == "F"

    def test_severity_weight_impact(self):
        """High severity issues should cause more deduction than low."""
        high_issues = [_make_issue(severity="high")]
        low_issues = [_make_issue(severity="low")]
        score_high, _ = compute_score(10, high_issues)
        score_low, _ = compute_score(10, low_issues)
        assert score_high < score_low

    def test_unknown_severity_defaults_to_1(self):
        issues = [_make_issue(severity="unknown")]
        score, _ = compute_score(10, issues)
        # deduction = 1, max_deduction = 30, score = 100 - int(100/30) = 97
        assert score == 97


# =============================================================================
# analyze_suite
# =============================================================================


class TestAnalyzeSuite:
    def test_clean_suite(self, tmp_path):
        (tmp_path / "test_a.py").write_text(GOOD_TEST_FILE)
        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 1
        assert report.total_tests == 2
        assert report.total_assertions == 2
        assert report.total_issues == 0
        assert report.score == 100
        assert report.grade == "A"

    def test_suite_with_issues(self, tmp_path):
        (tmp_path / "test_a.py").write_text(EMPTY_TEST_FILE)
        report = analyze_suite(str(tmp_path))
        assert report.total_issues > 0
        assert report.score < 100

    def test_empty_directory(self, tmp_path):
        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 0
        assert report.total_tests == 0
        assert report.score == 100

    def test_severity_filter(self, tmp_path):
        # File with both high and low severity issues
        (tmp_path / "test_mixed.py").write_text(textwrap.dedent("""\
            def test_1():
                assert True
                time.sleep(1)
        """))
        report = analyze_suite(str(tmp_path), severity_filter="high")
        # Only high severity issues (bare_assert is high)
        for sev in report.issues_by_severity:
            assert sev == "high"

    def test_issues_by_check_populated(self, tmp_path):
        (tmp_path / "test_empty.py").write_text(EMPTY_TEST_FILE)
        report = analyze_suite(str(tmp_path))
        assert "empty_test" in report.issues_by_check

    def test_issues_by_severity_populated(self, tmp_path):
        (tmp_path / "test_empty.py").write_text(EMPTY_TEST_FILE)
        report = analyze_suite(str(tmp_path))
        assert "high" in report.issues_by_severity

    def test_multiple_files(self, tmp_path):
        (tmp_path / "test_a.py").write_text(GOOD_TEST_FILE)
        (tmp_path / "test_b.py").write_text(textwrap.dedent("""\
            def test_third():
                assert 3 == 3
        """))
        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 2
        assert report.total_tests == 3

    def test_custom_max_length(self, tmp_path):
        (tmp_path / "test_len.py").write_text(textwrap.dedent("""\
            def test_x():
                a = 1
                b = 2
                c = 3
                d = 4
                assert a + b + c + d == 10
        """))
        report = analyze_suite(str(tmp_path), max_length=3)
        checks = set()
        for fr in report.files:
            for i in fr.issues:
                checks.add(i.check)
        assert "too_long" in checks

    def test_syntax_error_file_skipped(self, tmp_path):
        (tmp_path / "test_bad.py").write_text("def test_broken(:\n    pass")
        (tmp_path / "test_good.py").write_text(GOOD_TEST_FILE)
        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 1  # bad file skipped

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("home = /usr")
        (venv / "test_venv.py").write_text(GOOD_TEST_FILE)
        (tmp_path / "test_real.py").write_text(GOOD_TEST_FILE)
        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 1


# =============================================================================
# format_text
# =============================================================================


class TestFormatText:
    def test_clean_report(self):
        report = SuiteReport(
            files_analyzed=3, total_tests=10,
            total_assertions=15, total_issues=0,
            score=100, grade="A",
        )
        text = format_text(report)
        assert "Score: 100/100 (A)" in text
        assert "Files: 3" in text
        assert "Tests: 10" in text
        assert "Assertions: 15" in text
        assert "No issues found" in text

    def test_report_with_issues(self):
        issue = _make_issue(check="empty_test", severity="high",
                            message="Test body is empty", line=5,
                            function="test_x", file="/path/test.py")
        fr = FileReport(
            file="/path/test.py", test_count=1,
            total_assertions=0, assertion_density=0.0,
            issues=[issue],
        )
        report = SuiteReport(
            files_analyzed=1, total_tests=1,
            total_assertions=0, total_issues=1,
            issues_by_severity={"high": 1},
            issues_by_check={"empty_test": 1},
            score=80, grade="B",
            files=[fr],
        )
        text = format_text(report)
        assert "Score: 80/100 (B)" in text
        assert "empty_test" in text
        assert "L5" in text
        assert "test_x" in text

    def test_severity_icons(self):
        issues = [
            _make_issue(severity="high", line=1, function="test_h"),
            _make_issue(severity="medium", line=2, function="test_m"),
            _make_issue(severity="low", line=3, function="test_l"),
        ]
        fr = FileReport(
            file="test.py", test_count=3,
            total_assertions=3, assertion_density=1.0,
            issues=issues,
        )
        report = SuiteReport(
            files_analyzed=1, total_tests=3,
            total_assertions=3, total_issues=3,
            issues_by_severity={"high": 1, "medium": 1, "low": 1},
            issues_by_check={"check": 3},
            score=70, grade="C",
            files=[fr],
        )
        text = format_text(report)
        assert "[!]" in text  # high
        assert "[~]" in text  # medium
        assert "[.]" in text  # low

    def test_verbose_shows_clean_files(self):
        fr = FileReport(
            file="test_clean.py", test_count=5,
            total_assertions=10, assertion_density=2.0,
            issues=[],
        )
        report = SuiteReport(
            files_analyzed=1, total_tests=5,
            total_assertions=10, total_issues=0,
            score=100, grade="A",
            files=[fr],
        )
        text_normal = format_text(report, verbose=False)
        text_verbose = format_text(report, verbose=True)
        # verbose should include the clean file
        assert "test_clean.py" not in text_normal
        assert "test_clean.py" in text_verbose

    def test_assertion_density_displayed(self):
        report = SuiteReport(
            files_analyzed=1, total_tests=4,
            total_assertions=12, total_issues=0,
            score=100, grade="A",
        )
        text = format_text(report)
        assert "Assertion density: 3.0 per test" in text

    def test_zero_tests_no_density(self):
        report = SuiteReport(
            files_analyzed=0, total_tests=0,
            total_assertions=0, total_issues=0,
            score=100, grade="A",
        )
        text = format_text(report)
        assert "Assertion density" not in text

    def test_issues_by_check_section(self):
        report = SuiteReport(
            files_analyzed=1, total_tests=2,
            total_assertions=1, total_issues=2,
            issues_by_check={"empty_test": 1, "no_assertions": 1},
            issues_by_severity={"high": 2},
            score=60, grade="D",
        )
        text = format_text(report)
        assert "Issues by check:" in text
        assert "empty_test" in text
        assert "no_assertions" in text


# =============================================================================
# format_json
# =============================================================================


class TestFormatJson:
    def test_json_structure(self):
        report = SuiteReport(
            files_analyzed=2, total_tests=5,
            total_assertions=8, total_issues=0,
            score=100, grade="A",
        )
        data = json.loads(format_json(report))
        assert data["files_analyzed"] == 2
        assert data["total_tests"] == 5
        assert data["total_assertions"] == 8
        assert data["total_issues"] == 0
        assert data["score"] == 100
        assert data["grade"] == "A"
        assert data["files"] == []

    def test_json_with_issues(self):
        issue = _make_issue(check="empty_test", severity="high")
        fr = FileReport(
            file="test.py", test_count=1,
            total_assertions=0, assertion_density=0.0,
            issues=[issue],
        )
        report = SuiteReport(
            files_analyzed=1, total_tests=1,
            total_assertions=0, total_issues=1,
            issues_by_severity={"high": 1},
            issues_by_check={"empty_test": 1},
            score=80, grade="B",
            files=[fr],
        )
        data = json.loads(format_json(report))
        assert len(data["files"]) == 1
        assert data["files"][0]["test_count"] == 1
        assert len(data["files"][0]["issues"]) == 1
        assert data["files"][0]["issues"][0]["check"] == "empty_test"

    def test_json_excludes_clean_files(self):
        clean_fr = FileReport(
            file="test_clean.py", test_count=5,
            total_assertions=10, assertion_density=2.0,
            issues=[],
        )
        report = SuiteReport(
            files_analyzed=1, total_tests=5,
            total_assertions=10, total_issues=0,
            score=100, grade="A",
            files=[clean_fr],
        )
        data = json.loads(format_json(report))
        assert data["files"] == []

    def test_json_valid_format(self):
        report = SuiteReport(
            files_analyzed=0, total_tests=0,
            total_assertions=0, total_issues=0,
        )
        output = format_json(report)
        # Should not raise
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_json_issues_by_severity(self):
        report = SuiteReport(
            files_analyzed=1, total_tests=3,
            total_assertions=1, total_issues=2,
            issues_by_severity={"high": 1, "low": 1},
            issues_by_check={"empty_test": 1, "no_description": 1},
            score=70, grade="C",
        )
        data = json.loads(format_json(report))
        assert data["issues_by_severity"]["high"] == 1
        assert data["issues_by_severity"]["low"] == 1


# =============================================================================
# _relative_path
# =============================================================================


class TestRelativePath:
    def test_absolute_within_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "subdir" / "test.py"
        result = _relative_path(str(f))
        assert result == str(Path("subdir") / "test.py")

    def test_absolute_outside_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        result = _relative_path("/some/other/path/test.py")
        assert result == "/some/other/path/test.py"

    def test_already_relative(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # If the path is already relative, relative_to may fail
        # and it should return the path as-is
        result = _relative_path("test.py")
        # Could be "test.py" if not resolvable
        assert isinstance(result, str)


# =============================================================================
# build_parser
# =============================================================================


class TestBuildParser:
    def test_default_path(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.path == "."

    def test_custom_path(self):
        parser = build_parser()
        args = parser.parse_args(["src/tests"])
        assert args.path == "src/tests"

    def test_single_file(self):
        parser = build_parser()
        args = parser.parse_args(["-f", "test_foo.py"])
        assert args.single_file == "test_foo.py"

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json is True

    def test_threshold(self):
        parser = build_parser()
        args = parser.parse_args(["--threshold", "80"])
        assert args.threshold == 80

    def test_max_length(self):
        parser = build_parser()
        args = parser.parse_args(["--max-length", "30"])
        assert args.max_length == 30

    def test_max_assertions(self):
        parser = build_parser()
        args = parser.parse_args(["--max-assertions", "10"])
        assert args.max_assertions == 10

    def test_severity_filter(self):
        parser = build_parser()
        args = parser.parse_args(["--severity", "high"])
        assert args.severity == "high"

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_default_threshold_zero(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.threshold == 0

    def test_default_max_length(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.max_length == DEFAULT_MAX_TEST_LENGTH

    def test_default_max_assertions(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.max_assertions == DEFAULT_MAX_ASSERTIONS

    def test_all_args_combined(self):
        parser = build_parser()
        args = parser.parse_args([
            "path/to/tests",
            "--json",
            "--threshold", "70",
            "--max-length", "40",
            "--max-assertions", "8",
            "--severity", "medium",
            "-v",
        ])
        assert args.path == "path/to/tests"
        assert args.json is True
        assert args.threshold == 70
        assert args.max_length == 40
        assert args.max_assertions == 8
        assert args.severity == "medium"
        assert args.verbose is True


# =============================================================================
# main
# =============================================================================


class TestMain:
    def test_clean_directory_returns_zero(self, tmp_path):
        (tmp_path / "test_ok.py").write_text(GOOD_TEST_FILE)
        ret = main([str(tmp_path)])
        assert ret == 0

    def test_threshold_exceeded_returns_one(self, tmp_path):
        (tmp_path / "test_bad.py").write_text(EMPTY_TEST_FILE)
        ret = main([str(tmp_path), "--threshold", "90"])
        assert ret == 1

    def test_threshold_not_exceeded_returns_zero(self, tmp_path):
        (tmp_path / "test_ok.py").write_text(GOOD_TEST_FILE)
        ret = main([str(tmp_path), "--threshold", "90"])
        assert ret == 0

    def test_zero_threshold_always_passes(self, tmp_path):
        (tmp_path / "test_bad.py").write_text(EMPTY_TEST_FILE)
        ret = main([str(tmp_path), "--threshold", "0"])
        assert ret == 0

    def test_json_output(self, tmp_path, capsys):
        (tmp_path / "test_ok.py").write_text(GOOD_TEST_FILE)
        main([str(tmp_path), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "score" in data
        assert "grade" in data

    def test_text_output(self, tmp_path, capsys):
        (tmp_path / "test_ok.py").write_text(GOOD_TEST_FILE)
        main([str(tmp_path)])
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Test Quality Report" in captured.out

    def test_single_file_flag(self, tmp_path):
        f = tmp_path / "test_single.py"
        f.write_text(GOOD_TEST_FILE)
        ret = main(["-f", str(f)])
        assert ret == 0

    def test_empty_directory(self, tmp_path):
        ret = main([str(tmp_path)])
        assert ret == 0

    def test_verbose_flag(self, tmp_path, capsys):
        (tmp_path / "test_ok.py").write_text(GOOD_TEST_FILE)
        main([str(tmp_path), "-v"])
        captured = capsys.readouterr()
        assert "test_ok.py" in captured.out

    def test_severity_filter(self, tmp_path, capsys):
        (tmp_path / "test_mixed.py").write_text(textwrap.dedent("""\
            def test_1():
                assert True
                time.sleep(1)
        """))
        main([str(tmp_path), "--severity", "low", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Should still report issues (severity filter affects count display)
        assert isinstance(data["total_issues"], int)

    def test_custom_max_length(self, tmp_path, capsys):
        (tmp_path / "test_len.py").write_text(textwrap.dedent("""\
            def test_x():
                a = 1
                b = 2
                c = 3
                d = 4
                assert a + b + c + d == 10
        """))
        main([str(tmp_path), "--max-length", "3", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_issues"] > 0

    def test_custom_max_assertions(self, tmp_path, capsys):
        lines = ["def test_many():"]
        for i in range(5):
            lines.append(f"    assert {i} == {i}")
        (tmp_path / "test_many.py").write_text("\n".join(lines) + "\n")
        main([str(tmp_path), "--max-assertions", "3", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "too_many_asserts" in data.get("issues_by_check", {})


# =============================================================================
# Integration: full pipeline
# =============================================================================


class TestIntegration:
    def test_full_pipeline_with_mixed_quality(self, tmp_path):
        """End-to-end test: suite with good and bad tests."""
        (tmp_path / "test_good.py").write_text(GOOD_TEST_FILE)
        (tmp_path / "test_empty.py").write_text(EMPTY_TEST_FILE)
        (tmp_path / "test_noassert.py").write_text(NO_ASSERTION_TEST)

        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 3
        assert report.total_tests == 4  # 2 + 1 + 1
        assert report.total_issues > 0
        assert report.score < 100

        # Verify text output works
        text = format_text(report)
        assert "Score:" in text

        # Verify JSON output works
        json_str = format_json(report)
        data = json.loads(json_str)
        assert data["total_tests"] == 4

    def test_all_check_types_in_one_suite(self, tmp_path):
        """Suite with examples of every check type."""
        # empty_test
        (tmp_path / "test_empty.py").write_text("def test_e():\n    pass\n")

        # no_assertions
        (tmp_path / "test_noassert.py").write_text(
            "def test_na():\n    x = 1\n    print(x)\n"
        )

        # bare_assert + no_description + sleep_in_test
        (tmp_path / "test_multi.py").write_text(textwrap.dedent("""\
            def test_1():
                assert True
                time.sleep(1)
        """))

        # broad_except
        (tmp_path / "test_except.py").write_text(textwrap.dedent("""\
            def test_broad():
                try:
                    risky()
                except:
                    pass
                assert True
        """))

        # duplicate_name
        (tmp_path / "test_dup.py").write_text(textwrap.dedent("""\
            def test_dup():
                assert 1 == 1

            def test_dup():
                assert 2 == 2
        """))

        report = analyze_suite(str(tmp_path))
        all_checks = set()
        for fr in report.files:
            for issue in fr.issues:
                all_checks.add(issue.check)

        assert "empty_test" in all_checks
        assert "no_assertions" in all_checks
        assert "bare_assert" in all_checks
        assert "no_description" in all_checks
        assert "sleep_in_test" in all_checks
        assert "broad_except" in all_checks
        assert "duplicate_name" in all_checks

    def test_too_long_and_too_many_asserts(self, tmp_path):
        """Tests for too_long and too_many_asserts."""
        lines = ["def test_overloaded():"]
        for i in range(60):
            lines.append(f"    assert {i} == {i}")
        (tmp_path / "test_overloaded.py").write_text("\n".join(lines) + "\n")

        report = analyze_suite(str(tmp_path), max_length=50, max_assertions=15)
        all_checks = set()
        for fr in report.files:
            for issue in fr.issues:
                all_checks.add(issue.check)
        assert "too_long" in all_checks
        assert "too_many_asserts" in all_checks

    def test_conftest_not_analyzed_as_tests(self, tmp_path):
        """conftest.py typically has no test functions."""
        (tmp_path / "conftest.py").write_text(textwrap.dedent("""\
            import pytest

            @pytest.fixture
            def db():
                return {}
        """))
        report = analyze_suite(str(tmp_path))
        # conftest discovered but no test functions => not analyzed
        assert report.files_analyzed == 0

    def test_nested_test_directories(self, tmp_path):
        """Tests in nested directories are discovered."""
        unit = tmp_path / "tests" / "unit"
        unit.mkdir(parents=True)
        (unit / "test_unit.py").write_text(GOOD_TEST_FILE)

        integ = tmp_path / "tests" / "integration"
        integ.mkdir(parents=True)
        (integ / "test_integ.py").write_text(GOOD_TEST_FILE)

        report = analyze_suite(str(tmp_path))
        assert report.files_analyzed == 2
        assert report.total_tests == 4


