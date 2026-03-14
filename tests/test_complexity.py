"""Tests for the complexity analysis engine."""
from __future__ import annotations

import ast
import textwrap

import pytest

from code_health_suite.engines import complexity
from code_health_suite.engines.complexity import (
    AnalysisResult,
    ComplexityScore,
    FunctionMetrics,
    ModuleMetrics,
    analyze_module,
    classify_complexity_profile,
    compute_cognitive,
    compute_complexity_score,
    compute_cyclomatic,
    compute_length,
    compute_max_nesting,
    count_parameters,
    find_python_files,
    _is_venv_path,
)


# --- Helper ---

def _parse_func(code: str) -> ast.AST:
    """Parse a code snippet and return the first function node."""
    tree = ast.parse(textwrap.dedent(code))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("No function found in code")


def _make_fn(cyclomatic: int = 1, cognitive: int = 0, length: int = 5,
             max_nesting: int = 0, grade: str = "A") -> FunctionMetrics:
    """Create a FunctionMetrics with specific values for testing aggregation."""
    # Reverse-engineer cyclomatic from grade if needed
    return FunctionMetrics(
        file="test.py", name="f", qualified_name="f",
        line=1, end_line=length,
        cyclomatic=cyclomatic, cognitive=cognitive,
        length=length, max_nesting=max_nesting,
    )


# ============================================================
# Cyclomatic Complexity
# ============================================================


class TestComputeCyclomatic:
    """Tests for compute_cyclomatic()."""

    def test_empty_function(self):
        node = _parse_func("def f(): pass")
        assert compute_cyclomatic(node) == 1  # base path

    def test_single_if(self):
        node = _parse_func("""
            def f(x):
                if x > 0:
                    return x
                return -x
        """)
        assert compute_cyclomatic(node) == 2

    def test_if_elif_else(self):
        node = _parse_func("""
            def f(x):
                if x > 0:
                    return 1
                elif x == 0:
                    return 0
                else:
                    return -1
        """)
        # if + elif = 2 branches, else doesn't count
        assert compute_cyclomatic(node) == 3

    def test_for_loop(self):
        node = _parse_func("""
            def f(items):
                for item in items:
                    print(item)
        """)
        assert compute_cyclomatic(node) == 2

    def test_while_loop(self):
        node = _parse_func("""
            def f(n):
                while n > 0:
                    n -= 1
        """)
        assert compute_cyclomatic(node) == 2

    def test_try_except(self):
        node = _parse_func("""
            def f():
                try:
                    do_something()
                except ValueError:
                    handle_error()
        """)
        # except handler = +1
        assert compute_cyclomatic(node) == 2

    def test_try_multiple_except(self):
        node = _parse_func("""
            def f():
                try:
                    do_something()
                except ValueError:
                    pass
                except TypeError:
                    pass
                except (OSError, IOError):
                    pass
        """)
        assert compute_cyclomatic(node) == 4  # 1 + 3 except handlers

    def test_with_statement(self):
        node = _parse_func("""
            def f():
                with open("file") as f:
                    data = f.read()
        """)
        assert compute_cyclomatic(node) == 2

    def test_async_for(self):
        node = _parse_func("""
            async def f(items):
                async for item in items:
                    await process(item)
        """)
        assert compute_cyclomatic(node) == 2

    def test_async_with(self):
        node = _parse_func("""
            async def f():
                async with aopen("file") as f:
                    data = await f.read()
        """)
        assert compute_cyclomatic(node) == 2

    def test_assert(self):
        node = _parse_func("""
            def f(x):
                assert x > 0
                return x
        """)
        assert compute_cyclomatic(node) == 2

    def test_boolean_and(self):
        node = _parse_func("""
            def f(a, b):
                if a and b:
                    return True
        """)
        # if (+1) + and (+1) = base 1 + 2 = 3
        assert compute_cyclomatic(node) == 3

    def test_boolean_or(self):
        node = _parse_func("""
            def f(a, b):
                if a or b:
                    return True
        """)
        assert compute_cyclomatic(node) == 3

    def test_boolean_chain(self):
        node = _parse_func("""
            def f(a, b, c):
                if a and b and c:
                    return True
        """)
        # if (+1) + and with 3 values (+2) = 1 + 3 = 4
        assert compute_cyclomatic(node) == 4

    def test_ternary(self):
        node = _parse_func("""
            def f(x):
                return x if x > 0 else -x
        """)
        assert compute_cyclomatic(node) == 2  # base + IfExp

    def test_list_comprehension_with_if(self):
        node = _parse_func("""
            def f(items):
                return [x for x in items if x > 0]
        """)
        # comprehension: +1 for the for, +1 for the if filter
        assert compute_cyclomatic(node) == 3

    def test_nested_comprehension(self):
        node = _parse_func("""
            def f(matrix):
                return [x for row in matrix for x in row]
        """)
        # Two comprehension clauses: +1 each for the for
        assert compute_cyclomatic(node) == 3

    def test_nested_function_not_counted(self):
        """Nested function's complexity should NOT leak into outer."""
        node = _parse_func("""
            def outer():
                def inner():
                    if True:
                        pass
                    if True:
                        pass
                return inner
        """)
        # outer has no branches itself (nested inner is skipped by walk_scope_bfs)
        assert compute_cyclomatic(node) == 1

    def test_complex_function(self):
        """A realistic complex function."""
        node = _parse_func("""
            def process(data, mode):
                if not data:
                    return None
                for item in data:
                    if mode == "a":
                        try:
                            handle_a(item)
                        except ValueError:
                            log(item)
                    elif mode == "b":
                        handle_b(item)
                    else:
                        if item and item.valid:
                            handle_default(item)
                return data
        """)
        # 1 (base) + 1 (if not data) + 1 (for) + 1 (if mode==a) + 1 (except)
        # + 1 (elif mode==b) + 1 (if item and) + 1 (and) = 8
        assert compute_cyclomatic(node) == 8


# ============================================================
# Cognitive Complexity
# ============================================================


class TestComputeCognitive:
    """Tests for compute_cognitive()."""

    def test_empty_function(self):
        node = _parse_func("def f(): pass")
        assert compute_cognitive(node) == 0

    def test_single_if(self):
        node = _parse_func("""
            def f(x):
                if x > 0:
                    return x
        """)
        assert compute_cognitive(node) == 1  # +1 structural, nesting=0

    def test_nested_if(self):
        node = _parse_func("""
            def f(x, y):
                if x > 0:
                    if y > 0:
                        return x + y
        """)
        # outer if: +1 (nesting=0), inner if: +1 + 1 (nesting=1) = 3
        assert compute_cognitive(node) == 3

    def test_deeply_nested(self):
        node = _parse_func("""
            def f(a, b, c):
                if a:
                    if b:
                        if c:
                            return True
        """)
        # if a: +1 (n=0), if b: +1+1 (n=1), if c: +1+2 (n=2) = 6
        assert compute_cognitive(node) == 6

    def test_for_adds_nesting(self):
        node = _parse_func("""
            def f(items):
                for item in items:
                    if item > 0:
                        print(item)
        """)
        # for: +1 (n=0), if: +1+1 (n=1) = 3
        assert compute_cognitive(node) == 3

    def test_break_adds_fundamental(self):
        node = _parse_func("""
            def f(items):
                for item in items:
                    if item < 0:
                        break
        """)
        # for: +1 (n=0), if: +1+1 (n=1), break: +1 = 4
        assert compute_cognitive(node) == 4

    def test_continue_adds_fundamental(self):
        node = _parse_func("""
            def f(items):
                for item in items:
                    if item < 0:
                        continue
                    process(item)
        """)
        # for: +1 (n=0), if: +1+1 (n=1), continue: +1 = 4
        assert compute_cognitive(node) == 4

    def test_boolean_op(self):
        node = _parse_func("""
            def f(a, b):
                if a and b:
                    return True
        """)
        # if: +1 (n=0), and: +1 = 2
        assert compute_cognitive(node) == 2

    def test_ternary(self):
        node = _parse_func("""
            def f(x):
                return x if x > 0 else -x
        """)
        # IfExp: +1
        assert compute_cognitive(node) == 1

    def test_nested_function_increments(self):
        """Nested function definition adds +1 structural."""
        node = _parse_func("""
            def outer():
                def inner():
                    pass
                return inner
        """)
        assert compute_cognitive(node) == 1

    def test_except_handler_nesting(self):
        node = _parse_func("""
            def f():
                try:
                    x = 1
                except ValueError:
                    if True:
                        pass
        """)
        # except: +1 (n=0), if: +1+1 (n=1) = 3
        assert compute_cognitive(node) == 3


# ============================================================
# Max Nesting Depth
# ============================================================


class TestComputeMaxNesting:
    def test_flat_function(self):
        node = _parse_func("""
            def f():
                x = 1
                y = 2
                return x + y
        """)
        assert compute_max_nesting(node) == 0

    def test_single_if(self):
        node = _parse_func("def f():\n    if True: pass")
        assert compute_max_nesting(node) == 1

    def test_nested_ifs(self):
        node = _parse_func("""
            def f():
                if True:
                    if True:
                        if True:
                            pass
        """)
        assert compute_max_nesting(node) == 3

    def test_try_except_nesting(self):
        node = _parse_func("""
            def f():
                try:
                    if True:
                        pass
                except:
                    pass
        """)
        # try: depth 1, if inside try: depth 2, except: depth 1
        assert compute_max_nesting(node) == 2

    def test_for_with_if(self):
        node = _parse_func("""
            def f(items):
                for item in items:
                    if item:
                        while True:
                            break
        """)
        assert compute_max_nesting(node) == 3

    def test_nested_function_resets(self):
        """Nested function resets nesting depth."""
        node = _parse_func("""
            def outer():
                if True:
                    def inner():
                        if True:
                            pass
        """)
        # outer nesting: if=1, inner resets to 0 then if=1
        # max nesting in outer scope is 1
        assert compute_max_nesting(node) == 1


# ============================================================
# Function Length
# ============================================================


class TestComputeLength:
    def test_one_liner(self):
        node = _parse_func("def f(): pass")
        assert compute_length(node) == 1

    def test_multi_line(self):
        node = _parse_func("""
            def f():
                x = 1
                y = 2
                z = 3
                return x + y + z
        """)
        assert compute_length(node) == 5


# ============================================================
# Parameter Count
# ============================================================


class TestCountParameters:
    def test_no_params(self):
        node = _parse_func("def f(): pass")
        assert count_parameters(node) == 0

    def test_positional_params(self):
        node = _parse_func("def f(a, b, c): pass")
        assert count_parameters(node) == 3

    def test_self_counts(self):
        """self parameter is counted."""
        node = _parse_func("def f(self, x): pass")
        assert count_parameters(node) == 2

    def test_varargs(self):
        node = _parse_func("def f(*args): pass")
        assert count_parameters(node) == 1

    def test_kwargs(self):
        node = _parse_func("def f(**kwargs): pass")
        assert count_parameters(node) == 1

    def test_mixed(self):
        node = _parse_func("def f(a, b, *args, c=1, **kwargs): pass")
        # a, b (args), *args (vararg), c (kwonlyargs), **kwargs (kwarg)
        assert count_parameters(node) == 5

    def test_non_function_returns_zero(self):
        tree = ast.parse("x = 1")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assert count_parameters(node) == 0
                break


# ============================================================
# FunctionMetrics Grade
# ============================================================


class TestFunctionMetricsGrade:
    def test_grade_a(self):
        fn = _make_fn(cyclomatic=1)
        assert fn.grade == "A"
        assert fn.grade_label == "simple"

    def test_grade_b(self):
        fn = _make_fn(cyclomatic=7)
        assert fn.grade == "B"
        assert fn.grade_label == "moderate"

    def test_grade_c(self):
        fn = _make_fn(cyclomatic=15)
        assert fn.grade == "C"
        assert fn.grade_label == "complex"

    def test_grade_d(self):
        fn = _make_fn(cyclomatic=30)
        assert fn.grade == "D"
        assert fn.grade_label == "very complex"

    def test_grade_f(self):
        fn = _make_fn(cyclomatic=60)
        assert fn.grade == "F"
        assert fn.grade_label == "untestable"

    def test_grade_boundaries(self):
        assert _make_fn(cyclomatic=5).grade == "A"
        assert _make_fn(cyclomatic=6).grade == "B"
        assert _make_fn(cyclomatic=10).grade == "B"
        assert _make_fn(cyclomatic=11).grade == "C"
        assert _make_fn(cyclomatic=20).grade == "C"
        assert _make_fn(cyclomatic=21).grade == "D"
        assert _make_fn(cyclomatic=50).grade == "D"
        assert _make_fn(cyclomatic=51).grade == "F"

    def test_to_dict_includes_grade(self):
        fn = _make_fn(cyclomatic=3)
        d = fn.to_dict()
        assert d["grade"] == "A"
        assert d["grade_label"] == "simple"
        assert d["cyclomatic"] == 3


# ============================================================
# analyze_module
# ============================================================


class TestAnalyzeModule:
    def test_empty_file(self):
        mod = analyze_module("empty.py", "")
        assert mod.total_lines == 0
        assert mod.function_count == 0
        assert mod.functions == []

    def test_simple_function(self):
        source = textwrap.dedent("""\
            def add(a, b):
                return a + b
        """)
        mod = analyze_module("add.py", source)
        assert mod.function_count == 1
        assert mod.functions[0].name == "add"
        assert mod.functions[0].cyclomatic == 1
        assert mod.functions[0].parameter_count == 2

    def test_class_method(self):
        source = textwrap.dedent("""\
            class Calc:
                def add(self, a, b):
                    return a + b
        """)
        mod = analyze_module("calc.py", source)
        assert mod.class_count == 1
        assert mod.function_count == 1
        fn = mod.functions[0]
        assert fn.qualified_name == "Calc.add"
        assert fn.is_method is True
        assert fn.parameter_count == 3  # self, a, b

    def test_multiple_functions(self):
        source = textwrap.dedent("""\
            import os

            def foo():
                pass

            def bar(x):
                if x:
                    return 1
                return 0

            class C:
                def baz(self):
                    pass
        """)
        mod = analyze_module("multi.py", source)
        assert mod.function_count == 3
        assert mod.import_count == 1
        assert mod.class_count == 1
        names = [f.name for f in mod.functions]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names

    def test_module_aggregates(self):
        source = textwrap.dedent("""\
            def simple():
                pass

            def branchy(x, y):
                if x:
                    if y:
                        return 1
                return 0
        """)
        mod = analyze_module("agg.py", source)
        assert mod.function_count == 2
        assert mod.max_cyclomatic >= 1
        assert mod.avg_cyclomatic > 0

    def test_syntax_error_raises(self):
        with pytest.raises(ValueError, match="Syntax error"):
            analyze_module("bad.py", "def f(\n")

    def test_line_counts(self):
        source = textwrap.dedent("""\
            # comment
            import os

            def f():
                x = 1  # inline comment
                return x
        """)
        mod = analyze_module("lines.py", source)
        assert mod.total_lines == 6
        # code_lines excludes blank lines and comment-only lines
        # code_lines: import, def, x=1, return = 4 (comment-only line excluded, blank excluded)
        assert mod.code_lines == 4

    def test_async_function(self):
        source = textwrap.dedent("""\
            async def fetch(url):
                async with aiohttp.get(url) as resp:
                    return await resp.text()
        """)
        mod = analyze_module("async.py", source)
        assert mod.function_count == 1
        fn = mod.functions[0]
        assert fn.name == "fetch"
        assert fn.cyclomatic >= 2  # async with


# ============================================================
# ModuleMetrics.to_dict
# ============================================================


class TestModuleMetricsToDict:
    def test_to_dict_structure(self):
        mod = analyze_module("t.py", "def f(): pass\n")
        d = mod.to_dict()
        assert "file" in d
        assert "total_lines" in d
        assert "functions" in d
        assert isinstance(d["functions"], list)


# ============================================================
# AnalysisResult
# ============================================================


class TestAnalysisResult:
    def test_empty_result(self):
        r = AnalysisResult()
        assert r.all_functions == []
        d = r.to_dict()
        assert d["files_analyzed"] == 0
        assert d["summary"]["avg_cyclomatic"] == 0

    def test_all_functions_aggregation(self):
        mod1 = analyze_module("a.py", "def f(): pass\n")
        mod2 = analyze_module("b.py", "def g(): pass\ndef h(): pass\n")
        r = AnalysisResult(files_analyzed=2, total_functions=3, modules=[mod1, mod2])
        assert len(r.all_functions) == 3

    def test_summary_grade_distribution(self):
        mod = analyze_module("s.py", "def f(): pass\ndef g(): pass\n")
        r = AnalysisResult(files_analyzed=1, total_functions=2, modules=[mod])
        d = r.to_dict()
        assert "A" in d["summary"]["grade_distribution"]


# ============================================================
# compute_complexity_score
# ============================================================


class TestComputeComplexityScore:
    def test_empty_project(self):
        r = AnalysisResult()
        score = compute_complexity_score(r)
        assert score.score == 100
        assert score.grade == "A"
        assert score.profile == "clean"
        assert score.violations_count == 0

    def test_clean_project(self):
        source = "def f(): pass\ndef g(x): return x\n"
        mod = analyze_module("clean.py", source)
        r = AnalysisResult(files_analyzed=1, total_functions=2, modules=[mod])
        score = compute_complexity_score(r)
        assert score.score >= 90
        assert score.grade == "A"

    def test_complex_project_lowers_score(self):
        """A project with many complex functions should have a lower score."""
        # Create functions with high cyclomatic complexity
        source = textwrap.dedent("""\
            def complex_func(a, b, c, d, e, f, g, h, i, j, k):
                if a: pass
                if b: pass
                if c: pass
                if d: pass
                if e: pass
                if f: pass
                if g: pass
                if h: pass
                if i: pass
                if j: pass
                if k: pass
                for x in []:
                    if x: pass
                    for y in []:
                        if y: pass
                while True:
                    break
                try:
                    pass
                except ValueError:
                    pass
                except TypeError:
                    pass
                except OSError:
                    pass
        """)
        mod = analyze_module("complex.py", source)
        r = AnalysisResult(files_analyzed=1, total_functions=1, modules=[mod])
        score = compute_complexity_score(r)
        assert score.score < 90  # definitely not clean

    def test_top_offenders_limited_to_five(self):
        funcs = "\n".join(f"def f{i}(): pass" for i in range(10))
        mod = analyze_module("many.py", funcs + "\n")
        r = AnalysisResult(files_analyzed=1, total_functions=10, modules=[mod])
        score = compute_complexity_score(r)
        assert len(score.top_offenders) <= 5

    def test_score_to_dict(self):
        r = AnalysisResult()
        score = compute_complexity_score(r)
        d = score.to_dict()
        assert "score" in d
        assert "grade" in d
        assert "profile" in d
        assert "violations_count" in d

    def test_custom_thresholds(self):
        source = textwrap.dedent("""\
            def f(x):
                if x > 0:
                    if x > 10:
                        return 1
                return 0
        """)
        mod = analyze_module("thresh.py", source)
        r = AnalysisResult(files_analyzed=1, total_functions=1, modules=[mod])
        # With very low threshold, it should be a violation
        score = compute_complexity_score(r, cc_threshold=2)
        assert score.violations_count >= 1


# ============================================================
# classify_complexity_profile
# ============================================================


class TestClassifyProfile:
    def test_empty(self):
        assert classify_complexity_profile([]) == "clean"

    def test_clean_profile(self):
        fns = [_make_fn(cyclomatic=2) for _ in range(20)]
        assert classify_complexity_profile(fns) == "clean"

    def test_god_functions(self):
        """Few extreme outliers among many simple functions."""
        fns = [_make_fn(cyclomatic=2) for _ in range(100)]
        fns.append(_make_fn(cyclomatic=25))  # D grade
        assert classify_complexity_profile(fns) == "god_functions"

    def test_cyclomatic_heavy(self):
        """Many cyclomatic violations, few cognitive."""
        fns = [_make_fn(cyclomatic=12, cognitive=5) for _ in range(10)]
        fns += [_make_fn(cyclomatic=2, cognitive=2) for _ in range(10)]
        profile = classify_complexity_profile(fns, cc_threshold=10, cog_threshold=15)
        assert profile in ("cyclomatic_heavy", "god_functions", "uniformly_complex")

    def test_uniformly_complex(self):
        """Many C+ grade functions spread across the project."""
        # Need both CC and cognitive violations to avoid hitting cyclomatic_heavy first
        fns = [_make_fn(cyclomatic=15, cognitive=20) for _ in range(10)]  # all C grade
        profile = classify_complexity_profile(fns)
        assert profile == "uniformly_complex"


# ============================================================
# _is_venv_path
# ============================================================


class TestIsVenvPath:
    def test_venv_dir(self):
        assert _is_venv_path("/project/.venv/lib/python3.11/site-packages/foo.py")
        assert _is_venv_path("/project/venv/lib/foo.py")
        assert _is_venv_path("/project/.env/lib/foo.py")

    def test_site_packages(self):
        assert _is_venv_path("/usr/lib/python3.11/site-packages/foo.py")

    def test_normal_path(self):
        assert not _is_venv_path("/project/src/main.py")
        assert not _is_venv_path("/project/tests/test_foo.py")

    def test_env_suffix(self):
        assert _is_venv_path("/project/my_env/lib/foo.py")


# ============================================================
# find_python_files
# ============================================================


class TestFindPythonFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        result = find_python_files(str(f))
        assert len(result) == 1

    def test_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.txt").write_text("not python")
        result = find_python_files(str(tmp_path))
        assert len(result) == 2

    def test_skips_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("x = 1")
        (tmp_path / "real.py").write_text("y = 2")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "pkg.py").write_text("x = 1")
        (tmp_path / "main.py").write_text("y = 2")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_nonexistent_returns_empty(self):
        result = find_python_files("/nonexistent/path/12345")
        assert result == []

    def test_sorted_output(self, tmp_path):
        (tmp_path / "z.py").write_text("x = 1")
        (tmp_path / "a.py").write_text("y = 2")
        (tmp_path / "m.py").write_text("z = 3")
        result = find_python_files(str(tmp_path))
        basenames = [r.split("/")[-1] for r in result]
        assert basenames == sorted(basenames)


# ============================================================
# Edge Cases
# ============================================================


class TestEdgeCases:
    def test_lambda_not_counted(self):
        """Lambdas are not function definitions."""
        source = "fn = lambda x: x + 1\n"
        mod = analyze_module("lambda.py", source)
        assert mod.function_count == 0

    def test_decorator(self):
        source = textwrap.dedent("""\
            @decorator
            def f(x):
                return x
        """)
        mod = analyze_module("deco.py", source)
        assert mod.function_count == 1

    def test_nested_class(self):
        source = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """)
        mod = analyze_module("nested.py", source)
        assert mod.class_count == 1  # only top-level counted
        assert mod.function_count == 1
        assert mod.functions[0].qualified_name == "Inner.method"

    def test_property(self):
        source = textwrap.dedent("""\
            class C:
                @property
                def value(self):
                    return self._value
        """)
        mod = analyze_module("prop.py", source)
        assert mod.function_count == 1
        assert mod.functions[0].is_method is True

    def test_star_imports_counted(self):
        source = textwrap.dedent("""\
            from os import *
            from sys import path
        """)
        mod = analyze_module("imports.py", source)
        assert mod.import_count == 2

    def test_only_comments(self):
        source = "# just a comment\n# another comment\n"
        mod = analyze_module("comments.py", source)
        assert mod.total_lines == 2
        assert mod.code_lines == 0
        assert mod.function_count == 0
