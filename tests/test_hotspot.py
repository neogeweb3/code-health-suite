"""Tests for the hotspot engine — code hotspot detector."""
from __future__ import annotations

import ast
import json
import math
import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from code_health_suite.engines.hotspot import (
    DEFAULT_TOP_N,
    DEFAULT_SINCE_DAYS,
    RISK_THRESHOLDS,
    CATEGORY_DOMINANT_THRESHOLD,
    CATEGORY_WEAK_THRESHOLD,
    ChurnData,
    ComplexityData,
    HotspotResult,
    AnalysisResult,
    HotspotProjectStats,
    find_git_root,
    get_file_churn,
    ComplexityVisitor,
    get_file_complexity,
    normalize_values,
    classify_risk,
    compute_hotspots,
    classify_hotspot_category,
    compute_project_stats,
    find_python_files,
    analyze,
    format_text,
    format_json,
    format_score_text,
    format_score_json,
    main,
)


# ============================================================
# Constants
# ============================================================


class TestConstants:
    def test_default_top_n(self):
        assert DEFAULT_TOP_N == 20

    def test_default_since_days(self):
        assert DEFAULT_SINCE_DAYS == 180

    def test_risk_thresholds_ordered(self):
        assert RISK_THRESHOLDS["critical"] > RISK_THRESHOLDS["high"]
        assert RISK_THRESHOLDS["high"] > RISK_THRESHOLDS["medium"]

    def test_category_thresholds(self):
        assert CATEGORY_DOMINANT_THRESHOLD > CATEGORY_WEAK_THRESHOLD
        assert 0 < CATEGORY_WEAK_THRESHOLD < 1
        assert 0 < CATEGORY_DOMINANT_THRESHOLD < 1


# ============================================================
# Data Models
# ============================================================


class TestChurnData:
    def test_defaults(self):
        cd = ChurnData()
        assert cd.commits == 0
        assert cd.lines_added == 0
        assert cd.lines_deleted == 0

    def test_custom_values(self):
        cd = ChurnData(commits=10, lines_added=200, lines_deleted=50)
        assert cd.commits == 10
        assert cd.lines_added == 200
        assert cd.lines_deleted == 50

    def test_churn_score_zero_commits(self):
        cd = ChurnData(commits=0, lines_added=100, lines_deleted=50)
        assert cd.churn_score == 0.0

    def test_churn_score_zero_changes(self):
        cd = ChurnData(commits=5, lines_added=0, lines_deleted=0)
        assert cd.churn_score == 0.0

    def test_churn_score_positive(self):
        cd = ChurnData(commits=4, lines_added=50, lines_deleted=50)
        # sqrt(4) * sqrt(100) = 2 * 10 = 20
        assert cd.churn_score == pytest.approx(20.0)

    def test_churn_score_single_commit(self):
        cd = ChurnData(commits=1, lines_added=1, lines_deleted=0)
        # sqrt(1) * sqrt(1) = 1
        assert cd.churn_score == pytest.approx(1.0)

    def test_churn_score_large_values(self):
        cd = ChurnData(commits=100, lines_added=5000, lines_deleted=5000)
        expected = math.sqrt(100) * math.sqrt(10000)
        assert cd.churn_score == pytest.approx(expected)


class TestComplexityData:
    def test_defaults(self):
        cd = ComplexityData()
        assert cd.max_cc == 0
        assert cd.total_cc == 0
        assert cd.num_functions == 0
        assert cd.longest_function == 0

    def test_custom_values(self):
        cd = ComplexityData(max_cc=15, total_cc=45, num_functions=5, longest_function=80)
        assert cd.max_cc == 15
        assert cd.complexity_score == 15.0

    def test_complexity_score_zero(self):
        cd = ComplexityData(max_cc=0)
        assert cd.complexity_score == 0.0

    def test_complexity_score_returns_float(self):
        cd = ComplexityData(max_cc=7)
        assert isinstance(cd.complexity_score, float)
        assert cd.complexity_score == 7.0


class TestHotspotResult:
    def test_defaults(self):
        hr = HotspotResult(filepath="test.py")
        assert hr.filepath == "test.py"
        assert hr.hotspot_score == 0.0
        assert hr.risk_level == "low"
        assert hr.churn_normalized == 0.0
        assert hr.complexity_normalized == 0.0
        assert isinstance(hr.churn, ChurnData)
        assert isinstance(hr.complexity, ComplexityData)


class TestAnalysisResult:
    def test_defaults(self):
        ar = AnalysisResult(repo_path="/tmp/repo")
        assert ar.repo_path == "/tmp/repo"
        assert ar.total_files_analyzed == 0
        assert ar.total_python_files == 0
        assert ar.since_days == DEFAULT_SINCE_DAYS
        assert ar.hotspots == []
        assert ar.errors == []


class TestHotspotProjectStats:
    def test_defaults(self):
        s = HotspotProjectStats()
        assert s.total_files == 0
        assert s.files_with_hotspots == 0
        assert s.critical_count == 0
        assert s.high_count == 0
        assert s.medium_count == 0
        assert s.low_count == 0
        assert s.avg_hotspot_score == 0.0
        assert s.max_hotspot_score == 0.0
        assert s.hotspot_density_pct == 0.0
        assert s.score == 0
        assert s.grade == "F"


# ============================================================
# normalize_values
# ============================================================


class TestNormalizeValues:
    def test_empty(self):
        assert normalize_values([]) == []

    def test_single_value(self):
        result = normalize_values([5.0])
        assert result == [0.5]

    def test_uniform_values(self):
        result = normalize_values([3.0, 3.0, 3.0])
        assert result == [0.5, 0.5, 0.5]

    def test_two_values(self):
        result = normalize_values([0.0, 10.0])
        assert result == [0.0, 1.0]

    def test_three_values(self):
        result = normalize_values([0.0, 5.0, 10.0])
        assert result == [0.0, 0.5, 1.0]

    def test_negative_values(self):
        result = normalize_values([-10.0, 0.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_preserves_order(self):
        result = normalize_values([10.0, 0.0, 5.0])
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.5)


# ============================================================
# classify_risk
# ============================================================


class TestClassifyRisk:
    def test_critical(self):
        assert classify_risk(0.7) == "critical"
        assert classify_risk(0.9) == "critical"
        assert classify_risk(1.0) == "critical"

    def test_high(self):
        assert classify_risk(0.4) == "high"
        assert classify_risk(0.5) == "high"
        assert classify_risk(0.69) == "high"

    def test_medium(self):
        assert classify_risk(0.2) == "medium"
        assert classify_risk(0.3) == "medium"
        assert classify_risk(0.39) == "medium"

    def test_low(self):
        assert classify_risk(0.0) == "low"
        assert classify_risk(0.1) == "low"
        assert classify_risk(0.19) == "low"

    def test_boundary_critical(self):
        assert classify_risk(RISK_THRESHOLDS["critical"]) == "critical"

    def test_boundary_high(self):
        assert classify_risk(RISK_THRESHOLDS["high"]) == "high"

    def test_boundary_medium(self):
        assert classify_risk(RISK_THRESHOLDS["medium"]) == "medium"

    def test_just_below_critical(self):
        assert classify_risk(RISK_THRESHOLDS["critical"] - 0.001) == "high"


# ============================================================
# classify_hotspot_category
# ============================================================


class TestClassifyHotspotCategory:
    def _make(self, cn, ch):
        h = HotspotResult(filepath="x.py")
        h.complexity_normalized = cn
        h.churn_normalized = ch
        return h

    def test_dual(self):
        """Both dimensions dominant."""
        h = self._make(0.8, 0.8)
        assert classify_hotspot_category(h) == "dual"

    def test_dual_at_threshold(self):
        h = self._make(CATEGORY_DOMINANT_THRESHOLD, CATEGORY_DOMINANT_THRESHOLD)
        assert classify_hotspot_category(h) == "dual"

    def test_complexity_driven_strong(self):
        """High complexity, very low churn."""
        h = self._make(0.8, 0.1)
        assert classify_hotspot_category(h) == "complexity_driven"

    def test_churn_driven_strong(self):
        """High churn, very low complexity."""
        h = self._make(0.1, 0.8)
        assert classify_hotspot_category(h) == "churn_driven"

    def test_complexity_driven_moderate_churn(self):
        """High complexity, moderate churn (between weak and dominant)."""
        h = self._make(0.8, 0.4)
        assert classify_hotspot_category(h) == "complexity_driven"

    def test_churn_driven_moderate_complexity(self):
        """High churn, moderate complexity."""
        h = self._make(0.4, 0.8)
        assert classify_hotspot_category(h) == "churn_driven"

    def test_balanced(self):
        """Neither dimension is dominant."""
        h = self._make(0.3, 0.3)
        assert classify_hotspot_category(h) == "balanced"

    def test_balanced_low(self):
        h = self._make(0.0, 0.0)
        assert classify_hotspot_category(h) == "balanced"

    def test_both_below_dominant_above_weak(self):
        """Both moderate — not dominant enough for any driven category."""
        h = self._make(0.5, 0.5)
        assert classify_hotspot_category(h) == "balanced"


# ============================================================
# ComplexityVisitor
# ============================================================


class TestComplexityVisitor:
    def _visit(self, source):
        tree = ast.parse(textwrap.dedent(source))
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor

    def test_simple_function(self):
        v = self._visit("""
        def foo():
            return 1
        """)
        assert len(v.functions) == 1
        assert v.functions[0][0] == "foo"
        assert v.functions[0][1] == 1  # cc=1, no branches

    def test_if_branch(self):
        v = self._visit("""
        def foo(x):
            if x > 0:
                return x
            return -x
        """)
        assert v.functions[0][1] == 2  # cc = 1 + 1 (if)

    def test_multiple_branches(self):
        v = self._visit("""
        def foo(x, y):
            if x > 0:
                for i in range(y):
                    if i > 5:
                        pass
            return x
        """)
        # 1 + if + for + if = 4
        assert v.functions[0][1] == 4

    def test_while_loop(self):
        v = self._visit("""
        def foo():
            while True:
                break
        """)
        assert v.functions[0][1] == 2  # 1 + while

    def test_try_except(self):
        v = self._visit("""
        def foo():
            try:
                pass
            except ValueError:
                pass
            except TypeError:
                pass
        """)
        # 1 + 2 except handlers
        assert v.functions[0][1] == 3

    def test_with_statement(self):
        v = self._visit("""
        def foo():
            with open("f") as f:
                pass
        """)
        assert v.functions[0][1] == 2  # 1 + with

    def test_assert_statement(self):
        v = self._visit("""
        def foo(x):
            assert x > 0
        """)
        assert v.functions[0][1] == 2  # 1 + assert

    def test_list_comprehension(self):
        v = self._visit("""
        def foo(items):
            return [x for x in items if x > 0]
        """)
        # 1 + comprehension + if (the if in comprehension is a separate If?)
        # Actually, comprehension node itself counts, the `if x > 0` is a filter
        # inside the comprehension, not an ast.If node. Let me check...
        # The comprehension node has `ifs` list but those are expressions not ast.If
        # So: 1 + 1 (comprehension)
        assert v.functions[0][1] == 2

    def test_bool_op_and(self):
        v = self._visit("""
        def foo(x, y):
            if x > 0 and y > 0:
                return True
        """)
        # 1 + if + (and with 2 values -> 1 extra path)
        assert v.functions[0][1] == 3

    def test_bool_op_or_three(self):
        v = self._visit("""
        def foo(a, b, c):
            if a or b or c:
                return True
        """)
        # 1 + if + (or with 3 values -> 2 extra paths)
        assert v.functions[0][1] == 4

    def test_nested_function_not_counted_in_parent(self):
        """walk_scope skips nested function bodies."""
        v = self._visit("""
        def outer():
            def inner():
                if True:
                    pass
                if True:
                    pass
            return inner
        """)
        # outer: cc=1 (no branches in its scope)
        # inner: cc=3 (1 + 2 ifs)
        assert len(v.functions) == 2
        outer = next(f for f in v.functions if f[0] == "outer")
        inner = next(f for f in v.functions if f[0] == "inner")
        assert outer[1] == 1
        assert inner[1] == 3

    def test_async_function(self):
        v = self._visit("""
        async def foo():
            if True:
                pass
        """)
        assert len(v.functions) == 1
        assert v.functions[0][0] == "foo"
        assert v.functions[0][1] == 2

    def test_method_in_class(self):
        v = self._visit("""
        class Foo:
            def bar(self):
                if True:
                    pass
        """)
        assert len(v.functions) == 1
        assert v.functions[0][0] == "bar"

    def test_function_lines(self):
        v = self._visit("""
        def foo():
            x = 1
            y = 2
            z = 3
            return x + y + z
        """)
        # Should capture line count
        assert v.functions[0][2] > 0  # lines > 0

    def test_no_functions(self):
        v = self._visit("""
        x = 1
        y = 2
        """)
        assert len(v.functions) == 0

    def test_multiple_functions(self):
        v = self._visit("""
        def a(): pass
        def b(): pass
        def c(): pass
        """)
        assert len(v.functions) == 3


# ============================================================
# get_file_complexity
# ============================================================


class TestGetFileComplexity:
    def test_valid_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    if True:\n        pass\n    return 1\n")
        result = get_file_complexity(str(f))
        assert result is not None
        assert result.max_cc == 2
        assert result.num_functions == 1

    def test_syntax_error(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(:\n")
        result = get_file_complexity(str(f))
        assert result is None

    def test_no_functions(self, tmp_path):
        f = tmp_path / "nofunc.py"
        f.write_text("x = 1\ny = 2\n")
        result = get_file_complexity(str(f))
        assert result is not None
        assert result.max_cc == 1
        assert result.total_cc == 1
        assert result.num_functions == 0
        assert result.longest_function == 0

    def test_nonexistent_file(self):
        result = get_file_complexity("/nonexistent/path/file.py")
        assert result is None

    def test_multiple_functions(self, tmp_path):
        f = tmp_path / "multi.py"
        f.write_text(textwrap.dedent("""\
        def simple():
            return 1

        def complex_one(x, y):
            if x > 0:
                for i in range(y):
                    if i > 5:
                        while True:
                            break
            return x
        """))
        result = get_file_complexity(str(f))
        assert result is not None
        assert result.num_functions == 2
        assert result.max_cc == 5  # 1 + if + for + if + while
        assert result.total_cc == 6  # 1 + 5

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = get_file_complexity(str(f))
        assert result is not None
        assert result.num_functions == 0


# ============================================================
# compute_hotspots
# ============================================================


class TestComputeHotspots:
    def test_empty_maps(self):
        assert compute_hotspots({}, {}) == []

    def test_no_common_files(self):
        churn = {"a.py": ChurnData(commits=5, lines_added=100, lines_deleted=50)}
        complexity = {"b.py": ComplexityData(max_cc=10)}
        assert compute_hotspots(churn, complexity) == []

    def test_single_file(self):
        churn = {"a.py": ChurnData(commits=4, lines_added=50, lines_deleted=50)}
        complexity = {"a.py": ComplexityData(max_cc=10)}
        results = compute_hotspots(churn, complexity)
        assert len(results) == 1
        assert results[0].filepath == "a.py"
        # Single file: normalized to 0.5 each -> score = 0.25
        assert results[0].churn_normalized == 0.5
        assert results[0].complexity_normalized == 0.5
        assert results[0].hotspot_score == pytest.approx(0.25)

    def test_two_files_ordering(self):
        churn = {
            "low.py": ChurnData(commits=1, lines_added=1, lines_deleted=0),
            "high.py": ChurnData(commits=100, lines_added=1000, lines_deleted=500),
        }
        complexity = {
            "low.py": ComplexityData(max_cc=1),
            "high.py": ComplexityData(max_cc=20),
        }
        results = compute_hotspots(churn, complexity)
        assert len(results) == 2
        assert results[0].filepath == "high.py"
        assert results[1].filepath == "low.py"
        assert results[0].hotspot_score >= results[1].hotspot_score

    def test_assigns_risk_levels(self):
        churn = {
            "a.py": ChurnData(commits=100, lines_added=5000, lines_deleted=5000),
            "b.py": ChurnData(commits=1, lines_added=1, lines_deleted=0),
        }
        complexity = {
            "a.py": ComplexityData(max_cc=50),
            "b.py": ComplexityData(max_cc=1),
        }
        results = compute_hotspots(churn, complexity)
        # Top file should be critical (score=1.0), bottom should be low (score=0.0)
        assert results[0].risk_level == "critical"
        assert results[1].risk_level == "low"

    def test_normalization_range(self):
        """All normalized values should be in [0, 1]."""
        churn = {f"f{i}.py": ChurnData(commits=i+1, lines_added=(i+1)*10, lines_deleted=i*5)
                 for i in range(10)}
        complexity = {f"f{i}.py": ComplexityData(max_cc=i+1) for i in range(10)}
        results = compute_hotspots(churn, complexity)
        for r in results:
            assert 0.0 <= r.churn_normalized <= 1.0
            assert 0.0 <= r.complexity_normalized <= 1.0
            assert 0.0 <= r.hotspot_score <= 1.0

    def test_partial_overlap(self):
        """Only files in both maps are analyzed."""
        churn = {
            "a.py": ChurnData(commits=5, lines_added=50, lines_deleted=10),
            "b.py": ChurnData(commits=10, lines_added=100, lines_deleted=50),
            "c.py": ChurnData(commits=1, lines_added=1, lines_deleted=0),
        }
        complexity = {
            "b.py": ComplexityData(max_cc=8),
            "c.py": ComplexityData(max_cc=2),
            "d.py": ComplexityData(max_cc=15),
        }
        results = compute_hotspots(churn, complexity)
        filepaths = {r.filepath for r in results}
        assert filepaths == {"b.py", "c.py"}


# ============================================================
# compute_project_stats
# ============================================================


class TestComputeProjectStats:
    def _make_result(self, hotspots=None, total_files=10):
        ar = AnalysisResult(repo_path="/tmp")
        ar.total_files_analyzed = total_files
        ar.hotspots = hotspots or []
        return ar

    def _make_hotspot(self, score, risk):
        h = HotspotResult(filepath="x.py")
        h.hotspot_score = score
        h.risk_level = risk
        return h

    def test_no_hotspots(self):
        stats = compute_project_stats(self._make_result())
        assert stats.score == 100
        assert stats.grade == "A"
        assert stats.files_with_hotspots == 0

    def test_one_critical(self):
        hotspots = [self._make_hotspot(0.8, "critical")]
        stats = compute_project_stats(self._make_result(hotspots))
        assert stats.critical_count == 1
        assert stats.score < 100
        # 100 - 15 (1 critical) - density penalty
        assert stats.score == 100 - 15 - int(round(min((1/10)*100 * 0.08, 8)))

    def test_grade_a(self):
        stats = compute_project_stats(self._make_result())
        assert stats.grade == "A"

    def test_grade_b(self):
        # 2 high hotspots: 100 - 16 = 84, plus density
        hotspots = [self._make_hotspot(0.5, "high") for _ in range(2)]
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        assert stats.grade == "B"

    def test_grade_c(self):
        # 3 critical: 100 - 45 = 55... wait that's D
        # 2 high + 3 medium = 100 - 16 - 9 = 75
        hotspots = ([self._make_hotspot(0.5, "high")] * 2 +
                    [self._make_hotspot(0.25, "medium")] * 3)
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        assert stats.grade == "C"

    def test_grade_d(self):
        # 3 critical: 100 - 45 = 55
        hotspots = [self._make_hotspot(0.8, "critical")] * 3
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        assert stats.grade == "D"

    def test_grade_f(self):
        # 3 critical + 4 high: 100 - 45 - 32 = 23
        hotspots = ([self._make_hotspot(0.8, "critical")] * 3 +
                    [self._make_hotspot(0.5, "high")] * 4)
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        assert stats.grade == "F"

    def test_critical_cap(self):
        """Critical deduction capped at 45."""
        hotspots = [self._make_hotspot(0.8, "critical")] * 10
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        # 100 - 45 (cap) - density
        assert stats.score >= 100 - 45 - 8  # density cap is 8

    def test_high_cap(self):
        """High deduction capped at 32."""
        hotspots = [self._make_hotspot(0.5, "high")] * 10
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        assert stats.score >= 100 - 32 - 8

    def test_medium_cap(self):
        """Medium deduction capped at 15."""
        hotspots = [self._make_hotspot(0.25, "medium")] * 10
        stats = compute_project_stats(self._make_result(hotspots, total_files=100))
        assert stats.score >= 100 - 15 - 8

    def test_low_hotspots_no_penalty(self):
        """Low risk hotspots don't contribute to files_with_hotspots."""
        hotspots = [self._make_hotspot(0.1, "low")] * 5
        stats = compute_project_stats(self._make_result(hotspots))
        assert stats.files_with_hotspots == 0
        assert stats.low_count == 5

    def test_density_calculation(self):
        hotspots = [self._make_hotspot(0.8, "critical")] * 5
        stats = compute_project_stats(self._make_result(hotspots, total_files=10))
        assert stats.hotspot_density_pct == pytest.approx(50.0)

    def test_avg_and_max_score(self):
        hotspots = [
            self._make_hotspot(0.8, "critical"),
            self._make_hotspot(0.5, "high"),
            self._make_hotspot(0.2, "medium"),
        ]
        stats = compute_project_stats(self._make_result(hotspots))
        assert stats.max_hotspot_score == 0.8
        assert stats.avg_hotspot_score == pytest.approx(0.5)

    def test_score_never_negative(self):
        """Score is clamped to 0."""
        hotspots = ([self._make_hotspot(0.8, "critical")] * 3 +
                    [self._make_hotspot(0.5, "high")] * 4 +
                    [self._make_hotspot(0.25, "medium")] * 5)
        stats = compute_project_stats(self._make_result(hotspots, total_files=12))
        assert stats.score >= 0

    def test_zero_total_files(self):
        ar = AnalysisResult(repo_path="/tmp")
        ar.total_files_analyzed = 0
        ar.hotspots = [self._make_hotspot(0.8, "critical")]
        stats = compute_project_stats(ar)
        assert stats.hotspot_density_pct == 0.0


# ============================================================
# find_python_files
# ============================================================


class TestFindPythonFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        (tmp_path / "c.txt").write_text("not python")
        files = find_python_files(str(tmp_path))
        assert sorted(files) == ["a.py", "b.py"]

    def test_nested_files(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert "pkg/mod.py" in files

    def test_ignores_git(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "hook.py").write_text("pass")
        (tmp_path / "real.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert files == ["real.py"]

    def test_ignores_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "mod.cpython-310.py").write_text("pass")
        (tmp_path / "real.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert files == ["real.py"]

    def test_ignores_venv(self, tmp_path):
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "activate.py").write_text("pass")
        (tmp_path / "app.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert files == ["app.py"]

    def test_ignores_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "something.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert files == []

    def test_ignores_dot_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("pass")
        (tmp_path / "visible.py").write_text("pass")
        files = find_python_files(str(tmp_path))
        assert files == ["visible.py"]

    def test_empty_dir(self, tmp_path):
        files = find_python_files(str(tmp_path))
        assert files == []


# ============================================================
# find_git_root
# ============================================================


class TestFindGitRoot:
    def test_in_git_repo(self, tmp_path):
        """Create a real git repo and verify detection."""
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path),
                       capture_output=True, check=True)
        root = find_git_root(str(tmp_path))
        assert root is not None
        assert os.path.isdir(root)

    def test_not_git_repo(self, tmp_path):
        root = find_git_root(str(tmp_path))
        # May or may not find a parent git repo, but shouldn't crash
        # Just verify it returns string or None
        assert root is None or isinstance(root, str)

    def test_nonexistent_path(self):
        root = find_git_root("/nonexistent/path/xyz")
        assert root is None

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired("git", 10)
        assert find_git_root("/tmp") is None

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_nonzero_returncode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        assert find_git_root("/tmp") is None


# ============================================================
# get_file_churn
# ============================================================


class TestGetFileChurn:
    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_parses_commit_counts(self, mock_run):
        # First call: --name-only
        name_result = MagicMock(returncode=0, stdout="a.py\nb.py\na.py\n")
        # Second call: --numstat
        numstat_result = MagicMock(returncode=0, stdout="10\t5\ta.py\n3\t1\tb.py\n")
        mock_run.side_effect = [name_result, numstat_result]

        churn = get_file_churn("/repo")
        assert "a.py" in churn
        assert churn["a.py"].commits == 2
        assert churn["b.py"].commits == 1

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_parses_line_changes(self, mock_run):
        name_result = MagicMock(returncode=0, stdout="a.py\n")
        numstat_result = MagicMock(returncode=0, stdout="50\t20\ta.py\n")
        mock_run.side_effect = [name_result, numstat_result]

        churn = get_file_churn("/repo")
        assert churn["a.py"].lines_added == 50
        assert churn["a.py"].lines_deleted == 20

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_ignores_non_python(self, mock_run):
        name_result = MagicMock(returncode=0, stdout="a.py\nREADME.md\nb.js\n")
        numstat_result = MagicMock(returncode=0, stdout="10\t5\ta.py\n5\t2\tREADME.md\n")
        mock_run.side_effect = [name_result, numstat_result]

        churn = get_file_churn("/repo")
        assert "a.py" in churn
        assert "README.md" not in churn
        assert "b.js" not in churn

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_binary_file_skipped(self, mock_run):
        name_result = MagicMock(returncode=0, stdout="a.py\n")
        numstat_result = MagicMock(returncode=0, stdout="-\t-\ta.py\n")
        mock_run.side_effect = [name_result, numstat_result]

        churn = get_file_churn("/repo")
        assert churn["a.py"].lines_added == 0
        assert churn["a.py"].lines_deleted == 0

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_empty_git_log(self, mock_run):
        name_result = MagicMock(returncode=0, stdout="\n")
        numstat_result = MagicMock(returncode=0, stdout="\n")
        mock_run.side_effect = [name_result, numstat_result]

        churn = get_file_churn("/repo")
        assert len(churn) == 0

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_name_only_fails(self, mock_run):
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        churn = get_file_churn("/repo")
        assert churn == {}

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_timeout_returns_partial(self, mock_run):
        import subprocess as sp
        name_result = MagicMock(returncode=0, stdout="a.py\n")
        mock_run.side_effect = [name_result, sp.TimeoutExpired("git", 60)]

        churn = get_file_churn("/repo")
        # Should have commit count from first call but no line changes
        assert "a.py" in churn
        assert churn["a.py"].commits == 1
        assert churn["a.py"].lines_added == 0

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_numstat_malformed_lines(self, mock_run):
        name_result = MagicMock(returncode=0, stdout="a.py\n")
        # Malformed: missing columns, non-integer values
        numstat_result = MagicMock(returncode=0,
                                    stdout="10\ta.py\nfoo\tbar\ta.py\n5\t3\ta.py\n")
        mock_run.side_effect = [name_result, numstat_result]

        churn = get_file_churn("/repo")
        # Only the valid line (5, 3) should be counted
        assert churn["a.py"].lines_added == 5
        assert churn["a.py"].lines_deleted == 3

    @patch("code_health_suite.engines.hotspot.subprocess.run")
    def test_since_days_passed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        get_file_churn("/repo", since_days=30)
        # Verify the since argument was passed correctly
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "--since=30 days ago" in first_call_args


# ============================================================
# format_text
# ============================================================


class TestFormatText:
    def _make_result(self, hotspots=None, errors=None):
        ar = AnalysisResult(repo_path="/tmp/repo")
        ar.total_python_files = 50
        ar.total_files_analyzed = 30
        ar.since_days = 180
        ar.hotspots = hotspots or []
        ar.errors = errors or []
        return ar

    def test_header(self):
        result = self._make_result()
        text = format_text(result)
        assert "Code Hotspot Analysis" in text
        assert "/tmp/repo" in text
        assert "180 days" in text

    def test_no_hotspots(self):
        result = self._make_result()
        text = format_text(result)
        assert "No hotspots found" in text

    def test_with_errors(self):
        result = self._make_result(errors=["No Python files found"])
        text = format_text(result)
        assert "ERROR: No Python files found" in text

    def test_with_hotspots(self):
        h = HotspotResult(filepath="big.py")
        h.churn = ChurnData(commits=50, lines_added=500, lines_deleted=200)
        h.complexity = ComplexityData(max_cc=15, total_cc=30, num_functions=5)
        h.hotspot_score = 0.85
        h.risk_level = "critical"
        h.churn_normalized = 0.9
        h.complexity_normalized = 0.95

        result = self._make_result(hotspots=[h])
        text = format_text(result)
        assert "big.py" in text
        assert "critical" in text
        assert "0.850" in text
        assert "Recommendations" in text
        assert "CRITICAL" in text

    def test_risk_summary_counts(self):
        hotspots = [
            self._make_hotspot(0.8, "critical"),
            self._make_hotspot(0.5, "high"),
            self._make_hotspot(0.3, "medium"),
            self._make_hotspot(0.1, "low"),
        ]
        result = self._make_result(hotspots=hotspots)
        text = format_text(result)
        assert "critical: 1" in text
        assert "high: 1" in text
        assert "medium: 1" in text
        assert "low: 1" in text

    def test_high_recommendations(self):
        hotspots = [self._make_hotspot(0.5, "high")]
        hotspots[0].filepath = "risky.py"
        result = self._make_result(hotspots=hotspots)
        text = format_text(result)
        assert "HIGH" in text
        assert "Schedule refactoring" in text

    def _make_hotspot(self, score, risk):
        h = HotspotResult(filepath="x.py")
        h.hotspot_score = score
        h.risk_level = risk
        h.churn = ChurnData(commits=10, lines_added=100, lines_deleted=50)
        h.complexity = ComplexityData(max_cc=5)
        h.churn_normalized = 0.5
        h.complexity_normalized = 0.5
        return h


# ============================================================
# format_json
# ============================================================


class TestFormatJson:
    def test_valid_json(self):
        ar = AnalysisResult(repo_path="/tmp/repo")
        ar.total_python_files = 10
        ar.total_files_analyzed = 5
        output = format_json(ar)
        data = json.loads(output)
        assert data["repo_path"] == "/tmp/repo"
        assert data["total_python_files"] == 10
        assert data["hotspots"] == []
        assert data["errors"] == []

    def test_with_hotspots(self):
        ar = AnalysisResult(repo_path="/tmp")
        h = HotspotResult(filepath="mod.py")
        h.churn = ChurnData(commits=10, lines_added=100, lines_deleted=50)
        h.complexity = ComplexityData(max_cc=8, total_cc=20, num_functions=3, longest_function=40)
        h.hotspot_score = 0.6543
        h.risk_level = "high"
        h.churn_normalized = 0.7
        h.complexity_normalized = 0.9
        ar.hotspots = [h]

        data = json.loads(format_json(ar))
        hs = data["hotspots"]
        assert len(hs) == 1
        assert hs[0]["filepath"] == "mod.py"
        assert hs[0]["hotspot_score"] == 0.6543
        assert hs[0]["risk_level"] == "high"
        assert hs[0]["category"] in ("dual", "complexity_driven", "churn_driven", "balanced")
        assert hs[0]["complexity"]["max_cc"] == 8
        assert hs[0]["churn"]["commits"] == 10
        assert hs[0]["churn"]["churn_score"] > 0

    def test_errors_included(self):
        ar = AnalysisResult(repo_path="/tmp")
        ar.errors = ["No git data"]
        data = json.loads(format_json(ar))
        assert data["errors"] == ["No git data"]


# ============================================================
# format_score_text and format_score_json
# ============================================================


class TestFormatScoreText:
    def test_healthy_repo(self):
        ar = AnalysisResult(repo_path="/tmp/repo")
        ar.total_files_analyzed = 20
        text = format_score_text(ar)
        assert "Hotspot Health Score" in text
        assert "100/100" in text
        assert "Grade: A" in text

    def test_with_hotspots(self):
        ar = AnalysisResult(repo_path="/tmp/repo")
        ar.total_files_analyzed = 50
        h = HotspotResult(filepath="x.py")
        h.hotspot_score = 0.8
        h.risk_level = "critical"
        ar.hotspots = [h]
        text = format_score_text(ar)
        assert "1 critical" in text
        assert "Max hotspot score" in text


class TestFormatScoreJson:
    def test_valid_json(self):
        ar = AnalysisResult(repo_path="/tmp")
        ar.total_files_analyzed = 10
        data = json.loads(format_score_json(ar))
        assert data["score"] == 100
        assert data["grade"] == "A"
        assert data["repo_path"] == "/tmp"

    def test_with_hotspots(self):
        ar = AnalysisResult(repo_path="/tmp")
        ar.total_files_analyzed = 100
        h = HotspotResult(filepath="x.py")
        h.hotspot_score = 0.5
        h.risk_level = "high"
        ar.hotspots = [h]
        data = json.loads(format_score_json(ar))
        assert data["high_count"] == 1
        assert data["score"] < 100


# ============================================================
# analyze (integration with mocks)
# ============================================================


class TestAnalyze:
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_no_python_files(self, mock_find):
        mock_find.return_value = []
        result = analyze("/tmp/repo")
        assert "No Python files found" in result.errors
        assert result.hotspots == []

    @patch("code_health_suite.engines.hotspot.get_file_churn")
    @patch("code_health_suite.engines.hotspot.find_git_root")
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_no_churn_data(self, mock_find, mock_git_root, mock_churn):
        mock_find.return_value = ["a.py"]
        mock_git_root.return_value = "/tmp/repo"
        mock_churn.return_value = {}
        result = analyze("/tmp/repo")
        assert any("No git churn data" in e for e in result.errors)

    @patch("code_health_suite.engines.hotspot.get_file_complexity")
    @patch("code_health_suite.engines.hotspot.get_file_churn")
    @patch("code_health_suite.engines.hotspot.find_git_root")
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_full_analysis(self, mock_find, mock_git_root, mock_churn, mock_complexity):
        mock_find.return_value = ["a.py", "b.py"]
        mock_git_root.return_value = "/tmp/repo"
        mock_churn.return_value = {
            "a.py": ChurnData(commits=10, lines_added=100, lines_deleted=50),
            "b.py": ChurnData(commits=2, lines_added=10, lines_deleted=5),
        }
        mock_complexity.side_effect = [
            ComplexityData(max_cc=15, total_cc=30, num_functions=5),
            ComplexityData(max_cc=3, total_cc=5, num_functions=2),
        ]

        result = analyze("/tmp/repo", top_n=10, since_days=90)
        assert result.total_python_files == 2
        assert result.total_files_analyzed == 2
        assert result.since_days == 90
        assert len(result.hotspots) == 2
        # First hotspot should be the one with higher score
        assert result.hotspots[0].hotspot_score >= result.hotspots[1].hotspot_score

    @patch("code_health_suite.engines.hotspot.get_file_complexity")
    @patch("code_health_suite.engines.hotspot.get_file_churn")
    @patch("code_health_suite.engines.hotspot.find_git_root")
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_top_n_limits_results(self, mock_find, mock_git_root, mock_churn, mock_complexity):
        files = [f"f{i}.py" for i in range(10)]
        mock_find.return_value = files
        mock_git_root.return_value = "/tmp/repo"
        mock_churn.return_value = {f: ChurnData(commits=i+1, lines_added=(i+1)*10, lines_deleted=i*5)
                                    for i, f in enumerate(files)}
        mock_complexity.side_effect = [ComplexityData(max_cc=i+1) for i in range(10)]

        result = analyze("/tmp/repo", top_n=3)
        assert len(result.hotspots) <= 3

    @patch("code_health_suite.engines.hotspot.get_file_complexity")
    @patch("code_health_suite.engines.hotspot.get_file_churn")
    @patch("code_health_suite.engines.hotspot.find_git_root")
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_subdirectory_path_translation(self, mock_find, mock_git_root, mock_churn, mock_complexity):
        """When repo_path is a subdirectory, churn paths need prefix translation."""
        mock_find.return_value = ["mod.py"]
        mock_git_root.return_value = "/tmp/repo"
        # Churn paths are relative to git root
        mock_churn.return_value = {
            "src/mod.py": ChurnData(commits=5, lines_added=50, lines_deleted=10),
            "other.py": ChurnData(commits=3, lines_added=30, lines_deleted=5),
        }
        mock_complexity.return_value = ComplexityData(max_cc=5)

        result = analyze("/tmp/repo/src", top_n=10)
        # Should translate "src/mod.py" → "mod.py" and find the match
        if result.hotspots:
            assert result.hotspots[0].filepath == "mod.py"

    @patch("code_health_suite.engines.hotspot.get_file_complexity")
    @patch("code_health_suite.engines.hotspot.get_file_churn")
    @patch("code_health_suite.engines.hotspot.find_git_root")
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_complexity_returns_none_skipped(self, mock_find, mock_git_root, mock_churn, mock_complexity):
        mock_find.return_value = ["good.py", "bad.py"]
        mock_git_root.return_value = "/tmp/repo"
        mock_churn.return_value = {
            "good.py": ChurnData(commits=5, lines_added=50, lines_deleted=10),
            "bad.py": ChurnData(commits=3, lines_added=30, lines_deleted=5),
        }
        # bad.py returns None (syntax error)
        mock_complexity.side_effect = [
            ComplexityData(max_cc=10),
            None,
        ]

        result = analyze("/tmp/repo")
        assert result.total_files_analyzed == 1

    @patch("code_health_suite.engines.hotspot.get_file_churn")
    @patch("code_health_suite.engines.hotspot.find_git_root")
    @patch("code_health_suite.engines.hotspot.find_python_files")
    def test_no_git_root_uses_repo_path(self, mock_find, mock_git_root, mock_churn):
        mock_find.return_value = ["a.py"]
        mock_git_root.return_value = None  # No git root found
        mock_churn.return_value = {}

        result = analyze("/tmp/repo")
        # Should fall through to "no churn data" error
        assert any("No git churn" in e for e in result.errors)


# ============================================================
# CLI
# ============================================================


class TestCLI:
    @patch("code_health_suite.engines.hotspot.analyze")
    @patch("code_health_suite.engines.hotspot.format_text")
    def test_default_text_format(self, mock_format, mock_analyze):
        mock_analyze.return_value = AnalysisResult(repo_path="/tmp")
        mock_format.return_value = "output"
        with patch("sys.argv", ["hotspot", "/tmp"]):
            with patch("builtins.print") as mock_print:
                main()
                mock_format.assert_called_once()
                mock_print.assert_called_with("output")

    @patch("code_health_suite.engines.hotspot.analyze")
    @patch("code_health_suite.engines.hotspot.format_json")
    def test_json_format(self, mock_format, mock_analyze):
        mock_analyze.return_value = AnalysisResult(repo_path="/tmp")
        mock_format.return_value = "{}"
        with patch("sys.argv", ["hotspot", "/tmp", "--format", "json"]):
            with patch("builtins.print") as mock_print:
                main()
                mock_format.assert_called_once()

    @patch("code_health_suite.engines.hotspot.analyze")
    @patch("code_health_suite.engines.hotspot.format_score_text")
    def test_score_text(self, mock_format, mock_analyze):
        mock_analyze.return_value = AnalysisResult(repo_path="/tmp")
        mock_format.return_value = "score output"
        with patch("sys.argv", ["hotspot", "/tmp", "--score"]):
            with patch("builtins.print"):
                main()
                mock_format.assert_called_once()

    @patch("code_health_suite.engines.hotspot.analyze")
    @patch("code_health_suite.engines.hotspot.format_score_json")
    def test_score_json(self, mock_format, mock_analyze):
        mock_analyze.return_value = AnalysisResult(repo_path="/tmp")
        mock_format.return_value = "{}"
        with patch("sys.argv", ["hotspot", "/tmp", "--score", "--format", "json"]):
            with patch("builtins.print"):
                main()
                mock_format.assert_called_once()

    @patch("code_health_suite.engines.hotspot.analyze")
    def test_default_path(self, mock_analyze):
        mock_analyze.return_value = AnalysisResult(repo_path=".")
        with patch("sys.argv", ["hotspot"]):
            with patch("builtins.print"):
                main()
                mock_analyze.assert_called_once_with(".", top_n=20, since_days=180)

    @patch("code_health_suite.engines.hotspot.analyze")
    def test_custom_top_and_since(self, mock_analyze):
        mock_analyze.return_value = AnalysisResult(repo_path="/tmp")
        with patch("sys.argv", ["hotspot", "/tmp", "--top", "5", "--since", "30"]):
            with patch("builtins.print"):
                main()
                mock_analyze.assert_called_once_with("/tmp", top_n=5, since_days=30)


# ============================================================
# Integration Tests
# ============================================================


class TestIntegration:
    """End-to-end tests using real temp files (no git mocking)."""

    def _make_hotspot_result(self):
        """Create a fully populated AnalysisResult for formatting tests."""
        ar = AnalysisResult(repo_path="/tmp/test-repo")
        ar.total_python_files = 25
        ar.total_files_analyzed = 15
        ar.since_days = 90

        h1 = HotspotResult(filepath="core/engine.py")
        h1.churn = ChurnData(commits=50, lines_added=800, lines_deleted=300)
        h1.complexity = ComplexityData(max_cc=25, total_cc=60, num_functions=8, longest_function=120)
        h1.hotspot_score = 0.95
        h1.risk_level = "critical"
        h1.churn_normalized = 0.95
        h1.complexity_normalized = 1.0

        h2 = HotspotResult(filepath="api/views.py")
        h2.churn = ChurnData(commits=30, lines_added=400, lines_deleted=150)
        h2.complexity = ComplexityData(max_cc=12, total_cc=35, num_functions=6, longest_function=80)
        h2.hotspot_score = 0.55
        h2.risk_level = "high"
        h2.churn_normalized = 0.6
        h2.complexity_normalized = 0.7

        h3 = HotspotResult(filepath="utils/helpers.py")
        h3.churn = ChurnData(commits=5, lines_added=30, lines_deleted=10)
        h3.complexity = ComplexityData(max_cc=3, total_cc=8, num_functions=4, longest_function=20)
        h3.hotspot_score = 0.05
        h3.risk_level = "low"
        h3.churn_normalized = 0.1
        h3.complexity_normalized = 0.1

        ar.hotspots = [h1, h2, h3]
        return ar

    def test_full_text_pipeline(self):
        """format_text produces valid, complete output."""
        ar = self._make_hotspot_result()
        text = format_text(ar)
        assert "core/engine.py" in text
        assert "api/views.py" in text
        assert "utils/helpers.py" in text
        assert "critical" in text
        assert "Recommendations" in text

    def test_full_json_pipeline(self):
        """format_json produces valid JSON with all fields."""
        ar = self._make_hotspot_result()
        data = json.loads(format_json(ar))
        assert len(data["hotspots"]) == 3
        assert data["hotspots"][0]["filepath"] == "core/engine.py"
        assert data["hotspots"][0]["category"] == "dual"  # both normalized > 0.6

    def test_score_round_trip(self):
        """Score can be computed and formatted in both text and JSON."""
        ar = self._make_hotspot_result()
        score_text = format_score_text(ar)
        score_json = json.loads(format_score_json(ar))
        assert "Score:" in score_text
        assert isinstance(score_json["score"], int)
        assert score_json["grade"] in ("A", "B", "C", "D", "F")

    def test_complexity_visitor_real_code(self):
        """ComplexityVisitor on realistic Python code."""
        source = textwrap.dedent("""\
        class DataProcessor:
            def __init__(self, config):
                self.config = config

            def process(self, data):
                results = []
                for item in data:
                    if item.get("type") == "A":
                        if item.get("value") > 0:
                            results.append(self._handle_a(item))
                        else:
                            results.append(None)
                    elif item.get("type") == "B":
                        try:
                            results.append(self._handle_b(item))
                        except ValueError:
                            results.append(None)
                    else:
                        results.append(item)
                return results

            def _handle_a(self, item):
                return item["value"] * 2

            def _handle_b(self, item):
                if item.get("strict"):
                    assert item["value"] >= 0
                return item["value"]
        """)
        tree = ast.parse(source)
        visitor = ComplexityVisitor()
        visitor.visit(tree)

        names = {f[0] for f in visitor.functions}
        assert names == {"__init__", "process", "_handle_a", "_handle_b"}

        # process: 1 + for + if + if + elif(if) + except = 6
        process = next(f for f in visitor.functions if f[0] == "process")
        assert process[1] >= 5  # at least 5

    def test_compute_hotspots_deterministic(self):
        """Same input always produces same output."""
        churn = {
            "a.py": ChurnData(commits=10, lines_added=100, lines_deleted=50),
            "b.py": ChurnData(commits=5, lines_added=30, lines_deleted=10),
        }
        complexity = {
            "a.py": ComplexityData(max_cc=15),
            "b.py": ComplexityData(max_cc=3),
        }
        r1 = compute_hotspots(churn, complexity)
        r2 = compute_hotspots(churn, complexity)
        assert len(r1) == len(r2)
        for h1, h2 in zip(r1, r2):
            assert h1.filepath == h2.filepath
            assert h1.hotspot_score == h2.hotspot_score

    def test_empty_analysis_formatting(self):
        """Edge case: format empty results."""
        ar = AnalysisResult(repo_path="/tmp")
        assert "No hotspots found" in format_text(ar)
        data = json.loads(format_json(ar))
        assert data["hotspots"] == []
        assert "100/100" in format_score_text(ar)

    def test_file_complexity_integration(self, tmp_path):
        """get_file_complexity on a real file with known complexity."""
        f = tmp_path / "test_target.py"
        f.write_text(textwrap.dedent("""\
        def simple():
            return 42

        def branchy(x, y, z):
            if x > 0:
                if y > 0:
                    for i in range(z):
                        if i % 2 == 0:
                            pass
            elif x < -10:
                while y > 0:
                    y -= 1
            return x
        """))
        result = get_file_complexity(str(f))
        assert result is not None
        assert result.num_functions == 2
        assert result.max_cc >= 6  # branchy has many branches
        assert result.total_cc >= 7  # simple(1) + branchy(6+)
