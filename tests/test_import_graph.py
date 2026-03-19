"""Tests for the import graph analysis engine."""
from __future__ import annotations

import ast
import json
import os
import sys
import textwrap

import pytest

from code_health_suite.engines.import_graph import (
    __version__,
    DEFAULT_EXCLUDES,
    COLORS,
    ImportEdge,
    CycleInfo,
    ModuleMetrics,
    GraphResult,
    ImportHealthScore,
    _letter_grade,
    compute_import_health,
    classify_import_profile,
    format_score_text,
    format_score_json,
    find_python_files,
    path_to_module,
    extract_imports,
    resolve_import,
    get_top_level_package,
    build_graph,
    detect_cycles,
    calculate_metrics,
    analyze_external_deps,
    analyze,
    colorize,
    format_result,
    format_json,
    build_parser,
    main,
)


# ============================================================
# Helpers
# ============================================================

def _make_edge(
    source_module="mod_a",
    source_path="a.py",
    target_raw="mod_b",
    target_resolved="mod_b",
    names=None,
    is_relative=False,
    is_internal=True,
    line=1,
) -> ImportEdge:
    """Create an ImportEdge for testing."""
    return ImportEdge(
        source_module=source_module,
        source_path=source_path,
        target_raw=target_raw,
        target_resolved=target_resolved,
        names=names or [],
        is_relative=is_relative,
        is_internal=is_internal,
        line=line,
    )


def _make_metric(
    module="mod_a",
    path="a.py",
    afferent=0,
    efferent=0,
    instability=0.0,
    is_orphan=False,
) -> ModuleMetrics:
    """Create a ModuleMetrics for testing."""
    return ModuleMetrics(
        module=module,
        path=path,
        afferent=afferent,
        efferent=efferent,
        instability=instability,
        is_orphan=is_orphan,
    )


def _make_result(
    root="/tmp/project",
    total_modules=0,
    total_edges=0,
    internal_edges=0,
    external_packages=0,
    cycles=None,
    orphans=None,
    hub_modules=None,
    unstable_modules=None,
    metrics=None,
    external_deps=None,
) -> GraphResult:
    """Create a GraphResult for testing."""
    return GraphResult(
        root=root,
        total_modules=total_modules,
        total_edges=total_edges,
        internal_edges=internal_edges,
        external_packages=external_packages,
        cycles=cycles or [],
        orphans=orphans or [],
        hub_modules=hub_modules or [],
        unstable_modules=unstable_modules or [],
        metrics=metrics or [],
        external_deps=external_deps or [],
    )


def _write_py(path, content=""):
    """Write a Python file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))


# ============================================================
# Constants
# ============================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_version_string(self):
        assert __version__ == "0.2.0"

    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_default_excludes_is_set(self):
        assert isinstance(DEFAULT_EXCLUDES, set)

    def test_default_excludes_contains_venv(self):
        assert ".venv" in DEFAULT_EXCLUDES
        assert "venv" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_tox(self):
        assert ".tox" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_node_modules(self):
        assert "node_modules" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_pycache(self):
        assert "__pycache__" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_git(self):
        assert ".git" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_mypy_cache(self):
        assert ".mypy_cache" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_pytest_cache(self):
        assert ".pytest_cache" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_dist(self):
        assert "dist" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_build(self):
        assert "build" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_egg_info(self):
        assert "egg-info" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_site_packages(self):
        assert "site-packages" in DEFAULT_EXCLUDES

    def test_default_excludes_contains_eggs(self):
        assert ".eggs" in DEFAULT_EXCLUDES

    def test_default_excludes_count(self):
        assert len(DEFAULT_EXCLUDES) == 13

    def test_colors_has_red(self):
        assert "red" in COLORS

    def test_colors_has_green(self):
        assert "green" in COLORS

    def test_colors_has_yellow(self):
        assert "yellow" in COLORS

    def test_colors_has_blue(self):
        assert "blue" in COLORS

    def test_colors_has_cyan(self):
        assert "cyan" in COLORS

    def test_colors_has_bold(self):
        assert "bold" in COLORS

    def test_colors_has_dim(self):
        assert "dim" in COLORS

    def test_colors_has_reset(self):
        assert "reset" in COLORS

    def test_colors_values_are_ansi(self):
        for key, val in COLORS.items():
            assert val.startswith("\033["), f"COLORS['{key}'] should be ANSI escape"

    def test_colors_count(self):
        assert len(COLORS) == 8


# ============================================================
# Data Classes
# ============================================================


class TestImportEdge:
    """Tests for ImportEdge dataclass."""

    def test_basic_creation(self):
        edge = ImportEdge(
            source_module="foo",
            source_path="foo.py",
            target_raw="bar",
            target_resolved="bar",
            names=["baz"],
            is_relative=False,
            is_internal=True,
            line=5,
        )
        assert edge.source_module == "foo"
        assert edge.source_path == "foo.py"
        assert edge.target_raw == "bar"
        assert edge.target_resolved == "bar"
        assert edge.names == ["baz"]
        assert edge.is_relative is False
        assert edge.is_internal is True
        assert edge.line == 5

    def test_empty_names(self):
        edge = _make_edge(names=[])
        assert edge.names == []

    def test_multiple_names(self):
        edge = _make_edge(names=["a", "b", "c"])
        assert edge.names == ["a", "b", "c"]

    def test_relative_import(self):
        edge = _make_edge(is_relative=True)
        assert edge.is_relative is True

    def test_external_import(self):
        edge = _make_edge(is_internal=False)
        assert edge.is_internal is False


class TestCycleInfo:
    """Tests for CycleInfo dataclass."""

    def test_basic_creation(self):
        cycle = CycleInfo(path=["a", "b", "a"], length=2)
        assert cycle.path == ["a", "b", "a"]
        assert cycle.length == 2

    def test_single_node_cycle(self):
        cycle = CycleInfo(path=["a", "a"], length=1)
        assert cycle.length == 1

    def test_long_cycle(self):
        path = ["a", "b", "c", "d", "a"]
        cycle = CycleInfo(path=path, length=4)
        assert cycle.length == 4
        assert cycle.path[0] == cycle.path[-1]


class TestModuleMetrics:
    """Tests for ModuleMetrics dataclass."""

    def test_basic_creation(self):
        m = ModuleMetrics(
            module="foo", path="foo.py", afferent=3, efferent=2,
            instability=0.4, is_orphan=False,
        )
        assert m.module == "foo"
        assert m.afferent == 3
        assert m.efferent == 2
        assert m.instability == 0.4
        assert m.is_orphan is False

    def test_orphan_module(self):
        m = _make_metric(afferent=0, efferent=0, is_orphan=True)
        assert m.is_orphan is True

    def test_stable_module(self):
        m = _make_metric(instability=0.0)
        assert m.instability == 0.0

    def test_unstable_module(self):
        m = _make_metric(instability=1.0)
        assert m.instability == 1.0


class TestGraphResult:
    """Tests for GraphResult dataclass."""

    def test_empty_result(self):
        r = _make_result()
        assert r.total_modules == 0
        assert r.cycles == []
        assert r.orphans == []

    def test_with_cycles(self):
        c = CycleInfo(path=["a", "b", "a"], length=2)
        r = _make_result(cycles=[c])
        assert len(r.cycles) == 1

    def test_with_metrics(self):
        m = _make_metric()
        r = _make_result(metrics=[m])
        assert len(r.metrics) == 1


class TestImportHealthScore:
    """Tests for ImportHealthScore dataclass."""

    def test_basic_creation(self):
        h = ImportHealthScore(
            score=85, grade="B", total_modules=10,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=2, orphan_ratio=0.2, orphan_penalty=4.0,
            avg_instability=0.3, instability_penalty=4.5,
            hub_concentration=0.4, hub_penalty=6.0,
            external_dep_count=5, profile="clean",
        )
        assert h.score == 85
        assert h.grade == "B"
        assert h.profile == "clean"

    def test_perfect_score(self):
        h = ImportHealthScore(
            score=100, grade="A", total_modules=5,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.0, instability_penalty=0.0,
            hub_concentration=0.0, hub_penalty=0.0,
            external_dep_count=0, profile="clean",
        )
        assert h.score == 100
        assert h.grade == "A"


# ============================================================
# _letter_grade
# ============================================================


class TestLetterGrade:
    """Tests for _letter_grade()."""

    def test_score_100_is_A(self):
        assert _letter_grade(100) == "A"

    def test_score_95_is_A(self):
        assert _letter_grade(95) == "A"

    def test_score_90_is_A(self):
        assert _letter_grade(90) == "A"

    def test_score_89_is_B(self):
        assert _letter_grade(89) == "B"

    def test_score_85_is_B(self):
        assert _letter_grade(85) == "B"

    def test_score_80_is_B(self):
        assert _letter_grade(80) == "B"

    def test_score_79_is_C(self):
        assert _letter_grade(79) == "C"

    def test_score_75_is_C(self):
        assert _letter_grade(75) == "C"

    def test_score_70_is_C(self):
        assert _letter_grade(70) == "C"

    def test_score_69_is_D(self):
        assert _letter_grade(69) == "D"

    def test_score_65_is_D(self):
        assert _letter_grade(65) == "D"

    def test_score_60_is_D(self):
        assert _letter_grade(60) == "D"

    def test_score_59_is_F(self):
        assert _letter_grade(59) == "F"

    def test_score_50_is_F(self):
        assert _letter_grade(50) == "F"

    def test_score_0_is_F(self):
        assert _letter_grade(0) == "F"

    def test_score_negative_is_F(self):
        assert _letter_grade(-10) == "F"


# ============================================================
# classify_import_profile
# ============================================================


class TestClassifyImportProfile:
    """Tests for classify_import_profile()."""

    def test_empty_project(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.0, total_modules=0,
        ) == "empty"

    def test_clean_project(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.1, avg_instability=0.3,
            hub_concentration=0.2, total_modules=10,
        ) == "clean"

    def test_cycle_heavy_two_cycles(self):
        assert classify_import_profile(
            cycle_count=2, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "cycle_heavy"

    def test_cycle_heavy_many_cycles(self):
        assert classify_import_profile(
            cycle_count=10, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "cycle_heavy"

    def test_has_cycle_exactly_one(self):
        assert classify_import_profile(
            cycle_count=1, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "has_cycle"

    def test_orphan_heavy(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.5, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "orphan_heavy"

    def test_orphan_heavy_boundary(self):
        # Boundary: > 0.4, so 0.4 is NOT orphan_heavy
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.4, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "clean"

    def test_orphan_heavy_just_above(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.41, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "orphan_heavy"

    def test_unstable(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.7,
            hub_concentration=0.0, total_modules=10,
        ) == "unstable"

    def test_unstable_boundary(self):
        # Boundary: > 0.6, so 0.6 is NOT unstable
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.6,
            hub_concentration=0.0, total_modules=10,
        ) == "clean"

    def test_unstable_just_above(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.61,
            hub_concentration=0.0, total_modules=10,
        ) == "unstable"

    def test_hub_concentrated(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.7, total_modules=10,
        ) == "hub_concentrated"

    def test_hub_concentrated_boundary(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.6, total_modules=10,
        ) == "clean"

    def test_hub_concentrated_just_above(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.61, total_modules=10,
        ) == "hub_concentrated"

    def test_fragmented_orphan_and_unstable(self):
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.5, avg_instability=0.7,
            hub_concentration=0.0, total_modules=10,
        ) == "fragmented"

    def test_mixed_cycle_heavy_and_orphan(self):
        assert classify_import_profile(
            cycle_count=3, orphan_ratio=0.5, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "mixed"

    def test_mixed_cycle_and_hub(self):
        assert classify_import_profile(
            cycle_count=2, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.7, total_modules=10,
        ) == "mixed"

    def test_mixed_all_issues(self):
        # orphan_heavy + unstable triggers "fragmented" special case,
        # which is checked before the generic "mixed" fallback
        assert classify_import_profile(
            cycle_count=5, orphan_ratio=0.5, avg_instability=0.8,
            hub_concentration=0.9, total_modules=10,
        ) == "fragmented"

    def test_mixed_truly_mixed(self):
        # cycle_heavy + hub_concentrated (no orphan+unstable combo)
        assert classify_import_profile(
            cycle_count=3, orphan_ratio=0.0, avg_instability=0.0,
            hub_concentration=0.7, total_modules=10,
        ) == "mixed"

    def test_fragmented_beats_mixed_for_orphan_unstable(self):
        # orphan_heavy + unstable = fragmented (special case), not mixed
        assert classify_import_profile(
            cycle_count=0, orphan_ratio=0.5, avg_instability=0.7,
            hub_concentration=0.0, total_modules=10,
        ) == "fragmented"

    def test_mixed_when_has_cycle_plus_orphan(self):
        assert classify_import_profile(
            cycle_count=1, orphan_ratio=0.5, avg_instability=0.0,
            hub_concentration=0.0, total_modules=10,
        ) == "mixed"

    def test_mixed_when_has_cycle_plus_unstable(self):
        assert classify_import_profile(
            cycle_count=1, orphan_ratio=0.0, avg_instability=0.7,
            hub_concentration=0.0, total_modules=10,
        ) == "mixed"


# ============================================================
# compute_import_health
# ============================================================


class TestComputeImportHealth:
    """Tests for compute_import_health()."""

    def test_empty_project_returns_100(self):
        r = _make_result(total_modules=0)
        h = compute_import_health(r)
        assert h.score == 100
        assert h.grade == "A"
        assert h.profile == "empty"

    def test_empty_project_zeroed_penalties(self):
        r = _make_result(total_modules=0)
        h = compute_import_health(r)
        assert h.cycle_penalty == 0.0
        assert h.orphan_penalty == 0.0
        assert h.instability_penalty == 0.0
        assert h.hub_penalty == 0.0

    def test_perfect_project_no_penalties(self):
        metrics = [
            _make_metric("a", afferent=1, efferent=1, instability=0.5),
            _make_metric("b", afferent=1, efferent=1, instability=0.5),
        ]
        r = _make_result(
            total_modules=2,
            metrics=metrics,
            hub_modules=[{"module": "a", "afferent": 1, "path": "a.py"}],
        )
        h = compute_import_health(r)
        assert h.cycle_count == 0
        assert h.cycle_penalty == 0.0
        assert h.orphan_count == 0

    def test_one_cycle_penalty(self):
        cycle = CycleInfo(path=["a", "b", "a"], length=2)
        metrics = [_make_metric("a", afferent=1, efferent=1, instability=0.5),
                   _make_metric("b", afferent=1, efferent=1, instability=0.5)]
        r = _make_result(
            total_modules=2, cycles=[cycle], metrics=metrics,
            hub_modules=[{"module": "a", "afferent": 1, "path": "a.py"}],
        )
        h = compute_import_health(r)
        assert h.cycle_count == 1
        assert h.cycle_penalty == 15.0

    def test_two_cycles_penalty(self):
        c1 = CycleInfo(path=["a", "b", "a"], length=2)
        c2 = CycleInfo(path=["c", "d", "c"], length=2)
        metrics = [_make_metric("a"), _make_metric("b"),
                   _make_metric("c"), _make_metric("d")]
        r = _make_result(total_modules=4, cycles=[c1, c2], metrics=metrics)
        h = compute_import_health(r)
        assert h.cycle_penalty == 30.0

    def test_cycle_penalty_capped_at_40(self):
        cycles = [CycleInfo(path=["a", "b", "a"], length=2) for _ in range(5)]
        metrics = [_make_metric("a"), _make_metric("b")]
        r = _make_result(total_modules=2, cycles=cycles, metrics=metrics)
        h = compute_import_health(r)
        assert h.cycle_penalty == 40.0

    def test_orphan_penalty_half_orphans(self):
        metrics = [
            _make_metric("a", is_orphan=True),
            _make_metric("b", afferent=1, efferent=0, instability=0.0),
        ]
        r = _make_result(
            total_modules=2, orphans=["a"], metrics=metrics,
            hub_modules=[{"module": "b", "afferent": 1, "path": "b.py"}],
        )
        h = compute_import_health(r)
        assert h.orphan_ratio == 0.5
        assert h.orphan_penalty == 10.0

    def test_orphan_penalty_all_orphans(self):
        metrics = [
            _make_metric("a", is_orphan=True),
            _make_metric("b", is_orphan=True),
        ]
        r = _make_result(
            total_modules=2, orphans=["a", "b"], metrics=metrics,
        )
        h = compute_import_health(r)
        assert h.orphan_ratio == 1.0
        assert h.orphan_penalty == 20.0

    def test_orphan_penalty_capped_at_20(self):
        metrics = [_make_metric(f"m{i}", is_orphan=True) for i in range(10)]
        r = _make_result(
            total_modules=10,
            orphans=[f"m{i}" for i in range(10)],
            metrics=metrics,
        )
        h = compute_import_health(r)
        assert h.orphan_penalty == 20.0

    def test_instability_penalty_high(self):
        metrics = [
            _make_metric("a", afferent=0, efferent=3, instability=1.0),
            _make_metric("b", afferent=0, efferent=3, instability=1.0),
        ]
        r = _make_result(total_modules=2, metrics=metrics)
        h = compute_import_health(r)
        assert h.avg_instability == 1.0
        assert h.instability_penalty == 15.0

    def test_instability_penalty_capped_at_15(self):
        metrics = [
            _make_metric("a", afferent=0, efferent=5, instability=1.0),
        ]
        r = _make_result(total_modules=1, metrics=metrics)
        h = compute_import_health(r)
        assert h.instability_penalty == 15.0

    def test_instability_skips_orphans(self):
        metrics = [
            _make_metric("a", is_orphan=True, instability=0.0),
            _make_metric("b", afferent=1, efferent=1, instability=0.5),
        ]
        r = _make_result(
            total_modules=2, orphans=["a"], metrics=metrics,
            hub_modules=[{"module": "b", "afferent": 1, "path": "b.py"}],
        )
        h = compute_import_health(r)
        assert h.avg_instability == 0.5

    def test_hub_penalty_single_hub(self):
        metrics = [
            _make_metric("hub", afferent=4, efferent=0, instability=0.0),
            _make_metric("a", afferent=0, efferent=1, instability=1.0),
            _make_metric("b", afferent=0, efferent=1, instability=1.0),
            _make_metric("c", afferent=0, efferent=1, instability=1.0),
            _make_metric("d", afferent=0, efferent=1, instability=1.0),
        ]
        r = _make_result(
            total_modules=5, metrics=metrics,
            hub_modules=[{"module": "hub", "afferent": 4, "path": "hub.py"}],
        )
        h = compute_import_health(r)
        assert h.hub_concentration == 1.0  # 4 / (5-1)
        assert h.hub_penalty == 15.0

    def test_hub_penalty_no_hubs(self):
        metrics = [_make_metric("a", is_orphan=True)]
        r = _make_result(total_modules=1, orphans=["a"], metrics=metrics)
        h = compute_import_health(r)
        assert h.hub_concentration == 0.0
        assert h.hub_penalty == 0.0

    def test_hub_penalty_single_module(self):
        # total==1 means total-1==0, hub_concentration should be 0
        metrics = [_make_metric("a", afferent=1)]
        r = _make_result(
            total_modules=1, metrics=metrics,
            hub_modules=[{"module": "a", "afferent": 1, "path": "a.py"}],
        )
        h = compute_import_health(r)
        assert h.hub_concentration == 0.0

    def test_score_clamped_at_zero(self):
        cycles = [CycleInfo(path=["a", "b", "a"], length=2) for _ in range(5)]
        metrics = [
            _make_metric("a", is_orphan=True),
            _make_metric("b", is_orphan=True),
        ]
        r = _make_result(
            total_modules=2, cycles=cycles,
            orphans=["a", "b"], metrics=metrics,
        )
        h = compute_import_health(r)
        assert h.score >= 0

    def test_score_clamped_at_100(self):
        metrics = [
            _make_metric("a", afferent=1, efferent=1, instability=0.5),
            _make_metric("b", afferent=1, efferent=1, instability=0.5),
        ]
        r = _make_result(
            total_modules=2, metrics=metrics,
            hub_modules=[{"module": "a", "afferent": 1, "path": "a.py"}],
        )
        h = compute_import_health(r)
        assert h.score <= 100

    def test_grade_matches_score(self):
        metrics = [_make_metric("a", afferent=1, efferent=1, instability=0.5)]
        r = _make_result(
            total_modules=1, metrics=metrics,
            hub_modules=[{"module": "a", "afferent": 1, "path": "a.py"}],
        )
        h = compute_import_health(r)
        assert h.grade == _letter_grade(h.score)

    def test_all_orphans_instability_zero(self):
        # All modules are orphans => avg_instability from non-orphan = 0
        metrics = [
            _make_metric("a", is_orphan=True),
            _make_metric("b", is_orphan=True),
        ]
        r = _make_result(
            total_modules=2, orphans=["a", "b"], metrics=metrics,
        )
        h = compute_import_health(r)
        assert h.avg_instability == 0.0
        assert h.instability_penalty == 0.0


# ============================================================
# format_score_text / format_score_json
# ============================================================


class TestFormatScoreText:
    """Tests for format_score_text()."""

    def test_returns_string(self):
        h = ImportHealthScore(
            score=85, grade="B", total_modules=10,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=2, orphan_ratio=0.2, orphan_penalty=4.0,
            avg_instability=0.3, instability_penalty=4.5,
            hub_concentration=0.4, hub_penalty=6.0,
            external_dep_count=5, profile="clean",
        )
        result = format_score_text(h)
        assert isinstance(result, str)

    def test_contains_score(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        h = ImportHealthScore(
            score=92, grade="A", total_modules=5,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.2, instability_penalty=3.0,
            hub_concentration=0.1, hub_penalty=1.5,
            external_dep_count=2, profile="clean",
        )
        text = format_score_text(h)
        assert "92" in text
        assert "/100" in text
        assert "A" in text

    def test_contains_penalties(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        h = ImportHealthScore(
            score=70, grade="C", total_modules=10,
            cycle_count=1, cycle_penalty=15.0,
            orphan_count=3, orphan_ratio=0.3, orphan_penalty=6.0,
            avg_instability=0.4, instability_penalty=6.0,
            hub_concentration=0.2, hub_penalty=3.0,
            external_dep_count=8, profile="has_cycle",
        )
        text = format_score_text(h)
        assert "15.0" in text
        assert "Penalty" in text

    def test_contains_profile(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        h = ImportHealthScore(
            score=50, grade="F", total_modules=10,
            cycle_count=3, cycle_penalty=40.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.3, instability_penalty=4.5,
            hub_concentration=0.2, hub_penalty=3.0,
            external_dep_count=5, profile="cycle_heavy",
        )
        text = format_score_text(h)
        assert "cycle_heavy" in text

    def test_contains_external_deps_count(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        h = ImportHealthScore(
            score=90, grade="A", total_modules=5,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.0, instability_penalty=0.0,
            hub_concentration=0.0, hub_penalty=0.0,
            external_dep_count=42, profile="clean",
        )
        text = format_score_text(h)
        assert "42" in text


class TestFormatScoreJson:
    """Tests for format_score_json()."""

    def test_returns_valid_json(self):
        h = ImportHealthScore(
            score=85, grade="B", total_modules=10,
            cycle_count=1, cycle_penalty=15.0,
            orphan_count=2, orphan_ratio=0.2, orphan_penalty=4.0,
            avg_instability=0.3, instability_penalty=4.5,
            hub_concentration=0.4, hub_penalty=6.0,
            external_dep_count=5, profile="has_cycle",
        )
        text = format_score_json(h)
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_json_has_version(self):
        h = ImportHealthScore(
            score=100, grade="A", total_modules=0,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.0, instability_penalty=0.0,
            hub_concentration=0.0, hub_penalty=0.0,
            external_dep_count=0, profile="empty",
        )
        data = json.loads(format_score_json(h))
        assert data["version"] == __version__

    def test_json_has_score(self):
        h = ImportHealthScore(
            score=75, grade="C", total_modules=8,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=1, orphan_ratio=0.125, orphan_penalty=2.5,
            avg_instability=0.5, instability_penalty=7.5,
            hub_concentration=0.3, hub_penalty=4.5,
            external_dep_count=3, profile="clean",
        )
        data = json.loads(format_score_json(h))
        assert data["score"] == 75
        assert data["grade"] == "C"

    def test_json_has_penalties(self):
        h = ImportHealthScore(
            score=60, grade="D", total_modules=10,
            cycle_count=2, cycle_penalty=30.0,
            orphan_count=1, orphan_ratio=0.1, orphan_penalty=2.0,
            avg_instability=0.4, instability_penalty=6.0,
            hub_concentration=0.1, hub_penalty=1.5,
            external_dep_count=5, profile="cycle_heavy",
        )
        data = json.loads(format_score_json(h))
        assert "penalties" in data
        assert data["penalties"]["cycles"]["count"] == 2
        assert data["penalties"]["cycles"]["penalty"] == 30.0

    def test_json_has_profile(self):
        h = ImportHealthScore(
            score=100, grade="A", total_modules=0,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.0, instability_penalty=0.0,
            hub_concentration=0.0, hub_penalty=0.0,
            external_dep_count=0, profile="empty",
        )
        data = json.loads(format_score_json(h))
        assert data["profile"] == "empty"


# ============================================================
# find_python_files
# ============================================================


class TestFindPythonFiles:
    """Tests for find_python_files()."""

    def test_empty_directory(self, tmp_path):
        assert find_python_files(str(tmp_path)) == []

    def test_single_file(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("foo.py")

    def test_ignores_non_python(self, tmp_path):
        (tmp_path / "foo.py").write_text("x = 1")
        (tmp_path / "bar.txt").write_text("hello")
        (tmp_path / "baz.js").write_text("var x;")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_nested_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "b.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 2

    def test_results_are_sorted(self, tmp_path):
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        (tmp_path / "m.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert result == sorted(result)

    def test_excludes_venv(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "lib.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_dot_venv(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_node_modules(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "x.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_pycache(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "app.cpython-311.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_git(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_hidden_dirs(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        hid = tmp_path / ".hidden"
        hid.mkdir()
        (hid / "secret.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_env_suffix_dirs(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        env = tmp_path / "test_env"
        env.mkdir()
        (env / "lib.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_site_packages_in_name(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        sp = tmp_path / "my-site-packages-dir"
        sp.mkdir()
        (sp / "pkg.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_dist(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        d = tmp_path / "dist"
        d.mkdir()
        (d / "pkg.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_build(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        b = tmp_path / "build"
        b.mkdir()
        (b / "pkg.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_custom_excludes_override_defaults(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "mod.py").write_text("")
        result = find_python_files(str(tmp_path), excludes={"custom"})
        assert len(result) == 1

    def test_empty_excludes(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        # hidden dirs are STILL excluded even with empty excludes set
        # because the filter also checks startswith(".")
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "lib.py").write_text("")
        # venv is only in DEFAULT_EXCLUDES, not in empty set
        # but the filter also doesn't include it by name in our custom empty set
        result = find_python_files(str(tmp_path), excludes=set())
        assert len(result) == 2  # app.py + venv/lib.py

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "foo.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert os.path.isabs(result[0])

    def test_includes_init_py(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 2

    def test_deeply_nested(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "mod.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_tox(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        tox = tmp_path / ".tox"
        tox.mkdir()
        (tox / "env.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_mypy_cache(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        mc = tmp_path / ".mypy_cache"
        mc.mkdir()
        (mc / "cache.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_pytest_cache(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        pc = tmp_path / ".pytest_cache"
        pc.mkdir()
        (pc / "cache.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_eggs(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        eggs = tmp_path / ".eggs"
        eggs.mkdir()
        (eggs / "egg.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_egg_info(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        ei = tmp_path / "egg-info"
        ei.mkdir()
        (ei / "pkg.py").write_text("")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1


# ============================================================
# path_to_module
# ============================================================


class TestPathToModule:
    """Tests for path_to_module()."""

    def test_simple_file(self, tmp_path):
        f = tmp_path / "foo.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "foo"

    def test_nested_file(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        f = sub / "mod.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "pkg.mod"

    def test_init_py_becomes_package(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        f = pkg / "__init__.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "pkg"

    def test_deeply_nested(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        f = deep / "mod.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "a.b.c.mod"

    def test_deeply_nested_init(self, tmp_path):
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        f = deep / "__init__.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "a.b"

    def test_src_layout_without_init(self, tmp_path):
        # src/ without __init__.py => strip src/
        src = tmp_path / "src" / "mylib"
        src.mkdir(parents=True)
        f = src / "core.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "mylib.core"

    def test_src_layout_with_init(self, tmp_path):
        # src/ with __init__.py => src is a package, keep it
        src = tmp_path / "src"
        src.mkdir()
        (src / "__init__.py").write_text("")
        f = src / "mod.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "src.mod"

    def test_file_outside_root(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        f = other / "mod.py"
        f.write_text("")
        # root is a subdir that doesn't contain the file
        root = tmp_path / "project"
        root.mkdir()
        result = path_to_module(str(f), str(root))
        assert result == "mod"

    def test_root_init(self, tmp_path):
        f = tmp_path / "__init__.py"
        f.write_text("")
        result = path_to_module(str(f), str(tmp_path))
        # parts becomes empty after removing __init__, falls back to stem
        assert result == "__init__"

    def test_src_layout_init_in_subpackage(self, tmp_path):
        src = tmp_path / "src" / "mylib"
        src.mkdir(parents=True)
        f = src / "__init__.py"
        f.write_text("")
        assert path_to_module(str(f), str(tmp_path)) == "mylib"


# ============================================================
# extract_imports
# ============================================================


class TestExtractImports:
    """Tests for extract_imports()."""

    def test_simple_import(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("import os\n")
        result = extract_imports(str(f))
        assert len(result) == 1
        assert result[0]["module"] == "os"
        assert result[0]["is_relative"] is False
        assert result[0]["level"] == 0

    def test_import_from(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from os.path import join, exists\n")
        result = extract_imports(str(f))
        assert len(result) == 1
        assert result[0]["module"] == "os.path"
        assert result[0]["names"] == ["join", "exists"]

    def test_relative_import(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from . import utils\n")
        result = extract_imports(str(f))
        assert len(result) == 1
        assert result[0]["is_relative"] is True
        assert result[0]["level"] == 1

    def test_double_relative_import(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from .. import base\n")
        result = extract_imports(str(f))
        assert result[0]["level"] == 2
        assert result[0]["is_relative"] is True

    def test_relative_import_with_module(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from .core import Engine\n")
        result = extract_imports(str(f))
        assert result[0]["module"] == "core"
        assert result[0]["names"] == ["Engine"]
        assert result[0]["is_relative"] is True
        assert result[0]["level"] == 1

    def test_multiple_imports(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("import os\nimport sys\nfrom pathlib import Path\n")
        result = extract_imports(str(f))
        assert len(result) == 3

    def test_import_with_alias(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("import numpy as np\n")
        result = extract_imports(str(f))
        assert result[0]["module"] == "numpy"
        assert result[0]["names"] == ["np"]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("")
        result = extract_imports(str(f))
        assert result == []

    def test_no_imports(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\nprint(x)\n")
        result = extract_imports(str(f))
        assert result == []

    def test_syntax_error_returns_empty(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def broken(:\n  pass\n")
        result = extract_imports(str(f))
        assert result == []

    def test_nonexistent_file_returns_empty(self):
        result = extract_imports("/nonexistent/path/file.py")
        assert result == []

    def test_binary_file_returns_empty(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_bytes(b"\x00\x01\x02\x03")
        result = extract_imports(str(f))
        assert result == []

    def test_line_numbers(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("# comment\nimport os\n\nimport sys\n")
        result = extract_imports(str(f))
        assert result[0]["line"] == 2
        assert result[1]["line"] == 4

    def test_from_import_no_module(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from . import something\n")
        result = extract_imports(str(f))
        assert result[0]["module"] == ""
        assert result[0]["names"] == ["something"]

    def test_future_import(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from __future__ import annotations\nimport os\n")
        result = extract_imports(str(f))
        assert len(result) == 2
        assert result[0]["module"] == "__future__"

    def test_star_import(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("from os.path import *\n")
        result = extract_imports(str(f))
        assert result[0]["names"] == ["*"]

    def test_multi_import_one_line(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("import os, sys\n")
        result = extract_imports(str(f))
        # ast.Import with multiple names creates separate entries per alias
        assert len(result) == 2

    def test_import_inside_function(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    import os\n")
        result = extract_imports(str(f))
        assert len(result) == 1

    def test_import_inside_class(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class Foo:\n    import os\n")
        result = extract_imports(str(f))
        assert len(result) == 1

    def test_conditional_import(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("try:\n    import fast\nexcept ImportError:\n    import slow\n")
        result = extract_imports(str(f))
        assert len(result) == 2


# ============================================================
# resolve_import
# ============================================================


class TestResolveImport:
    """Tests for resolve_import()."""

    def test_absolute_direct_match(self):
        module_map = {"foo": "foo.py", "bar": "bar.py"}
        imp = {"module": "foo", "names": [], "is_relative": False, "level": 0, "line": 1}
        resolved, is_internal = resolve_import("bar", "bar.py", imp, module_map, "/root")
        assert resolved == "foo"
        assert is_internal is True

    def test_absolute_no_match(self):
        module_map = {"foo": "foo.py"}
        imp = {"module": "requests", "names": [], "is_relative": False, "level": 0, "line": 1}
        resolved, is_internal = resolve_import("foo", "foo.py", imp, module_map, "/root")
        assert resolved == "requests"
        assert is_internal is False

    def test_absolute_package_prefix(self):
        module_map = {"pkg.sub": "pkg/sub.py", "other": "other.py"}
        imp = {"module": "pkg", "names": [], "is_relative": False, "level": 0, "line": 1}
        resolved, is_internal = resolve_import("other", "other.py", imp, module_map, "/root")
        assert resolved == "pkg"
        assert is_internal is True

    def test_absolute_partial_match(self):
        module_map = {"pkg.sub.mod": "pkg/sub/mod.py", "main": "main.py"}
        imp = {"module": "pkg.sub.mod.func", "names": ["func"], "is_relative": False, "level": 0, "line": 1}
        resolved, is_internal = resolve_import("main", "main.py", imp, module_map, "/root")
        assert resolved == "pkg.sub.mod"
        assert is_internal is True

    def test_relative_level_one(self):
        module_map = {"pkg.a": "pkg/a.py", "pkg.b": "pkg/b.py"}
        imp = {"module": "b", "names": [], "is_relative": True, "level": 1, "line": 1}
        resolved, is_internal = resolve_import("pkg.a", "pkg/a.py", imp, module_map, "/root")
        assert resolved == "pkg.b"
        assert is_internal is True

    def test_relative_level_two(self):
        module_map = {"a.b.c": "a/b/c.py", "a.d": "a/d.py"}
        imp = {"module": "d", "names": [], "is_relative": True, "level": 2, "line": 1}
        resolved, is_internal = resolve_import("a.b.c", "a/b/c.py", imp, module_map, "/root")
        assert resolved == "a.d"
        assert is_internal is True

    def test_relative_from_init(self):
        # from .core import X in pkg/__init__.py => pkg.core
        module_map = {"pkg": "pkg/__init__.py", "pkg.core": "pkg/core.py"}
        imp = {"module": "core", "names": ["X"], "is_relative": True, "level": 1, "line": 1}
        resolved, is_internal = resolve_import("pkg", "pkg/__init__.py", imp, module_map, "/root")
        assert resolved == "pkg.core"
        assert is_internal is True

    def test_relative_no_module(self):
        # from . import utils => resolves based on package
        module_map = {"pkg.a": "pkg/a.py", "pkg.utils": "pkg/utils.py"}
        imp = {"module": "", "names": ["utils"], "is_relative": True, "level": 1, "line": 1}
        resolved, is_internal = resolve_import("pkg.a", "pkg/a.py", imp, module_map, "/root")
        assert is_internal is True

    def test_relative_level_exceeds_parts(self):
        module_map = {"mod": "mod.py"}
        imp = {"module": "x", "names": [], "is_relative": True, "level": 5, "line": 1}
        resolved, is_internal = resolve_import("mod", "mod.py", imp, module_map, "/root")
        assert is_internal is False

    def test_relative_submodule_name_match(self):
        # from . import sub => candidate="pkg", which is a prefix of "pkg.a" and "pkg.sub"
        # so it matches as a package before checking individual names
        module_map = {"pkg.a": "pkg/a.py", "pkg.sub": "pkg/sub.py"}
        imp = {"module": "", "names": ["sub"], "is_relative": True, "level": 1, "line": 1}
        resolved, is_internal = resolve_import("pkg.a", "pkg/a.py", imp, module_map, "/root")
        assert resolved == "pkg"
        assert is_internal is True

    def test_relative_name_match_falls_through_to_names(self):
        # When candidate matches as a package prefix (known modules start with "pkg."),
        # it resolves to the package, not the individual name
        module_map = {"pkg.a": "pkg/a.py", "pkg.sub": "pkg/sub.py"}
        imp = {"module": "", "names": ["sub"], "is_relative": True, "level": 1, "line": 1}
        resolved, is_internal = resolve_import("pkg.a", "pkg/a.py", imp, module_map, "/root")
        # "pkg" is prefix of "pkg.a", so it returns ("pkg", True) before name check
        assert resolved == "pkg"
        assert is_internal is True

    def test_relative_import_external_when_nothing_matches(self):
        # When candidate doesn't match anything in module_map
        module_map = {"other.mod": "other/mod.py"}
        imp = {"module": "unknown", "names": [], "is_relative": True, "level": 1, "line": 1}
        resolved, is_internal = resolve_import("pkg.a", "pkg/a.py", imp, module_map, "/root")
        assert is_internal is False

    def test_absolute_dotted_direct(self):
        module_map = {"pkg.sub.mod": "pkg/sub/mod.py"}
        imp = {"module": "pkg.sub.mod", "names": ["X"], "is_relative": False, "level": 0, "line": 1}
        resolved, is_internal = resolve_import("main", "main.py", imp, module_map, "/root")
        assert resolved == "pkg.sub.mod"
        assert is_internal is True


# ============================================================
# get_top_level_package
# ============================================================


class TestGetTopLevelPackage:
    """Tests for get_top_level_package()."""

    def test_simple_module(self):
        assert get_top_level_package("os") == "os"

    def test_dotted_module(self):
        assert get_top_level_package("os.path") == "os"

    def test_deeply_dotted(self):
        assert get_top_level_package("a.b.c.d.e") == "a"

    def test_empty_string(self):
        assert get_top_level_package("") == ""

    def test_single_part(self):
        assert get_top_level_package("requests") == "requests"


# ============================================================
# build_graph
# ============================================================


class TestBuildGraph:
    """Tests for build_graph()."""

    def test_empty_dir(self, tmp_path):
        edges, module_map = build_graph(str(tmp_path))
        assert edges == []
        assert module_map == {}

    def test_single_file_no_imports(self, tmp_path):
        _write_py(str(tmp_path / "foo.py"), "x = 1\n")
        edges, module_map = build_graph(str(tmp_path))
        assert len(module_map) == 1
        assert "foo" in module_map
        assert edges == []

    def test_two_files_one_import(self, tmp_path):
        _write_py(str(tmp_path / "a.py"), "from b import x\n")
        _write_py(str(tmp_path / "b.py"), "x = 1\n")
        edges, module_map = build_graph(str(tmp_path))
        assert len(module_map) == 2
        assert len(edges) == 1
        assert edges[0].source_module == "a"
        assert edges[0].is_internal is True

    def test_external_import(self, tmp_path):
        _write_py(str(tmp_path / "app.py"), "import requests\n")
        edges, module_map = build_graph(str(tmp_path))
        assert len(edges) == 1
        assert edges[0].is_internal is False

    def test_package_structure(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        _write_py(str(pkg / "__init__.py"), "")
        _write_py(str(pkg / "core.py"), "x = 1\n")
        _write_py(str(pkg / "utils.py"), "from .core import x\n")
        edges, module_map = build_graph(str(tmp_path))
        assert "pkg" in module_map
        assert "pkg.core" in module_map
        assert "pkg.utils" in module_map
        internal = [e for e in edges if e.is_internal]
        assert len(internal) >= 1

    def test_module_map_has_paths(self, tmp_path):
        _write_py(str(tmp_path / "mod.py"), "x = 1\n")
        _, module_map = build_graph(str(tmp_path))
        assert module_map["mod"].endswith("mod.py")

    def test_custom_excludes(self, tmp_path):
        _write_py(str(tmp_path / "app.py"), "")
        skip = tmp_path / "skip_this"
        skip.mkdir()
        _write_py(str(skip / "hidden.py"), "")
        _, module_map = build_graph(str(tmp_path), excludes={"skip_this"})
        assert len(module_map) == 1


# ============================================================
# detect_cycles
# ============================================================


class TestDetectCycles:
    """Tests for detect_cycles()."""

    def test_no_cycles(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="c"),
        ]
        module_map = {"a": "a.py", "b": "b.py", "c": "c.py"}
        cycles = detect_cycles(edges, module_map)
        assert cycles == []

    def test_simple_cycle(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="a"),
        ]
        module_map = {"a": "a.py", "b": "b.py"}
        cycles = detect_cycles(edges, module_map)
        assert len(cycles) == 1
        assert cycles[0].length == 2

    def test_three_node_cycle(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="c"),
            _make_edge(source_module="c", target_resolved="a"),
        ]
        module_map = {"a": "a.py", "b": "b.py", "c": "c.py"}
        cycles = detect_cycles(edges, module_map)
        assert len(cycles) == 1
        assert cycles[0].length == 3

    def test_self_import_ignored(self):
        edges = [
            _make_edge(source_module="a", target_resolved="a"),
        ]
        module_map = {"a": "a.py"}
        cycles = detect_cycles(edges, module_map)
        assert cycles == []

    def test_external_edges_ignored(self):
        edges = [
            _make_edge(source_module="a", target_resolved="ext", is_internal=False),
        ]
        module_map = {"a": "a.py"}
        cycles = detect_cycles(edges, module_map)
        assert cycles == []

    def test_cycle_path_ends_with_start(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="a"),
        ]
        module_map = {"a": "a.py", "b": "b.py"}
        cycles = detect_cycles(edges, module_map)
        assert cycles[0].path[0] == cycles[0].path[-1]

    def test_cycles_normalized_lexicographic(self):
        edges = [
            _make_edge(source_module="z", target_resolved="a"),
            _make_edge(source_module="a", target_resolved="z"),
        ]
        module_map = {"a": "a.py", "z": "z.py"}
        cycles = detect_cycles(edges, module_map)
        assert cycles[0].path[0] == "a"

    def test_cycles_sorted_by_length_desc(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="a"),
            _make_edge(source_module="c", target_resolved="d"),
            _make_edge(source_module="d", target_resolved="e"),
            _make_edge(source_module="e", target_resolved="c"),
        ]
        module_map = {"a": "a.py", "b": "b.py", "c": "c.py", "d": "d.py", "e": "e.py"}
        cycles = detect_cycles(edges, module_map)
        assert len(cycles) == 2
        assert cycles[0].length >= cycles[1].length

    def test_empty_edges(self):
        module_map = {"a": "a.py"}
        cycles = detect_cycles([], module_map)
        assert cycles == []

    def test_empty_module_map(self):
        cycles = detect_cycles([], {})
        assert cycles == []

    def test_unresolved_target_not_in_map(self):
        edges = [
            _make_edge(source_module="a", target_resolved="unknown", is_internal=True),
        ]
        module_map = {"a": "a.py"}
        cycles = detect_cycles(edges, module_map)
        assert cycles == []

    def test_two_separate_cycles(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="a"),
            _make_edge(source_module="x", target_resolved="y"),
            _make_edge(source_module="y", target_resolved="x"),
        ]
        module_map = {"a": "a.py", "b": "b.py", "x": "x.py", "y": "y.py"}
        cycles = detect_cycles(edges, module_map)
        assert len(cycles) == 2


# ============================================================
# calculate_metrics
# ============================================================


class TestCalculateMetrics:
    """Tests for calculate_metrics()."""

    def test_empty(self):
        metrics = calculate_metrics([], {})
        assert metrics == []

    def test_single_orphan(self):
        module_map = {"a": "a.py"}
        metrics = calculate_metrics([], module_map)
        assert len(metrics) == 1
        assert metrics[0].is_orphan is True
        assert metrics[0].afferent == 0
        assert metrics[0].efferent == 0

    def test_one_imports_another(self):
        edges = [_make_edge(source_module="a", target_resolved="b")]
        module_map = {"a": "a.py", "b": "b.py"}
        metrics = calculate_metrics(edges, module_map)
        a = next(m for m in metrics if m.module == "a")
        b = next(m for m in metrics if m.module == "b")
        assert a.efferent == 1
        assert a.afferent == 0
        assert b.afferent == 1
        assert b.efferent == 0

    def test_instability_calculation(self):
        edges = [_make_edge(source_module="a", target_resolved="b")]
        module_map = {"a": "a.py", "b": "b.py"}
        metrics = calculate_metrics(edges, module_map)
        a = next(m for m in metrics if m.module == "a")
        b = next(m for m in metrics if m.module == "b")
        assert a.instability == 1.0  # Ce=1, Ca=0 => 1/(0+1)
        assert b.instability == 0.0  # Ce=0, Ca=1 => 0/(1+0)

    def test_self_import_ignored(self):
        edges = [_make_edge(source_module="a", target_resolved="a")]
        module_map = {"a": "a.py"}
        metrics = calculate_metrics(edges, module_map)
        assert metrics[0].afferent == 0
        assert metrics[0].efferent == 0
        assert metrics[0].is_orphan is True

    def test_external_edges_ignored(self):
        edges = [_make_edge(source_module="a", target_resolved="requests", is_internal=False)]
        module_map = {"a": "a.py"}
        metrics = calculate_metrics(edges, module_map)
        assert metrics[0].efferent == 0

    def test_sorted_by_module_name(self):
        module_map = {"z": "z.py", "a": "a.py", "m": "m.py"}
        metrics = calculate_metrics([], module_map)
        assert [m.module for m in metrics] == ["a", "m", "z"]

    def test_mutual_import(self):
        edges = [
            _make_edge(source_module="a", target_resolved="b"),
            _make_edge(source_module="b", target_resolved="a"),
        ]
        module_map = {"a": "a.py", "b": "b.py"}
        metrics = calculate_metrics(edges, module_map)
        a = next(m for m in metrics if m.module == "a")
        b = next(m for m in metrics if m.module == "b")
        assert a.afferent == 1 and a.efferent == 1
        assert b.afferent == 1 and b.efferent == 1
        assert a.instability == 0.5
        assert b.instability == 0.5

    def test_not_orphan_when_imported(self):
        edges = [_make_edge(source_module="a", target_resolved="b")]
        module_map = {"a": "a.py", "b": "b.py"}
        metrics = calculate_metrics(edges, module_map)
        a = next(m for m in metrics if m.module == "a")
        b = next(m for m in metrics if m.module == "b")
        assert a.is_orphan is False
        assert b.is_orphan is False

    def test_instability_zero_for_orphan(self):
        module_map = {"a": "a.py"}
        metrics = calculate_metrics([], module_map)
        assert metrics[0].instability == 0.0

    def test_target_not_in_module_map_ignored(self):
        edges = [_make_edge(source_module="a", target_resolved="unknown", is_internal=True)]
        module_map = {"a": "a.py"}
        metrics = calculate_metrics(edges, module_map)
        assert metrics[0].efferent == 0

    def test_hub_module_high_afferent(self):
        edges = [
            _make_edge(source_module="a", target_resolved="hub"),
            _make_edge(source_module="b", target_resolved="hub"),
            _make_edge(source_module="c", target_resolved="hub"),
        ]
        module_map = {"a": "a.py", "b": "b.py", "c": "c.py", "hub": "hub.py"}
        metrics = calculate_metrics(edges, module_map)
        hub = next(m for m in metrics if m.module == "hub")
        assert hub.afferent == 3
        assert hub.instability == 0.0


# ============================================================
# analyze_external_deps
# ============================================================


class TestAnalyzeExternalDeps:
    """Tests for analyze_external_deps()."""

    def test_no_external(self):
        edges = [_make_edge(is_internal=True)]
        result = analyze_external_deps(edges)
        assert result == []

    def test_one_external(self):
        edges = [_make_edge(target_raw="requests", is_internal=False)]
        result = analyze_external_deps(edges)
        assert len(result) == 1
        assert result[0]["package"] == "requests"
        assert result[0]["imported_by_count"] == 1

    def test_skips_future(self):
        edges = [_make_edge(target_raw="__future__", is_internal=False)]
        result = analyze_external_deps(edges)
        assert result == []

    def test_groups_by_top_level(self):
        edges = [
            _make_edge(source_module="a", target_raw="requests.auth", is_internal=False),
            _make_edge(source_module="b", target_raw="requests.sessions", is_internal=False),
        ]
        result = analyze_external_deps(edges)
        assert len(result) == 1
        assert result[0]["package"] == "requests"
        assert result[0]["imported_by_count"] == 2

    def test_sorted_by_count_desc(self):
        edges = [
            _make_edge(source_module="a", target_raw="requests", is_internal=False),
            _make_edge(source_module="b", target_raw="requests", is_internal=False),
            _make_edge(source_module="c", target_raw="flask", is_internal=False),
        ]
        result = analyze_external_deps(edges)
        assert result[0]["package"] == "requests"
        assert result[0]["imported_by_count"] == 2

    def test_sorted_alphabetically_on_tie(self):
        edges = [
            _make_edge(source_module="a", target_raw="blib", is_internal=False),
            _make_edge(source_module="b", target_raw="alib", is_internal=False),
        ]
        result = analyze_external_deps(edges)
        assert result[0]["package"] == "alib"
        assert result[1]["package"] == "blib"

    def test_empty_edges(self):
        result = analyze_external_deps([])
        assert result == []

    def test_same_module_imports_same_external(self):
        # Same source imports same external package twice via different submodules
        edges = [
            _make_edge(source_module="a", target_raw="requests.auth", is_internal=False),
            _make_edge(source_module="a", target_raw="requests.sessions", is_internal=False),
        ]
        result = analyze_external_deps(edges)
        # Still only 1 source module for requests (set dedup)
        assert result[0]["imported_by_count"] == 1

    def test_skips_empty_target(self):
        edges = [_make_edge(target_raw="", is_internal=False)]
        result = analyze_external_deps(edges)
        assert result == []


# ============================================================
# analyze (integration)
# ============================================================


class TestAnalyze:
    """Tests for analyze() — integration tests."""

    def test_empty_dir(self, tmp_path):
        result = analyze(str(tmp_path))
        assert result.total_modules == 0
        assert result.total_edges == 0
        assert result.cycles == []

    def test_single_file(self, tmp_path):
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        result = analyze(str(tmp_path))
        assert result.total_modules == 1
        assert result.total_edges == 0

    def test_root_is_resolved_path(self, tmp_path):
        result = analyze(str(tmp_path))
        assert os.path.isabs(result.root)

    def test_internal_edge_count(self, tmp_path):
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "x = 1\n")
        result = analyze(str(tmp_path))
        assert result.internal_edges == 1

    def test_external_package_count(self, tmp_path):
        _write_py(str(tmp_path / "app.py"), "import requests\nimport flask\n")
        result = analyze(str(tmp_path))
        assert result.external_packages == 2

    def test_circular_imports_detected(self, tmp_path):
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import a\n")
        result = analyze(str(tmp_path))
        assert len(result.cycles) == 1
        assert result.cycles[0].length == 2

    def test_orphans_detected(self, tmp_path):
        _write_py(str(tmp_path / "used.py"), "x = 1\n")
        _write_py(str(tmp_path / "orphan.py"), "y = 2\n")
        _write_py(str(tmp_path / "main.py"), "import used\n")
        result = analyze(str(tmp_path))
        assert "orphan" in result.orphans

    def test_hub_modules_detected(self, tmp_path):
        _write_py(str(tmp_path / "hub.py"), "x = 1\n")
        _write_py(str(tmp_path / "a.py"), "import hub\n")
        _write_py(str(tmp_path / "b.py"), "import hub\n")
        _write_py(str(tmp_path / "c.py"), "import hub\n")
        result = analyze(str(tmp_path))
        assert len(result.hub_modules) > 0
        assert result.hub_modules[0]["module"] == "hub"
        assert result.hub_modules[0]["afferent"] == 3

    def test_hub_modules_capped_at_20(self, tmp_path):
        # Create many hub modules
        for i in range(25):
            _write_py(str(tmp_path / f"hub{i}.py"), "x = 1\n")
            _write_py(str(tmp_path / f"user{i}.py"), f"import hub{i}\n")
        result = analyze(str(tmp_path))
        assert len(result.hub_modules) <= 20

    def test_unstable_modules_detected(self, tmp_path):
        # Create a module that imports many others (high efferent) but nobody imports it
        _write_py(str(tmp_path / "stable1.py"), "x = 1\n")
        _write_py(str(tmp_path / "stable2.py"), "y = 2\n")
        _write_py(str(tmp_path / "stable3.py"), "z = 3\n")
        _write_py(str(tmp_path / "unstable.py"), "import stable1\nimport stable2\nimport stable3\n")
        result = analyze(str(tmp_path))
        unstable_names = [m["module"] for m in result.unstable_modules]
        # unstable: instability > 0.8, efferent > 1
        # unstable.py: Ce=3, Ca=0 => instability=1.0, efferent=3 => qualifies
        assert "unstable" in unstable_names

    def test_unstable_modules_capped_at_20(self, tmp_path):
        # Create many unstable modules
        _write_py(str(tmp_path / "lib1.py"), "x = 1\n")
        _write_py(str(tmp_path / "lib2.py"), "y = 2\n")
        for i in range(25):
            _write_py(str(tmp_path / f"consumer{i}.py"), "import lib1\nimport lib2\n")
        result = analyze(str(tmp_path))
        assert len(result.unstable_modules) <= 20

    def test_package_with_init(self, tmp_path):
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        _write_py(str(pkg / "__init__.py"), "")
        _write_py(str(pkg / "core.py"), "value = 42\n")
        _write_py(str(pkg / "api.py"), "from .core import value\n")
        result = analyze(str(tmp_path))
        assert result.total_modules >= 3
        assert result.internal_edges >= 1

    def test_src_layout_project(self, tmp_path):
        src = tmp_path / "src" / "mylib"
        src.mkdir(parents=True)
        _write_py(str(src / "__init__.py"), "")
        _write_py(str(src / "core.py"), "x = 1\n")
        _write_py(str(src / "utils.py"), "from .core import x\n")
        result = analyze(str(tmp_path))
        modules = [m.module for m in result.metrics]
        assert "mylib" in modules
        assert "mylib.core" in modules
        assert "mylib.utils" in modules

    def test_metrics_populated(self, tmp_path):
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "x = 1\n")
        result = analyze(str(tmp_path))
        assert len(result.metrics) == 2

    def test_external_deps_populated(self, tmp_path):
        _write_py(str(tmp_path / "app.py"), "import requests\n")
        result = analyze(str(tmp_path))
        assert len(result.external_deps) == 1
        assert result.external_deps[0]["package"] == "requests"

    def test_custom_excludes(self, tmp_path):
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        skip = tmp_path / "vendor"
        skip.mkdir()
        _write_py(str(skip / "lib.py"), "y = 2\n")
        result = analyze(str(tmp_path), excludes={"vendor"})
        assert result.total_modules == 1

    def test_three_node_cycle(self, tmp_path):
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import c\n")
        _write_py(str(tmp_path / "c.py"), "import a\n")
        result = analyze(str(tmp_path))
        assert len(result.cycles) == 1
        assert result.cycles[0].length == 3


# ============================================================
# colorize
# ============================================================


class TestColorize:
    """Tests for colorize()."""

    def test_no_color_when_not_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        assert colorize("hello", "red") == "hello"

    def test_color_when_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = colorize("hello", "red")
        assert "\033[91m" in result
        assert "\033[0m" in result
        assert "hello" in result

    def test_unknown_color_no_crash(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = colorize("hello", "nonexistent")
        assert "hello" in result

    def test_all_colors_apply(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        for color_name in COLORS:
            if color_name == "reset":
                continue
            result = colorize("test", color_name)
            assert COLORS[color_name] in result

    def test_empty_text(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        assert colorize("", "red") == ""

    def test_bold(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = colorize("text", "bold")
        assert "\033[1m" in result

    def test_dim(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        result = colorize("text", "dim")
        assert "\033[2m" in result


# ============================================================
# format_result
# ============================================================


class TestFormatResult:
    """Tests for format_result()."""

    def test_returns_string(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result()
        assert isinstance(format_result(r), str)

    def test_contains_summary(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result(total_modules=5, total_edges=10, internal_edges=8)
        text = format_result(r)
        assert "5" in text
        assert "10" in text

    def test_no_cycles_message(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result()
        text = format_result(r)
        assert "No circular imports" in text

    def test_cycles_shown(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        c = CycleInfo(path=["a", "b", "a"], length=2)
        r = _make_result(cycles=[c])
        text = format_result(r)
        assert "a" in text
        assert "b" in text

    def test_hub_modules_shown(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result(
            hub_modules=[{"module": "hub", "afferent": 5, "path": "hub.py"}],
        )
        text = format_result(r)
        assert "hub" in text
        assert "5" in text

    def test_unstable_modules_shown(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result(
            unstable_modules=[{"module": "fragile", "instability": 0.95, "efferent": 3, "afferent": 0}],
        )
        text = format_result(r)
        assert "fragile" in text

    def test_orphans_shown(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result(orphans=["lonely"])
        text = format_result(r)
        assert "lonely" in text

    def test_external_deps_shown(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        r = _make_result(
            external_packages=1,
            external_deps=[{"package": "requests", "imported_by_count": 3, "imported_by": ["a", "b", "c"]}],
        )
        text = format_result(r)
        assert "requests" in text

    def test_top_n_limits_cycles(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        cycles = [CycleInfo(path=[f"m{i}", f"m{i+1}", f"m{i}"], length=2) for i in range(20)]
        r = _make_result(cycles=cycles)
        text = format_result(r, top_n=5)
        assert "... and 15 more" in text

    def test_top_n_limits_orphans(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        orphans = [f"orphan_{i}" for i in range(20)]
        r = _make_result(orphans=orphans)
        text = format_result(r, top_n=5)
        assert "... and 15 more" in text

    def test_top_n_default_is_10(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        cycles = [CycleInfo(path=[f"m{i}", f"n{i}", f"m{i}"], length=2) for i in range(15)]
        r = _make_result(cycles=cycles)
        text = format_result(r)
        assert "... and 5 more" in text


# ============================================================
# format_json
# ============================================================


class TestFormatJson:
    """Tests for format_json()."""

    def test_valid_json(self):
        r = _make_result()
        data = json.loads(format_json(r))
        assert isinstance(data, dict)

    def test_has_version(self):
        r = _make_result()
        data = json.loads(format_json(r))
        assert data["version"] == __version__

    def test_has_summary(self):
        r = _make_result(total_modules=10, total_edges=20)
        data = json.loads(format_json(r))
        assert data["summary"]["total_modules"] == 10
        assert data["summary"]["total_edges"] == 20

    def test_has_cycles(self):
        c = CycleInfo(path=["a", "b", "a"], length=2)
        r = _make_result(cycles=[c])
        data = json.loads(format_json(r))
        assert len(data["cycles"]) == 1
        assert data["cycles"][0]["length"] == 2

    def test_has_orphans(self):
        r = _make_result(orphans=["lone"])
        data = json.loads(format_json(r))
        assert data["orphans"] == ["lone"]

    def test_has_hub_modules(self):
        r = _make_result(
            hub_modules=[{"module": "hub", "afferent": 5, "path": "hub.py"}],
        )
        data = json.loads(format_json(r))
        assert len(data["hub_modules"]) == 1

    def test_has_unstable_modules(self):
        r = _make_result(
            unstable_modules=[{"module": "u", "instability": 0.9, "efferent": 3, "afferent": 0}],
        )
        data = json.loads(format_json(r))
        assert len(data["unstable_modules"]) == 1

    def test_has_external_deps(self):
        r = _make_result(
            external_deps=[{"package": "flask", "imported_by_count": 2, "imported_by": ["a", "b"]}],
        )
        data = json.loads(format_json(r))
        assert len(data["external_deps"]) == 1

    def test_has_metrics(self):
        m = _make_metric("mod_a")
        r = _make_result(metrics=[m])
        data = json.loads(format_json(r))
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["module"] == "mod_a"

    def test_root_in_output(self):
        r = _make_result(root="/my/project")
        data = json.loads(format_json(r))
        assert data["root"] == "/my/project"

    def test_circular_imports_count(self):
        c1 = CycleInfo(path=["a", "b", "a"], length=2)
        c2 = CycleInfo(path=["c", "d", "c"], length=2)
        r = _make_result(cycles=[c1, c2])
        data = json.loads(format_json(r))
        assert data["summary"]["circular_imports"] == 2

    def test_orphan_modules_count(self):
        r = _make_result(orphans=["a", "b", "c"])
        data = json.loads(format_json(r))
        assert data["summary"]["orphan_modules"] == 3


# ============================================================
# build_parser
# ============================================================


class TestBuildParser:
    """Tests for build_parser()."""

    def test_default_path(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.path == "."

    def test_custom_path(self):
        parser = build_parser()
        args = parser.parse_args(["/some/path"])
        assert args.path == "/some/path"

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json_output is True

    def test_json_flag_default_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.json_output is False

    def test_cycles_only_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--cycles-only"])
        assert args.cycles_only is True

    def test_cycles_only_default_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.cycles_only is False

    def test_top_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--top", "5"])
        assert args.top == 5

    def test_top_default_is_10(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.top == 10

    def test_score_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--score"])
        assert args.score is True

    def test_score_default_false(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.score is False

    def test_version_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_all_flags_combined(self):
        parser = build_parser()
        args = parser.parse_args(["--json", "--score", "--top", "20", "/path"])
        assert args.json_output is True
        assert args.score is True
        assert args.top == 20
        assert args.path == "/path"


# ============================================================
# main (CLI)
# ============================================================


class TestMain:
    """Tests for main() CLI entry point."""

    def test_nonexistent_path_returns_1(self, capsys):
        ret = main(["/nonexistent/path/that/does/not/exist"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "not a directory" in err

    def test_empty_dir_returns_0(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        ret = main([str(tmp_path)])
        assert ret == 0

    def test_json_output(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([str(tmp_path), "--json"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data

    def test_cycles_only_no_cycles(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([str(tmp_path), "--cycles-only"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "No circular imports" in out

    def test_cycles_only_with_cycles_returns_2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import a\n")
        ret = main([str(tmp_path), "--cycles-only"])
        assert ret == 2

    def test_normal_mode_with_cycles_returns_2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import a\n")
        ret = main([str(tmp_path)])
        assert ret == 2

    def test_normal_mode_no_cycles_returns_0(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([str(tmp_path)])
        assert ret == 0

    def test_score_flag(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([str(tmp_path), "--score"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "/100" in out

    def test_score_json_flag(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([str(tmp_path), "--score", "--json"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "score" in data
        assert "grade" in data

    def test_score_returns_0_even_with_cycles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import a\n")
        ret = main([str(tmp_path), "--score"])
        assert ret == 0

    def test_json_returns_0_even_with_cycles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import a\n")
        ret = main([str(tmp_path), "--json"])
        assert ret == 0

    def test_top_flag_works(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([str(tmp_path), "--top", "3"])
        assert ret == 0

    def test_default_path_dot(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        monkeypatch.chdir(tmp_path)
        _write_py(str(tmp_path / "app.py"), "x = 1\n")
        ret = main([])
        assert ret == 0

    def test_file_path_returns_1(self, tmp_path, capsys):
        f = tmp_path / "file.py"
        f.write_text("x = 1")
        ret = main([str(f)])
        assert ret == 1


# ============================================================
# Integration: complex project structures
# ============================================================


class TestIntegrationComplexProjects:
    """Integration tests with realistic project structures."""

    def test_django_like_structure(self, tmp_path):
        """Simulate a Django-like app with models, views, urls."""
        app = tmp_path / "myapp"
        app.mkdir()
        _write_py(str(app / "__init__.py"), "")
        _write_py(str(app / "models.py"), """\
            class User:
                pass
        """)
        _write_py(str(app / "views.py"), """\
            from .models import User
            def index():
                return User()
        """)
        _write_py(str(app / "urls.py"), """\
            from .views import index
        """)
        _write_py(str(app / "admin.py"), """\
            from .models import User
        """)
        result = analyze(str(tmp_path))
        assert result.total_modules >= 5
        assert result.internal_edges >= 3
        assert len(result.cycles) == 0
        # models should be a hub
        hub_names = [h["module"] for h in result.hub_modules]
        assert "myapp.models" in hub_names

    def test_deeply_nested_package(self, tmp_path):
        """Test a deeply nested package hierarchy."""
        deep = tmp_path / "pkg" / "sub1" / "sub2" / "sub3"
        deep.mkdir(parents=True)
        for d in [tmp_path / "pkg", tmp_path / "pkg" / "sub1",
                  tmp_path / "pkg" / "sub1" / "sub2", deep]:
            _write_py(str(d / "__init__.py"), "")
        _write_py(str(deep / "mod.py"), "x = 1\n")
        result = analyze(str(tmp_path))
        modules = [m.module for m in result.metrics]
        assert "pkg.sub1.sub2.sub3.mod" in modules

    def test_mixed_internal_external(self, tmp_path):
        """Project with both internal and external imports."""
        _write_py(str(tmp_path / "core.py"), "value = 42\n")
        _write_py(str(tmp_path / "app.py"), """\
            import os
            import json
            import requests
            from core import value
        """)
        result = analyze(str(tmp_path))
        assert result.internal_edges >= 1
        assert result.external_packages >= 1

    def test_self_importing_module(self, tmp_path):
        """Module that imports itself should not cause cycles."""
        _write_py(str(tmp_path / "selfish.py"), "import selfish\n")
        result = analyze(str(tmp_path))
        assert len(result.cycles) == 0

    def test_many_orphans(self, tmp_path):
        """Project with many unconnected modules."""
        for i in range(10):
            _write_py(str(tmp_path / f"module_{i}.py"), f"x_{i} = {i}\n")
        result = analyze(str(tmp_path))
        assert len(result.orphans) == 10
        health = compute_import_health(result)
        assert health.orphan_ratio == 1.0

    def test_star_topology(self, tmp_path):
        """Star topology: one hub module imported by all others."""
        _write_py(str(tmp_path / "hub.py"), "shared = True\n")
        for i in range(8):
            _write_py(str(tmp_path / f"spoke_{i}.py"), "from hub import shared\n")
        result = analyze(str(tmp_path))
        assert result.hub_modules[0]["module"] == "hub"
        assert result.hub_modules[0]["afferent"] == 8

    def test_chain_topology(self, tmp_path):
        """Linear chain: a -> b -> c -> d."""
        _write_py(str(tmp_path / "d.py"), "x = 1\n")
        _write_py(str(tmp_path / "c.py"), "import d\n")
        _write_py(str(tmp_path / "b.py"), "import c\n")
        _write_py(str(tmp_path / "a.py"), "import b\n")
        result = analyze(str(tmp_path))
        assert len(result.cycles) == 0
        assert result.internal_edges == 3

    def test_diamond_dependency(self, tmp_path):
        """Diamond: a -> b, a -> c, b -> d, c -> d."""
        _write_py(str(tmp_path / "d.py"), "x = 1\n")
        _write_py(str(tmp_path / "b.py"), "import d\n")
        _write_py(str(tmp_path / "c.py"), "import d\n")
        _write_py(str(tmp_path / "a.py"), "import b\nimport c\n")
        result = analyze(str(tmp_path))
        assert len(result.cycles) == 0
        # d is imported by b and c
        d_metrics = next(m for m in result.metrics if m.module == "d")
        assert d_metrics.afferent == 2

    def test_complex_cycle_detection(self, tmp_path):
        """Multiple overlapping cycles."""
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import c\nimport a\n")
        _write_py(str(tmp_path / "c.py"), "import a\n")
        result = analyze(str(tmp_path))
        assert len(result.cycles) >= 1

    def test_syntax_error_file_gracefully_skipped(self, tmp_path):
        """Files with syntax errors should not crash analysis."""
        _write_py(str(tmp_path / "good.py"), "x = 1\n")
        _write_py(str(tmp_path / "bad.py"), "def broken(:\n")
        result = analyze(str(tmp_path))
        # bad.py is in module_map but has no imports
        assert result.total_modules == 2

    def test_binary_file_in_project(self, tmp_path):
        """Binary files with .py extension should be handled."""
        _write_py(str(tmp_path / "good.py"), "x = 1\n")
        (tmp_path / "binary.py").write_bytes(b"\x00\x01\x89PNG")
        result = analyze(str(tmp_path))
        assert result.total_modules == 2

    def test_empty_init_files_only(self, tmp_path):
        """Project with only __init__.py files."""
        pkg = tmp_path / "pkg" / "sub"
        pkg.mkdir(parents=True)
        _write_py(str(tmp_path / "pkg" / "__init__.py"), "")
        _write_py(str(pkg / "__init__.py"), "")
        result = analyze(str(tmp_path))
        assert result.total_modules == 2

    def test_relative_imports_across_packages(self, tmp_path):
        """Relative imports between subpackages."""
        pkg = tmp_path / "pkg"
        sub_a = pkg / "a"
        sub_b = pkg / "b"
        for d in [pkg, sub_a, sub_b]:
            d.mkdir(parents=True)
            _write_py(str(d / "__init__.py"), "")
        _write_py(str(sub_a / "mod.py"), "x = 1\n")
        _write_py(str(sub_b / "mod.py"), "from ..a.mod import x\n")
        result = analyze(str(tmp_path))
        internal = [e for e in result.metrics if not e.is_orphan]
        assert len(internal) > 0

    def test_full_health_pipeline(self, tmp_path):
        """End-to-end: analyze -> health score -> format."""
        _write_py(str(tmp_path / "a.py"), "import b\n")
        _write_py(str(tmp_path / "b.py"), "import a\n")
        _write_py(str(tmp_path / "orphan.py"), "x = 1\n")
        result = analyze(str(tmp_path))
        health = compute_import_health(result)
        assert 0 <= health.score <= 100
        assert health.grade in ("A", "B", "C", "D", "F")
        text = format_score_text(health)
        assert isinstance(text, str)
        j = format_score_json(health)
        data = json.loads(j)
        assert data["score"] == health.score

    def test_large_project_does_not_crash(self, tmp_path):
        """50 modules with various dependencies."""
        for i in range(50):
            if i > 0:
                _write_py(str(tmp_path / f"mod_{i}.py"), f"import mod_{i - 1}\n")
            else:
                _write_py(str(tmp_path / f"mod_{i}.py"), "base = True\n")
        result = analyze(str(tmp_path))
        assert result.total_modules == 50
        assert result.total_edges == 49
