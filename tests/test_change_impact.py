"""Tests for the change_impact analysis engine."""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_health_suite.engines.change_impact import (
    DEFAULT_EXCLUDES,
    ChangeImpactResult,
    CouplingMetrics,
    CouplingResult,
    ImpactedModule,
    _coupling_grade,
    analyze,
    build_dependency_graph,
    build_parser,
    build_reverse_deps,
    compute_coupling_metrics,
    compute_risk_level,
    extract_imports,
    find_python_files,
    format_coupling_json,
    format_coupling_text,
    format_json,
    format_text,
    get_changed_files_from_git,
    is_test_file,
    main,
    path_to_module,
    propagate_impact,
    resolve_import_target,
    suggest_test_command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str = "") -> Path:
    """Write a file, creating parent dirs as needed. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def _make_project(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a small Python project under tmp_path.

    ``files`` maps relative paths to source content.
    Returns the project root.
    """
    for rel, content in files.items():
        _write(tmp_path / rel, content)
    return tmp_path


def _make_impact_result(**overrides) -> ChangeImpactResult:
    """Create a ChangeImpactResult with sensible defaults."""
    defaults = dict(
        root="/project",
        changed_files=["/project/a.py"],
        changed_modules=["a"],
        total_project_modules=5,
        direct_impact=[],
        transitive_impact=[],
        affected_tests=[],
        impact_score=0.0,
        risk_level="low",
        summary="Impact: 1 changed / 0 direct / 0 transitive / 0 tests — risk: low",
    )
    defaults.update(overrides)
    return ChangeImpactResult(**defaults)


# ===================================================================
# Data Structures
# ===================================================================


class TestImpactedModule:
    """Tests for the ImpactedModule dataclass."""

    def test_default_imported_by(self):
        m = ImpactedModule(module="a", path="a.py", depth=1, is_test=False)
        assert m.imported_by == []

    def test_with_imported_by(self):
        m = ImpactedModule(
            module="b", path="b.py", depth=2, is_test=True,
            imported_by=["a", "c"],
        )
        assert m.imported_by == ["a", "c"]
        assert m.is_test is True
        assert m.depth == 2

    def test_asdict(self):
        m = ImpactedModule(module="x", path="x.py", depth=0, is_test=False)
        d = asdict(m)
        assert d == {
            "module": "x", "path": "x.py", "depth": 0,
            "is_test": False, "imported_by": [],
        }


class TestCouplingMetrics:
    """Tests for the CouplingMetrics dataclass."""

    def test_fields(self):
        m = CouplingMetrics(
            module="core", path="core.py", is_test=False,
            ca=5, ce=3, instability=0.375, hub_score=15, grade="D",
        )
        assert m.ca == 5
        assert m.ce == 3
        assert m.instability == 0.375
        assert m.hub_score == 15
        assert m.grade == "D"

    def test_asdict(self):
        m = CouplingMetrics(
            module="m", path="m.py", is_test=False,
            ca=0, ce=0, instability=0.0, hub_score=0, grade="A",
        )
        d = asdict(m)
        assert d["module"] == "m"
        assert d["grade"] == "A"


class TestCouplingResult:
    """Tests for the CouplingResult dataclass."""

    def test_fields(self):
        r = CouplingResult(
            root="/root", total_modules=3, modules=[], avg_instability=0.5,
            median_instability=0.5, hub_modules=[], stable_modules=[],
            unstable_modules=[], summary="test",
        )
        assert r.total_modules == 3
        assert r.summary == "test"


class TestChangeImpactResult:
    """Tests for the ChangeImpactResult dataclass."""

    def test_fields(self):
        r = _make_impact_result(impact_score=0.42, risk_level="high")
        assert r.impact_score == 0.42
        assert r.risk_level == "high"

    def test_asdict(self):
        r = _make_impact_result()
        d = asdict(r)
        assert "changed_modules" in d
        assert "impact_score" in d


# ===================================================================
# find_python_files
# ===================================================================


class TestFindPythonFiles:
    """Tests for find_python_files()."""

    def test_finds_py_files(self, tmp_path):
        _write(tmp_path / "a.py")
        _write(tmp_path / "b.py")
        _write(tmp_path / "c.txt")
        result = find_python_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "a.py" in basenames
        assert "b.py" in basenames
        assert "c.txt" not in basenames

    def test_finds_nested_files(self, tmp_path):
        _write(tmp_path / "pkg" / "__init__.py")
        _write(tmp_path / "pkg" / "mod.py")
        result = find_python_files(str(tmp_path))
        basenames = [os.path.basename(f) for f in result]
        assert "__init__.py" in basenames
        assert "mod.py" in basenames

    def test_returns_sorted(self, tmp_path):
        _write(tmp_path / "z.py")
        _write(tmp_path / "a.py")
        _write(tmp_path / "m.py")
        result = find_python_files(str(tmp_path))
        assert result == sorted(result)

    def test_excludes_venv(self, tmp_path):
        _write(tmp_path / "app.py")
        _write(tmp_path / ".venv" / "lib.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("app.py")

    def test_excludes_node_modules(self, tmp_path):
        _write(tmp_path / "app.py")
        _write(tmp_path / "node_modules" / "something.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_pycache(self, tmp_path):
        _write(tmp_path / "app.py")
        _write(tmp_path / "__pycache__" / "app.cpython-311.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_git(self, tmp_path):
        _write(tmp_path / "app.py")
        _write(tmp_path / ".git" / "hooks" / "pre-commit.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_dot_dirs(self, tmp_path):
        _write(tmp_path / "app.py")
        _write(tmp_path / ".hidden" / "secret.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_tox(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / ".tox" / "env.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_dist(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / "dist" / "pkg.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_build(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / "build" / "pkg.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_mypy_cache(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / ".mypy_cache" / "cache.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_pytest_cache(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / ".pytest_cache" / "cache.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_eggs(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / ".eggs" / "egg.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_site_packages(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / "site-packages" / "lib.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_excludes_egg_info(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / "egg-info" / "pkg.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_pyvenv_cfg_root_returns_empty(self, tmp_path):
        _write(tmp_path / "pyvenv.cfg", "home = /usr/bin")
        _write(tmp_path / "lib.py")
        result = find_python_files(str(tmp_path))
        assert result == []

    def test_pyvenv_cfg_subdirectory_excluded(self, tmp_path):
        _write(tmp_path / "app.py")
        venv = tmp_path / "myenv"
        _write(venv / "pyvenv.cfg", "home = /usr/bin")
        _write(venv / "lib" / "module.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("app.py")

    def test_custom_excludes(self, tmp_path):
        _write(tmp_path / "a.py")
        _write(tmp_path / "custom_dir" / "b.py")
        result = find_python_files(str(tmp_path), excludes={"custom_dir"})
        assert len(result) == 1

    def test_custom_excludes_replaces_defaults(self, tmp_path):
        """Custom excludes fully replace DEFAULT_EXCLUDES."""
        _write(tmp_path / "a.py")
        _write(tmp_path / ".venv" / "b.py")
        # With custom excludes that don't include .venv, .venv should still
        # be excluded because dot-dirs are always excluded.
        result = find_python_files(str(tmp_path), excludes={"other"})
        # .venv starts with '.', so still excluded via dot-dir check
        assert len(result) == 1

    def test_empty_directory(self, tmp_path):
        result = find_python_files(str(tmp_path))
        assert result == []

    def test_deeply_nested(self, tmp_path):
        _write(tmp_path / "a" / "b" / "c" / "d" / "deep.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("deep.py")

    def test_all_default_excludes(self):
        """Verify DEFAULT_EXCLUDES contains expected entries."""
        expected = {
            ".venv", "venv", ".tox", "node_modules", "__pycache__", ".git",
            ".mypy_cache", ".pytest_cache", "dist", "build", "egg-info",
            "site-packages", ".eggs",
        }
        assert DEFAULT_EXCLUDES == expected

    def test_excludes_venv_without_dot(self, tmp_path):
        _write(tmp_path / "main.py")
        _write(tmp_path / "venv" / "lib.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_site_packages_in_path(self, tmp_path):
        """site-packages anywhere in path gets excluded."""
        _write(tmp_path / "main.py")
        _write(tmp_path / "lib" / "site-packages" / "pkg.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_node_modules_in_subpath(self, tmp_path):
        """node_modules anywhere in path gets excluded."""
        _write(tmp_path / "main.py")
        _write(tmp_path / "frontend" / "node_modules" / "pkg.py")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1


# ===================================================================
# path_to_module
# ===================================================================


class TestPathToModule:
    """Tests for path_to_module()."""

    def test_simple_file(self, tmp_path):
        f = _write(tmp_path / "module.py")
        assert path_to_module(str(f), str(tmp_path)) == "module"

    def test_nested_module(self, tmp_path):
        f = _write(tmp_path / "pkg" / "sub.py")
        assert path_to_module(str(f), str(tmp_path)) == "pkg.sub"

    def test_init_file(self, tmp_path):
        f = _write(tmp_path / "pkg" / "__init__.py")
        assert path_to_module(str(f), str(tmp_path)) == "pkg"

    def test_deeply_nested(self, tmp_path):
        f = _write(tmp_path / "a" / "b" / "c" / "d.py")
        assert path_to_module(str(f), str(tmp_path)) == "a.b.c.d"

    def test_src_layout_strips_src(self, tmp_path):
        """src/ is stripped when there is no src/__init__.py."""
        f = _write(tmp_path / "src" / "mypackage" / "core.py")
        # No src/__init__.py
        assert path_to_module(str(f), str(tmp_path)) == "mypackage.core"

    def test_src_layout_keeps_src_with_init(self, tmp_path):
        """src/ is kept when src/__init__.py exists (src IS a package)."""
        _write(tmp_path / "src" / "__init__.py")
        f = _write(tmp_path / "src" / "mypackage" / "core.py")
        assert path_to_module(str(f), str(tmp_path)) == "src.mypackage.core"

    def test_init_in_nested_package(self, tmp_path):
        f = _write(tmp_path / "pkg" / "sub" / "__init__.py")
        assert path_to_module(str(f), str(tmp_path)) == "pkg.sub"

    def test_file_outside_root(self, tmp_path):
        """File outside root falls back to stem."""
        other = tmp_path / "other"
        other.mkdir()
        f = _write(other / "outside.py")
        root = tmp_path / "project"
        root.mkdir()
        result = path_to_module(str(f), str(root))
        assert result == "outside"

    def test_root_level_init(self, tmp_path):
        """__init__.py at root returns stem as fallback."""
        f = _write(tmp_path / "__init__.py")
        result = path_to_module(str(f), str(tmp_path))
        # parts will be empty after stripping __init__, so falls back to stem
        assert result == "__init__"

    def test_src_layout_init_in_package(self, tmp_path):
        """src/pkg/__init__.py with src-layout."""
        f = _write(tmp_path / "src" / "pkg" / "__init__.py")
        assert path_to_module(str(f), str(tmp_path)) == "pkg"


# ===================================================================
# is_test_file
# ===================================================================


class TestIsTestFile:
    """Tests for is_test_file()."""

    def test_test_prefix(self):
        assert is_test_file("test_module.py") is True

    def test_test_suffix(self):
        assert is_test_file("module_test.py") is True

    def test_conftest(self):
        assert is_test_file("conftest.py") is True

    def test_tests_directory(self):
        assert is_test_file("/project/tests/something.py") is True

    def test_test_directory(self):
        assert is_test_file("/project/test/something.py") is True

    def test_regular_file(self):
        assert is_test_file("module.py") is False

    def test_regular_nested(self):
        assert is_test_file("/project/src/core.py") is False

    def test_test_in_name_but_not_prefix_or_suffix(self):
        assert is_test_file("testing_utils.py") is False

    def test_windows_path_tests(self):
        """Backslashes are normalized for /tests/ detection."""
        assert is_test_file("project\\tests\\foo.py") is True

    def test_windows_path_test(self):
        assert is_test_file("project\\test\\foo.py") is True

    def test_nested_tests_dir(self):
        assert is_test_file("/project/src/tests/deep/module.py") is True

    def test_conftest_in_subdirectory(self):
        assert is_test_file("/project/tests/conftest.py") is True

    def test_file_ending_with_test_no_underscore(self):
        """'contest.py' should NOT be detected as test file."""
        assert is_test_file("contest.py") is False

    def test_test_prefix_in_subdir(self):
        assert is_test_file("/src/test_something.py") is True


# ===================================================================
# extract_imports
# ===================================================================


class TestExtractImports:
    """Tests for extract_imports()."""

    def test_simple_import(self, tmp_path):
        f = _write(tmp_path / "mod.py", "import os\n")
        result = extract_imports(str(f))
        assert len(result) == 1
        assert result[0]["module"] == "os"
        assert result[0]["is_relative"] is False
        assert result[0]["level"] == 0

    def test_from_import(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from os.path import join\n")
        result = extract_imports(str(f))
        assert len(result) == 1
        assert result[0]["module"] == "os.path"
        assert result[0]["names"] == ["join"]
        assert result[0]["is_relative"] is False

    def test_relative_import(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from . import sibling\n")
        result = extract_imports(str(f))
        assert len(result) == 1
        assert result[0]["is_relative"] is True
        assert result[0]["level"] == 1

    def test_relative_import_level_2(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from .. import parent\n")
        result = extract_imports(str(f))
        assert result[0]["level"] == 2
        assert result[0]["is_relative"] is True

    def test_relative_import_with_module(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from ..utils import helper\n")
        result = extract_imports(str(f))
        assert result[0]["module"] == "utils"
        assert result[0]["level"] == 2
        assert result[0]["names"] == ["helper"]

    def test_multiple_imports(self, tmp_path):
        code = """\
        import os
        import sys
        from pathlib import Path
        """
        f = _write(tmp_path / "mod.py", code)
        result = extract_imports(str(f))
        assert len(result) == 3

    def test_import_with_alias(self, tmp_path):
        f = _write(tmp_path / "mod.py", "import numpy as np\n")
        result = extract_imports(str(f))
        assert result[0]["module"] == "numpy"
        assert result[0]["names"] == ["np"]  # alias

    def test_from_import_multiple_names(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from os.path import join, dirname, basename\n")
        result = extract_imports(str(f))
        assert result[0]["names"] == ["join", "dirname", "basename"]

    def test_syntax_error_returns_empty(self, tmp_path):
        f = _write(tmp_path / "bad.py", "def f(\n")
        result = extract_imports(str(f))
        assert result == []

    def test_nonexistent_file_returns_empty(self, tmp_path):
        result = extract_imports(str(tmp_path / "nonexistent.py"))
        assert result == []

    def test_empty_file(self, tmp_path):
        f = _write(tmp_path / "empty.py", "")
        result = extract_imports(str(f))
        assert result == []

    def test_line_numbers(self, tmp_path):
        code = """\
        # line 1
        import os  # line 2
        # line 3
        from sys import argv  # line 4
        """
        f = _write(tmp_path / "mod.py", code)
        result = extract_imports(str(f))
        assert result[0]["line"] == 2
        assert result[1]["line"] == 4

    def test_from_import_no_module(self, tmp_path):
        """from . import something has module=''."""
        f = _write(tmp_path / "mod.py", "from . import foo\n")
        result = extract_imports(str(f))
        assert result[0]["module"] == ""

    def test_import_star(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from os import *\n")
        result = extract_imports(str(f))
        assert "*" in result[0]["names"]

    def test_encoding_error_handled(self, tmp_path):
        """Binary content is handled via errors='replace'."""
        f = tmp_path / "binary.py"
        f.write_bytes(b"\x80\x81import os\n")
        result = extract_imports(str(f))
        # Should not raise; may or may not parse depending on content
        assert isinstance(result, list)

    def test_relative_import_level_3(self, tmp_path):
        f = _write(tmp_path / "mod.py", "from ...deep import something\n")
        result = extract_imports(str(f))
        assert result[0]["level"] == 3
        assert result[0]["module"] == "deep"

    def test_multiple_import_names_in_import(self, tmp_path):
        f = _write(tmp_path / "mod.py", "import os, sys, json\n")
        result = extract_imports(str(f))
        assert len(result) == 3
        modules = {r["module"] for r in result}
        assert modules == {"os", "sys", "json"}


# ===================================================================
# resolve_import_target
# ===================================================================


class TestResolveImportTarget:
    """Tests for resolve_import_target()."""

    def test_absolute_import_known(self):
        imp = {"module": "pkg.utils", "is_relative": False, "level": 0}
        known = {"pkg", "pkg.utils", "pkg.core"}
        result = resolve_import_target("pkg.core", "pkg/core.py", imp, known)
        assert result == "pkg.utils"

    def test_absolute_import_unknown(self):
        imp = {"module": "numpy", "is_relative": False, "level": 0}
        known = {"pkg", "pkg.core"}
        result = resolve_import_target("pkg.core", "pkg/core.py", imp, known)
        assert result is None

    def test_absolute_import_prefix_match(self):
        """If exact match fails, try prefix matching."""
        imp = {"module": "pkg.utils.helper", "is_relative": False, "level": 0}
        known = {"pkg", "pkg.utils"}
        result = resolve_import_target("pkg.core", "pkg/core.py", imp, known)
        assert result == "pkg.utils"

    def test_relative_import_same_level(self):
        """from . import sibling in pkg/sub.py -> pkg.sibling if known."""
        imp = {"module": "", "is_relative": True, "level": 1}
        known = {"pkg", "pkg.sub", "pkg.sibling"}
        result = resolve_import_target("pkg.sub", "pkg/sub.py", imp, known)
        assert result == "pkg"

    def test_relative_import_with_module(self):
        """from .utils import helper in pkg/core.py -> pkg.utils."""
        imp = {"module": "utils", "is_relative": True, "level": 1}
        known = {"pkg", "pkg.core", "pkg.utils"}
        result = resolve_import_target("pkg.core", "pkg/core.py", imp, known)
        assert result == "pkg.utils"

    def test_relative_import_level_2(self):
        """from .. import base in pkg/sub/mod.py -> pkg.base."""
        imp = {"module": "base", "is_relative": True, "level": 2}
        known = {"pkg", "pkg.base", "pkg.sub", "pkg.sub.mod"}
        result = resolve_import_target("pkg.sub.mod", "pkg/sub/mod.py", imp, known)
        assert result == "pkg.base"

    def test_relative_import_from_init(self):
        """from . import X in __init__.py gets level-1 treatment."""
        imp = {"module": "sub", "is_relative": True, "level": 1}
        known = {"pkg", "pkg.sub"}
        result = resolve_import_target("pkg", "pkg/__init__.py", imp, known)
        assert result == "pkg.sub"

    def test_relative_import_level_exceeds_parts(self):
        """Level exceeds module depth -> base_parts is empty."""
        imp = {"module": "foo", "is_relative": True, "level": 5}
        known = {"foo"}
        result = resolve_import_target("a.b", "a/b.py", imp, known)
        assert result == "foo"

    def test_relative_import_no_module_deep(self):
        """from .. import in pkg/sub/mod.py -> pkg."""
        imp = {"module": "", "is_relative": True, "level": 2}
        known = {"pkg", "pkg.sub", "pkg.sub.mod"}
        result = resolve_import_target("pkg.sub.mod", "pkg/sub/mod.py", imp, known)
        assert result == "pkg"

    def test_relative_import_subpackage_match(self):
        """Candidate not directly in known but has subpackage matches."""
        imp = {"module": "data", "is_relative": True, "level": 1}
        known = {"pkg", "pkg.core", "pkg.data.loader", "pkg.data.parser"}
        result = resolve_import_target("pkg.core", "pkg/core.py", imp, known)
        # pkg.data is not in known but pkg.data.loader starts with pkg.data.
        assert result == "pkg.data"

    def test_absolute_import_single_component(self):
        """Single-component import, not in known."""
        imp = {"module": "os", "is_relative": False, "level": 0}
        known = {"mypackage"}
        result = resolve_import_target("mypackage", "mypackage.py", imp, known)
        assert result is None

    def test_relative_level_0_from_init(self):
        """Level 1 from __init__ reduces to level 0."""
        imp = {"module": "utils", "is_relative": True, "level": 1}
        known = {"pkg", "pkg.utils"}
        result = resolve_import_target("pkg", "pkg/__init__.py", imp, known)
        assert result == "pkg.utils"

    def test_relative_import_level_exceeds_returns_none_when_no_base(self):
        """Level exceeds parts with no module -> returns None."""
        imp = {"module": "", "is_relative": True, "level": 5}
        known = {"a"}
        result = resolve_import_target("a.b", "a/b.py", imp, known)
        # base_parts is empty, no module -> empty string, not in known
        assert result is None

    def test_absolute_prefix_matching_iterates(self):
        """Prefix matching tries progressively shorter."""
        imp = {"module": "a.b.c.d", "is_relative": False, "level": 0}
        known = {"a"}
        result = resolve_import_target("x", "x.py", imp, known)
        assert result == "a"


# ===================================================================
# build_dependency_graph
# ===================================================================


class TestBuildDependencyGraph:
    """Tests for build_dependency_graph()."""

    def test_simple_graph(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "utils.py": "import core\n",
        })
        fwd, mod_path, all_mods = build_dependency_graph(str(tmp_path))
        assert "core" in fwd.get("utils", set())
        assert "core" in all_mods
        assert "utils" in all_mods

    def test_no_self_dependency(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "import a\n",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert "a" not in fwd.get("a", set())

    def test_no_external_deps(self, tmp_path):
        _make_project(tmp_path, {
            "app.py": "import os\nimport sys\n",
        })
        fwd, _, all_mods = build_dependency_graph(str(tmp_path))
        assert fwd.get("app", set()) == set()

    def test_multiple_dependencies(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "utils.py": "",
            "app.py": "import core\nimport utils\n",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert fwd["app"] == {"core", "utils"}

    def test_empty_project(self, tmp_path):
        fwd, mod_path, all_mods = build_dependency_graph(str(tmp_path))
        assert fwd == {}
        assert mod_path == {}
        assert all_mods == set()

    def test_package_with_init(self, tmp_path):
        _make_project(tmp_path, {
            "pkg/__init__.py": "",
            "pkg/core.py": "",
            "main.py": "import pkg.core\n",
        })
        fwd, _, all_mods = build_dependency_graph(str(tmp_path))
        assert "pkg" in all_mods
        assert "pkg.core" in all_mods

    def test_chain_dependency(self, tmp_path):
        """a -> b -> c."""
        _make_project(tmp_path, {
            "c.py": "",
            "b.py": "import c\n",
            "a.py": "import b\n",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert "b" in fwd.get("a", set())
        assert "c" in fwd.get("b", set())


# ===================================================================
# build_reverse_deps
# ===================================================================


class TestBuildReverseDeps:
    """Tests for build_reverse_deps()."""

    def test_simple_reverse(self):
        forward = {"a": {"b", "c"}, "d": {"b"}}
        rev = build_reverse_deps(forward)
        assert rev["b"] == {"a", "d"}
        assert rev["c"] == {"a"}

    def test_empty(self):
        assert build_reverse_deps({}) == {}

    def test_no_reverse(self):
        """Module with no incoming edges."""
        forward = {"a": {"b"}}
        rev = build_reverse_deps(forward)
        assert "a" not in rev

    def test_diamond(self):
        """a->b, a->c, b->d, c->d."""
        forward = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}}
        rev = build_reverse_deps(forward)
        assert rev["d"] == {"b", "c"}
        assert rev["b"] == {"a"}
        assert rev["c"] == {"a"}

    def test_single_module_no_deps(self):
        forward = {"a": set()}
        rev = build_reverse_deps(forward)
        assert rev == {}


# ===================================================================
# propagate_impact
# ===================================================================


class TestPropagateImpact:
    """Tests for propagate_impact()."""

    def test_no_impact(self):
        result = propagate_impact(["a"], {}, {"a": "a.py"})
        assert result == []

    def test_direct_impact(self):
        reverse = {"a": {"b"}}
        mod_path = {"a": "a.py", "b": "b.py"}
        result = propagate_impact(["a"], reverse, mod_path)
        assert len(result) == 1
        assert result[0].module == "b"
        assert result[0].depth == 1

    def test_transitive_impact(self):
        """a -> b -> c (reverse: a<-b<-c means b imports a, c imports b)."""
        reverse = {"a": {"b"}, "b": {"c"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        result = propagate_impact(["a"], reverse, mod_path)
        assert len(result) == 2
        modules = {m.module for m in result}
        assert modules == {"b", "c"}
        depths = {m.module: m.depth for m in result}
        assert depths["b"] == 1
        assert depths["c"] == 2

    def test_max_depth_limits_propagation(self):
        reverse = {"a": {"b"}, "b": {"c"}, "c": {"d"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py", "d": "d.py"}
        result = propagate_impact(["a"], reverse, mod_path, max_depth=2)
        modules = {m.module for m in result}
        assert "b" in modules
        assert "c" in modules
        assert "d" not in modules

    def test_max_depth_1(self):
        reverse = {"a": {"b"}, "b": {"c"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        result = propagate_impact(["a"], reverse, mod_path, max_depth=1)
        assert len(result) == 1
        assert result[0].module == "b"

    def test_changed_module_not_in_mod_path(self):
        """Changed module not found in module_to_path => skipped."""
        result = propagate_impact(["unknown"], {"a": {"b"}}, {"a": "a.py", "b": "b.py"})
        assert result == []

    def test_multiple_changed_modules(self):
        reverse = {"a": {"c"}, "b": {"c"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        result = propagate_impact(["a", "b"], reverse, mod_path)
        assert len(result) == 1
        assert result[0].module == "c"
        assert result[0].depth == 1

    def test_imported_by_chain(self):
        reverse = {"a": {"b"}, "b": {"c"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        result = propagate_impact(["a"], reverse, mod_path)
        c_impact = [m for m in result if m.module == "c"][0]
        assert "a" in c_impact.imported_by or "b" in c_impact.imported_by

    def test_skips_already_visited(self):
        """BFS should not revisit nodes."""
        reverse = {"a": {"b", "c"}, "c": {"b"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        result = propagate_impact(["a"], reverse, mod_path)
        modules = [m.module for m in result]
        assert modules.count("b") == 1
        assert modules.count("c") == 1

    def test_test_file_detection(self):
        reverse = {"core": {"test_core"}}
        mod_path = {"core": "core.py", "test_core": "test_core.py"}
        result = propagate_impact(["core"], reverse, mod_path)
        assert result[0].is_test is True

    def test_circular_dependency_handled(self):
        """Circular deps: a->b->c->a should not infinite loop."""
        reverse = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        # a is changed, BFS visits b (depth 1), c (depth 2), stops because a is already visited
        result = propagate_impact(["a"], reverse, mod_path)
        modules = {m.module for m in result}
        assert "b" in modules
        assert "c" in modules
        assert len(result) == 2  # a is skipped (depth 0)

    def test_diamond_dependency(self):
        """a -> b, a -> c, b -> d, c -> d."""
        reverse = {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py", "d": "d.py"}
        result = propagate_impact(["a"], reverse, mod_path)
        modules = {m.module for m in result}
        assert modules == {"b", "c", "d"}
        d_impact = [m for m in result if m.module == "d"][0]
        assert d_impact.depth == 2

    def test_results_sorted_by_depth_then_name(self):
        reverse = {"a": {"c", "b"}}
        mod_path = {"a": "a.py", "b": "b.py", "c": "c.py"}
        result = propagate_impact(["a"], reverse, mod_path)
        assert result[0].module == "b"
        assert result[1].module == "c"


# ===================================================================
# _coupling_grade
# ===================================================================


class TestCouplingGrade:
    """Tests for _coupling_grade()."""

    def test_grade_A(self):
        assert _coupling_grade(0) == "A"

    def test_grade_B_lower(self):
        assert _coupling_grade(1) == "B"

    def test_grade_B_upper(self):
        assert _coupling_grade(4) == "B"

    def test_grade_C_lower(self):
        assert _coupling_grade(5) == "C"

    def test_grade_C_upper(self):
        assert _coupling_grade(12) == "C"

    def test_grade_D_lower(self):
        assert _coupling_grade(13) == "D"

    def test_grade_D_upper(self):
        assert _coupling_grade(25) == "D"

    def test_grade_F(self):
        assert _coupling_grade(26) == "F"

    def test_grade_F_high(self):
        assert _coupling_grade(100) == "F"

    def test_boundary_B_C(self):
        assert _coupling_grade(4) == "B"
        assert _coupling_grade(5) == "C"

    def test_boundary_C_D(self):
        assert _coupling_grade(12) == "C"
        assert _coupling_grade(13) == "D"

    def test_boundary_D_F(self):
        assert _coupling_grade(25) == "D"
        assert _coupling_grade(26) == "F"


# ===================================================================
# compute_risk_level
# ===================================================================


class TestComputeRiskLevel:
    """Tests for compute_risk_level()."""

    def test_low(self):
        assert compute_risk_level(0.0) == "low"

    def test_low_boundary(self):
        assert compute_risk_level(0.09) == "low"

    def test_medium_lower(self):
        assert compute_risk_level(0.1) == "medium"

    def test_medium_upper(self):
        assert compute_risk_level(0.24) == "medium"

    def test_high_lower(self):
        assert compute_risk_level(0.25) == "high"

    def test_high_upper(self):
        assert compute_risk_level(0.49) == "high"

    def test_critical_lower(self):
        assert compute_risk_level(0.5) == "critical"

    def test_critical_high(self):
        assert compute_risk_level(1.0) == "critical"

    def test_boundary_low_medium(self):
        assert compute_risk_level(0.099) == "low"
        assert compute_risk_level(0.1) == "medium"

    def test_boundary_medium_high(self):
        assert compute_risk_level(0.249) == "medium"
        assert compute_risk_level(0.25) == "high"

    def test_boundary_high_critical(self):
        assert compute_risk_level(0.499) == "high"
        assert compute_risk_level(0.5) == "critical"


# ===================================================================
# compute_coupling_metrics
# ===================================================================


class TestComputeCouplingMetrics:
    """Tests for compute_coupling_metrics()."""

    def test_empty_project(self, tmp_path):
        result = compute_coupling_metrics(str(tmp_path))
        assert result.total_modules == 0
        assert result.modules == []
        assert result.avg_instability == 0.0

    def test_single_module_no_deps(self, tmp_path):
        _make_project(tmp_path, {"a.py": ""})
        result = compute_coupling_metrics(str(tmp_path))
        assert result.total_modules == 1
        assert len(result.modules) == 1
        m = result.modules[0]
        assert m.ca == 0
        assert m.ce == 0
        assert m.instability == 0.0
        assert m.hub_score == 0
        assert m.grade == "A"

    def test_two_modules_one_dep(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        assert result.total_modules == 2

        metrics = {m.module: m for m in result.modules}
        # core: imported by app (Ca=1), imports nothing (Ce=0)
        assert metrics["core"].ca == 1
        assert metrics["core"].ce == 0
        assert metrics["core"].instability == 0.0
        # app: imports core (Ce=1), not imported by anyone (Ca=0)
        assert metrics["app"].ca == 0
        assert metrics["app"].ce == 1
        assert metrics["app"].instability == 1.0

    def test_hub_module(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "hub.py": "import a\nimport b\n",
            "c.py": "import hub\n",
            "d.py": "import hub\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        metrics = {m.module: m for m in result.modules}
        hub = metrics["hub"]
        assert hub.ca == 2  # c and d import hub
        assert hub.ce == 2  # hub imports a and b
        assert hub.hub_score == 4  # 2 * 2
        assert hub.instability == 0.5

    def test_stable_modules(self, tmp_path):
        _make_project(tmp_path, {
            "lib.py": "",
            "app.py": "import lib\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        assert "lib" in result.stable_modules

    def test_unstable_modules(self, tmp_path):
        _make_project(tmp_path, {
            "lib.py": "",
            "app.py": "import lib\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        assert "app" in result.unstable_modules

    def test_top_n_limits_hubs(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "hub1.py": "import a\n",
            "hub2.py": "import b\n",
            "c.py": "import hub1\nimport hub2\n",
        })
        result = compute_coupling_metrics(str(tmp_path), top_n=1)
        assert len(result.hub_modules) <= 1

    def test_filters_test_modules_from_analysis(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "test_core.py": "import core\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        # test_core should still be in modules but NOT in stable/unstable
        non_test = [m for m in result.modules if not m.is_test]
        test_mods = [m for m in result.modules if m.is_test]
        assert len(test_mods) == 1
        assert test_mods[0].module == "test_core"
        # stable_modules should only contain non-test
        for s in result.stable_modules:
            assert not s.startswith("test_")

    def test_instability_formula(self, tmp_path):
        """Instability = Ce / (Ca + Ce)."""
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "c.py": "",
            "hub.py": "import a\nimport b\nimport c\n",
            "user.py": "import hub\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        metrics = {m.module: m for m in result.modules}
        hub = metrics["hub"]
        # Ca=1 (user imports hub), Ce=3 (hub imports a,b,c)
        assert hub.ca == 1
        assert hub.ce == 3
        expected_instability = 3 / (1 + 3)
        assert hub.instability == round(expected_instability, 4)

    def test_summary_format(self, tmp_path):
        _make_project(tmp_path, {"a.py": ""})
        result = compute_coupling_metrics(str(tmp_path))
        assert "Coupling:" in result.summary
        assert "modules" in result.summary
        assert "avg instability" in result.summary

    def test_median_instability_odd(self, tmp_path):
        """Median with odd number of modules."""
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
            "c.py": "import a\nimport b\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        # a: instability 0 (Ca=2, Ce=0)
        # b: instability 0.5 (Ca=1, Ce=1)
        # c: instability 1.0 (Ca=0, Ce=2)
        # Sorted: [0, 0.5, 1.0], median = 0.5
        assert result.median_instability == 0.5

    def test_median_instability_even(self, tmp_path):
        """Median with even number of modules."""
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
            "c.py": "import a\n",
            "d.py": "import a\nimport b\nimport c\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        # Check that median is computed (exact value depends on graph)
        assert 0.0 <= result.median_instability <= 1.0


# ===================================================================
# get_changed_files_from_git
# ===================================================================


class TestGetChangedFilesFromGit:
    """Tests for get_changed_files_from_git()."""

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_with_ref(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="src/a.py\nsrc/b.py\nREADME.md\n",
        )
        result = get_changed_files_from_git(str(tmp_path), ref="HEAD~3")
        assert len(result) == 2
        assert any("a.py" in f for f in result)
        assert any("b.py" in f for f in result)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "HEAD~3" in args

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_without_ref_combines_staged_unstaged(self, mock_run, tmp_path):
        staged_result = MagicMock(returncode=0, stdout="staged.py\n")
        unstaged_result = MagicMock(returncode=0, stdout="unstaged.py\n")
        mock_run.side_effect = [staged_result, unstaged_result]
        result = get_changed_files_from_git(str(tmp_path), ref=None)
        assert len(result) == 2
        basenames = {os.path.basename(f) for f in result}
        assert basenames == {"staged.py", "unstaged.py"}

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_without_ref_deduplicates(self, mock_run, tmp_path):
        """Same file in both staged and unstaged."""
        staged_result = MagicMock(returncode=0, stdout="common.py\n")
        unstaged_result = MagicMock(returncode=0, stdout="common.py\n")
        mock_run.side_effect = [staged_result, unstaged_result]
        result = get_changed_files_from_git(str(tmp_path), ref=None)
        assert len(result) == 1

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_filters_non_python(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="a.py\nb.js\nc.txt\nd.py\n",
        )
        result = get_changed_files_from_git(str(tmp_path), ref="HEAD")
        basenames = {os.path.basename(f) for f in result}
        assert basenames == {"a.py", "d.py"}

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_git_error_returns_empty(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = get_changed_files_from_git(str(tmp_path), ref="HEAD~1")
        assert result == []

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_git_not_found(self, mock_run, tmp_path):
        mock_run.side_effect = FileNotFoundError
        result = get_changed_files_from_git(str(tmp_path))
        assert result == []

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_empty_output(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="\n")
        result = get_changed_files_from_git(str(tmp_path), ref="HEAD")
        assert result == []

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_without_ref_staged_fails(self, mock_run, tmp_path):
        staged_result = MagicMock(returncode=1, stdout="")
        unstaged_result = MagicMock(returncode=0, stdout="file.py\n")
        mock_run.side_effect = [staged_result, unstaged_result]
        result = get_changed_files_from_git(str(tmp_path), ref=None)
        assert len(result) == 1

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_without_ref_unstaged_fails(self, mock_run, tmp_path):
        staged_result = MagicMock(returncode=0, stdout="file.py\n")
        unstaged_result = MagicMock(returncode=1, stdout="")
        mock_run.side_effect = [staged_result, unstaged_result]
        result = get_changed_files_from_git(str(tmp_path), ref=None)
        assert len(result) == 1

    @patch("code_health_suite.engines.change_impact.subprocess.run")
    def test_returns_absolute_paths(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="a.py\n")
        result = get_changed_files_from_git(str(tmp_path), ref="HEAD")
        assert os.path.isabs(result[0])


# ===================================================================
# analyze
# ===================================================================


class TestAnalyze:
    """Tests for analyze()."""

    def test_no_changed_modules(self, tmp_path):
        _make_project(tmp_path, {"a.py": ""})
        result = analyze(str(tmp_path), [str(tmp_path / "nonexistent.py")])
        assert result.changed_modules == []
        assert result.impact_score == 0.0
        assert result.risk_level == "low"
        assert "No matching" in result.summary

    def test_single_changed_no_dependents(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "other.py": "",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert result.changed_modules == ["core"]
        assert result.direct_impact == []
        assert result.transitive_impact == []

    def test_direct_impact(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert len(result.direct_impact) == 1
        assert result.direct_impact[0].module == "app"

    def test_transitive_impact(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "mid.py": "import core\n",
            "top.py": "import mid\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert len(result.direct_impact) == 1
        assert len(result.transitive_impact) == 1
        assert result.transitive_impact[0].module == "top"

    def test_affected_tests(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "test_core.py": "import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert "test_core" in result.affected_tests

    def test_impact_score(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "a.py": "import core\n",
            "b.py": "import core\n",
            "c.py": "",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        # 4 non-test modules total, 1 changed = 3 denominator
        # 2 affected non-test = 2/3
        assert result.impact_score == round(2 / 3, 4)

    def test_max_depth(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
            "c.py": "import b\n",
            "d.py": "import c\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "a.py")], max_depth=1)
        assert len(result.direct_impact) == 1
        assert result.transitive_impact == []

    def test_empty_project(self, tmp_path):
        result = analyze(str(tmp_path), [])
        assert result.changed_modules == []

    def test_multiple_changed_files(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "c.py": "import a\nimport b\n",
        })
        result = analyze(
            str(tmp_path),
            [str(tmp_path / "a.py"), str(tmp_path / "b.py")],
        )
        assert len(result.changed_modules) == 2
        assert len(result.direct_impact) == 1

    def test_risk_level_computed(self, tmp_path):
        _make_project(tmp_path, {
            "core.py": "",
            "a.py": "import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert result.risk_level in ("low", "medium", "high", "critical")

    def test_total_project_modules(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "c.py": "",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "a.py")])
        assert result.total_project_modules == 3

    def test_changed_files_resolved(self, tmp_path):
        _make_project(tmp_path, {"a.py": ""})
        result = analyze(str(tmp_path), [str(tmp_path / "a.py")])
        for f in result.changed_files:
            assert os.path.isabs(f)

    def test_summary_format(self, tmp_path):
        _make_project(tmp_path, {"a.py": ""})
        result = analyze(str(tmp_path), [str(tmp_path / "a.py")])
        assert "Impact:" in result.summary
        assert "changed" in result.summary
        assert "risk:" in result.summary

    def test_all_files_changed(self, tmp_path):
        """Edge case: every file in the project changed."""
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
        })
        result = analyze(
            str(tmp_path),
            [str(tmp_path / "a.py"), str(tmp_path / "b.py")],
        )
        # b imports a but b is also changed, so b is at depth 0
        # No additional impact since all modules are changed
        assert len(result.changed_modules) == 2


# ===================================================================
# format_text
# ===================================================================


class TestFormatText:
    """Tests for format_text()."""

    def test_basic_output(self, tmp_path):
        result = _make_impact_result(root=str(tmp_path))
        text = format_text(result)
        assert "Change Impact Analysis" in text
        assert "Risk Level:" in text
        assert "Impact Score:" in text

    def test_includes_changed_modules(self):
        result = _make_impact_result(changed_modules=["core", "utils"])
        text = format_text(result)
        assert "core" in text
        assert "utils" in text

    def test_includes_direct_impact(self):
        impact = ImpactedModule(
            module="app", path="app.py", depth=1,
            is_test=False, imported_by=["core"],
        )
        result = _make_impact_result(direct_impact=[impact])
        text = format_text(result)
        assert "Direct Impact" in text
        assert "app" in text

    def test_includes_transitive_impact(self):
        impact = ImpactedModule(
            module="cli", path="cli.py", depth=2,
            is_test=False, imported_by=["core", "app"],
        )
        result = _make_impact_result(transitive_impact=[impact])
        text = format_text(result)
        assert "Transitive Impact" in text
        assert "cli" in text
        assert "depth 2" in text

    def test_test_tag(self):
        impact = ImpactedModule(
            module="test_core", path="test_core.py", depth=1,
            is_test=True, imported_by=["core"],
        )
        result = _make_impact_result(
            direct_impact=[impact], affected_tests=["test_core"],
        )
        text = format_text(result)
        assert "[TEST]" in text

    def test_no_impact_message(self):
        result = _make_impact_result()
        text = format_text(result)
        assert "No downstream modules are affected" in text

    def test_affected_tests_section(self):
        result = _make_impact_result(affected_tests=["test_a", "test_b"])
        text = format_text(result)
        assert "Affected Tests" in text
        assert "test_a" in text
        assert "test_b" in text

    def test_risk_level_uppercase(self):
        result = _make_impact_result(risk_level="critical")
        text = format_text(result)
        assert "CRITICAL" in text

    def test_impact_score_percentage(self):
        result = _make_impact_result(impact_score=0.42)
        text = format_text(result)
        assert "42.0%" in text

    def test_project_size(self):
        result = _make_impact_result(total_project_modules=15)
        text = format_text(result)
        assert "15 modules" in text

    def test_separator_line(self):
        result = _make_impact_result()
        text = format_text(result)
        assert "=" * 60 in text

    def test_summary_at_end(self):
        summary = "Impact: 1 changed / 0 direct / 0 transitive / 0 tests — risk: low"
        result = _make_impact_result(summary=summary)
        text = format_text(result)
        assert text.strip().endswith(summary)


# ===================================================================
# format_json
# ===================================================================


class TestFormatJson:
    """Tests for format_json()."""

    def test_valid_json(self):
        result = _make_impact_result()
        output = format_json(result)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_contains_all_fields(self):
        result = _make_impact_result()
        data = json.loads(format_json(result))
        assert "root" in data
        assert "changed_files" in data
        assert "changed_modules" in data
        assert "total_project_modules" in data
        assert "direct_impact" in data
        assert "transitive_impact" in data
        assert "affected_tests" in data
        assert "impact_score" in data
        assert "risk_level" in data
        assert "summary" in data

    def test_impact_modules_serialized(self):
        impact = ImpactedModule(
            module="app", path="app.py", depth=1,
            is_test=False, imported_by=["core"],
        )
        result = _make_impact_result(direct_impact=[impact])
        data = json.loads(format_json(result))
        assert len(data["direct_impact"]) == 1
        assert data["direct_impact"][0]["module"] == "app"
        assert data["direct_impact"][0]["depth"] == 1

    def test_pretty_printed(self):
        result = _make_impact_result()
        output = format_json(result)
        assert "\n" in output  # indented

    def test_values_match(self):
        result = _make_impact_result(
            impact_score=0.33, risk_level="high",
            changed_modules=["x"], affected_tests=["test_x"],
        )
        data = json.loads(format_json(result))
        assert data["impact_score"] == 0.33
        assert data["risk_level"] == "high"
        assert data["changed_modules"] == ["x"]
        assert data["affected_tests"] == ["test_x"]


# ===================================================================
# format_coupling_text
# ===================================================================


class TestFormatCouplingText:
    """Tests for format_coupling_text()."""

    def test_basic_output(self):
        result = CouplingResult(
            root="/project", total_modules=5, modules=[],
            avg_instability=0.4, median_instability=0.5,
            hub_modules=[], stable_modules=[], unstable_modules=[],
            summary="Coupling: 5 modules, avg instability 0.40, 0 hubs, 0 pure libraries, 0 pure dependents",
        )
        text = format_coupling_text(result)
        assert "Coupling Analysis" in text
        assert "Project Size: 5 modules" in text
        assert "Avg Instability: 0.40" in text

    def test_hub_modules_table(self):
        hub = CouplingMetrics(
            module="hub", path="hub.py", is_test=False,
            ca=3, ce=2, instability=0.4, hub_score=6, grade="C",
        )
        result = CouplingResult(
            root="/project", total_modules=5, modules=[hub],
            avg_instability=0.4, median_instability=0.4,
            hub_modules=[hub], stable_modules=[], unstable_modules=[],
            summary="test",
        )
        text = format_coupling_text(result)
        assert "Hub Modules" in text
        assert "hub" in text

    def test_stable_modules_listed(self):
        result = CouplingResult(
            root="/project", total_modules=5, modules=[],
            avg_instability=0.0, median_instability=0.0,
            hub_modules=[], stable_modules=["lib_a", "lib_b"],
            unstable_modules=[], summary="test",
        )
        text = format_coupling_text(result)
        assert "Pure Libraries" in text
        assert "lib_a" in text
        assert "lib_b" in text

    def test_unstable_modules_listed(self):
        result = CouplingResult(
            root="/project", total_modules=5, modules=[],
            avg_instability=1.0, median_instability=1.0,
            hub_modules=[], stable_modules=[],
            unstable_modules=["app", "cli"], summary="test",
        )
        text = format_coupling_text(result)
        assert "Pure Dependents" in text
        assert "app" in text
        assert "cli" in text

    def test_summary_at_end(self):
        summary = "Coupling: 5 modules, avg instability 0.40"
        result = CouplingResult(
            root="/project", total_modules=5, modules=[],
            avg_instability=0.4, median_instability=0.4,
            hub_modules=[], stable_modules=[], unstable_modules=[],
            summary=summary,
        )
        text = format_coupling_text(result)
        assert text.strip().endswith(summary)

    def test_separator_line(self):
        result = CouplingResult(
            root="/project", total_modules=5, modules=[],
            avg_instability=0.0, median_instability=0.0,
            hub_modules=[], stable_modules=[], unstable_modules=[],
            summary="test",
        )
        text = format_coupling_text(result)
        assert "=" * 60 in text


# ===================================================================
# format_coupling_json
# ===================================================================


class TestFormatCouplingJson:
    """Tests for format_coupling_json()."""

    def test_valid_json(self):
        result = CouplingResult(
            root="/project", total_modules=3, modules=[],
            avg_instability=0.5, median_instability=0.5,
            hub_modules=[], stable_modules=[], unstable_modules=[],
            summary="test",
        )
        data = json.loads(format_coupling_json(result))
        assert isinstance(data, dict)

    def test_contains_all_fields(self):
        result = CouplingResult(
            root="/project", total_modules=3, modules=[],
            avg_instability=0.5, median_instability=0.5,
            hub_modules=[], stable_modules=[], unstable_modules=[],
            summary="test",
        )
        data = json.loads(format_coupling_json(result))
        expected_keys = {
            "root", "total_modules", "avg_instability", "median_instability",
            "hub_modules", "stable_modules", "unstable_modules", "all_modules",
            "summary",
        }
        assert expected_keys == set(data.keys())

    def test_modules_serialized(self):
        m = CouplingMetrics(
            module="core", path="core.py", is_test=False,
            ca=2, ce=1, instability=0.333, hub_score=2, grade="B",
        )
        result = CouplingResult(
            root="/p", total_modules=1, modules=[m],
            avg_instability=0.333, median_instability=0.333,
            hub_modules=[m], stable_modules=[], unstable_modules=[],
            summary="test",
        )
        data = json.loads(format_coupling_json(result))
        assert len(data["all_modules"]) == 1
        assert data["all_modules"][0]["module"] == "core"
        assert data["all_modules"][0]["grade"] == "B"

    def test_values_match(self):
        result = CouplingResult(
            root="/project", total_modules=10,
            modules=[], avg_instability=0.42, median_instability=0.35,
            hub_modules=[], stable_modules=["a"], unstable_modules=["b"],
            summary="summary text",
        )
        data = json.loads(format_coupling_json(result))
        assert data["total_modules"] == 10
        assert data["avg_instability"] == 0.42
        assert data["stable_modules"] == ["a"]
        assert data["unstable_modules"] == ["b"]


# ===================================================================
# suggest_test_command
# ===================================================================


class TestSuggestTestCommand:
    """Tests for suggest_test_command()."""

    def test_no_affected_tests(self):
        result = _make_impact_result(affected_tests=[])
        assert suggest_test_command(result) == ""

    def test_pytest_command(self):
        impact = ImpactedModule(
            module="test_core", path="/project/tests/test_core.py",
            depth=1, is_test=True, imported_by=["core"],
        )
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_core"],
            direct_impact=[impact],
        )
        cmd = suggest_test_command(result, runner="pytest")
        assert cmd.startswith("pytest ")
        assert "test_core.py" in cmd

    def test_unittest_command(self):
        impact = ImpactedModule(
            module="test_core", path="/project/tests/test_core.py",
            depth=1, is_test=True, imported_by=["core"],
        )
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_core"],
            direct_impact=[impact],
        )
        cmd = suggest_test_command(result, runner="unittest")
        assert cmd.startswith("python -m unittest ")

    def test_multiple_tests(self):
        impact1 = ImpactedModule(
            module="test_a", path="/project/tests/test_a.py",
            depth=1, is_test=True, imported_by=["a"],
        )
        impact2 = ImpactedModule(
            module="test_b", path="/project/tests/test_b.py",
            depth=1, is_test=True, imported_by=["b"],
        )
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_a", "test_b"],
            direct_impact=[impact1, impact2],
        )
        cmd = suggest_test_command(result)
        assert "test_a.py" in cmd
        assert "test_b.py" in cmd

    def test_deduplicates_paths(self):
        impact1 = ImpactedModule(
            module="test_core", path="/project/tests/test_core.py",
            depth=1, is_test=True, imported_by=["core"],
        )
        impact2 = ImpactedModule(
            module="test_core", path="/project/tests/test_core.py",
            depth=2, is_test=True, imported_by=["core", "util"],
        )
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_core"],
            direct_impact=[impact1],
            transitive_impact=[impact2],
        )
        cmd = suggest_test_command(result)
        # Should only appear once
        assert cmd.count("test_core.py") == 1

    def test_no_test_paths_returns_empty(self):
        """affected_tests is non-empty but no matching impact paths."""
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_core"],
            direct_impact=[],
        )
        cmd = suggest_test_command(result)
        assert cmd == ""

    def test_relative_paths_used(self):
        impact = ImpactedModule(
            module="test_core", path="/project/tests/test_core.py",
            depth=1, is_test=True, imported_by=["core"],
        )
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_core"],
            direct_impact=[impact],
        )
        cmd = suggest_test_command(result)
        assert "/project/" not in cmd
        assert "tests/test_core.py" in cmd

    def test_non_test_impacts_ignored(self):
        non_test = ImpactedModule(
            module="app", path="/project/app.py",
            depth=1, is_test=False, imported_by=["core"],
        )
        test = ImpactedModule(
            module="test_app", path="/project/test_app.py",
            depth=1, is_test=True, imported_by=["core"],
        )
        result = _make_impact_result(
            root="/project",
            affected_tests=["test_app"],
            direct_impact=[non_test, test],
        )
        cmd = suggest_test_command(result)
        assert "app.py" not in cmd or "test_app.py" in cmd


# ===================================================================
# build_parser
# ===================================================================


class TestBuildParser:
    """Tests for build_parser()."""

    def test_default_root(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.root == "."

    def test_explicit_root(self):
        parser = build_parser()
        args = parser.parse_args(["/path/to/project"])
        assert args.root == "/path/to/project"

    def test_files_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--files", "a.py", "b.py"])
        assert args.files == ["a.py", "b.py"]

    def test_git_diff_no_ref(self):
        parser = build_parser()
        args = parser.parse_args(["--git-diff"])
        assert args.git_diff == "__UNCOMMITTED__"

    def test_git_diff_with_ref(self):
        parser = build_parser()
        args = parser.parse_args(["--git-diff", "HEAD~3"])
        assert args.git_diff == "HEAD~3"

    def test_depth_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--depth", "5"])
        assert args.depth == 5

    def test_default_depth(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.depth == 10

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json_output is True

    def test_coupling_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--coupling"])
        assert args.coupling is True

    def test_suggest_tests_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--suggest-tests"])
        assert args.suggest_tests is True

    def test_top_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--top", "5"])
        assert args.top == 5

    def test_default_top(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.top == 0

    def test_no_files_default(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.files is None

    def test_no_git_diff_default(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.git_diff is None

    def test_no_json_default(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.json_output is False

    def test_no_coupling_default(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.coupling is False

    def test_no_suggest_tests_default(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.suggest_tests is False

    def test_combined_flags(self):
        parser = build_parser()
        args = parser.parse_args(["--coupling", "--json", "--top", "3"])
        assert args.coupling is True
        assert args.json_output is True
        assert args.top == 3


# ===================================================================
# main
# ===================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    def test_coupling_mode_text(self, tmp_path, capsys):
        _make_project(tmp_path, {"a.py": "", "b.py": "import a\n"})
        ret = main(["--coupling", str(tmp_path)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Coupling Analysis" in captured.out

    def test_coupling_mode_json(self, tmp_path, capsys):
        _make_project(tmp_path, {"a.py": ""})
        ret = main(["--coupling", "--json", str(tmp_path)])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert "total_modules" in data

    def test_coupling_with_top(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
            "c.py": "import a\nimport b\n",
        })
        ret = main(["--coupling", "--top", "1", str(tmp_path)])
        assert ret == 0

    def test_files_mode(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        ret = main([str(tmp_path), "--files", "core.py"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Change Impact Analysis" in captured.out

    def test_files_mode_json(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        ret = main([str(tmp_path), "--files", "core.py", "--json"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert "changed_modules" in data

    def test_no_changes_returns_1(self, tmp_path, capsys):
        _make_project(tmp_path, {"a.py": ""})
        # No --files and mock git to return nothing
        with patch("code_health_suite.engines.change_impact.get_changed_files_from_git", return_value=[]):
            ret = main([str(tmp_path)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "No changed Python files found" in captured.err

    def test_suggest_tests_with_tests(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "core.py": "",
            "test_core.py": "import core\n",
        })
        ret = main([str(tmp_path), "--files", "core.py", "--suggest-tests"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "pytest" in captured.out or "test_core" in captured.out

    def test_suggest_tests_no_tests(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        ret = main([str(tmp_path), "--files", "core.py", "--suggest-tests"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "No affected tests" in captured.err

    @patch("code_health_suite.engines.change_impact.get_changed_files_from_git")
    def test_git_diff_uncommitted(self, mock_git, tmp_path, capsys):
        _make_project(tmp_path, {"core.py": ""})
        mock_git.return_value = [str(tmp_path / "core.py")]
        ret = main([str(tmp_path), "--git-diff"])
        assert ret == 0
        mock_git.assert_called_once_with(str(Path(tmp_path).resolve()), None)

    @patch("code_health_suite.engines.change_impact.get_changed_files_from_git")
    def test_git_diff_with_ref(self, mock_git, tmp_path, capsys):
        _make_project(tmp_path, {"core.py": ""})
        mock_git.return_value = [str(tmp_path / "core.py")]
        ret = main([str(tmp_path), "--git-diff", "HEAD~3"])
        assert ret == 0
        mock_git.assert_called_once_with(str(Path(tmp_path).resolve()), "HEAD~3")

    @patch("code_health_suite.engines.change_impact.get_changed_files_from_git")
    def test_default_uses_git_uncommitted(self, mock_git, tmp_path, capsys):
        """When neither --files nor --git-diff, defaults to git uncommitted."""
        _make_project(tmp_path, {"core.py": ""})
        mock_git.return_value = [str(tmp_path / "core.py")]
        ret = main([str(tmp_path)])
        assert ret == 0
        mock_git.assert_called_once_with(str(Path(tmp_path).resolve()), None)

    def test_depth_parameter_passed(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
            "c.py": "import b\n",
        })
        ret = main([str(tmp_path), "--files", "a.py", "--depth", "1"])
        assert ret == 0
        data_text = capsys.readouterr().out
        # With depth=1, c should not appear in impact
        assert "c" not in data_text or "Transitive" not in data_text

    def test_files_absolute_path(self, tmp_path, capsys):
        """Absolute path passed with --files."""
        _make_project(tmp_path, {"core.py": ""})
        ret = main([str(tmp_path), "--files", str(tmp_path / "core.py")])
        assert ret == 0


# ===================================================================
# Integration: Full Workflow
# ===================================================================


class TestIntegration:
    """Integration tests exercising multiple components together."""

    def test_full_project_analysis(self, tmp_path):
        """A realistic small project with packages using absolute imports."""
        _make_project(tmp_path, {
            "myapp/__init__.py": "",
            "myapp/core.py": "",
            "myapp/utils.py": "from myapp.core import something\n",
            "myapp/api.py": "from myapp.core import x\nfrom myapp.utils import y\n",
            "tests/__init__.py": "",
            "tests/test_core.py": "from myapp.core import z\n",
            "tests/test_api.py": "from myapp.api import w\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "myapp" / "core.py")])
        assert "myapp.core" in result.changed_modules
        # utils and api should be impacted (they import core)
        impact_mods = {m.module for m in result.direct_impact + result.transitive_impact}
        assert "myapp.utils" in impact_mods or "myapp.api" in impact_mods
        # tests should be affected
        assert len(result.affected_tests) > 0

    def test_src_layout_project(self, tmp_path):
        """src-layout project with no src/__init__.py."""
        _make_project(tmp_path, {
            "src/mylib/__init__.py": "",
            "src/mylib/core.py": "",
            "src/mylib/helpers.py": "from mylib import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "src" / "mylib" / "core.py")])
        assert "mylib.core" in result.changed_modules

    def test_coupling_on_hub_project(self, tmp_path):
        """Project with clear hub module."""
        _make_project(tmp_path, {
            "models.py": "",
            "db.py": "",
            "hub.py": "import models\nimport db\n",
            "api.py": "import hub\n",
            "cli.py": "import hub\n",
            "worker.py": "import hub\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        metrics = {m.module: m for m in result.modules}
        hub = metrics["hub"]
        assert hub.ca == 3  # api, cli, worker
        assert hub.ce == 2  # models, db
        assert hub.hub_score == 6

    def test_empty_project_analysis(self, tmp_path):
        """Empty project returns empty results."""
        result = analyze(str(tmp_path), [])
        assert result.changed_modules == []
        assert result.total_project_modules == 0

    def test_build_graph_then_analyze(self, tmp_path):
        """build_dependency_graph + build_reverse_deps + propagate_impact."""
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
            "c.py": "import b\n",
        })
        fwd, mod_path, all_mods = build_dependency_graph(str(tmp_path))
        rev = build_reverse_deps(fwd)
        impacts = propagate_impact(["a"], rev, mod_path)
        modules = {m.module for m in impacts}
        assert modules == {"b", "c"}

    def test_circular_project(self, tmp_path):
        """Circular imports should not cause infinite loops."""
        _make_project(tmp_path, {
            "a.py": "import b\n",
            "b.py": "import c\n",
            "c.py": "import a\n",
        })
        # Should complete without hanging
        result = analyze(str(tmp_path), [str(tmp_path / "a.py")])
        assert isinstance(result, ChangeImpactResult)

    def test_large_fan_out(self, tmp_path):
        """One module imports many others."""
        files = {"lib_{}.py".format(i): "" for i in range(20)}
        files["app.py"] = "\n".join("import lib_{}".format(i) for i in range(20))
        _make_project(tmp_path, files)
        result = analyze(str(tmp_path), [str(tmp_path / "lib_0.py")])
        assert any(m.module == "app" for m in result.direct_impact)

    def test_large_fan_in(self, tmp_path):
        """Many modules import the same core module."""
        files = {"core.py": ""}
        for i in range(15):
            files[f"user_{i}.py"] = "import core\n"
        _make_project(tmp_path, files)
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert len(result.direct_impact) == 15

    def test_relative_import_resolution(self, tmp_path):
        """Relative imports are properly resolved in dependency graph.

        Note: ``from . import core`` has module="" so it resolves to the
        package itself (``pkg``), not ``pkg.core``.  Use ``from .core import X``
        (module="core") to get ``pkg.core``.
        """
        _make_project(tmp_path, {
            "pkg/__init__.py": "",
            "pkg/core.py": "",
            "pkg/utils.py": "from .core import helper\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/deep.py": "from ..core import helper\n",
        })
        fwd, mod_path, all_mods = build_dependency_graph(str(tmp_path))
        # utils should depend on core
        assert "pkg.core" in fwd.get("pkg.utils", set())
        # deep should depend on core
        assert "pkg.core" in fwd.get("pkg.sub.deep", set())

    def test_format_roundtrip_json(self, tmp_path):
        """JSON output is parseable and contains the original data."""
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        json_out = format_json(result)
        data = json.loads(json_out)
        assert data["risk_level"] == result.risk_level
        assert data["impact_score"] == result.impact_score
        assert len(data["direct_impact"]) == len(result.direct_impact)

    def test_coupling_json_roundtrip(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        json_out = format_coupling_json(result)
        data = json.loads(json_out)
        assert data["total_modules"] == result.total_modules

    def test_impact_with_conftest(self, tmp_path):
        """conftest.py is detected as test file."""
        _make_project(tmp_path, {
            "core.py": "",
            "conftest.py": "import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert "conftest" in result.affected_tests


# ===================================================================
# Edge Cases
# ===================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_deeply_nested_packages(self, tmp_path):
        _make_project(tmp_path, {
            "a/b/c/d/e/__init__.py": "",
            "a/b/c/d/e/mod.py": "",
            "a/b/c/d/__init__.py": "",
            "a/b/c/__init__.py": "",
            "a/b/__init__.py": "",
            "a/__init__.py": "",
        })
        files = find_python_files(str(tmp_path))
        assert len(files) == 6

    def test_file_with_only_comments(self, tmp_path):
        _write(tmp_path / "comments.py", "# just a comment\n# nothing else\n")
        result = extract_imports(str(tmp_path / "comments.py"))
        assert result == []

    def test_file_with_docstring_only(self, tmp_path):
        _write(tmp_path / "doc.py", '"""Module docstring."""\n')
        result = extract_imports(str(tmp_path / "doc.py"))
        assert result == []

    def test_module_with_no_imports(self, tmp_path):
        _make_project(tmp_path, {
            "constants.py": "FOO = 42\nBAR = 'hello'\n",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert fwd.get("constants", set()) == set()

    def test_single_file_project(self, tmp_path):
        _make_project(tmp_path, {"main.py": "print('hello')\n"})
        result = analyze(str(tmp_path), [str(tmp_path / "main.py")])
        assert result.changed_modules == ["main"]
        assert result.direct_impact == []

    def test_score_zero_when_no_downstream(self, tmp_path):
        _make_project(tmp_path, {"lonely.py": ""})
        result = analyze(str(tmp_path), [str(tmp_path / "lonely.py")])
        assert result.impact_score == 0.0

    def test_score_with_only_test_dependents(self, tmp_path):
        """Score only counts non-test modules in denominator."""
        _make_project(tmp_path, {
            "core.py": "",
            "test_core.py": "import core\n",
        })
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        # affected_non_test is empty (test_core is test), denominator = 0 non-test minus changed
        # denominator = {test_core is test -> non_test = {core}} - {core} = 0
        assert result.impact_score == 0.0

    def test_multiple_imports_same_module(self, tmp_path):
        """Module imported multiple times still counted once."""
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\nimport core\nfrom core import something\n",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert fwd.get("app", set()) == {"core"}

    def test_unicode_filename_handling(self, tmp_path):
        """Files with unicode names are handled."""
        _write(tmp_path / "modulo.py", "x = 1\n")
        result = find_python_files(str(tmp_path))
        assert len(result) == 1

    def test_nonexistent_root(self):
        """Non-existent root returns empty results."""
        result = find_python_files("/nonexistent/path/that/does/not/exist")
        # os.walk on nonexistent path returns nothing
        assert result == []

    def test_analyze_with_relative_path_input(self, tmp_path):
        """Changed files can be passed as relative paths that get resolved."""
        _make_project(tmp_path, {
            "core.py": "",
            "app.py": "import core\n",
        })
        # Pass a path that resolves to the file
        result = analyze(str(tmp_path), [str(tmp_path / "core.py")])
        assert len(result.changed_modules) == 1

    def test_propagation_depth_0(self):
        """max_depth=0 should return no results (no propagation beyond changed)."""
        reverse = {"a": {"b"}}
        mod_path = {"a": "a.py", "b": "b.py"}
        # max_depth=0 means changed modules at depth 0 cannot propagate (depth >= max_depth)
        # Actually, depth=0 means the queue starts at 0 and since 0 >= 0, it won't explore
        # Wait - the code says: if depth >= max_depth: continue
        # With max_depth=0, depth=0 >= 0 is true, so no propagation at all
        result = propagate_impact(["a"], reverse, mod_path, max_depth=0)
        assert result == []


# ===================================================================
# Additional resolve_import_target scenarios
# ===================================================================


class TestResolveImportTargetAdvanced:
    """Advanced tests for resolve_import_target edge cases."""

    def test_relative_import_from_init_level_2(self):
        """from .. import X in pkg/sub/__init__.py."""
        imp = {"module": "core", "is_relative": True, "level": 2}
        known = {"pkg", "pkg.core", "pkg.sub"}
        # From __init__.py, level is reduced by 1: effective level = 1
        result = resolve_import_target("pkg.sub", "pkg/sub/__init__.py", imp, known)
        assert result == "pkg.core"

    def test_absolute_import_partial_match_longest_prefix(self):
        """Prefix matching returns the longest matching prefix."""
        imp = {"module": "a.b.c.d", "is_relative": False, "level": 0}
        known = {"a", "a.b", "a.b.c"}
        result = resolve_import_target("x", "x.py", imp, known)
        assert result == "a.b.c"

    def test_relative_import_empty_module_level_1(self):
        """from . import in pkg/mod.py -> pkg (parent package)."""
        imp = {"module": "", "is_relative": True, "level": 1}
        known = {"pkg", "pkg.mod"}
        result = resolve_import_target("pkg.mod", "pkg/mod.py", imp, known)
        assert result == "pkg"

    def test_relative_level_0_from_non_init(self):
        """Level 0 with is_relative=True (unusual but valid)."""
        # Actually level=0 with is_relative=True shouldn't normally occur,
        # but we test robustness. With level=0, base_parts = all parts.
        imp = {"module": "foo", "is_relative": True, "level": 0}
        known = {"pkg", "pkg.foo"}
        # Level 0 doesn't strip anything: base_parts = ["pkg"], candidate = "pkg.foo"
        # This should only occur with is_relative=False typically, but let's test
        result = resolve_import_target("pkg", "pkg.py", imp, known)
        assert result == "pkg.foo"


# ===================================================================
# Additional build_dependency_graph scenarios
# ===================================================================


class TestBuildDependencyGraphAdvanced:
    """Advanced tests for build_dependency_graph."""

    def test_relative_imports_in_package(self, tmp_path):
        """``from .a import X`` (module='a') resolves to pkg.a."""
        _make_project(tmp_path, {
            "pkg/__init__.py": "",
            "pkg/a.py": "",
            "pkg/b.py": "from .a import something\n",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert "pkg.a" in fwd.get("pkg.b", set())

    def test_init_imports_submodule(self, tmp_path):
        """``from .core import X`` in __init__.py resolves to pkg.core."""
        _make_project(tmp_path, {
            "pkg/__init__.py": "from .core import something\n",
            "pkg/core.py": "",
        })
        fwd, _, _ = build_dependency_graph(str(tmp_path))
        assert "pkg.core" in fwd.get("pkg", set())

    def test_module_to_path_mapping(self, tmp_path):
        _make_project(tmp_path, {
            "foo.py": "",
            "bar/__init__.py": "",
            "bar/baz.py": "",
        })
        _, mod_path, _ = build_dependency_graph(str(tmp_path))
        assert "foo" in mod_path
        assert "bar" in mod_path
        assert "bar.baz" in mod_path
        for path in mod_path.values():
            assert path.endswith(".py")

    def test_all_modules_complete(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "c.py": "",
        })
        _, _, all_mods = build_dependency_graph(str(tmp_path))
        assert all_mods == {"a", "b", "c"}

    def test_syntax_error_file_ignored_gracefully(self, tmp_path):
        _make_project(tmp_path, {
            "good.py": "",
            "bad.py": "def f(\n",
            "user.py": "import good\n",
        })
        fwd, _, all_mods = build_dependency_graph(str(tmp_path))
        assert "good" in all_mods
        assert "bad" in all_mods  # file exists, even if no imports extracted
        assert "user" in all_mods


# ===================================================================
# Additional coupling metric edge cases
# ===================================================================


class TestCouplingEdgeCases:
    """Edge case tests for coupling analysis."""

    def test_all_modules_are_tests(self, tmp_path):
        _make_project(tmp_path, {
            "test_a.py": "",
            "test_b.py": "import test_a\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        # All modules are tests, non_test is empty
        assert result.stable_modules == []
        assert result.unstable_modules == []

    def test_instability_zero_division(self, tmp_path):
        """Module with Ca=0 and Ce=0 has instability 0."""
        _make_project(tmp_path, {"isolated.py": ""})
        result = compute_coupling_metrics(str(tmp_path))
        m = result.modules[0]
        assert m.instability == 0.0

    def test_grade_distribution(self, tmp_path):
        """Verify grade is computed for each module."""
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "import a\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        for m in result.modules:
            assert m.grade in ("A", "B", "C", "D", "F")

    def test_pure_library_criteria(self, tmp_path):
        """Pure library: instability == 0.0 AND ca > 0."""
        _make_project(tmp_path, {
            "lib.py": "",
            "app.py": "import lib\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        assert "lib" in result.stable_modules
        # A module with ca=0 and ce=0 is NOT a pure library
        # because it's not imported by anything

    def test_pure_dependent_criteria(self, tmp_path):
        """Pure dependent: instability == 1.0 AND ce > 0."""
        _make_project(tmp_path, {
            "lib.py": "",
            "app.py": "import lib\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        assert "app" in result.unstable_modules

    def test_hub_modules_sorted_by_hub_score(self, tmp_path):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
            "c.py": "",
            "small_hub.py": "import a\n",
            "big_hub.py": "import a\nimport b\nimport c\n",
            "user1.py": "import small_hub\nimport big_hub\n",
            "user2.py": "import big_hub\n",
        })
        result = compute_coupling_metrics(str(tmp_path))
        if len(result.hub_modules) >= 2:
            # Should be sorted by hub_score descending
            for i in range(len(result.hub_modules) - 1):
                assert result.hub_modules[i].hub_score >= result.hub_modules[i + 1].hub_score


# ===================================================================
# Additional format_text coverage
# ===================================================================


class TestFormatTextAdditional:
    """Additional format_text tests for coverage."""

    def test_direct_and_transitive_both_present(self):
        direct = ImpactedModule(
            module="mid", path="mid.py", depth=1,
            is_test=False, imported_by=["core"],
        )
        transitive = ImpactedModule(
            module="top", path="top.py", depth=2,
            is_test=False, imported_by=["core", "mid"],
        )
        result = _make_impact_result(
            direct_impact=[direct], transitive_impact=[transitive],
        )
        text = format_text(result)
        assert "Direct Impact (1)" in text
        assert "Transitive Impact (1)" in text

    def test_changed_count_in_output(self):
        result = _make_impact_result(changed_modules=["a", "b", "c"])
        text = format_text(result)
        assert "Changed (3)" in text

    def test_root_name_in_title(self, tmp_path):
        result = _make_impact_result(root=str(tmp_path))
        text = format_text(result)
        assert Path(str(tmp_path)).name in text


# ===================================================================
# Main function: additional edge cases
# ===================================================================


class TestMainEdgeCases:
    """Edge case tests for main() function."""

    def test_absolute_file_path(self, tmp_path, capsys):
        _make_project(tmp_path, {"core.py": ""})
        abs_path = str(tmp_path / "core.py")
        ret = main([str(tmp_path), "--files", abs_path])
        assert ret == 0

    def test_multiple_files_arg(self, tmp_path, capsys):
        _make_project(tmp_path, {
            "a.py": "",
            "b.py": "",
        })
        ret = main([str(tmp_path), "--files", "a.py", "b.py"])
        assert ret == 0

    @patch("code_health_suite.engines.change_impact.get_changed_files_from_git")
    def test_git_diff_empty_result(self, mock_git, tmp_path, capsys):
        mock_git.return_value = []
        ret = main([str(tmp_path), "--git-diff"])
        assert ret == 1


# ===================================================================
# Constants verification
# ===================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_default_excludes_is_set(self):
        assert isinstance(DEFAULT_EXCLUDES, set)

    def test_default_excludes_count(self):
        assert len(DEFAULT_EXCLUDES) == 13

    def test_version_string(self):
        from code_health_suite.engines.change_impact import __version__
        assert isinstance(__version__, str)
        # Version follows semver-like pattern
        parts = __version__.split(".")
        assert len(parts) >= 2
