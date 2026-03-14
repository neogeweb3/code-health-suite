#!/usr/bin/env python3
"""ai-change-impact: Static change-impact analyzer for Python projects.

Given a set of changed files, computes which other modules are affected
via import dependency propagation. Zero external dependencies — pure stdlib.

Usage:
    ai-change-impact path/to/project --files a.py b.py   # explicit files
    ai-change-impact --git-diff                           # from uncommitted changes
    ai-change-impact --git-diff HEAD~3                    # from recent commits
    ai-change-impact --json                               # JSON output
    ai-change-impact --depth 3                            # limit propagation depth
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


__version__ = "0.2.0"

# --- Exclusion Patterns ---

DEFAULT_EXCLUDES = {
    ".venv", "venv", ".tox", "node_modules", "__pycache__", ".git",
    ".mypy_cache", ".pytest_cache", "dist", "build", "egg-info",
    "site-packages", ".eggs",
}


# --- Data Structures ---

@dataclass
class ImpactedModule:
    """A module affected by a change."""
    module: str
    path: str
    depth: int          # 0 = directly changed, 1 = direct importer, etc.
    is_test: bool
    imported_by: list[str] = field(default_factory=list)  # chain from changed file


@dataclass
class CouplingMetrics:
    """Coupling metrics for a single module (Robert Martin's packaging principles)."""
    module: str
    path: str
    is_test: bool
    ca: int             # Afferent coupling: modules that depend on this (fan-in)
    ce: int             # Efferent coupling: modules this depends on (fan-out)
    instability: float  # Ce / (Ca + Ce), 0=stable, 1=unstable
    hub_score: int      # Ca * Ce — high = risky hub (both imported and imports many)
    grade: str          # A-F based on hub_score


@dataclass
class CouplingResult:
    """Complete coupling analysis result."""
    root: str
    total_modules: int
    modules: list[CouplingMetrics]
    avg_instability: float
    median_instability: float
    hub_modules: list[CouplingMetrics]  # top hub_score modules
    stable_modules: list[str]           # instability == 0 (pure libraries)
    unstable_modules: list[str]         # instability == 1 (pure dependents)
    summary: str


@dataclass
class ChangeImpactResult:
    """Complete impact analysis result."""
    root: str
    changed_files: list[str]
    changed_modules: list[str]
    total_project_modules: int
    direct_impact: list[ImpactedModule]
    transitive_impact: list[ImpactedModule]
    affected_tests: list[str]
    impact_score: float         # 0.0 - 1.0, fraction of project affected
    risk_level: str             # low / medium / high / critical
    summary: str


# --- File Discovery ---

def find_python_files(root: str, excludes: Optional[set] = None) -> list[str]:
    """Find all .py files under root, excluding common non-source dirs."""
    if excludes is None:
        excludes = DEFAULT_EXCLUDES
    results = []
    root_path = Path(root).resolve()
    # Handle pyvenv.cfg detection (virtual env directory)
    if (root_path / "pyvenv.cfg").exists():
        return []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [
            d for d in dirnames
            if d not in excludes
            and not d.startswith(".")
            and "site-packages" not in d
            and "node_modules" not in d
            and not (Path(dirpath) / d / "pyvenv.cfg").exists()
        ]
        for f in filenames:
            if f.endswith(".py"):
                results.append(os.path.join(dirpath, f))
    return sorted(results)


def path_to_module(filepath: str, root: str) -> str:
    """Convert a file path to a dotted module name relative to root."""
    root_path = Path(root).resolve()
    file_path = Path(filepath).resolve()
    try:
        rel = file_path.relative_to(root_path)
    except ValueError:
        return file_path.stem
    parts = list(rel.parts)
    # Handle src-layout: if first part is 'src' and it has no __init__.py, strip it
    if (
        len(parts) > 1
        and parts[0] == "src"
        and not (root_path / "src" / "__init__.py").exists()
    ):
        parts = parts[1:]
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else file_path.stem


def is_test_file(filepath: str) -> bool:
    """Check if a file is a test file."""
    name = Path(filepath).name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or "/tests/" in filepath.replace("\\", "/")
        or "/test/" in filepath.replace("\\", "/")
    )


# --- Import Extraction ---

def extract_imports(filepath: str) -> list[dict]:
    """Parse a Python file and extract all import statements."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except (OSError, IOError):
        return []
    try:
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, ValueError):
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "names": [alias.asname or alias.name],
                    "is_relative": False,
                    "level": 0,
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names] if node.names else []
            imports.append({
                "module": module,
                "names": names,
                "is_relative": (node.level or 0) > 0,
                "level": node.level or 0,
                "line": node.lineno,
            })
    return imports


# --- Import Graph Building ---

def resolve_import_target(
    source_module: str,
    source_path: str,
    imp: dict,
    known_modules: set[str],
) -> Optional[str]:
    """Resolve an import to an internal module name, or None if external."""
    if imp["is_relative"]:
        parts = source_module.split(".")
        level = imp["level"]
        if source_path.endswith("__init__.py") and level > 0:
            level -= 1
        if level == 0:
            base_parts = list(parts)
        elif level <= len(parts):
            base_parts = parts[:-level]
        else:
            base_parts = []
        if imp["module"]:
            candidate = ".".join(base_parts + [imp["module"]]) if base_parts else imp["module"]
        else:
            candidate = ".".join(base_parts)
        if candidate in known_modules:
            return candidate
        # Try as subpackage
        for km in known_modules:
            if km.startswith(candidate + "."):
                return candidate
        return candidate if base_parts else None
    else:
        mod = imp["module"]
        if mod in known_modules:
            return mod
        # Try progressively shorter prefixes
        parts = mod.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in known_modules:
                return prefix
        return None


def build_dependency_graph(root: str) -> tuple[dict[str, set[str]], dict[str, str], set[str]]:
    """Build forward dependency graph: module -> set of modules it imports.

    Returns:
        (forward_deps, module_to_path, all_modules)
    """
    files = find_python_files(root)
    module_to_path: dict[str, str] = {}
    for f in files:
        mod = path_to_module(f, root)
        module_to_path[mod] = f

    known_modules = set(module_to_path.keys())
    forward_deps: dict[str, set[str]] = defaultdict(set)

    for mod, filepath in module_to_path.items():
        imports = extract_imports(filepath)
        for imp in imports:
            target = resolve_import_target(mod, filepath, imp, known_modules)
            if target and target in known_modules and target != mod:
                forward_deps[mod].add(target)

    return dict(forward_deps), module_to_path, known_modules


def build_reverse_deps(forward_deps: dict[str, set[str]]) -> dict[str, set[str]]:
    """Invert dependency graph: module -> set of modules that import it."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for source, targets in forward_deps.items():
        for target in targets:
            reverse[target].add(source)
    return dict(reverse)


# --- Impact Propagation ---

def propagate_impact(
    changed_modules: list[str],
    reverse_deps: dict[str, set[str]],
    module_to_path: dict[str, str],
    max_depth: int = 10,
) -> list[ImpactedModule]:
    """BFS propagation from changed modules through reverse dependencies."""
    visited: dict[str, int] = {}   # module -> depth
    parent: dict[str, str] = {}    # module -> parent in BFS
    queue: deque[tuple[str, int]] = deque()

    for mod in changed_modules:
        if mod in module_to_path:
            visited[mod] = 0
            queue.append((mod, 0))

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for importer in reverse_deps.get(current, set()):
            if importer not in visited:
                visited[importer] = depth + 1
                parent[importer] = current
                queue.append((importer, depth + 1))

    results = []
    for mod, depth in sorted(visited.items(), key=lambda x: (x[1], x[0])):
        if depth == 0:
            continue  # skip changed files themselves
        path = module_to_path.get(mod, "")
        chain = []
        cur = mod
        while cur in parent:
            chain.append(cur)
            cur = parent[cur]
        chain.append(cur)
        chain.reverse()
        results.append(ImpactedModule(
            module=mod,
            path=path,
            depth=depth,
            is_test=is_test_file(path),
            imported_by=[c for c in chain if c != mod],
        ))
    return results


# --- Coupling Analysis ---

def _coupling_grade(hub_score: int) -> str:
    """Classify hub_score into a letter grade."""
    if hub_score == 0:
        return "A"
    elif hub_score <= 4:
        return "B"
    elif hub_score <= 12:
        return "C"
    elif hub_score <= 25:
        return "D"
    return "F"


def compute_coupling_metrics(
    root: str,
    top_n: int = 0,
) -> CouplingResult:
    """Compute coupling metrics for all modules in a project.

    Uses Robert Martin's packaging principles:
    - Ca (Afferent coupling): # of modules that depend on this one (fan-in)
    - Ce (Efferent coupling): # of modules this one depends on (fan-out)
    - Instability = Ce / (Ca + Ce): 0 = maximally stable, 1 = maximally unstable
    - Hub Score = Ca * Ce: high values indicate risky hub modules

    Args:
        root: Project root directory
        top_n: If > 0, limit hub_modules to top N by hub_score
    """
    root = str(Path(root).resolve())
    forward_deps, module_to_path, all_modules = build_dependency_graph(root)
    reverse_deps = build_reverse_deps(forward_deps)

    metrics = []
    for mod in sorted(all_modules):
        path = module_to_path.get(mod, "")
        ce = len(forward_deps.get(mod, set()))   # what I import
        ca = len(reverse_deps.get(mod, set()))    # who imports me
        total = ca + ce
        instability = ce / total if total > 0 else 0.0
        hub = ca * ce
        metrics.append(CouplingMetrics(
            module=mod,
            path=path,
            is_test=is_test_file(path),
            ca=ca,
            ce=ce,
            instability=round(instability, 4),
            hub_score=hub,
            grade=_coupling_grade(hub),
        ))

    # Filter out test modules for analysis
    non_test = [m for m in metrics if not m.is_test]
    instabilities = [m.instability for m in non_test] if non_test else [0.0]

    avg_inst = sum(instabilities) / len(instabilities) if instabilities else 0.0
    sorted_inst = sorted(instabilities)
    mid = len(sorted_inst) // 2
    median_inst = (
        sorted_inst[mid]
        if len(sorted_inst) % 2 == 1
        else (sorted_inst[mid - 1] + sorted_inst[mid]) / 2
        if sorted_inst
        else 0.0
    )

    # Hub modules: sorted by hub_score descending
    hub_sorted = sorted(non_test, key=lambda m: m.hub_score, reverse=True)
    hub_modules = [m for m in hub_sorted if m.hub_score > 0]
    if top_n > 0:
        hub_modules = hub_modules[:top_n]

    stable = sorted(m.module for m in non_test if m.instability == 0.0 and m.ca > 0)
    unstable = sorted(m.module for m in non_test if m.instability == 1.0 and m.ce > 0)

    n_hubs = len([m for m in non_test if m.hub_score > 0])
    summary = (
        f"Coupling: {len(non_test)} modules, "
        f"avg instability {avg_inst:.2f}, "
        f"{n_hubs} hubs, "
        f"{len(stable)} pure libraries, "
        f"{len(unstable)} pure dependents"
    )

    return CouplingResult(
        root=root,
        total_modules=len(all_modules),
        modules=metrics,
        avg_instability=round(avg_inst, 4),
        median_instability=round(median_inst, 4),
        hub_modules=hub_modules,
        stable_modules=stable,
        unstable_modules=unstable,
        summary=summary,
    )


def format_coupling_text(result: CouplingResult) -> str:
    """Format coupling result as human-readable text."""
    lines = []
    lines.append(f"Coupling Analysis — {Path(result.root).name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Project Size: {result.total_modules} modules")
    lines.append(f"Avg Instability: {result.avg_instability:.2f} (0=stable, 1=unstable)")
    lines.append(f"Median Instability: {result.median_instability:.2f}")
    lines.append("")

    if result.hub_modules:
        lines.append("Hub Modules (highest coupling risk):")
        lines.append(f"  {'Module':<35} {'Ca':>4} {'Ce':>4} {'Hub':>5} {'I':>5} {'Grade':>5}")
        lines.append(f"  {'-'*35} {'-'*4} {'-'*4} {'-'*5} {'-'*5} {'-'*5}")
        for m in result.hub_modules:
            lines.append(
                f"  {m.module:<35} {m.ca:>4} {m.ce:>4} {m.hub_score:>5} "
                f"{m.instability:>5.2f} {m.grade:>5}"
            )
        lines.append("")

    if result.stable_modules:
        lines.append(f"Pure Libraries (I=0, only imported, never import): {len(result.stable_modules)}")
        for mod in result.stable_modules:
            lines.append(f"  {mod}")
        lines.append("")

    if result.unstable_modules:
        lines.append(f"Pure Dependents (I=1, only import, never imported): {len(result.unstable_modules)}")
        for mod in result.unstable_modules:
            lines.append(f"  {mod}")
        lines.append("")

    lines.append(result.summary)
    return "\n".join(lines)


def format_coupling_json(result: CouplingResult) -> str:
    """Format coupling result as JSON."""
    data = {
        "root": result.root,
        "total_modules": result.total_modules,
        "avg_instability": result.avg_instability,
        "median_instability": result.median_instability,
        "hub_modules": [asdict(m) for m in result.hub_modules],
        "stable_modules": result.stable_modules,
        "unstable_modules": result.unstable_modules,
        "all_modules": [asdict(m) for m in result.modules],
        "summary": result.summary,
    }
    return json.dumps(data, indent=2)


# --- Test Suggestion ---

def suggest_test_command(result: ChangeImpactResult, runner: str = "pytest") -> str:
    """Generate a test runner command for affected tests.

    Args:
        result: Change impact analysis result
        runner: Test runner to use (pytest or unittest)
    """
    if not result.affected_tests:
        return ""

    # Convert module names to file paths
    test_paths = []
    for m in result.direct_impact + result.transitive_impact:
        if m.is_test and m.path:
            # Use relative path from root
            try:
                rel = str(Path(m.path).relative_to(result.root))
            except ValueError:
                rel = m.path
            test_paths.append(rel)

    if not test_paths:
        return ""

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in test_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if runner == "pytest":
        return f"pytest {' '.join(unique_paths)}"
    else:
        return f"python -m unittest {' '.join(unique_paths)}"


def compute_risk_level(score: float) -> str:
    """Classify impact score into risk level."""
    if score >= 0.5:
        return "critical"
    elif score >= 0.25:
        return "high"
    elif score >= 0.1:
        return "medium"
    return "low"


# --- Git Integration ---

def get_changed_files_from_git(root: str, ref: Optional[str] = None) -> list[str]:
    """Get list of changed Python files from git.

    If ref is None, gets uncommitted changes (staged + unstaged).
    If ref is provided (e.g. HEAD~3), gets changes since that ref.
    """
    root_path = Path(root).resolve()
    try:
        if ref:
            result = subprocess.run(
                ["git", "diff", "--name-only", ref],
                capture_output=True, text=True, cwd=str(root_path),
            )
        else:
            # Combine staged + unstaged
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True, text=True, cwd=str(root_path),
            )
            unstaged = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, cwd=str(root_path),
            )
            combined = set()
            if staged.returncode == 0:
                combined.update(staged.stdout.strip().split("\n"))
            if unstaged.returncode == 0:
                combined.update(unstaged.stdout.strip().split("\n"))
            combined.discard("")
            return [
                str(root_path / f) for f in sorted(combined) if f.endswith(".py")
            ]

        if result.returncode != 0:
            return []
        files = result.stdout.strip().split("\n")
        return [
            str(root_path / f) for f in files if f.strip() and f.endswith(".py")
        ]
    except FileNotFoundError:
        return []


# --- Main Analysis ---

def analyze(
    root: str,
    changed_files: list[str],
    max_depth: int = 10,
) -> ChangeImpactResult:
    """Run complete change impact analysis.

    Args:
        root: Project root directory
        changed_files: List of file paths that changed
        max_depth: Maximum propagation depth for transitive impact
    """
    root = str(Path(root).resolve())
    forward_deps, module_to_path, all_modules = build_dependency_graph(root)
    reverse_deps = build_reverse_deps(forward_deps)

    # Resolve changed files to modules
    changed_modules = []
    changed_paths = []
    for f in changed_files:
        fpath = str(Path(f).resolve())
        mod = path_to_module(fpath, root)
        if mod in all_modules:
            changed_modules.append(mod)
            changed_paths.append(fpath)

    if not changed_modules:
        return ChangeImpactResult(
            root=root,
            changed_files=[str(Path(f).resolve()) for f in changed_files],
            changed_modules=[],
            total_project_modules=len(all_modules),
            direct_impact=[],
            transitive_impact=[],
            affected_tests=[],
            impact_score=0.0,
            risk_level="low",
            summary="No matching project modules found for changed files.",
        )

    # Propagate impact
    all_impacted = propagate_impact(changed_modules, reverse_deps, module_to_path, max_depth)

    direct = [m for m in all_impacted if m.depth == 1]
    transitive = [m for m in all_impacted if m.depth > 1]
    affected_tests = sorted(set(
        m.module for m in all_impacted if m.is_test
    ))

    # Score: fraction of non-changed, non-test modules affected
    non_test_modules = {m for m, p in module_to_path.items() if not is_test_file(p)}
    changed_set = set(changed_modules)
    affected_non_test = {m.module for m in all_impacted if not m.is_test and m.module not in changed_set}
    denominator = len(non_test_modules - changed_set)
    score = len(affected_non_test) / denominator if denominator > 0 else 0.0

    total_affected = len(changed_modules) + len(all_impacted)
    summary_parts = [
        f"{len(changed_modules)} changed",
        f"{len(direct)} direct",
        f"{len(transitive)} transitive",
        f"{len(affected_tests)} tests",
    ]
    summary = f"Impact: {' / '.join(summary_parts)} — risk: {compute_risk_level(score)}"

    return ChangeImpactResult(
        root=root,
        changed_files=changed_paths or [str(Path(f).resolve()) for f in changed_files],
        changed_modules=changed_modules,
        total_project_modules=len(all_modules),
        direct_impact=direct,
        transitive_impact=transitive,
        affected_tests=affected_tests,
        impact_score=round(score, 4),
        risk_level=compute_risk_level(score),
        summary=summary,
    )


# --- Output Formatting ---

def format_text(result: ChangeImpactResult) -> str:
    """Format result as human-readable text."""
    lines = []
    lines.append(f"Change Impact Analysis — {Path(result.root).name}")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(f"Risk Level: {result.risk_level.upper()}")
    lines.append(f"Impact Score: {result.impact_score:.1%} of project affected")
    lines.append(f"Project Size: {result.total_project_modules} modules")
    lines.append("")

    # Changed files
    lines.append(f"Changed ({len(result.changed_modules)}):")
    for mod in result.changed_modules:
        lines.append(f"  * {mod}")
    lines.append("")

    # Direct impact
    if result.direct_impact:
        lines.append(f"Direct Impact ({len(result.direct_impact)}):")
        for m in result.direct_impact:
            test_tag = " [TEST]" if m.is_test else ""
            lines.append(f"  <- {m.module}{test_tag}")
        lines.append("")

    # Transitive impact
    if result.transitive_impact:
        lines.append(f"Transitive Impact ({len(result.transitive_impact)}):")
        for m in result.transitive_impact:
            test_tag = " [TEST]" if m.is_test else ""
            chain = " -> ".join(m.imported_by)
            lines.append(f"  {'  ' * (m.depth - 1)}<- {m.module} (depth {m.depth}){test_tag}")
        lines.append("")

    # Affected tests
    if result.affected_tests:
        lines.append(f"Affected Tests ({len(result.affected_tests)}):")
        for t in result.affected_tests:
            lines.append(f"  [TEST] {t}")
        lines.append("")

    if not result.direct_impact and not result.transitive_impact:
        lines.append("No downstream modules are affected by these changes.")
        lines.append("")

    lines.append(result.summary)
    return "\n".join(lines)


def format_json(result: ChangeImpactResult) -> str:
    """Format result as JSON."""
    data = {
        "root": result.root,
        "changed_files": result.changed_files,
        "changed_modules": result.changed_modules,
        "total_project_modules": result.total_project_modules,
        "direct_impact": [asdict(m) for m in result.direct_impact],
        "transitive_impact": [asdict(m) for m in result.transitive_impact],
        "affected_tests": result.affected_tests,
        "impact_score": result.impact_score,
        "risk_level": result.risk_level,
        "summary": result.summary,
    }
    return json.dumps(data, indent=2)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-change-impact",
        description="Static change-impact analyzer for Python projects.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit list of changed files",
    )
    parser.add_argument(
        "--git-diff",
        nargs="?",
        const="__UNCOMMITTED__",
        default=None,
        metavar="REF",
        help="Get changed files from git diff (optionally vs a ref like HEAD~3)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Maximum propagation depth (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--coupling",
        action="store_true",
        help="Run coupling analysis instead of change impact",
    )
    parser.add_argument(
        "--suggest-tests",
        action="store_true",
        dest="suggest_tests",
        help="Output a pytest command for affected tests",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Limit coupling hub output to top N modules (default: all)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    root = str(Path(args.root).resolve())

    # Coupling analysis mode
    if args.coupling:
        coupling = compute_coupling_metrics(root, top_n=args.top)
        if args.json_output:
            print(format_coupling_json(coupling))
        else:
            print(format_coupling_text(coupling))
        return 0

    # Determine changed files
    changed_files: list[str] = []
    if args.files:
        changed_files = [
            str(Path(f).resolve()) if os.path.isabs(f) else str(Path(root) / f)
            for f in args.files
        ]
    elif args.git_diff is not None:
        ref = None if args.git_diff == "__UNCOMMITTED__" else args.git_diff
        changed_files = get_changed_files_from_git(root, ref)
    else:
        # Default: get git uncommitted changes
        changed_files = get_changed_files_from_git(root, None)

    if not changed_files:
        print("No changed Python files found.", file=sys.stderr)
        return 1

    result = analyze(root, changed_files, max_depth=args.depth)

    if args.suggest_tests:
        cmd = suggest_test_command(result)
        if cmd:
            print(cmd)
        else:
            print("No affected tests found.", file=sys.stderr)
            return 1
        return 0

    if args.json_output:
        print(format_json(result))
    else:
        print(format_text(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
