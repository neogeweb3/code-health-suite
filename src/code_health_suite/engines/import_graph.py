#!/usr/bin/env python3
"""ai-import-graph: Python cross-module import graph analyzer.

Builds a directed import graph from Python source files, detects circular
imports, identifies orphan modules, and calculates coupling metrics.
Zero external dependencies — pure Python stdlib.

Usage:
    ai-import-graph                         # analyze current directory
    ai-import-graph path/to/project         # analyze specific directory
    ai-import-graph --json                  # JSON output
    ai-import-graph --cycles-only           # only show circular imports
    ai-import-graph --top N                 # show top N hub modules
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
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
class ImportEdge:
    """A single import relationship."""
    source_module: str       # importing module (dotted name)
    source_path: str         # file path of source
    target_raw: str          # raw import string (as written in code)
    target_resolved: str     # resolved to internal module name (or '' if external)
    names: list[str]         # specific names imported (from X import a, b)
    is_relative: bool        # relative import
    is_internal: bool        # resolved to a module within the project
    line: int                # line number


@dataclass
class CycleInfo:
    """A circular import cycle."""
    path: list[str]          # module names forming the cycle
    length: int              # number of modules in cycle


@dataclass
class ModuleMetrics:
    """Coupling metrics for a single module."""
    module: str
    path: str
    afferent: int            # Ca — how many internal modules import this one
    efferent: int            # Ce — how many internal modules this one imports
    instability: float       # Ce / (Ca + Ce), 0.0=stable, 1.0=unstable
    is_orphan: bool          # no internal imports in or out


@dataclass
class GraphResult:
    """Complete analysis result."""
    root: str
    total_modules: int
    total_edges: int
    internal_edges: int
    external_packages: int
    cycles: list[CycleInfo]
    orphans: list[str]
    hub_modules: list[dict]       # [{module, afferent, path}]
    unstable_modules: list[dict]  # [{module, instability, efferent, afferent}]
    metrics: list[ModuleMetrics]
    external_deps: list[dict]     # [{package, imported_by_count}]


# --- Health Scoring ---

@dataclass
class ImportHealthScore:
    """Project-level import health score."""
    score: int                      # 0-100
    grade: str                      # A-F
    total_modules: int
    cycle_count: int
    cycle_penalty: float
    orphan_count: int
    orphan_ratio: float
    orphan_penalty: float
    avg_instability: float
    instability_penalty: float
    hub_concentration: float        # Gini-like: top hub afferent / total modules
    hub_penalty: float
    external_dep_count: int
    profile: str                    # classification label


def _letter_grade(score: int) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_import_health(result: GraphResult) -> ImportHealthScore:
    """Compute import health score (0-100) from analysis result.

    Penalty model:
    - Circular imports: 15 pts per cycle (most critical)
    - Orphan ratio: up to 20 pts (>50% orphans = max penalty)
    - Avg instability: up to 15 pts (high instability = fragile codebase)
    - Hub concentration: up to 15 pts (one module depended on by everyone)
    """
    total = result.total_modules
    if total == 0:
        score = ImportHealthScore(
            score=100, grade="A", total_modules=0,
            cycle_count=0, cycle_penalty=0.0,
            orphan_count=0, orphan_ratio=0.0, orphan_penalty=0.0,
            avg_instability=0.0, instability_penalty=0.0,
            hub_concentration=0.0, hub_penalty=0.0,
            external_dep_count=0, profile="empty",
        )
        return score

    # 1. Cycle penalty: 15 per cycle, max 40
    cycle_count = len(result.cycles)
    cycle_penalty = min(cycle_count * 15.0, 40.0)

    # 2. Orphan penalty: orphan_ratio * 20, max 20
    orphan_count = len(result.orphans)
    orphan_ratio = orphan_count / total if total > 0 else 0.0
    orphan_penalty = min(orphan_ratio * 20.0, 20.0)

    # 3. Instability penalty: avg instability of non-orphan modules * 15
    non_orphan_metrics = [m for m in result.metrics if not m.is_orphan]
    if non_orphan_metrics:
        avg_instability = sum(m.instability for m in non_orphan_metrics) / len(non_orphan_metrics)
    else:
        avg_instability = 0.0
    instability_penalty = min(avg_instability * 15.0, 15.0)

    # 4. Hub concentration penalty: top hub's afferent / total * 15
    if result.hub_modules and total > 1:
        max_afferent = result.hub_modules[0]["afferent"]
        hub_concentration = max_afferent / (total - 1)  # exclude self
    else:
        hub_concentration = 0.0
    hub_penalty = min(hub_concentration * 15.0, 15.0)

    raw_score = 100.0 - cycle_penalty - orphan_penalty - instability_penalty - hub_penalty
    final_score = max(0, min(100, int(round(raw_score))))

    profile = classify_import_profile(
        cycle_count=cycle_count,
        orphan_ratio=orphan_ratio,
        avg_instability=avg_instability,
        hub_concentration=hub_concentration,
        total_modules=total,
    )

    return ImportHealthScore(
        score=final_score,
        grade=_letter_grade(final_score),
        total_modules=total,
        cycle_count=cycle_count,
        cycle_penalty=round(cycle_penalty, 1),
        orphan_count=orphan_count,
        orphan_ratio=round(orphan_ratio, 3),
        orphan_penalty=round(orphan_penalty, 1),
        avg_instability=round(avg_instability, 3),
        instability_penalty=round(instability_penalty, 1),
        hub_concentration=round(hub_concentration, 3),
        hub_penalty=round(hub_penalty, 1),
        external_dep_count=result.external_packages,
        profile=profile,
    )


def classify_import_profile(
    *,
    cycle_count: int,
    orphan_ratio: float,
    avg_instability: float,
    hub_concentration: float,
    total_modules: int,
) -> str:
    """Classify the project's import structure profile.

    Profiles:
    - clean: no major issues (score would be >=90)
    - cycle_heavy: circular imports dominate
    - orphan_heavy: many disconnected modules
    - hub_concentrated: few modules carry most dependencies
    - unstable: high average instability
    - fragmented: orphan + unstable combination
    - mixed: multiple issue types
    """
    if total_modules == 0:
        return "empty"

    issues = []
    if cycle_count >= 2:
        issues.append("cycle_heavy")
    elif cycle_count == 1:
        issues.append("has_cycle")
    if orphan_ratio > 0.4:
        issues.append("orphan_heavy")
    if avg_instability > 0.6:
        issues.append("unstable")
    if hub_concentration > 0.6:
        issues.append("hub_concentrated")

    if not issues:
        return "clean"
    if len(issues) == 1:
        return issues[0]
    if "orphan_heavy" in issues and "unstable" in issues:
        return "fragmented"
    return "mixed"


def format_score_text(health: ImportHealthScore) -> str:
    """Format health score for terminal display."""
    lines = []
    lines.append(colorize("=" * 60, "bold"))
    lines.append(colorize("  ai-import-graph — Import Health Score", "bold"))
    lines.append(colorize("=" * 60, "bold"))
    lines.append("")

    grade_color = {"A": "green", "B": "green", "C": "yellow", "D": "yellow"}.get(health.grade, "red")
    lines.append(f"  Score: {colorize(str(health.score), grade_color)}/100  "
                 f"Grade: {colorize(health.grade, grade_color)}  "
                 f"Profile: {colorize(health.profile, 'cyan')}")
    lines.append("")

    lines.append(colorize("  Penalty Breakdown:", "bold"))
    lines.append(f"    Circular imports: {health.cycle_count} cycles → -{health.cycle_penalty} pts")
    lines.append(f"    Orphan modules:   {health.orphan_count}/{health.total_modules} "
                 f"({health.orphan_ratio:.1%}) → -{health.orphan_penalty} pts")
    lines.append(f"    Avg instability:  {health.avg_instability:.3f} → -{health.instability_penalty} pts")
    lines.append(f"    Hub concentration: {health.hub_concentration:.3f} → -{health.hub_penalty} pts")
    lines.append("")
    lines.append(f"  External deps: {health.external_dep_count}")
    lines.append("")
    return "\n".join(lines)


def format_score_json(health: ImportHealthScore) -> str:
    """Format health score as JSON."""
    data = {
        "version": __version__,
        "score": health.score,
        "grade": health.grade,
        "profile": health.profile,
        "total_modules": health.total_modules,
        "penalties": {
            "cycles": {"count": health.cycle_count, "penalty": health.cycle_penalty},
            "orphans": {"count": health.orphan_count, "ratio": health.orphan_ratio, "penalty": health.orphan_penalty},
            "instability": {"avg": health.avg_instability, "penalty": health.instability_penalty},
            "hub_concentration": {"value": health.hub_concentration, "penalty": health.hub_penalty},
        },
        "external_dep_count": health.external_dep_count,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# --- File Discovery ---

def find_python_files(root: str, excludes: Optional[set] = None) -> list[str]:
    """Find all .py files under root, excluding common non-source dirs."""
    if excludes is None:
        excludes = DEFAULT_EXCLUDES
    results = []
    root_path = Path(root).resolve()
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in excludes
            and not d.startswith(".")
            and "site-packages" not in d
            and "node_modules" not in d
            and not (d.endswith("_env") or d.startswith(".venv"))
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
    # Remove .py extension from last part
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # Convert __init__ to package name
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    # Handle src-layout: if first component is 'src' and src/__init__.py
    # doesn't exist, strip it (src is a namespace directory, not a package)
    if parts and parts[0] == "src" and not (root_path / "src" / "__init__.py").exists():
        parts = parts[1:]
    return ".".join(parts) if parts else file_path.stem


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
        # ValueError: source code string cannot contain null bytes (binary files)
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


# --- Module Resolution ---

def resolve_import(
    source_module: str,
    source_path: str,
    imp: dict,
    module_map: dict[str, str],
    root: str,
) -> tuple[str, bool]:
    """Resolve an import to an internal module name.

    Returns (resolved_name, is_internal).
    """
    if imp["is_relative"]:
        # Relative import: resolve based on source module
        parts = source_module.split(".")
        level = imp["level"]
        # __init__.py IS the package — level 1 means "current package", not "go up"
        # e.g. from .core import X in pkg/__init__.py → pkg.core, not core
        if source_path.endswith("__init__.py") and level > 0:
            level -= 1
        # Go up 'level' packages
        if level == 0:
            base_parts = list(parts)
        elif level <= len(parts):
            base_parts = parts[:-level]
        else:
            base_parts = []
        if imp["module"]:
            candidate = ".".join(base_parts + [imp["module"]]) if base_parts else imp["module"]
        else:
            candidate = ".".join(base_parts) if base_parts else ""

        # Check if this resolves to a known module
        if candidate in module_map:
            return candidate, True
        # Maybe it's a package — check with submodule names
        for known in module_map:
            if known.startswith(candidate + "."):
                return candidate, True
        # Check if any imported name matches a submodule
        for name in imp["names"]:
            sub = f"{candidate}.{name}" if candidate else name
            if sub in module_map:
                return sub, True
        return candidate or imp["module"], False

    # Absolute import
    module = imp["module"]
    # Direct match
    if module in module_map:
        return module, True
    # Check if it's a package prefix
    for known in module_map:
        if known.startswith(module + "."):
            return module, True
    # Check top-level parts (from foo.bar import X → foo.bar might be internal)
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in module_map:
            return prefix, True
    return module, False


def get_top_level_package(module_name: str) -> str:
    """Extract top-level package from dotted name."""
    return module_name.split(".")[0] if module_name else module_name


# --- Graph Building ---

def build_graph(root: str, excludes: Optional[set] = None) -> tuple[list[ImportEdge], dict[str, str]]:
    """Build the import graph for all Python files under root.

    Returns (edges, module_map) where module_map is {module_name: filepath}.
    """
    files = find_python_files(root, excludes)
    # Build module map: dotted name → file path
    module_map: dict[str, str] = {}
    for fp in files:
        mod_name = path_to_module(fp, root)
        if mod_name:
            module_map[mod_name] = fp

    edges: list[ImportEdge] = []
    for fp in files:
        source_mod = path_to_module(fp, root)
        raw_imports = extract_imports(fp)
        for imp in raw_imports:
            resolved, is_internal = resolve_import(
                source_mod, fp, imp, module_map, root
            )
            edge = ImportEdge(
                source_module=source_mod,
                source_path=fp,
                target_raw=imp["module"],
                target_resolved=resolved,
                names=imp["names"],
                is_relative=imp["is_relative"],
                is_internal=is_internal,
                line=imp["line"],
            )
            edges.append(edge)
    return edges, module_map


# --- Cycle Detection ---

def detect_cycles(edges: list[ImportEdge], module_map: dict[str, str]) -> list[CycleInfo]:
    """Detect circular imports using DFS with coloring."""
    # Build adjacency list (internal edges only)
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.is_internal and e.target_resolved in module_map:
            if e.source_module != e.target_resolved:  # skip self-imports
                adj[e.source_module].add(e.target_resolved)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {m: WHITE for m in module_map}
    parent: dict[str, Optional[str]] = {m: None for m in module_map}
    cycles: list[CycleInfo] = []
    seen_cycles: set[tuple] = set()

    def dfs(u: str, path: list[str]):
        color[u] = GRAY
        path.append(u)
        for v in sorted(adj.get(u, [])):
            if color.get(v) == GRAY:
                # Found a cycle — extract it
                cycle_start = path.index(v)
                cycle_path = path[cycle_start:] + [v]
                # Normalize: start from lexicographically smallest
                min_idx = cycle_path[:-1].index(min(cycle_path[:-1]))
                normalized = tuple(cycle_path[min_idx:-1])
                if normalized not in seen_cycles:
                    seen_cycles.add(normalized)
                    cycles.append(CycleInfo(
                        path=list(normalized) + [normalized[0]],
                        length=len(normalized),
                    ))
            elif color.get(v, WHITE) == WHITE:
                dfs(v, path)
        path.pop()
        color[u] = BLACK

    for module in sorted(module_map.keys()):
        if color.get(module, WHITE) == WHITE:
            dfs(module, [])

    return sorted(cycles, key=lambda c: (-c.length, c.path[0]))


# --- Metrics Calculation ---

def calculate_metrics(
    edges: list[ImportEdge],
    module_map: dict[str, str],
) -> list[ModuleMetrics]:
    """Calculate coupling metrics for each module."""
    # Only count internal edges
    afferent: dict[str, set[str]] = defaultdict(set)   # who imports me
    efferent: dict[str, set[str]] = defaultdict(set)    # who do I import

    for e in edges:
        if e.is_internal and e.target_resolved in module_map:
            target = e.target_resolved
            source = e.source_module
            if source != target:  # skip self-imports
                afferent[target].add(source)
                efferent[source].add(target)

    metrics = []
    for mod, path in sorted(module_map.items()):
        ca = len(afferent.get(mod, set()))
        ce = len(efferent.get(mod, set()))
        instability = ce / (ca + ce) if (ca + ce) > 0 else 0.0
        is_orphan = ca == 0 and ce == 0
        metrics.append(ModuleMetrics(
            module=mod,
            path=path,
            afferent=ca,
            efferent=ce,
            instability=round(instability, 3),
            is_orphan=is_orphan,
        ))
    return metrics


# --- External Dependency Analysis ---

def analyze_external_deps(edges: list[ImportEdge]) -> list[dict]:
    """Analyze external (non-internal) imports."""
    ext_counts: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if not e.is_internal:
            pkg = get_top_level_package(e.target_raw)
            if pkg and pkg not in ("", "__future__"):
                ext_counts[pkg].add(e.source_module)
    result = [
        {"package": pkg, "imported_by_count": len(modules), "imported_by": sorted(modules)}
        for pkg, modules in ext_counts.items()
    ]
    return sorted(result, key=lambda x: (-x["imported_by_count"], x["package"]))


# --- Main Analysis ---

def analyze(root: str, excludes: Optional[set] = None) -> GraphResult:
    """Run complete import graph analysis."""
    root = str(Path(root).resolve())
    edges, module_map = build_graph(root, excludes)
    cycles = detect_cycles(edges, module_map)
    metrics = calculate_metrics(edges, module_map)
    external_deps = analyze_external_deps(edges)

    internal_edges = [e for e in edges if e.is_internal]
    orphans = [m.module for m in metrics if m.is_orphan]
    hub_modules = sorted(
        [{"module": m.module, "afferent": m.afferent, "path": m.path}
         for m in metrics if m.afferent > 0],
        key=lambda x: (-x["afferent"], x["module"]),
    )
    unstable_modules = sorted(
        [{"module": m.module, "instability": m.instability,
          "efferent": m.efferent, "afferent": m.afferent}
         for m in metrics if m.instability > 0.8 and m.efferent > 1],
        key=lambda x: (-x["instability"], -x["efferent"]),
    )

    return GraphResult(
        root=root,
        total_modules=len(module_map),
        total_edges=len(edges),
        internal_edges=len(internal_edges),
        external_packages=len(external_deps),
        cycles=cycles,
        orphans=orphans,
        hub_modules=hub_modules[:20],
        unstable_modules=unstable_modules[:20],
        metrics=metrics,
        external_deps=external_deps,
    )


# --- Formatting ---

COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def colorize(text: str, color: str) -> str:
    """Apply terminal color if stdout is a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def format_result(result: GraphResult, top_n: int = 10) -> str:
    """Format analysis result for terminal display."""
    lines = []
    lines.append(colorize("=" * 60, "bold"))
    lines.append(colorize("  ai-import-graph — Python Import Graph Analysis", "bold"))
    lines.append(colorize("=" * 60, "bold"))
    lines.append("")

    # Summary
    lines.append(colorize("📊 Summary", "bold"))
    lines.append(f"  Root:              {result.root}")
    lines.append(f"  Modules:           {result.total_modules}")
    lines.append(f"  Import edges:      {result.total_edges} ({result.internal_edges} internal)")
    lines.append(f"  External packages: {result.external_packages}")
    lines.append(f"  Circular imports:  {len(result.cycles)}")
    lines.append(f"  Orphan modules:    {len(result.orphans)}")
    lines.append("")

    # Circular imports
    if result.cycles:
        lines.append(colorize(f"🔄 Circular Imports ({len(result.cycles)})", "red"))
        for i, cycle in enumerate(result.cycles[:top_n], 1):
            chain = " → ".join(cycle.path)
            lines.append(f"  {i}. [{cycle.length} modules] {chain}")
        if len(result.cycles) > top_n:
            lines.append(f"  ... and {len(result.cycles) - top_n} more")
        lines.append("")
    else:
        lines.append(colorize("✅ No circular imports detected", "green"))
        lines.append("")

    # Hub modules (most imported)
    if result.hub_modules:
        lines.append(colorize(f"🏗️  Hub Modules (most depended on)", "blue"))
        for i, hub in enumerate(result.hub_modules[:top_n], 1):
            lines.append(f"  {i}. {hub['module']} — {hub['afferent']} dependents")
        lines.append("")

    # Unstable modules
    if result.unstable_modules:
        lines.append(colorize(f"⚠️  Unstable Modules (instability > 0.8, efferent > 1)", "yellow"))
        for i, mod in enumerate(result.unstable_modules[:top_n], 1):
            lines.append(
                f"  {i}. {mod['module']} — I={mod['instability']:.2f} "
                f"(Ce={mod['efferent']}, Ca={mod['afferent']})"
            )
        lines.append("")

    # Orphan modules
    if result.orphans:
        lines.append(colorize(f"🏝️  Orphan Modules ({len(result.orphans)} — no internal imports in/out)", "dim"))
        for orphan in result.orphans[:top_n]:
            lines.append(f"  - {orphan}")
        if len(result.orphans) > top_n:
            lines.append(f"  ... and {len(result.orphans) - top_n} more")
        lines.append("")

    # External dependencies
    if result.external_deps:
        lines.append(colorize(f"📦 External Dependencies ({result.external_packages})", "cyan"))
        for dep in result.external_deps[:top_n]:
            lines.append(f"  {dep['package']:30s} used by {dep['imported_by_count']} module(s)")
        if len(result.external_deps) > top_n:
            lines.append(f"  ... and {len(result.external_deps) - top_n} more")
        lines.append("")

    return "\n".join(lines)


def format_json(result: GraphResult) -> str:
    """Format result as JSON."""
    data = {
        "version": __version__,
        "root": result.root,
        "summary": {
            "total_modules": result.total_modules,
            "total_edges": result.total_edges,
            "internal_edges": result.internal_edges,
            "external_packages": result.external_packages,
            "circular_imports": len(result.cycles),
            "orphan_modules": len(result.orphans),
        },
        "cycles": [asdict(c) for c in result.cycles],
        "hub_modules": result.hub_modules,
        "unstable_modules": result.unstable_modules,
        "orphans": result.orphans,
        "external_deps": result.external_deps,
        "metrics": [asdict(m) for m in result.metrics],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-import-graph",
        description="Python cross-module import graph analyzer",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory to analyze (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--cycles-only",
        action="store_true",
        help="Only show circular imports",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Show top N items in each category (default: 10)",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Show project import health score (0-100)",
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

    root = args.path
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory", file=sys.stderr)
        return 1

    result = analyze(root)

    if args.score:
        health = compute_import_health(result)
        if args.json_output:
            print(format_score_json(health))
        else:
            print(format_score_text(health))
        return 0

    if args.json_output:
        print(format_json(result))
    elif args.cycles_only:
        if result.cycles:
            for cycle in result.cycles:
                chain = " → ".join(cycle.path)
                print(f"[{cycle.length}] {chain}")
            return 2 if result.cycles else 0
        else:
            print("No circular imports detected.")
    else:
        print(format_result(result, top_n=args.top))
        if result.cycles:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
