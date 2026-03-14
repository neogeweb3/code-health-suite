#!/usr/bin/env python3
"""ai-hotspot — Code hotspot detector.

Finds files that are both complex AND frequently changed.
High churn + high complexity = highest bug risk, best ROI for code review.

Dependencies: ai-ast-utils (sibling package, scope-safe AST traversal).
"""

import argparse
import ast
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Import shared scope-safe AST traversal (sibling ai-ast-utils package)
# sys.path hack removed — using package import
from code_health_suite.ast_utils import walk_scope as _walk_scope  # noqa: E402


# --- Constants ---

DEFAULT_TOP_N = 20
DEFAULT_SINCE_DAYS = 180
RISK_THRESHOLDS = {"critical": 0.7, "high": 0.4, "medium": 0.2}

# Category thresholds: a hotspot is "driven" by whichever normalized dimension ≥ this
CATEGORY_DOMINANT_THRESHOLD = 0.6
CATEGORY_WEAK_THRESHOLD = 0.3


# --- Data Classes ---

@dataclass
class ChurnData:
    """Git churn metrics for a single file."""
    commits: int = 0
    lines_added: int = 0
    lines_deleted: int = 0

    @property
    def churn_score(self) -> float:
        """Weighted churn: sqrt(commits) * sqrt(total_changes)."""
        total = self.lines_added + self.lines_deleted
        return math.sqrt(self.commits) * math.sqrt(total) if self.commits > 0 and total > 0 else 0.0


@dataclass
class ComplexityData:
    """Complexity metrics for a single file."""
    max_cc: int = 0
    total_cc: int = 0
    num_functions: int = 0
    longest_function: int = 0

    @property
    def complexity_score(self) -> float:
        """Primary score is max CC across all functions."""
        return float(self.max_cc)


@dataclass
class HotspotResult:
    """Combined hotspot analysis for a single file."""
    filepath: str
    churn: ChurnData = field(default_factory=ChurnData)
    complexity: ComplexityData = field(default_factory=ComplexityData)
    hotspot_score: float = 0.0
    risk_level: str = "low"
    churn_normalized: float = 0.0
    complexity_normalized: float = 0.0


@dataclass
class AnalysisResult:
    """Full analysis output."""
    repo_path: str
    total_files_analyzed: int = 0
    total_python_files: int = 0
    since_days: int = DEFAULT_SINCE_DAYS
    hotspots: List[HotspotResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class HotspotProjectStats:
    """Project-level hotspot health summary."""
    total_files: int = 0
    files_with_hotspots: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    avg_hotspot_score: float = 0.0
    max_hotspot_score: float = 0.0
    hotspot_density_pct: float = 0.0  # % of files that are medium+ hotspots
    score: int = 0        # 0-100, higher = healthier
    grade: str = "F"      # A-F


# --- Git Root Detection ---

def find_git_root(path: str) -> Optional[str]:
    """Find the git repository root for the given path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=path, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, NotADirectoryError):
        pass
    return None


# --- Git Churn Analysis ---

def get_file_churn(repo_path: str, since_days: int = DEFAULT_SINCE_DAYS) -> Dict[str, ChurnData]:
    """Get commit count and line changes per file from git log."""
    churn_map: Dict[str, ChurnData] = {}

    since_arg = f"--since={since_days} days ago"

    # Step 1: Get commit counts per file
    try:
        result = subprocess.run(
            ["git", "log", since_arg, "--format=", "--name-only"],
            capture_output=True, text=True, cwd=repo_path, timeout=60
        )
        if result.returncode != 0:
            return churn_map

        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line.endswith(".py"):
                if line not in churn_map:
                    churn_map[line] = ChurnData()
                churn_map[line].commits += 1
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return churn_map

    # Step 2: Get line changes per file
    try:
        result = subprocess.run(
            ["git", "log", since_arg, "--format=", "--numstat"],
            capture_output=True, text=True, cwd=repo_path, timeout=60
        )
        if result.returncode != 0:
            return churn_map

        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added_str, deleted_str, filepath = parts
            if not filepath.endswith(".py"):
                continue
            # Binary files show "-" for added/deleted
            if added_str == "-" or deleted_str == "-":
                continue
            try:
                added = int(added_str)
                deleted = int(deleted_str)
            except ValueError:
                continue
            if filepath not in churn_map:
                churn_map[filepath] = ChurnData()
            churn_map[filepath].lines_added += added
            churn_map[filepath].lines_deleted += deleted
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return churn_map


# --- AST Complexity Analysis ---

class ComplexityVisitor(ast.NodeVisitor):
    """McCabe cyclomatic complexity calculator."""

    BRANCH_NODES = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )

    def __init__(self):
        self.functions: List[Tuple[str, int, int]] = []  # (name, cc, lines)
        self._current_cc = 0

    def _count_branches(self, node: ast.AST) -> int:
        """Count branching statements in a node (excludes nested scopes).

        Uses shared walk_scope from ai-ast-utils which correctly skips
        nested FunctionDef, AsyncFunctionDef, and ClassDef bodies.
        Previously used a local _walk_excluding_nested that duplicated
        this logic and was the root cause of BUG-48.
        """
        count = 0
        for child in _walk_scope(node):
            if isinstance(child, self.BRANCH_NODES):
                count += 1
            elif isinstance(child, ast.BoolOp):
                # Each 'and'/'or' adds a path
                count += len(child.values) - 1
        return count

    def _visit_function(self, node):
        """Process a function/method definition."""
        name = getattr(node, "name", "<lambda>")
        cc = 1 + self._count_branches(node)
        # Calculate function length
        if hasattr(node, "end_lineno") and node.end_lineno and node.lineno:
            lines = node.end_lineno - node.lineno + 1
        else:
            lines = 0
        self.functions.append((name, cc, lines))
        self.generic_visit(node)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def get_file_complexity(filepath: str) -> Optional[ComplexityData]:
    """Compute complexity metrics for a single Python file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None

    visitor = ComplexityVisitor()
    visitor.visit(tree)

    if not visitor.functions:
        return ComplexityData(max_cc=1, total_cc=1, num_functions=0, longest_function=0)

    ccs = [cc for _, cc, _ in visitor.functions]
    lines = [ln for _, _, ln in visitor.functions]

    return ComplexityData(
        max_cc=max(ccs),
        total_cc=sum(ccs),
        num_functions=len(visitor.functions),
        longest_function=max(lines) if lines else 0,
    )


# --- Hotspot Computation ---

def normalize_values(values: List[float]) -> List[float]:
    """Min-max normalize a list of values to [0, 1]."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def classify_risk(score: float) -> str:
    """Classify hotspot score into risk level."""
    if score >= RISK_THRESHOLDS["critical"]:
        return "critical"
    elif score >= RISK_THRESHOLDS["high"]:
        return "high"
    elif score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def compute_hotspots(
    churn_map: Dict[str, ChurnData],
    complexity_map: Dict[str, ComplexityData],
) -> List[HotspotResult]:
    """Compute hotspot scores by combining churn and complexity."""
    # Only files present in BOTH maps
    common_files = set(churn_map.keys()) & set(complexity_map.keys())
    if not common_files:
        return []

    results = []
    for filepath in common_files:
        results.append(HotspotResult(
            filepath=filepath,
            churn=churn_map[filepath],
            complexity=complexity_map[filepath],
        ))

    # Normalize churn scores
    churn_scores = [r.churn.churn_score for r in results]
    churn_norm = normalize_values(churn_scores)

    # Normalize complexity scores
    complexity_scores = [r.complexity.complexity_score for r in results]
    complexity_norm = normalize_values(complexity_scores)

    # Compute hotspot = churn_norm * complexity_norm
    for i, result in enumerate(results):
        result.churn_normalized = churn_norm[i]
        result.complexity_normalized = complexity_norm[i]
        result.hotspot_score = churn_norm[i] * complexity_norm[i]
        result.risk_level = classify_risk(result.hotspot_score)

    # Sort descending by hotspot score
    results.sort(key=lambda r: r.hotspot_score, reverse=True)
    return results


# --- Hotspot Categories ---

def classify_hotspot_category(h: HotspotResult) -> str:
    """Classify a hotspot by its dominant dimension.

    Returns one of:
      - "complexity_driven": high complexity, moderate/low churn
      - "churn_driven": high churn, moderate/low complexity
      - "dual": both dimensions are strong
      - "balanced": neither dimension dominates (low hotspot score)
    """
    cn = h.complexity_normalized
    ch = h.churn_normalized
    c_dom = CATEGORY_DOMINANT_THRESHOLD
    c_weak = CATEGORY_WEAK_THRESHOLD

    if cn >= c_dom and ch >= c_dom:
        return "dual"
    elif cn >= c_dom and ch < c_weak:
        return "complexity_driven"
    elif ch >= c_dom and cn < c_weak:
        return "churn_driven"
    elif cn >= c_dom:
        return "complexity_driven"
    elif ch >= c_dom:
        return "churn_driven"
    else:
        return "balanced"


# --- Project-Level Scoring ---

def compute_project_stats(result: AnalysisResult) -> HotspotProjectStats:
    """Compute project-level health score from hotspot analysis.

    Score formula (0-100, higher = healthier):
      - Start at 100
      - Deduct 15 per critical hotspot (max 45)
      - Deduct 8 per high hotspot (max 32)
      - Deduct 3 per medium hotspot (max 15)
      - Deduct up to 8 based on hotspot density (% of medium+ files)

    Grade: A (90-100), B (80-89), C (65-79), D (50-64), F (<50)
    """
    stats = HotspotProjectStats()

    if not result.hotspots:
        stats.total_files = result.total_files_analyzed
        stats.score = 100
        stats.grade = "A"
        return stats

    stats.total_files = result.total_files_analyzed

    for h in result.hotspots:
        if h.risk_level == "critical":
            stats.critical_count += 1
        elif h.risk_level == "high":
            stats.high_count += 1
        elif h.risk_level == "medium":
            stats.medium_count += 1
        else:
            stats.low_count += 1

    stats.files_with_hotspots = stats.critical_count + stats.high_count + stats.medium_count
    scores = [h.hotspot_score for h in result.hotspots]
    stats.avg_hotspot_score = sum(scores) / len(scores)
    stats.max_hotspot_score = max(scores)

    if stats.total_files > 0:
        stats.hotspot_density_pct = (stats.files_with_hotspots / stats.total_files) * 100
    else:
        stats.hotspot_density_pct = 0.0

    # Score calculation
    raw = 100.0
    raw -= min(stats.critical_count * 15, 45)
    raw -= min(stats.high_count * 8, 32)
    raw -= min(stats.medium_count * 3, 15)
    raw -= min(stats.hotspot_density_pct * 0.08, 8)

    stats.score = max(0, min(100, int(round(raw))))

    # Grade
    if stats.score >= 90:
        stats.grade = "A"
    elif stats.score >= 80:
        stats.grade = "B"
    elif stats.score >= 65:
        stats.grade = "C"
    elif stats.score >= 50:
        stats.grade = "D"
    else:
        stats.grade = "F"

    return stats


# --- Main Analysis ---

def find_python_files(repo_path: str) -> List[str]:
    """Find all Python files in repo, respecting common ignore patterns."""
    py_files = []
    ignore_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                   ".tox", ".eggs", "build", "dist", ".mypy_cache", "_archive"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                py_files.append(rel)
    return py_files


def analyze(
    repo_path: str,
    top_n: int = DEFAULT_TOP_N,
    since_days: int = DEFAULT_SINCE_DAYS,
) -> AnalysisResult:
    """Run full hotspot analysis on a repository."""
    repo_path = os.path.abspath(repo_path)
    result = AnalysisResult(repo_path=repo_path, since_days=since_days)

    # Step 1: Find Python files (relative to repo_path)
    py_files = find_python_files(repo_path)
    result.total_python_files = len(py_files)

    if not py_files:
        result.errors.append("No Python files found")
        return result

    # Step 2: Detect git root — churn paths are always relative to git root,
    # but find_python_files returns paths relative to repo_path.
    # When repo_path is a subdirectory (e.g. /repo/src), we need to translate.
    git_root = find_git_root(repo_path) or repo_path
    git_root = os.path.abspath(git_root)

    # Compute prefix to translate between git-root-relative and repo_path-relative paths
    if git_root != repo_path:
        rel_prefix = os.path.relpath(repo_path, git_root)
    else:
        rel_prefix = ""

    # Step 3: Get git churn data (paths are relative to git root)
    churn_map = get_file_churn(git_root, since_days)
    if not churn_map:
        result.errors.append("No git churn data (not a git repo or no commits in time range)")
        return result

    # Step 4: Translate churn_map to repo_path-relative paths, filtering to our scope
    if rel_prefix:
        scoped_churn: Dict[str, ChurnData] = {}
        prefix_with_sep = rel_prefix + os.sep
        for git_path, churn in churn_map.items():
            if git_path.startswith(prefix_with_sep):
                local_path = git_path[len(prefix_with_sep):]
                scoped_churn[local_path] = churn
        churn_map = scoped_churn

    # Step 5: Compute complexity for files that have churn
    complexity_map: Dict[str, ComplexityData] = {}
    for filepath in py_files:
        if filepath in churn_map:
            abs_path = os.path.join(repo_path, filepath)
            cd = get_file_complexity(abs_path)
            if cd is not None:
                complexity_map[filepath] = cd

    result.total_files_analyzed = len(complexity_map)

    # Step 6: Compute hotspot scores
    all_hotspots = compute_hotspots(churn_map, complexity_map)
    result.hotspots = all_hotspots[:top_n]

    return result


# --- Formatting ---

def format_text(result: AnalysisResult) -> str:
    """Format analysis result as human-readable text."""
    lines = []
    lines.append(f"=== Code Hotspot Analysis ===")
    lines.append(f"Repository: {result.repo_path}")
    lines.append(f"Python files: {result.total_python_files} total, "
                 f"{result.total_files_analyzed} with git history")
    lines.append(f"Time range: last {result.since_days} days")
    lines.append("")

    if result.errors:
        for err in result.errors:
            lines.append(f"ERROR: {err}")
        return "\n".join(lines)

    if not result.hotspots:
        lines.append("No hotspots found.")
        return "\n".join(lines)

    # Summary
    risk_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for h in result.hotspots:
        risk_counts[h.risk_level] += 1

    lines.append(f"Top {len(result.hotspots)} hotspots "
                 f"(critical: {risk_counts['critical']}, high: {risk_counts['high']}, "
                 f"medium: {risk_counts['medium']}, low: {risk_counts['low']}):")
    lines.append("")

    # Header
    lines.append(f"{'#':>3}  {'Score':>6}  {'Risk':>8}  {'Category':>18}  {'CC':>4}  {'Churn':>6}  {'Commits':>7}  File")
    lines.append(f"{'─' * 3}  {'─' * 6}  {'─' * 8}  {'─' * 18}  {'─' * 4}  {'─' * 6}  {'─' * 7}  {'─' * 40}")

    for i, h in enumerate(result.hotspots, 1):
        risk_marker = {"critical": "!!!", "high": " !!", "medium": "  !", "low": "   "}
        category = classify_hotspot_category(h)
        lines.append(
            f"{i:>3}  {h.hotspot_score:>6.3f}  {h.risk_level:>8}{risk_marker[h.risk_level]}  "
            f"{category:>18}  "
            f"{h.complexity.max_cc:>4}  {h.churn.churn_score:>6.1f}  "
            f"{h.churn.commits:>7}  {h.filepath}"
        )

    # Recommendations
    critical = [h for h in result.hotspots if h.risk_level == "critical"]
    high = [h for h in result.hotspots if h.risk_level == "high"]

    if critical or high:
        lines.append("")
        lines.append("=== Recommendations ===")
        if critical:
            lines.append(f"CRITICAL ({len(critical)} files): Refactor immediately — "
                         "high complexity + frequent changes = bugs waiting to happen")
            for h in critical:
                lines.append(f"  - {h.filepath} (CC={h.complexity.max_cc}, "
                             f"{h.churn.commits} commits, +{h.churn.lines_added}/-{h.churn.lines_deleted})")
        if high:
            lines.append(f"HIGH ({len(high)} files): Schedule refactoring — "
                         "significant risk accumulation")
            for h in high[:5]:  # Top 5 high-risk
                lines.append(f"  - {h.filepath} (CC={h.complexity.max_cc}, "
                             f"{h.churn.commits} commits)")

    return "\n".join(lines)


def format_json(result: AnalysisResult) -> str:
    """Format analysis result as JSON."""
    data = {
        "repo_path": result.repo_path,
        "total_python_files": result.total_python_files,
        "total_files_analyzed": result.total_files_analyzed,
        "since_days": result.since_days,
        "errors": result.errors,
        "hotspots": [
            {
                "filepath": h.filepath,
                "hotspot_score": round(h.hotspot_score, 4),
                "risk_level": h.risk_level,
                "category": classify_hotspot_category(h),
                "churn_normalized": round(h.churn_normalized, 4),
                "complexity_normalized": round(h.complexity_normalized, 4),
                "complexity": {
                    "max_cc": h.complexity.max_cc,
                    "total_cc": h.complexity.total_cc,
                    "num_functions": h.complexity.num_functions,
                    "longest_function": h.complexity.longest_function,
                },
                "churn": {
                    "commits": h.churn.commits,
                    "lines_added": h.churn.lines_added,
                    "lines_deleted": h.churn.lines_deleted,
                    "churn_score": round(h.churn.churn_score, 2),
                },
            }
            for h in result.hotspots
        ],
    }
    return json.dumps(data, indent=2)


def format_score_text(result: AnalysisResult) -> str:
    """Format project score as compact text summary."""
    stats = compute_project_stats(result)
    lines = []
    lines.append(f"=== Hotspot Health Score ===")
    lines.append(f"Repository: {result.repo_path}")
    lines.append(f"Score: {stats.score}/100 (Grade: {stats.grade})")
    lines.append(f"Files analyzed: {stats.total_files}")
    lines.append(f"Hotspot density: {stats.hotspot_density_pct:.1f}% of files are medium+ risk")
    lines.append(f"Risk breakdown: {stats.critical_count} critical, {stats.high_count} high, "
                 f"{stats.medium_count} medium, {stats.low_count} low")

    if stats.max_hotspot_score > 0:
        lines.append(f"Max hotspot score: {stats.max_hotspot_score:.3f}")
        lines.append(f"Avg hotspot score: {stats.avg_hotspot_score:.3f}")

    return "\n".join(lines)


def format_score_json(result: AnalysisResult) -> str:
    """Format project score as JSON."""
    stats = compute_project_stats(result)
    data = {
        "repo_path": result.repo_path,
        "score": stats.score,
        "grade": stats.grade,
        "total_files": stats.total_files,
        "hotspot_density_pct": round(stats.hotspot_density_pct, 2),
        "critical_count": stats.critical_count,
        "high_count": stats.high_count,
        "medium_count": stats.medium_count,
        "low_count": stats.low_count,
        "avg_hotspot_score": round(stats.avg_hotspot_score, 4),
        "max_hotspot_score": round(stats.max_hotspot_score, 4),
    }
    return json.dumps(data, indent=2)


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Code hotspot detector — finds files that are both complex AND frequently changed"
    )
    parser.add_argument("path", nargs="?", default=".",
                        help="Repository path to analyze (default: current directory)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of top hotspots to show (default: {DEFAULT_TOP_N})")
    parser.add_argument("--since", type=int, default=DEFAULT_SINCE_DAYS,
                        help=f"Analyze commits from last N days (default: {DEFAULT_SINCE_DAYS})")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--score", action="store_true",
                        help="Show project health score only (0-100 + grade)")

    args = parser.parse_args()
    result = analyze(args.path, top_n=args.top, since_days=args.since)

    if args.score:
        if args.format == "json":
            print(format_score_json(result))
        else:
            print(format_score_text(result))
    elif args.format == "json":
        print(format_json(result))
    else:
        print(format_text(result))


if __name__ == "__main__":
    main()
