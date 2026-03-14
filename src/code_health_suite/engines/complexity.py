#!/usr/bin/env python3
"""ai-complexity: AST-based Python code complexity analyzer.

Measures cyclomatic complexity, cognitive complexity, function length, and
nesting depth for Python code. Zero external dependencies.

Usage:
    ai-complexity                         # analyze current directory
    ai-complexity path/to/project         # analyze specific directory
    ai-complexity -f specific_file.py     # analyze single file
    ai-complexity --json                  # JSON output
    ai-complexity --threshold 10          # flag functions with CC >= 10
    ai-complexity --sort complexity       # sort by cyclomatic complexity
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Import shared scope-safe AST traversal (sibling ai-ast-utils package)
# sys.path hack removed — using package import
from code_health_suite.ast_utils import walk_scope_bfs as _walk_skip_nested  # noqa: E402


__version__ = "0.2.0"

# --- Thresholds ---

DEFAULT_CYCLOMATIC_THRESHOLD = 10  # McCabe standard
DEFAULT_COGNITIVE_THRESHOLD = 15
DEFAULT_LENGTH_THRESHOLD = 50  # lines

COMPLEXITY_RATINGS = {
    (0, 5): ("A", "simple"),
    (6, 10): ("B", "moderate"),
    (11, 20): ("C", "complex"),
    (21, 50): ("D", "very complex"),
    (51, float("inf")): ("F", "untestable"),
}

# Directories to always skip
SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", "dist", "build", "egg-info",
}

# Virtual environment patterns
VENV_PATTERNS = {".venv", "venv", ".env", "env"}


# --- Data models ---


@dataclass
class FunctionMetrics:
    """Metrics for a single function or method."""
    file: str
    name: str
    qualified_name: str  # Class.method or module-level function
    line: int
    end_line: int
    cyclomatic: int = 1  # starts at 1 (base path)
    cognitive: int = 0
    length: int = 0
    max_nesting: int = 0
    parameter_count: int = 0
    is_method: bool = False

    @property
    def grade(self) -> str:
        for (low, high), (grade, _) in COMPLEXITY_RATINGS.items():
            if low <= self.cyclomatic <= high:
                return grade
        return "F"

    @property
    def grade_label(self) -> str:
        for (low, high), (_, label) in COMPLEXITY_RATINGS.items():
            if low <= self.cyclomatic <= high:
                return label
        return "untestable"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grade"] = self.grade
        d["grade_label"] = self.grade_label
        return d


@dataclass
class ModuleMetrics:
    """Metrics for a single Python module (file)."""
    file: str
    total_lines: int = 0
    code_lines: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    avg_cyclomatic: float = 0.0
    max_cyclomatic: int = 0
    avg_cognitive: float = 0.0
    max_cognitive: int = 0
    functions: list[FunctionMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "file": self.file,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "avg_cyclomatic": round(self.avg_cyclomatic, 2),
            "max_cyclomatic": self.max_cyclomatic,
            "avg_cognitive": round(self.avg_cognitive, 2),
            "max_cognitive": self.max_cognitive,
            "functions": [f.to_dict() for f in self.functions],
        }
        return d


@dataclass
class AnalysisResult:
    """Overall analysis result."""
    files_analyzed: int = 0
    total_functions: int = 0
    modules: list[ModuleMetrics] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_functions(self) -> list[FunctionMetrics]:
        result = []
        for m in self.modules:
            result.extend(m.functions)
        return result

    @property
    def violations(self) -> list[FunctionMetrics]:
        """Functions exceeding any threshold (set externally)."""
        return []  # computed at output time

    def to_dict(self) -> dict:
        all_fns = self.all_functions
        return {
            "files_analyzed": self.files_analyzed,
            "total_functions": self.total_functions,
            "modules": [m.to_dict() for m in self.modules],
            "errors": self.errors,
            "summary": self._summary_dict(all_fns),
        }

    def _summary_dict(self, fns: list[FunctionMetrics]) -> dict:
        if not fns:
            return {"avg_cyclomatic": 0, "max_cyclomatic": 0,
                    "avg_cognitive": 0, "max_cognitive": 0,
                    "grade_distribution": {}}
        cc_vals = [f.cyclomatic for f in fns]
        cog_vals = [f.cognitive for f in fns]
        grades: dict[str, int] = {}
        for f in fns:
            g = f.grade
            grades[g] = grades.get(g, 0) + 1
        return {
            "avg_cyclomatic": round(sum(cc_vals) / len(cc_vals), 2),
            "max_cyclomatic": max(cc_vals),
            "avg_cognitive": round(sum(cog_vals) / len(cog_vals), 2),
            "max_cognitive": max(cog_vals),
            "grade_distribution": grades,
        }


# --- Project Complexity Score ---

GRADE_PENALTIES = {"A": 0, "B": 1, "C": 3, "D": 8, "F": 20}

COMPLEXITY_PROFILES = (
    "clean",              # mostly A/B, low avg complexity
    "cognitive_heavy",    # cognitive violations > cyclomatic violations
    "cyclomatic_heavy",   # cyclomatic violations > cognitive violations
    "deep_nesting",       # nesting depth is the dominant issue
    "god_functions",      # few functions with extreme complexity (D/F)
    "uniformly_complex",  # complexity spread across many functions
    "mixed",              # multiple issue types
)


@dataclass
class ComplexityScore:
    """Project-level complexity health score."""
    score: int  # 0-100
    grade: str  # A-F
    profile: str  # from COMPLEXITY_PROFILES
    total_functions: int
    files_analyzed: int
    avg_cyclomatic: float
    avg_cognitive: float
    violations_count: int
    top_offenders: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "grade": self.grade,
            "profile": self.profile,
            "total_functions": self.total_functions,
            "files_analyzed": self.files_analyzed,
            "avg_cyclomatic": round(self.avg_cyclomatic, 2),
            "avg_cognitive": round(self.avg_cognitive, 2),
            "violations_count": self.violations_count,
            "top_offenders": self.top_offenders,
        }


def compute_complexity_score(
    result: AnalysisResult,
    cc_threshold: int = DEFAULT_CYCLOMATIC_THRESHOLD,
    cog_threshold: int = DEFAULT_COGNITIVE_THRESHOLD,
    len_threshold: int = DEFAULT_LENGTH_THRESHOLD,
) -> ComplexityScore:
    """Compute a 0-100 project complexity health score.

    Scoring: starts at 100, deducts penalties per function based on grade.
    Penalties are density-normalized (per-function average).
    """
    all_fns = result.all_functions
    if not all_fns:
        return ComplexityScore(
            score=100, grade="A", profile="clean",
            total_functions=0, files_analyzed=result.files_analyzed,
            avg_cyclomatic=0.0, avg_cognitive=0.0,
            violations_count=0, top_offenders=[],
        )

    # Compute raw penalty from grade distribution
    total_penalty = sum(GRADE_PENALTIES.get(f.grade, 20) for f in all_fns)

    # Density-normalize: penalty per function, then scale
    density_penalty = (total_penalty / len(all_fns)) * 20

    # Bonus penalty for extreme outliers (D/F functions)
    extreme_count = sum(1 for f in all_fns if f.grade in ("D", "F"))
    extreme_penalty = min(extreme_count * 5, 30)

    score = max(0, min(100, round(100 - density_penalty - extreme_penalty)))

    # Grade from score
    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 25:
        grade = "D"
    else:
        grade = "F"

    # Violations
    violations = [
        f for f in all_fns
        if f.cyclomatic >= cc_threshold or f.cognitive >= cog_threshold
        or f.length >= len_threshold
    ]

    # Averages
    avg_cc = sum(f.cyclomatic for f in all_fns) / len(all_fns)
    avg_cog = sum(f.cognitive for f in all_fns) / len(all_fns)

    # Top offenders (top 5 by cyclomatic)
    sorted_fns = sorted(all_fns, key=lambda f: f.cyclomatic, reverse=True)
    top_offenders = [
        {
            "function": f.qualified_name,
            "file": f.file,
            "line": f.line,
            "cyclomatic": f.cyclomatic,
            "cognitive": f.cognitive,
            "grade": f.grade,
        }
        for f in sorted_fns[:5]
    ]

    profile = classify_complexity_profile(all_fns, cc_threshold, cog_threshold)

    return ComplexityScore(
        score=score,
        grade=grade,
        profile=profile,
        total_functions=len(all_fns),
        files_analyzed=result.files_analyzed,
        avg_cyclomatic=avg_cc,
        avg_cognitive=avg_cog,
        violations_count=len(violations),
        top_offenders=top_offenders,
    )


def classify_complexity_profile(
    functions: list[FunctionMetrics],
    cc_threshold: int = DEFAULT_CYCLOMATIC_THRESHOLD,
    cog_threshold: int = DEFAULT_COGNITIVE_THRESHOLD,
) -> str:
    """Classify project complexity into a named profile."""
    if not functions:
        return "clean"

    n = len(functions)
    cc_violations = [f for f in functions if f.cyclomatic >= cc_threshold]
    cog_violations = [f for f in functions if f.cognitive >= cog_threshold]
    nesting_violations = [f for f in functions if f.max_nesting >= 4]
    extreme = [f for f in functions if f.grade in ("D", "F")]
    ab_count = sum(1 for f in functions if f.grade in ("A", "B"))

    # Clean: >90% A/B functions, no extreme
    if ab_count / n > 0.9 and not extreme:
        return "clean"

    # God functions: few extreme outliers dominate
    if extreme and len(extreme) <= max(3, n * 0.05):
        return "god_functions"

    # Deep nesting: nesting violations > other violations
    if (len(nesting_violations) > len(cc_violations)
            and len(nesting_violations) > len(cog_violations)):
        return "deep_nesting"

    # Cognitive heavy: cognitive violations dominate
    if len(cog_violations) > len(cc_violations) * 1.5:
        return "cognitive_heavy"

    # Cyclomatic heavy: cyclomatic violations dominate
    if len(cc_violations) > len(cog_violations) * 1.5:
        return "cyclomatic_heavy"

    # Uniformly complex: many functions are C+ grade
    c_plus = sum(1 for f in functions if f.grade in ("C", "D", "F"))
    if c_plus / n > 0.3:
        return "uniformly_complex"

    # Mixed: multiple issue types
    if cc_violations and cog_violations:
        return "mixed"

    return "clean"


def format_score_text(score: ComplexityScore) -> str:
    """Format complexity score for terminal display."""
    lines = [
        f"\n{BOLD}Complexity Health Score{RESET}",
        f"  Score: {BOLD}{score.score}/100{RESET} ({score.grade})",
        f"  Profile: {score.profile}",
        f"  Functions: {score.total_functions} across {score.files_analyzed} files",
        f"  Avg cyclomatic: {score.avg_cyclomatic:.1f}  |  Avg cognitive: {score.avg_cognitive:.1f}",
        f"  Violations: {score.violations_count}",
    ]
    if score.top_offenders:
        lines.append(f"\n  {BOLD}Top offenders:{RESET}")
        for o in score.top_offenders:
            lines.append(
                f"    [{o['grade']}] {o['file']}:{o['line']} "
                f"{o['function']} (CC={o['cyclomatic']}, Cog={o['cognitive']})"
            )
    lines.append("")
    return "\n".join(lines)


def format_score_json(score: ComplexityScore) -> str:
    """Format complexity score as JSON."""
    return json.dumps(score.to_dict(), indent=2)


# --- Cyclomatic Complexity Calculator ---

# Nodes that add a decision point (branching)
CYCLOMATIC_INCREMENTORS = (
    ast.If, ast.IfExp,  # if / ternary
    ast.For, ast.AsyncFor,  # for loops
    ast.While,  # while loops
    ast.ExceptHandler,  # except clauses
    ast.With, ast.AsyncWith,  # context managers (implicit try)
    ast.Assert,  # assert (implicit if)
)

# Boolean operators that add decision paths
BOOL_OPS = (ast.And, ast.Or)



def compute_cyclomatic(node: ast.AST) -> int:
    """Compute McCabe cyclomatic complexity for a function/method node."""
    complexity = 1  # base path

    for child in _walk_skip_nested(node):
        if isinstance(child, CYCLOMATIC_INCREMENTORS):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # Each boolean operator adds (n-1) paths
            # e.g., `a and b and c` = 2 additional paths
            complexity += len(child.values) - 1
        elif isinstance(child, ast.comprehension):
            # List/set/dict comprehensions with conditions
            complexity += len(child.ifs)
            complexity += 1  # the for itself

    return complexity


# --- Cognitive Complexity Calculator ---

# Structural increments (things that add nesting)
NESTING_INCREMENTORS = (
    ast.If, ast.For, ast.AsyncFor, ast.While,
    ast.ExceptHandler, ast.With, ast.AsyncWith,
)

# Fundamental increments (not nesting-related)
FUNDAMENTAL_INCREMENTORS = (
    ast.Break, ast.Continue,
)


def compute_cognitive(node: ast.AST) -> int:
    """Compute cognitive complexity for a function/method node.

    Based on SonarSource cognitive complexity specification:
    1. Structural: +1 for control flow breaks, +nesting for nested ones
    2. Fundamental: +1 for breaks in linear flow (break, continue, goto)
    3. Hybrid: +1 for boolean operator sequences
    """
    total = 0

    def _walk(node: ast.AST, nesting: int = 0) -> None:
        nonlocal total

        for child in ast.iter_child_nodes(node):
            # Skip nested function/class definitions
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Nested definitions add +1 structural but don't increment nesting
                total += 1
                _walk(child, nesting + 1)
                continue

            # Structural increment + nesting penalty
            if isinstance(child, NESTING_INCREMENTORS):
                total += 1 + nesting
                _walk(child, nesting + 1)
                continue

            # Else/elif handling
            if isinstance(child, ast.If):
                # Already handled above for If
                pass

            # Fundamental increments (no nesting penalty)
            if isinstance(child, FUNDAMENTAL_INCREMENTORS):
                total += 1

            # Boolean operator sequences
            if isinstance(child, ast.BoolOp):
                total += 1  # one increment per sequence

            # Ternary (IfExp) - structural, no nesting increase
            if isinstance(child, ast.IfExp):
                total += 1

            # Recursion for non-nesting nodes
            if not isinstance(child, NESTING_INCREMENTORS):
                _walk(child, nesting)

    # Walk the function body
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            _walk(child, 1)
        elif isinstance(child, NESTING_INCREMENTORS):
            total += 1  # +1 structural, nesting=0 at top level
            _walk(child, 1)
        else:
            if isinstance(child, FUNDAMENTAL_INCREMENTORS):
                total += 1
            if isinstance(child, ast.BoolOp):
                total += 1
            if isinstance(child, ast.IfExp):
                total += 1
            _walk(child, 0)

    return total


# --- Nesting Depth Calculator ---


def compute_max_nesting(node: ast.AST) -> int:
    """Compute the maximum nesting depth of a function body."""
    max_depth = 0

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While,
                                  ast.With, ast.AsyncWith, ast.Try,
                                  ast.ExceptHandler)):
                new_depth = depth + 1
                if new_depth > max_depth:
                    max_depth = new_depth
                _walk(child, new_depth)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Nested function resets nesting for its body
                _walk(child, 0)
            else:
                _walk(child, depth)

    _walk(node, 0)
    return max_depth


# --- Function Length Calculator ---


def compute_length(node: ast.AST) -> int:
    """Compute the number of lines in a function body (excluding decorators)."""
    if not hasattr(node, "end_lineno") or not hasattr(node, "lineno"):
        return 0
    # Subtract decorator lines
    start = node.lineno
    if hasattr(node, "decorator_list") and node.decorator_list:
        # Body starts after the last decorator + def line
        start = node.lineno
    return (node.end_lineno or node.lineno) - start + 1


# --- Parameter Count ---


def count_parameters(node: ast.AST) -> int:
    """Count the number of parameters in a function definition."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0
    args = node.args
    count = len(args.args) + len(args.posonlyargs) + len(args.kwonlyargs)
    if args.vararg:
        count += 1
    if args.kwarg:
        count += 1
    return count


# --- Module Analyzer ---


def analyze_module(file_path: str, source: str) -> ModuleMetrics:
    """Analyze a single Python module."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in {file_path}: {e}") from e

    lines = source.splitlines()
    total_lines = len(lines)
    code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))

    module = ModuleMetrics(
        file=file_path,
        total_lines=total_lines,
        code_lines=code_lines,
    )

    # Count imports and classes at module level
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module.import_count += 1
        elif isinstance(node, ast.ClassDef):
            module.class_count += 1

    # Extract all function/method definitions
    _extract_functions(tree, file_path, module, class_name=None)

    # Compute module-level aggregates
    if module.functions:
        cc_vals = [f.cyclomatic for f in module.functions]
        cog_vals = [f.cognitive for f in module.functions]
        module.function_count = len(module.functions)
        module.avg_cyclomatic = sum(cc_vals) / len(cc_vals)
        module.max_cyclomatic = max(cc_vals)
        module.avg_cognitive = sum(cog_vals) / len(cog_vals)
        module.max_cognitive = max(cog_vals)

    return module


def _extract_functions(
    node: ast.AST,
    file_path: str,
    module: ModuleMetrics,
    class_name: Optional[str],
) -> None:
    """Recursively extract function definitions from AST."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            _extract_functions(child, file_path, module, class_name=child.name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = child.name
            qualified = f"{class_name}.{name}" if class_name else name

            metrics = FunctionMetrics(
                file=file_path,
                name=name,
                qualified_name=qualified,
                line=child.lineno,
                end_line=child.end_lineno or child.lineno,
                cyclomatic=compute_cyclomatic(child),
                cognitive=compute_cognitive(child),
                length=compute_length(child),
                max_nesting=compute_max_nesting(child),
                parameter_count=count_parameters(child),
                is_method=class_name is not None,
            )
            module.functions.append(metrics)

            # Also check for nested classes inside methods
            _extract_functions(child, file_path, module, class_name=class_name)


# --- File Discovery ---


def _is_venv_path(path: str) -> bool:
    """Check if path is inside a virtual environment."""
    parts = Path(path).parts
    for part in parts:
        if part in VENV_PATTERNS or part.startswith(".venv") or part.endswith("_env"):
            return True
        if part == "site-packages":
            return True
    return False


def find_python_files(target: str) -> list[str]:
    """Find all Python files in a directory, excluding common non-project dirs."""
    target_path = Path(target).resolve()

    if target_path.is_file():
        return [str(target_path)]

    if not target_path.is_dir():
        return []

    files = []
    for root, dirs, filenames in os.walk(target_path):
        # Filter out skip directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not d.endswith(".egg-info")
            and not _is_venv_path(os.path.join(root, d))
        ]

        for fname in sorted(filenames):
            if fname.endswith(".py"):
                full_path = os.path.join(root, fname)
                if not _is_venv_path(full_path):
                    files.append(full_path)

    return sorted(files)


# --- Main Analysis ---


def analyze(target: str) -> AnalysisResult:
    """Analyze all Python files in the target path."""
    result = AnalysisResult()
    files = find_python_files(target)

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            module = analyze_module(fpath, source)
            result.modules.append(module)
            result.files_analyzed += 1
            result.total_functions += module.function_count
        except ValueError as e:
            result.errors.append(str(e))
        except OSError as e:
            result.errors.append(f"Cannot read {fpath}: {e}")

    return result


# --- Output Formatting ---

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _grade_color(grade: str) -> str:
    if grade == "A":
        return GREEN
    elif grade == "B":
        return GREEN
    elif grade == "C":
        return YELLOW
    elif grade == "D":
        return RED
    return RED + BOLD


def format_terminal(
    result: AnalysisResult,
    cc_threshold: int = DEFAULT_CYCLOMATIC_THRESHOLD,
    cog_threshold: int = DEFAULT_COGNITIVE_THRESHOLD,
    len_threshold: int = DEFAULT_LENGTH_THRESHOLD,
    sort_by: str = "complexity",
    top_n: int = 0,
) -> str:
    """Format analysis results for terminal display."""
    lines: list[str] = []
    all_fns = result.all_functions

    if not all_fns:
        lines.append(f"{DIM}No Python functions found.{RESET}")
        return "\n".join(lines)

    # Sort functions
    if sort_by == "complexity":
        all_fns.sort(key=lambda f: f.cyclomatic, reverse=True)
    elif sort_by == "cognitive":
        all_fns.sort(key=lambda f: f.cognitive, reverse=True)
    elif sort_by == "length":
        all_fns.sort(key=lambda f: f.length, reverse=True)
    elif sort_by == "nesting":
        all_fns.sort(key=lambda f: f.max_nesting, reverse=True)

    # Apply top_n filter
    display_fns = all_fns[:top_n] if top_n > 0 else all_fns

    # Header
    lines.append(f"\n{BOLD}ai-complexity v{__version__}{RESET}")
    lines.append(f"Analyzed {result.files_analyzed} files, {result.total_functions} functions\n")

    # Summary stats
    summary = result.to_dict()["summary"]
    lines.append(f"  Avg cyclomatic: {summary['avg_cyclomatic']:.1f}  |  Max: {summary['max_cyclomatic']}")
    lines.append(f"  Avg cognitive:  {summary['avg_cognitive']:.1f}  |  Max: {summary['max_cognitive']}")
    if summary.get("grade_distribution"):
        dist_str = "  ".join(f"{g}:{c}" for g, c in sorted(summary["grade_distribution"].items()))
        lines.append(f"  Grade distribution: {dist_str}")

    # Violations
    violations = [
        f for f in all_fns
        if f.cyclomatic >= cc_threshold or f.cognitive >= cog_threshold or f.length >= len_threshold
    ]

    if violations:
        lines.append(f"\n{RED}{BOLD}Violations ({len(violations)} functions exceed thresholds):{RESET}")
        lines.append(f"  {DIM}Thresholds: CC>={cc_threshold}, Cognitive>={cog_threshold}, Length>={len_threshold}{RESET}")
        for fn in violations:
            flags = []
            if fn.cyclomatic >= cc_threshold:
                flags.append(f"CC={fn.cyclomatic}")
            if fn.cognitive >= cog_threshold:
                flags.append(f"Cog={fn.cognitive}")
            if fn.length >= len_threshold:
                flags.append(f"Len={fn.length}")
            color = _grade_color(fn.grade)
            lines.append(
                f"  {color}[{fn.grade}]{RESET} {fn.file}:{fn.line} "
                f"{BOLD}{fn.qualified_name}{RESET} — {', '.join(flags)}"
            )

    # Function table
    lines.append(f"\n{BOLD}Top functions (sorted by {sort_by}):{RESET}")
    lines.append(f"  {'Grade':<6} {'CC':>4} {'Cog':>4} {'Len':>4} {'Nest':>4} {'Params':>6}  {'Function'}")
    lines.append(f"  {'─'*6} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*6}  {'─'*40}")

    for fn in display_fns:
        color = _grade_color(fn.grade)
        lines.append(
            f"  {color}  {fn.grade:<4}{RESET} {fn.cyclomatic:>4} {fn.cognitive:>4} "
            f"{fn.length:>4} {fn.max_nesting:>4} {fn.parameter_count:>6}  "
            f"{fn.file}:{fn.line} {fn.qualified_name}"
        )

    if top_n > 0 and len(all_fns) > top_n:
        lines.append(f"  {DIM}... and {len(all_fns) - top_n} more functions{RESET}")

    # Errors
    if result.errors:
        lines.append(f"\n{YELLOW}Parse errors ({len(result.errors)}):{RESET}")
        for err in result.errors:
            lines.append(f"  {err}")

    lines.append("")
    return "\n".join(lines)


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-complexity",
        description="AST-based Python code complexity analyzer",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to analyze (default: current directory)",
    )
    parser.add_argument(
        "-f", "--file",
        help="Analyze a single file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--threshold", "--cc",
        type=int,
        default=DEFAULT_CYCLOMATIC_THRESHOLD,
        dest="cc_threshold",
        help=f"Cyclomatic complexity threshold (default: {DEFAULT_CYCLOMATIC_THRESHOLD})",
    )
    parser.add_argument(
        "--cognitive-threshold",
        type=int,
        default=DEFAULT_COGNITIVE_THRESHOLD,
        dest="cog_threshold",
        help=f"Cognitive complexity threshold (default: {DEFAULT_COGNITIVE_THRESHOLD})",
    )
    parser.add_argument(
        "--length-threshold",
        type=int,
        default=DEFAULT_LENGTH_THRESHOLD,
        dest="len_threshold",
        help=f"Function length threshold in lines (default: {DEFAULT_LENGTH_THRESHOLD})",
    )
    parser.add_argument(
        "--sort",
        choices=["complexity", "cognitive", "length", "nesting"],
        default="complexity",
        help="Sort order for results (default: complexity)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        dest="top_n",
        help="Show only top N functions (default: all)",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Show project complexity health score (0-100)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        _disable_colors()

    target = args.file if args.file else args.path
    result = analyze(target)

    if args.score:
        score = compute_complexity_score(
            result,
            cc_threshold=args.cc_threshold,
            cog_threshold=args.cog_threshold,
            len_threshold=args.len_threshold,
        )
        if args.json_output:
            print(format_score_json(score))
        else:
            print(format_score_text(score))
        return 0

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        output = format_terminal(
            result,
            cc_threshold=args.cc_threshold,
            cog_threshold=args.cog_threshold,
            len_threshold=args.len_threshold,
            sort_by=args.sort,
            top_n=args.top_n,
        )
        print(output)

    # Exit code: 1 if any function exceeds thresholds
    all_fns = result.all_functions
    violations = [
        f for f in all_fns
        if f.cyclomatic >= args.cc_threshold
        or f.cognitive >= args.cog_threshold
        or f.length >= args.len_threshold
    ]
    return 1 if violations else 0


def _disable_colors() -> None:
    """Replace ANSI color codes with empty strings."""
    global RESET, BOLD, RED, YELLOW, GREEN, CYAN, DIM
    RESET = BOLD = RED = YELLOW = GREEN = CYAN = DIM = ""


if __name__ == "__main__":
    sys.exit(main())
