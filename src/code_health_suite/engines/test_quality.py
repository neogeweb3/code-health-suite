#!/usr/bin/env python3
"""ai-test-quality: AST-based Python test suite quality analyzer.

Detects common test anti-patterns and scores test quality. Zero external
dependencies — pure stdlib AST analysis.

Checks:
    - no_assertions:    Test function with no assert/self.assert* calls
    - empty_test:       Test body is just `pass` or `...`
    - too_long:         Test function exceeds line threshold
    - too_many_asserts: Single test with excessive assertions (testing too much)
    - duplicate_name:   Same test function name appears multiple times
    - bare_assert:      Useless `assert True` / `assert 1`
    - broad_except:     try/except Exception swallows test failures
    - sleep_in_test:    time.sleep() — flaky test indicator
    - no_description:   Generic name like test_1, test_it, test_func

Usage:
    ai-test-quality                        # analyze current directory
    ai-test-quality path/to/tests          # analyze specific path
    ai-test-quality -f test_foo.py         # single file
    ai-test-quality --json                 # JSON output
    ai-test-quality --threshold 70         # fail if score < 70
    ai-test-quality --severity high        # only show high-severity issues
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

__version__ = "0.1.0"

# --- Configuration ---

DEFAULT_MAX_TEST_LENGTH = 50  # lines
DEFAULT_MAX_ASSERTIONS = 15
DEFAULT_SCORE_THRESHOLD = 0  # no threshold by default

SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages", "dist", "build", "egg-info",
}

VENV_PATTERNS = {".venv", "venv", ".env", "env"}

# Generic test names that don't describe behavior
GENERIC_NAME_RE = re.compile(
    r"^test_?\d*$|^test_it$|^test_func$|^test_method$|^test_case$|^test_something$"
)

# Severity weights for scoring (higher = worse)
SEVERITY_WEIGHTS = {
    "high": 5,
    "medium": 3,
    "low": 1,
}


# --- Data models ---


@dataclass
class TestIssue:
    """A single quality issue found in a test."""
    check: str
    severity: str  # high, medium, low
    message: str
    line: int
    function: str
    file: str


@dataclass
class TestFunctionInfo:
    """Metrics for a single test function."""
    name: str
    qualified_name: str  # TestClass.test_method or bare test_func
    file: str
    line: int
    end_line: int
    length: int
    assertion_count: int
    issues: list[TestIssue] = field(default_factory=list)


@dataclass
class FileReport:
    """Analysis report for a single test file."""
    file: str
    test_count: int
    total_assertions: int
    assertion_density: float  # avg assertions per test
    issues: list[TestIssue] = field(default_factory=list)
    tests: list[TestFunctionInfo] = field(default_factory=list)


@dataclass
class SuiteReport:
    """Overall test suite quality report."""
    files_analyzed: int
    total_tests: int
    total_assertions: int
    total_issues: int
    issues_by_severity: dict[str, int] = field(default_factory=dict)
    issues_by_check: dict[str, int] = field(default_factory=dict)
    score: int = 100  # starts at 100, deductions per issue
    grade: str = "A"
    files: list[FileReport] = field(default_factory=list)


# --- AST analysis helpers ---


def _count_assertions(node: ast.AST) -> int:
    """Count assert statements and self.assert*/pytest-style calls in a function."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            count += 1
        elif isinstance(child, ast.Call):
            name = _get_call_name(child)
            if name and ("assert" in name.lower()):
                count += 1
    return count


def _get_call_name(node: ast.Call) -> Optional[str]:
    """Extract function/method name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_empty_body(body: list[ast.stmt]) -> bool:
    """Check if a function body is just pass/... or a docstring + pass/..."""
    stmts = [s for s in body if not isinstance(s, ast.Expr)
             or not isinstance(s.value, (ast.Constant, ast.Str))]
    if not stmts:
        return True  # only docstring or empty
    return all(isinstance(s, ast.Pass) for s in stmts)


def _is_bare_assert(node: ast.Assert) -> bool:
    """Check if an assert is useless: assert True, assert 1, assert "string"."""
    if isinstance(node.test, ast.Constant):
        return bool(node.test.value)  # assert True, assert 1, assert "x"
    if isinstance(node.test, ast.NameConstant):  # Python 3.7
        return node.test.value is True
    return False


def _has_broad_except(node: ast.AST) -> list[int]:
    """Find try/except Exception or bare except in a function body."""
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.ExceptHandler):
            if child.type is None:  # bare except:
                lines.append(child.lineno)
            elif isinstance(child.type, ast.Name) and child.type.id == "Exception":
                lines.append(child.lineno)
    return lines


def _has_sleep_call(node: ast.AST) -> list[int]:
    """Find time.sleep() calls in a function."""
    lines = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _get_call_name(child)
            if name == "sleep":
                lines.append(child.lineno)
    return lines


def _is_test_function(node: ast.AST) -> bool:
    """Check if a node is a test function (starts with test_)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test_") or node.name == "test"
    return False


def _is_test_class(node: ast.AST) -> bool:
    """Check if a node is a test class (starts with Test)."""
    if isinstance(node, ast.ClassDef):
        return node.name.startswith("Test")
    return False


# --- Core analysis ---


def analyze_test_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    filepath: str,
    class_name: Optional[str] = None,
    max_length: int = DEFAULT_MAX_TEST_LENGTH,
    max_assertions: int = DEFAULT_MAX_ASSERTIONS,
) -> TestFunctionInfo:
    """Analyze a single test function for quality issues."""
    qualified = f"{class_name}.{func_node.name}" if class_name else func_node.name
    end_line = getattr(func_node, "end_lineno", func_node.lineno)
    length = end_line - func_node.lineno + 1
    assertion_count = _count_assertions(func_node)

    info = TestFunctionInfo(
        name=func_node.name,
        qualified_name=qualified,
        file=filepath,
        line=func_node.lineno,
        end_line=end_line,
        length=length,
        assertion_count=assertion_count,
    )

    # Check: empty_test
    if _is_empty_body(func_node.body):
        info.issues.append(TestIssue(
            check="empty_test", severity="high",
            message="Test body is empty (pass/...)",
            line=func_node.lineno, function=qualified, file=filepath,
        ))

    # Check: no_assertions (skip if empty — already flagged)
    elif assertion_count == 0:
        info.issues.append(TestIssue(
            check="no_assertions", severity="high",
            message="Test has no assertions — may silently pass",
            line=func_node.lineno, function=qualified, file=filepath,
        ))

    # Check: too_long
    if length > max_length:
        info.issues.append(TestIssue(
            check="too_long", severity="medium",
            message=f"Test is {length} lines (threshold: {max_length})",
            line=func_node.lineno, function=qualified, file=filepath,
        ))

    # Check: too_many_asserts
    if assertion_count > max_assertions:
        info.issues.append(TestIssue(
            check="too_many_asserts", severity="medium",
            message=f"Test has {assertion_count} assertions (threshold: {max_assertions}) — consider splitting",
            line=func_node.lineno, function=qualified, file=filepath,
        ))

    # Check: bare_assert
    for child in ast.walk(func_node):
        if isinstance(child, ast.Assert) and _is_bare_assert(child):
            info.issues.append(TestIssue(
                check="bare_assert", severity="high",
                message="Useless assertion: `assert True` / `assert <constant>`",
                line=child.lineno, function=qualified, file=filepath,
            ))
            break  # one is enough

    # Check: broad_except
    for exc_line in _has_broad_except(func_node):
        info.issues.append(TestIssue(
            check="broad_except", severity="medium",
            message="Broad except clause may swallow test failures",
            line=exc_line, function=qualified, file=filepath,
        ))

    # Check: sleep_in_test
    for sleep_line in _has_sleep_call(func_node):
        info.issues.append(TestIssue(
            check="sleep_in_test", severity="low",
            message="time.sleep() in test — may cause flakiness",
            line=sleep_line, function=qualified, file=filepath,
        ))

    # Check: no_description (generic name)
    if GENERIC_NAME_RE.match(func_node.name):
        info.issues.append(TestIssue(
            check="no_description", severity="low",
            message=f"Generic test name '{func_node.name}' — use descriptive names",
            line=func_node.lineno, function=qualified, file=filepath,
        ))

    return info


def analyze_file(
    filepath: str,
    max_length: int = DEFAULT_MAX_TEST_LENGTH,
    max_assertions: int = DEFAULT_MAX_ASSERTIONS,
) -> Optional[FileReport]:
    """Analyze a single test file."""
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return None

    tests: list[TestFunctionInfo] = []
    seen_names: dict[str, int] = {}  # name -> first line

    # Collect all test functions (top-level and in classes)
    for node in ast.iter_child_nodes(tree):
        if _is_test_function(node):
            info = analyze_test_function(node, filepath, None, max_length, max_assertions)
            tests.append(info)
            _track_duplicate(seen_names, node.name, node.lineno, info, filepath)

        elif _is_test_class(node):
            for child in ast.iter_child_nodes(node):
                if _is_test_function(child):
                    info = analyze_test_function(
                        child, filepath, node.name, max_length, max_assertions
                    )
                    tests.append(info)
                    qualified = f"{node.name}.{child.name}"
                    _track_duplicate(seen_names, qualified, child.lineno, info, filepath)

    if not tests:
        return None  # not a test file

    all_issues = []
    total_assertions = 0
    for t in tests:
        all_issues.extend(t.issues)
        total_assertions += t.assertion_count

    density = total_assertions / len(tests) if tests else 0.0

    return FileReport(
        file=filepath,
        test_count=len(tests),
        total_assertions=total_assertions,
        assertion_density=round(density, 1),
        issues=all_issues,
        tests=tests,
    )


def _track_duplicate(
    seen: dict[str, int], name: str, line: int,
    info: TestFunctionInfo, filepath: str,
) -> None:
    """Track and flag duplicate test names."""
    if name in seen:
        info.issues.append(TestIssue(
            check="duplicate_name", severity="medium",
            message=f"Duplicate test name '{name}' (first at line {seen[name]})",
            line=line, function=info.qualified_name, file=filepath,
        ))
    else:
        seen[name] = line


# --- File discovery ---


def _is_venv(path: Path) -> bool:
    """Check if path looks like a virtual environment."""
    return path.name in VENV_PATTERNS and (path / "pyvenv.cfg").exists()


def _is_test_file(path: Path) -> bool:
    """Check if a file looks like a test file."""
    name = path.name
    return (name.startswith("test_") or name.endswith("_test.py") or
            name == "conftest.py") and name.endswith(".py")


def discover_test_files(root: str) -> list[str]:
    """Find all Python test files under root."""
    root_path = Path(root)

    if root_path.is_file():
        return [str(root_path)] if root_path.suffix == ".py" else []

    files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        current = Path(dirpath)
        # Skip unwanted directories
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not _is_venv(current / d)
        ]
        for fname in filenames:
            fpath = current / fname
            if _is_test_file(fpath):
                files.append(str(fpath))
    return sorted(files)


# --- Scoring ---


def compute_score(total_tests: int, issues: list[TestIssue]) -> tuple[int, str]:
    """Compute quality score (0-100) and letter grade."""
    if total_tests == 0:
        return 100, "A"

    deduction = 0
    for issue in issues:
        weight = SEVERITY_WEIGHTS.get(issue.severity, 1)
        deduction += weight

    # Normalize: max deduction proportional to test count
    max_deduction = total_tests * 3  # 3 points per test = harsh but fair
    if max_deduction == 0:
        max_deduction = 1
    score = max(0, 100 - int(100 * deduction / max_deduction))

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "F"

    return score, grade


# --- Suite analysis ---


def analyze_suite(
    root: str,
    max_length: int = DEFAULT_MAX_TEST_LENGTH,
    max_assertions: int = DEFAULT_MAX_ASSERTIONS,
    severity_filter: Optional[str] = None,
) -> SuiteReport:
    """Analyze an entire test suite."""
    files = discover_test_files(root)
    file_reports: list[FileReport] = []
    all_issues: list[TestIssue] = []
    total_tests = 0
    total_assertions = 0

    for fpath in files:
        report = analyze_file(fpath, max_length, max_assertions)
        if report is None:
            continue
        file_reports.append(report)
        total_tests += report.test_count
        total_assertions += report.total_assertions
        all_issues.extend(report.issues)

    # Apply severity filter
    if severity_filter:
        filtered = [i for i in all_issues if i.severity == severity_filter]
    else:
        filtered = all_issues

    # Count by severity and check
    by_severity: dict[str, int] = {}
    by_check: dict[str, int] = {}
    for issue in filtered:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        by_check[issue.check] = by_check.get(issue.check, 0) + 1

    score, grade = compute_score(total_tests, all_issues)

    return SuiteReport(
        files_analyzed=len(file_reports),
        total_tests=total_tests,
        total_assertions=total_assertions,
        total_issues=len(filtered),
        issues_by_severity=by_severity,
        issues_by_check=by_check,
        score=score,
        grade=grade,
        files=file_reports,
    )


# --- Output formatting ---


def format_text(report: SuiteReport, verbose: bool = False) -> str:
    """Format report as human-readable text."""
    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  Test Quality Report — Score: {report.score}/100 ({report.grade})")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Files: {report.files_analyzed}  |  Tests: {report.total_tests}"
                 f"  |  Assertions: {report.total_assertions}"
                 f"  |  Issues: {report.total_issues}")
    if report.total_tests:
        density = report.total_assertions / report.total_tests
        lines.append(f"  Assertion density: {density:.1f} per test")
    lines.append("")

    if report.issues_by_check:
        lines.append("  Issues by check:")
        for check, count in sorted(report.issues_by_check.items(), key=lambda x: -x[1]):
            lines.append(f"    {check:20s}  {count}")
        lines.append("")

    # Per-file issues
    for fr in report.files:
        file_issues = fr.issues
        if not file_issues and not verbose:
            continue
        rel = _relative_path(fr.file)
        lines.append(f"  {rel} ({fr.test_count} tests, {fr.assertion_density} assert/test)")
        for issue in file_issues:
            sev_icon = {"high": "!", "medium": "~", "low": "."}[issue.severity]
            lines.append(f"    [{sev_icon}] L{issue.line} {issue.function}: {issue.message}")
        lines.append("")

    if not report.total_issues:
        lines.append("  No issues found. Test suite looks clean!")
        lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def format_json(report: SuiteReport) -> str:
    """Format report as JSON."""
    data = {
        "files_analyzed": report.files_analyzed,
        "total_tests": report.total_tests,
        "total_assertions": report.total_assertions,
        "total_issues": report.total_issues,
        "issues_by_severity": report.issues_by_severity,
        "issues_by_check": report.issues_by_check,
        "score": report.score,
        "grade": report.grade,
        "files": [
            {
                "file": _relative_path(fr.file),
                "test_count": fr.test_count,
                "total_assertions": fr.total_assertions,
                "assertion_density": fr.assertion_density,
                "issues": [asdict(i) for i in fr.issues],
            }
            for fr in report.files
            if fr.issues  # only include files with issues in JSON
        ],
    }
    return json.dumps(data, indent=2)


def _relative_path(path: str) -> str:
    """Convert to relative path if possible."""
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except ValueError:
        return path


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-test-quality",
        description="AST-based Python test suite quality analyzer",
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="directory or file to analyze (default: current directory)",
    )
    parser.add_argument(
        "-f", "--file", dest="single_file",
        help="analyze a single file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="output as JSON",
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_SCORE_THRESHOLD,
        help="exit with code 1 if score < threshold (default: 0 = no threshold)",
    )
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_TEST_LENGTH,
        help=f"max test function length in lines (default: {DEFAULT_MAX_TEST_LENGTH})",
    )
    parser.add_argument(
        "--max-assertions", type=int, default=DEFAULT_MAX_ASSERTIONS,
        help=f"max assertions per test (default: {DEFAULT_MAX_ASSERTIONS})",
    )
    parser.add_argument(
        "--severity", choices=["high", "medium", "low"],
        help="only show issues of this severity",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="show all files including those with no issues",
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target = args.single_file or args.path
    report = analyze_suite(
        target,
        max_length=args.max_length,
        max_assertions=args.max_assertions,
        severity_filter=args.severity,
    )

    if args.json:
        print(format_json(report))
    else:
        print(format_text(report, verbose=args.verbose))

    if args.threshold and report.score < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
