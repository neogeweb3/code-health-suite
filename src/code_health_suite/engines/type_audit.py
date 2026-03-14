#!/usr/bin/env python3
"""ai-type-audit: Python type annotation coverage analyzer.

Scans Python files to measure type annotation coverage, find untyped
functions, excessive Any usage, and type: ignore comments.

Usage:
    ai-type-audit                     # scan current directory
    ai-type-audit path/to/project     # scan specific directory
    ai-type-audit --json              # JSON output
    ai-type-audit --score             # health score (0-100 + A-F grade)
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


# --- Data models ---

@dataclass
class FunctionInfo:
    """Info about a function or method definition."""
    name: str
    file_path: str
    line_number: int
    is_method: bool = False
    has_return_annotation: bool = False
    total_params: int = 0  # excludes self/cls
    annotated_params: int = 0
    unannotated_param_names: tuple = ()

    @property
    def fully_typed(self) -> bool:
        return self.has_return_annotation and self.annotated_params == self.total_params

    @property
    def param_coverage(self) -> float:
        if self.total_params == 0:
            return 1.0
        return self.annotated_params / self.total_params


@dataclass
class AnyUsage:
    """A use of 'Any' type in annotations."""
    file_path: str
    line_number: int
    context: str


@dataclass
class TypeIgnore:
    """A '# type: ignore' comment."""
    file_path: str
    line_number: int
    context: str
    codes: str = ""  # e.g. "[assignment]"


@dataclass
class FileResult:
    """Results for a single file."""
    path: str
    functions: list = field(default_factory=list)
    any_usages: list = field(default_factory=list)
    type_ignores: list = field(default_factory=list)
    parse_error: str = ""


@dataclass
class AuditResult:
    """Overall audit result."""
    target: str
    files_scanned: int = 0
    files_with_issues: int = 0
    total_functions: int = 0
    fully_typed_functions: int = 0
    partially_typed_functions: int = 0
    untyped_functions: int = 0
    total_params: int = 0
    annotated_params: int = 0
    functions_with_return: int = 0
    any_count: int = 0
    type_ignore_count: int = 0
    parse_errors: int = 0
    file_results: list = field(default_factory=list)

    @property
    def function_coverage(self) -> float:
        if self.total_functions == 0:
            return 1.0
        return self.fully_typed_functions / self.total_functions

    @property
    def param_coverage(self) -> float:
        if self.total_params == 0:
            return 1.0
        return self.annotated_params / self.total_params

    @property
    def return_coverage(self) -> float:
        if self.total_functions == 0:
            return 1.0
        return self.functions_with_return / self.total_functions


# --- Configuration ---

SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules", ".tox",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env", ".env",
    "dist", "build", "egg-info", ".eggs", "site-packages",
}

SKIP_FUNCTION_NAMES = {
    "__repr__", "__str__", "__hash__", "__eq__", "__ne__", "__lt__",
    "__le__", "__gt__", "__ge__", "__bool__", "__len__", "__contains__",
    "__iter__", "__next__", "__enter__", "__exit__", "__del__",
    "__init_subclass__", "__class_getitem__",
}


# --- AST analysis ---

class TypeAnnotationVisitor(ast.NodeVisitor):
    """Walk AST to collect function signatures and Any usages."""

    def __init__(self, file_path: str, source_lines: list):
        self.file_path = file_path
        self.source_lines = source_lines
        self.functions: list[FunctionInfo] = []
        self.any_usages: list[AnyUsage] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = len(self._class_stack) > 0
        name = node.name

        # Skip dunder methods that rarely need annotations
        if name in SKIP_FUNCTION_NAMES:
            self.generic_visit(node)
            return

        # Check return annotation
        has_return = node.returns is not None

        # Check parameter annotations (skip self/cls for methods)
        args = node.args
        all_args = []
        for arg in args.posonlyargs:
            all_args.append(arg)
        for arg in args.args:
            all_args.append(arg)
        if args.vararg:
            all_args.append(args.vararg)
        for arg in args.kwonlyargs:
            all_args.append(arg)
        if args.kwarg:
            all_args.append(args.kwarg)

        # Skip self/cls
        skip_first = is_method and len(args.args) > 0 and args.args[0].arg in ("self", "cls")
        if skip_first and args.args[0] in all_args:
            all_args.remove(args.args[0])

        total_params = len(all_args)
        annotated = sum(1 for a in all_args if a.annotation is not None)
        unannotated = tuple(a.arg for a in all_args if a.annotation is None)

        # Check for Any in annotations
        self._check_any_in_annotations(node)

        func_name = f"{self._class_stack[-1]}.{name}" if self._class_stack else name
        info = FunctionInfo(
            name=func_name,
            file_path=self.file_path,
            line_number=node.lineno,
            is_method=is_method,
            has_return_annotation=has_return,
            total_params=total_params,
            annotated_params=annotated,
            unannotated_param_names=unannotated,
        )
        self.functions.append(info)

        # Continue visiting nested functions
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _check_any_in_annotations(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Check if Any is used in function annotations."""
        annotations_to_check = []
        if node.returns:
            annotations_to_check.append(node.returns)
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.annotation:
                annotations_to_check.append(arg.annotation)
        if node.args.vararg and node.args.vararg.annotation:
            annotations_to_check.append(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation:
            annotations_to_check.append(node.args.kwarg.annotation)

        for ann in annotations_to_check:
            for sub_node in ast.walk(ann):
                if isinstance(sub_node, ast.Name) and sub_node.id == "Any":
                    line = self.source_lines[sub_node.lineno - 1] if sub_node.lineno <= len(self.source_lines) else ""
                    self.any_usages.append(AnyUsage(
                        file_path=self.file_path,
                        line_number=sub_node.lineno,
                        context=line.strip(),
                    ))
                elif isinstance(sub_node, ast.Attribute) and sub_node.attr == "Any":
                    line = self.source_lines[sub_node.lineno - 1] if sub_node.lineno <= len(self.source_lines) else ""
                    self.any_usages.append(AnyUsage(
                        file_path=self.file_path,
                        line_number=sub_node.lineno,
                        context=line.strip(),
                    ))


def find_type_ignores(source: str, file_path: str) -> list[TypeIgnore]:
    """Find all # type: ignore comments in source."""
    results = []
    pattern = re.compile(r"#\s*type:\s*ignore(\[[\w,\s-]+\])?")
    for i, line in enumerate(source.splitlines(), 1):
        m = pattern.search(line)
        if m:
            results.append(TypeIgnore(
                file_path=file_path,
                line_number=i,
                context=line.strip(),
                codes=m.group(1) or "",
            ))
    return results


def analyze_file(file_path: str) -> FileResult:
    """Analyze a single Python file for type annotation coverage."""
    result = FileResult(path=file_path)
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        result.parse_error = str(e)
        return result

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        result.parse_error = f"SyntaxError: {e}"
        return result

    lines = source.splitlines()
    visitor = TypeAnnotationVisitor(file_path, lines)
    visitor.visit(tree)

    result.functions = visitor.functions
    result.any_usages = visitor.any_usages
    result.type_ignores = find_type_ignores(source, file_path)

    return result


# --- File discovery ---

def find_python_files(target: str) -> list[str]:
    """Find all Python files under target, respecting skip dirs."""
    target_path = Path(target).resolve()
    if target_path.is_file():
        return [str(target_path)] if target_path.suffix == ".py" else []

    files = []
    for root, dirs, filenames in os.walk(target_path):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))
    return files


# --- Aggregation ---

def aggregate_results(target: str, file_results: list[FileResult]) -> AuditResult:
    """Aggregate per-file results into an overall audit result."""
    result = AuditResult(target=target, file_results=file_results)

    for fr in file_results:
        result.files_scanned += 1
        if fr.parse_error:
            result.parse_errors += 1
            continue

        has_issue = False
        for func in fr.functions:
            result.total_functions += 1
            result.total_params += func.total_params
            result.annotated_params += func.annotated_params
            if func.has_return_annotation:
                result.functions_with_return += 1
            if func.fully_typed:
                result.fully_typed_functions += 1
            elif func.has_return_annotation or func.annotated_params > 0:
                result.partially_typed_functions += 1
                has_issue = True
            else:
                result.untyped_functions += 1
                has_issue = True

        result.any_count += len(fr.any_usages)
        result.type_ignore_count += len(fr.type_ignores)
        if fr.any_usages or fr.type_ignores:
            has_issue = True

        if has_issue:
            result.files_with_issues += 1

    return result


# --- Scoring ---

def compute_score(result: AuditResult) -> int:
    """Compute a 0-100 health score for type annotation coverage.

    Weights:
    - Function coverage (fully typed / total): 40%
    - Param coverage (annotated / total): 25%
    - Return coverage (has return / total): 25%
    - Deductions: Any usage (-1 per, max -5), type:ignore (-0.5 per, max -5)
    """
    if result.total_functions == 0:
        return 100

    fc = result.function_coverage * 40
    pc = result.param_coverage * 25
    rc = result.return_coverage * 25
    base = fc + pc + rc + 10  # +10 base for having code to analyze

    # Deductions
    any_penalty = min(result.any_count * 1.0, 5.0)
    ignore_penalty = min(result.type_ignore_count * 0.5, 5.0)

    score = base - any_penalty - ignore_penalty
    return max(0, min(100, round(score)))


def score_to_grade(score: int) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    return "F"


def classify_profile(result: AuditResult) -> str:
    """Classify the type annotation profile."""
    if result.total_functions == 0:
        return "no_functions"

    fc = result.function_coverage
    any_ratio = result.any_count / max(result.total_functions, 1)

    if fc >= 0.95 and any_ratio < 0.05:
        return "fully_typed"
    elif fc >= 0.80:
        return "well_typed"
    elif fc >= 0.50:
        return "partially_typed"
    elif any_ratio > 0.3:
        return "any_heavy"
    elif result.type_ignore_count > result.total_functions * 0.2:
        return "ignore_heavy"
    elif fc < 0.10:
        return "untyped"
    return "mixed"


PROFILE_DESCRIPTIONS = {
    "no_functions": "No functions found",
    "fully_typed": "Excellent — nearly all functions are fully annotated",
    "well_typed": "Good — most functions have type annotations",
    "partially_typed": "Fair — many functions lack complete annotations",
    "any_heavy": "Overuse of Any — annotations exist but lack precision",
    "ignore_heavy": "Heavy use of type: ignore — may indicate systemic type issues",
    "untyped": "Minimal — very few functions have type annotations",
    "mixed": "Mixed — inconsistent annotation practices",
}


# --- Output ---

def format_text(result: AuditResult) -> str:
    """Format results as human-readable text."""
    lines = []
    lines.append(f"ai-type-audit v{__version__} — {result.target}")
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append("")

    if result.total_functions == 0:
        lines.append("No Python functions found.")
        return "\n".join(lines)

    # Summary
    lines.append(f"Functions: {result.total_functions} total, "
                 f"{result.fully_typed_functions} fully typed, "
                 f"{result.partially_typed_functions} partial, "
                 f"{result.untyped_functions} untyped")
    lines.append(f"Parameters: {result.annotated_params}/{result.total_params} annotated "
                 f"({result.param_coverage:.0%})")
    lines.append(f"Return types: {result.functions_with_return}/{result.total_functions} "
                 f"({result.return_coverage:.0%})")
    lines.append(f"Any usages: {result.any_count}")
    lines.append(f"type: ignore: {result.type_ignore_count}")
    if result.parse_errors:
        lines.append(f"Parse errors: {result.parse_errors}")
    lines.append("")

    # Untyped functions (most actionable)
    untyped = [f for fr in result.file_results for f in fr.functions if not f.fully_typed]
    if untyped:
        lines.append("--- Untyped / Partially Typed Functions ---")
        # Group by file
        by_file: dict[str, list] = {}
        for f in untyped:
            by_file.setdefault(f.file_path, []).append(f)

        for fpath, funcs in sorted(by_file.items()):
            rel = os.path.relpath(fpath, result.target)
            lines.append(f"\n  {rel}:")
            for f in funcs:
                issues = []
                if not f.has_return_annotation:
                    issues.append("no return type")
                if f.unannotated_param_names:
                    issues.append(f"untyped params: {', '.join(f.unannotated_param_names)}")
                lines.append(f"    L{f.line_number} {f.name}: {'; '.join(issues)}")
        lines.append("")

    # Any usages
    all_any = [a for fr in result.file_results for a in fr.any_usages]
    if all_any:
        lines.append("--- Any Usages ---")
        for a in all_any[:20]:  # limit output
            rel = os.path.relpath(a.file_path, result.target)
            lines.append(f"  {rel}:{a.line_number}: {a.context}")
        if len(all_any) > 20:
            lines.append(f"  ... and {len(all_any) - 20} more")
        lines.append("")

    # Type ignores
    all_ignores = [t for fr in result.file_results for t in fr.type_ignores]
    if all_ignores:
        lines.append("--- type: ignore Comments ---")
        for t in all_ignores[:20]:
            rel = os.path.relpath(t.file_path, result.target)
            lines.append(f"  {rel}:{t.line_number}: {t.context}")
        if len(all_ignores) > 20:
            lines.append(f"  ... and {len(all_ignores) - 20} more")
        lines.append("")

    return "\n".join(lines)


def format_score(result: AuditResult) -> str:
    """Format the score summary."""
    score = compute_score(result)
    grade = score_to_grade(score)
    profile = classify_profile(result)
    desc = PROFILE_DESCRIPTIONS.get(profile, profile)

    lines = []
    lines.append(f"ai-type-audit v{__version__} — {result.target}")
    lines.append(f"Score: {score}/100 (Grade: {grade})")
    lines.append(f"Profile: {profile} — {desc}")
    lines.append("")
    lines.append(f"  Function coverage: {result.function_coverage:.0%} "
                 f"({result.fully_typed_functions}/{result.total_functions})")
    lines.append(f"  Parameter coverage: {result.param_coverage:.0%} "
                 f"({result.annotated_params}/{result.total_params})")
    lines.append(f"  Return type coverage: {result.return_coverage:.0%} "
                 f"({result.functions_with_return}/{result.total_functions})")
    lines.append(f"  Any usages: {result.any_count}")
    lines.append(f"  type: ignore: {result.type_ignore_count}")
    return "\n".join(lines)


def format_json(result: AuditResult, include_score: bool = False) -> str:
    """Format results as JSON."""
    data = {
        "version": __version__,
        "target": result.target,
        "summary": {
            "files_scanned": result.files_scanned,
            "files_with_issues": result.files_with_issues,
            "total_functions": result.total_functions,
            "fully_typed": result.fully_typed_functions,
            "partially_typed": result.partially_typed_functions,
            "untyped": result.untyped_functions,
            "total_params": result.total_params,
            "annotated_params": result.annotated_params,
            "functions_with_return": result.functions_with_return,
            "any_count": result.any_count,
            "type_ignore_count": result.type_ignore_count,
            "parse_errors": result.parse_errors,
            "function_coverage": round(result.function_coverage, 4),
            "param_coverage": round(result.param_coverage, 4),
            "return_coverage": round(result.return_coverage, 4),
        },
        "untyped_functions": [
            {
                "name": f.name,
                "file": f.file_path,
                "line": f.line_number,
                "has_return": f.has_return_annotation,
                "params": f.total_params,
                "annotated_params": f.annotated_params,
                "unannotated": list(f.unannotated_param_names),
            }
            for fr in result.file_results
            for f in fr.functions
            if not f.fully_typed
        ],
        "any_usages": [
            {"file": a.file_path, "line": a.line_number, "context": a.context}
            for fr in result.file_results
            for a in fr.any_usages
        ],
        "type_ignores": [
            {"file": t.file_path, "line": t.line_number, "context": t.context, "codes": t.codes}
            for fr in result.file_results
            for t in fr.type_ignores
        ],
    }
    if include_score:
        score = compute_score(result)
        data["score"] = {
            "value": score,
            "grade": score_to_grade(score),
            "profile": classify_profile(result),
        }
    return json.dumps(data, indent=2)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-type-audit",
        description="Python type annotation coverage analyzer",
    )
    parser.add_argument("target", nargs="?", default=".",
                        help="Directory or file to scan (default: current directory)")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output as JSON")
    parser.add_argument("--score", action="store_true",
                        help="Show health score (0-100 + A-F grade)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(args: Optional[list[str]] = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)

    target = os.path.abspath(opts.target)
    if not os.path.exists(target):
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 1

    files = find_python_files(target)
    if not files:
        print(f"No Python files found in {target}", file=sys.stderr)
        return 0

    file_results = [analyze_file(f) for f in files]
    result = aggregate_results(target, file_results)

    if opts.json_output:
        print(format_json(result, include_score=opts.score))
    elif opts.score:
        print(format_score(result))
    else:
        print(format_text(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
