#!/usr/bin/env python3
"""ai-dead-code: AST-based Python dead code detector.

Detects unused imports, functions, variables, and unreachable code using
abstract syntax tree analysis. Zero external dependencies.

Usage:
    ai-dead-code                          # scan current directory
    ai-dead-code path/to/project          # scan specific directory
    ai-dead-code -f specific_file.py      # scan single file
    ai-dead-code --json                   # JSON output
    ai-dead-code --category imports       # filter by category
    ai-dead-code --ignore __init__.py     # ignore specific files
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

# Import shared scope-safe AST traversal (sibling ai-ast-utils package)
# sys.path hack removed — using package import
from code_health_suite.ast_utils import walk_scope as _walk_current_scope  # noqa: E402


__version__ = "0.3.0"

# --- Categories ---

CATEGORIES = {
    "unused-import": "Import is never used in the module",
    "unused-function": "Function/method is defined but never called",
    "unused-variable": "Variable is assigned but never read",
    "unused-argument": "Function argument is never used in the body",
    "unreachable-code": "Code after return/raise/break/continue is unreachable",
}

SEVERITY_MAP = {
    "unused-import": "medium",
    "unused-function": "low",
    "unused-variable": "low",
    "unused-argument": "info",
    "unreachable-code": "medium",
}

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1, "info": 0}

# Names that are conventionally unused and should be ignored
IGNORED_NAMES = {
    "_",           # throwaway variable
    "__all__",     # export list
    "__version__", # package version
    "__author__",  # package author
    "__doc__",     # docstring
}

# Common dunder methods that are called implicitly
DUNDER_METHODS = {
    "__init__", "__del__", "__repr__", "__str__", "__bytes__", "__format__",
    "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__", "__hash__",
    "__bool__", "__len__", "__getitem__", "__setitem__", "__delitem__",
    "__iter__", "__next__", "__reversed__", "__contains__",
    "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__",
    "__mul__", "__rmul__", "__imul__", "__truediv__", "__rtruediv__",
    "__floordiv__", "__mod__", "__pow__", "__and__", "__or__", "__xor__",
    "__lshift__", "__rshift__", "__neg__", "__pos__", "__abs__", "__invert__",
    "__call__", "__getattr__", "__getattribute__", "__setattr__", "__delattr__",
    "__get__", "__set__", "__delete__", "__set_name__",
    "__enter__", "__exit__", "__aenter__", "__aexit__",
    "__await__", "__aiter__", "__anext__",
    "__new__", "__init_subclass__", "__class_getitem__",
    "__missing__", "__sizeof__", "__reduce__", "__reduce_ex__",
    "__copy__", "__deepcopy__", "__getnewargs__", "__getnewargs_ex__",
    "__fspath__", "__index__", "__int__", "__float__", "__complex__",
    "__round__", "__trunc__", "__floor__", "__ceil__",
    "__post_init__",  # dataclass
}

# Decorator names that imply the function is used externally
EXTERNAL_USE_DECORATORS = {
    "property", "staticmethod", "classmethod", "abstractmethod",
    "overload", "override",
    # Testing
    "pytest.fixture", "fixture",
    "pytest.mark.parametrize", "parametrize",
    # Web frameworks
    "app.route", "app.get", "app.post", "app.put", "app.delete",
    "router.get", "router.post", "router.put", "router.delete",
    "api_view", "action",
    # CLI
    "click.command", "click.group",
    "app.command",
    # Celery
    "shared_task", "task",
    # Signals
    "receiver",
}


@dataclass
class Finding:
    """A single dead code finding."""
    file: str
    line: int
    category: str
    name: str
    message: str
    severity: str = ""
    end_line: Optional[int] = None

    def __post_init__(self):
        if not self.severity:
            self.severity = SEVERITY_MAP.get(self.category, "low")


@dataclass
class FileReport:
    """Report for a single file."""
    file: str
    findings: list[Finding] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ScanResult:
    """Complete scan result."""
    files_scanned: int = 0
    total_findings: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    reports: list[FileReport] = field(default_factory=list)


# --- Project-Level Scoring ---

@dataclass
class DeadCodeProjectStats:
    """Project-level dead code health statistics."""
    score: int  # 0-100 (higher = cleaner)
    grade: str  # A-F
    profile: str  # import_heavy / function_heavy / variable_heavy / argument_heavy / mixed / clean
    files_scanned: int
    files_with_findings: int
    clean_file_pct: float
    total_findings: int
    density: float  # findings per file
    by_category: dict[str, int]
    by_severity: dict[str, int]
    dominant_category: str  # category with most findings
    dominant_pct: float  # percentage of dominant category


def _score_to_grade(score: int) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


_SEVERITY_WEIGHTS = {"high": 4.0, "medium": 2.0, "low": 1.0, "info": 0.5}


def classify_dead_code_profile(result: ScanResult) -> tuple[str, str, float]:
    """Classify a project's dead code profile.

    Returns:
        (profile, dominant_category, dominant_pct)

    Profiles:
        clean: <=2 total findings
        import_heavy: >50% unused-import
        function_heavy: >50% unused-function
        variable_heavy: >50% unused-variable
        argument_heavy: >50% unused-argument
        mixed: no single category >50%
    """
    if result.total_findings <= 2:
        dominant = max(result.by_category, key=result.by_category.get) if result.by_category else "none"
        dominant_pct = (result.by_category.get(dominant, 0) / max(result.total_findings, 1)) * 100
        return "clean", dominant, dominant_pct

    dominant = max(result.by_category, key=result.by_category.get) if result.by_category else "none"
    dominant_pct = (result.by_category.get(dominant, 0) / result.total_findings) * 100

    profile_map = {
        "unused-import": "import_heavy",
        "unused-function": "function_heavy",
        "unused-variable": "variable_heavy",
        "unused-argument": "argument_heavy",
        "unreachable-code": "mixed",  # rare to dominate
    }

    if dominant_pct > 50:
        return profile_map.get(dominant, "mixed"), dominant, dominant_pct
    return "mixed", dominant, dominant_pct


def compute_project_stats(result: ScanResult) -> DeadCodeProjectStats:
    """Compute project-level dead code health score.

    Scoring formula:
        1. weighted_sum = sum(severity_weight * count_per_severity)
        2. weighted_density = weighted_sum / files_scanned
        3. penalty = weighted_density * 8 (capped at 100)
        4. score = max(0, 100 - penalty)

    Clean projects (0 findings) score 100 (A).
    ~12 weighted findings per file = score 0 (F).
    """
    files_scanned = max(result.files_scanned, 1)
    files_with_findings = len([r for r in result.reports if r.findings])
    clean_pct = ((files_scanned - files_with_findings) / files_scanned) * 100
    density = result.total_findings / files_scanned

    # Severity-weighted penalty
    weighted_sum = sum(
        result.by_severity.get(sev, 0) * w
        for sev, w in _SEVERITY_WEIGHTS.items()
    )
    weighted_density = weighted_sum / files_scanned
    penalty = min(100, weighted_density * 8)
    score = max(0, min(100, round(100 - penalty)))

    grade = _score_to_grade(score)
    profile, dominant, dominant_pct = classify_dead_code_profile(result)

    return DeadCodeProjectStats(
        score=score,
        grade=grade,
        profile=profile,
        files_scanned=result.files_scanned,
        files_with_findings=files_with_findings,
        clean_file_pct=round(clean_pct, 1),
        total_findings=result.total_findings,
        density=round(density, 2),
        by_category=dict(result.by_category),
        by_severity=dict(result.by_severity),
        dominant_category=dominant,
        dominant_pct=round(dominant_pct, 1),
    )


def format_score_text(stats: DeadCodeProjectStats) -> str:
    """Format project score as human-readable text."""
    lines = [
        f"Dead Code Health Score: {stats.score}/100 ({stats.grade})",
        f"Profile: {stats.profile}",
        f"Files: {stats.files_scanned} scanned, "
        f"{stats.files_scanned - stats.files_with_findings} clean ({stats.clean_file_pct}%)",
        f"Findings: {stats.total_findings} total ({stats.density}/file)",
    ]
    if stats.by_category:
        lines.append("  By category:")
        for cat, count in sorted(stats.by_category.items(), key=lambda x: -x[1]):
            pct = (count / max(stats.total_findings, 1)) * 100
            lines.append(f"    {cat}: {count} ({pct:.1f}%)")
    if stats.by_severity:
        lines.append("  By severity:")
        for sev in ["high", "medium", "low", "info"]:
            if sev in stats.by_severity:
                lines.append(f"    {sev}: {stats.by_severity[sev]}")
    return "\n".join(lines)


def format_score_json(stats: DeadCodeProjectStats) -> str:
    """Format project score as JSON."""
    return json.dumps(asdict(stats), indent=2)


# --- AST Analysis ---

class NameCollector(ast.NodeVisitor):
    """Collects all name references (reads) in an AST subtree."""

    def __init__(self):
        self.names: set[str] = set()
        self.attr_names: set[str] = set()  # x.y -> "y"
        self.dotted_names: set[str] = set()  # x.y -> "x.y"

    def visit_Name(self, node: ast.Name):
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self.attr_names.add(node.attr)
        # Collect dotted name for "module.function" patterns
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            self.dotted_names.add(".".join(reversed(parts)))
        self.generic_visit(node)


def _get_decorator_names(decorators: list[ast.expr]) -> set[str]:
    """Extract decorator names from a list of decorator nodes."""
    names = set()
    for dec in decorators:
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            parts = []
            current = dec
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                names.add(".".join(reversed(parts)))
        elif isinstance(dec, ast.Call):
            # @decorator(args) -> extract decorator name
            if isinstance(dec.func, ast.Name):
                names.add(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                parts = []
                current = dec.func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    names.add(".".join(reversed(parts)))
    return names


def _has_external_decorator(decorators: list[ast.expr]) -> bool:
    """Check if any decorator implies external usage."""
    dec_names = _get_decorator_names(decorators)
    for name in dec_names:
        if name in EXTERNAL_USE_DECORATORS:
            return True
        # Check suffix match (e.g., "app.route" matches "flask_app.route")
        for ext_dec in EXTERNAL_USE_DECORATORS:
            if name.endswith("." + ext_dec.split(".")[-1]):
                return True
    return False


def find_unused_imports(tree: ast.Module) -> list[Finding]:
    """Find imports that are never referenced in the module."""
    findings = []
    # Collect all names used in the module (excluding import statements)
    collector = NameCollector()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            collector.visit(node)

    used_names = collector.names | collector.attr_names | collector.dotted_names

    # Check if __all__ is defined — if so, names in __all__ are "used"
    all_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_names.add(elt.value)

    used_names |= all_names

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                # For "import os.path", the usable name is "os"
                base_name = name.split(".")[0]
                if base_name not in used_names and name not in used_names and name not in IGNORED_NAMES:
                    findings.append(Finding(
                        file="", line=node.lineno, category="unused-import",
                        name=name,
                        message=f"'{name}' imported but never used",
                    ))
        elif isinstance(node, ast.ImportFrom):
            # __future__ imports are compile-time directives, not runtime references
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue  # Can't analyze star imports
                name = alias.asname or alias.name
                if name not in used_names and name not in IGNORED_NAMES:
                    findings.append(Finding(
                        file="", line=node.lineno, category="unused-import",
                        name=name,
                        message=f"'{name}' imported from '{node.module}' but never used",
                    ))

    return findings


def _is_test_file(filepath: str) -> bool:
    """Check if a file is a test file based on naming conventions."""
    basename = os.path.basename(filepath)
    return (
        basename.startswith("test_")
        or basename.endswith("_test.py")
        or "/tests/" in filepath
        or "/test/" in filepath
        or basename == "conftest.py"
    )


# Test function prefixes that are entry points (called by frameworks, not code)
_TEST_ENTRY_PREFIXES = ("test_", "setup_", "teardown_")


def find_unused_functions(
    tree: ast.Module,
    cross_module_names: Optional[set[str]] = None,
    is_wholesale_imported: bool = False,
    is_test_file: bool = False,
) -> list[Finding]:
    """Find module-level and class-level functions that are never referenced.

    Args:
        tree: Parsed AST module.
        cross_module_names: Names imported from this file by other modules.
            When provided, functions whose names appear here are NOT flagged.
        is_wholesale_imported: If True, another module does `import this_module`
            or `from this_module import *`, so all public names are potentially used.
        is_test_file: If True, skip test entry points (test_*, setup_*, teardown_*)
            since they are called by test frameworks via naming convention.
    """
    findings = []

    # If the entire module is wholesale imported, all public names are used
    if is_wholesale_imported:
        return findings

    # Collect all name references in the module
    collector = NameCollector()
    collector.visit(tree)
    used_names = collector.names | collector.attr_names

    # Find function definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name in IGNORED_NAMES or name in DUNDER_METHODS:
                continue
            # Skip test entry points in test files (called by pytest/unittest)
            if is_test_file and any(name.startswith(p) for p in _TEST_ENTRY_PREFIXES):
                continue
            if name.startswith("_") and not name.startswith("__"):
                # Private functions — check usage
                pass
            if _has_external_decorator(node.decorator_list):
                continue
            # Check if the function name appears anywhere as a reference
            # (not just its own definition)
            if name not in used_names:
                # Cross-module check: is this name imported by other files?
                if cross_module_names and name in cross_module_names:
                    continue
                findings.append(Finding(
                    file="", line=node.lineno, category="unused-function",
                    name=name,
                    message=f"Function '{name}' is defined but never used",
                ))

        elif isinstance(node, ast.ClassDef):
            # In test files, skip test classes entirely (Test* classes
            # contain test_* methods called by framework)
            if is_test_file and node.name.startswith("Test"):
                continue
            # Detect visitor pattern classes (ast.NodeVisitor, etc.)
            # Their visit_* methods are called dynamically by the framework
            _visitor_bases = {"NodeVisitor", "NodeTransformer"}
            is_visitor = any(
                (isinstance(b, ast.Attribute) and b.attr in _visitor_bases)
                or (isinstance(b, ast.Name) and b.id in _visitor_bases)
                for b in node.bases
            )
            # Check methods inside classes
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mname = item.name
                    if mname in DUNDER_METHODS or mname in IGNORED_NAMES:
                        continue
                    # Skip test methods in test files
                    if is_test_file and any(mname.startswith(p) for p in _TEST_ENTRY_PREFIXES):
                        continue
                    if _has_external_decorator(item.decorator_list):
                        continue
                    # Skip visit_* and generic_visit in visitor pattern classes
                    if is_visitor and (mname.startswith("visit_") or mname == "generic_visit"):
                        continue
                    # Class methods are harder — they might be called via self.method()
                    # Only flag if method name never appears as an attribute access
                    if mname not in used_names:
                        # Cross-module check for class names and method names
                        if cross_module_names and (
                            mname in cross_module_names
                            or node.name in cross_module_names
                        ):
                            continue
                        findings.append(Finding(
                            file="", line=item.lineno, category="unused-function",
                            name=f"{node.name}.{mname}",
                            message=f"Method '{mname}' in class '{node.name}' is never referenced",
                        ))

    return findings



def find_unused_variables(tree: ast.Module) -> list[Finding]:
    """Find variables that are assigned but never read."""
    findings = []

    # We analyze function/method bodies for local variable usage
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Collect global/nonlocal names — these write to outer scope, not local.
        # Assignments to these names should NOT be flagged as unused locals.
        global_names: set[str] = set()
        for child in _walk_current_scope(node):
            if isinstance(child, ast.Global):
                global_names.update(child.names)
            elif isinstance(child, ast.Nonlocal):
                global_names.update(child.names)

        # Collect assignments ONLY from this function's own scope
        # (skip nested function/class bodies — they have their own scope)
        assigned: dict[str, int] = {}  # name -> line number

        for child in _walk_current_scope(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name not in IGNORED_NAMES and not name.startswith("_") and name not in global_names:
                            assigned[name] = child.lineno
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                name = elt.id
                                if name not in IGNORED_NAMES and not name.startswith("_") and name not in global_names:
                                    assigned[name] = child.lineno
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                if isinstance(child.target, ast.Name):
                    name = child.target.id
                    if name not in IGNORED_NAMES and not name.startswith("_") and name not in global_names:
                        assigned[name] = child.lineno

        # Collect reads from ENTIRE subtree (closure reads count as usage)
        read_names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                read_names.add(child.id)

        # Find assigned but never read
        func_name = node.name
        for name, line in assigned.items():
            if name not in read_names:
                findings.append(Finding(
                    file="", line=line, category="unused-variable",
                    name=name,
                    message=f"Variable '{name}' in '{func_name}' is assigned but never read",
                ))

    return findings


def find_unused_arguments(tree: ast.Module) -> list[Finding]:
    """Find function arguments that are never used in the function body."""
    findings = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Skip if function body is just pass/... (abstract or placeholder)
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is Ellipsis:
                    continue
            # Also skip if body is just a docstring
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
        # Also handle docstring + pass/...
        if len(node.body) == 2:
            first = node.body[0]
            second = node.body[1]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                if isinstance(second, ast.Pass):
                    continue
                if isinstance(second, ast.Expr) and isinstance(second.value, ast.Constant) and second.value.value is Ellipsis:
                    continue

        # Skip methods with external decorators
        if _has_external_decorator(node.decorator_list):
            continue

        # Collect all name reads in the function body
        read_names: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                read_names.add(child.id)

        # Check each argument
        args = node.args
        all_args = []
        for arg in args.args:
            all_args.append(arg)
        for arg in args.posonlyargs:
            all_args.append(arg)
        for arg in args.kwonlyargs:
            all_args.append(arg)

        for arg in all_args:
            name = arg.arg
            if name == "self" or name == "cls":
                continue
            if name.startswith("_"):
                continue
            if name in IGNORED_NAMES:
                continue
            if name not in read_names:
                # Check if the function uses *args or **kwargs — might consume it
                if args.vararg or args.kwarg:
                    continue
                findings.append(Finding(
                    file="", line=node.lineno, category="unused-argument",
                    name=name,
                    message=f"Argument '{name}' of '{node.name}' is never used",
                ))

    return findings


def find_unreachable_code(tree: ast.Module) -> list[Finding]:
    """Find code after return/raise/break/continue that can never execute."""
    findings = []

    def check_body(body: list[ast.stmt], context: str):
        for i, stmt in enumerate(body):
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                # Check if there are statements after this one in the same block
                remaining = body[i + 1:]
                if remaining:
                    # Filter out string constants (might be documentation)
                    non_doc = [s for s in remaining
                              if not (isinstance(s, ast.Expr)
                                      and isinstance(s.value, ast.Constant)
                                      and isinstance(s.value.value, str))]
                    if non_doc:
                        next_stmt = non_doc[0]
                        end_stmt = non_doc[-1]
                        keyword = type(stmt).__name__.lower()
                        findings.append(Finding(
                            file="", line=next_stmt.lineno,
                            category="unreachable-code",
                            name=f"after_{keyword}",
                            message=f"Code after '{keyword}' is unreachable in {context}",
                            end_line=getattr(end_stmt, "end_lineno", end_stmt.lineno),
                        ))
                break  # Only report the first unreachable block per scope

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_body(node.body, f"function '{node.name}'")
            # Check inside if/elif/else, for/while, try/except bodies
            # Use _walk_current_scope to avoid entering nested function/class
            # bodies — they will be checked when the outer loop reaches them.
            # BUG-45: ast.walk entered nested scopes, causing duplicate findings
            # with wrong attribution (inner unreachable code attributed to outer func).
            for child in _walk_current_scope(node):
                if isinstance(child, ast.If):
                    check_body(child.body, f"if block in '{node.name}'")
                    check_body(child.orelse, f"else block in '{node.name}'")
                elif isinstance(child, (ast.For, ast.AsyncFor)):
                    check_body(child.body, f"for loop in '{node.name}'")
                elif isinstance(child, (ast.While,)):
                    check_body(child.body, f"while loop in '{node.name}'")
                elif isinstance(child, ast.With):
                    check_body(child.body, f"with block in '{node.name}'")
                elif isinstance(child, ast.ExceptHandler):
                    check_body(child.body, f"except block in '{node.name}'")
                elif isinstance(child, ast.Try):
                    check_body(child.body, f"try block in '{node.name}'")
                    check_body(child.orelse, f"try/else block in '{node.name}'")
                    check_body(child.finalbody, f"finally block in '{node.name}'")
                elif isinstance(child, ast.AsyncWith):
                    check_body(child.body, f"async with block in '{node.name}'")

    return findings


# --- Cross-Module Analysis ---

def _path_to_module(filepath: str, root: str) -> str:
    """Convert a file path to a dotted module name relative to root."""
    root_path = Path(root).resolve()
    file_path = Path(filepath).resolve()
    try:
        rel = file_path.relative_to(root_path)
    except ValueError:
        return file_path.stem
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    # Handle src-layout: if first component is 'src' and src/__init__.py
    # doesn't exist, strip it (src is a namespace directory, not a package)
    if parts and parts[0] == "src" and not (root_path / "src" / "__init__.py").exists():
        parts = parts[1:]
    return ".".join(parts) if parts else file_path.stem


def _build_module_file_map(files: list[str], root: str) -> dict[str, str]:
    """Build mapping from dotted module names to file paths."""
    module_map: dict[str, str] = {}
    for fp in files:
        mod_name = _path_to_module(fp, root)
        if mod_name:
            module_map[mod_name] = fp
    return module_map


def _resolve_import_target(
    source_module: str,
    imp_module: str,
    imp_level: int,
    module_map: dict[str, str],
) -> Optional[str]:
    """Resolve an import to an internal module file path, or None if external."""
    if imp_level > 0:
        # Relative import
        parts = source_module.split(".")
        base = parts[:-imp_level] if imp_level <= len(parts) else []
        if imp_module:
            target = ".".join(base + [imp_module]) if base else imp_module
        else:
            target = ".".join(base) if base else ""
    else:
        target = imp_module

    # Direct match
    if target in module_map:
        return module_map[target]
    # Package prefix match
    for known_mod, known_file in module_map.items():
        if known_mod.startswith(target + "."):
            return known_file
    # Partial resolution (from foo.bar.baz, try foo.bar, then foo)
    parts = target.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in module_map:
            return module_map[prefix]
    return None


def build_cross_module_refs(
    files: list[str], root: str,
) -> tuple[dict[str, set[str]], set[str]]:
    """Build cross-module reference map for unused-function FP elimination.

    Returns:
        imported_names: {filepath: set of names explicitly imported from this file}
        wholesale_imported: set of filepaths imported wholesale (import X / from X import *)
    """
    module_map = _build_module_file_map(files, root)
    file_to_module: dict[str, str] = {v: k for k, v in module_map.items()}

    imported_names: dict[str, set[str]] = defaultdict(set)
    wholesale_imported: set[str] = set()

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=filepath)
        except (OSError, SyntaxError, ValueError):
            continue

        source_module = file_to_module.get(filepath, "")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                level = node.level or 0
                target_file = _resolve_import_target(
                    source_module, node.module, level, module_map,
                )
                if target_file and target_file != filepath:
                    for alias in node.names:
                        if alias.name == "*":
                            wholesale_imported.add(target_file)
                        else:
                            imported_names[target_file].add(alias.name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target_file = _resolve_import_target(
                        source_module, alias.name, 0, module_map,
                    )
                    if target_file and target_file != filepath:
                        wholesale_imported.add(target_file)

    return imported_names, wholesale_imported


def analyze_file(filepath: str,
                 cross_module_names: Optional[set[str]] = None,
                 is_wholesale_imported: bool = False) -> FileReport:
    """Analyze a single Python file for dead code."""
    report = FileReport(file=filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except (OSError, IOError) as e:
        report.error = str(e)
        return report

    try:
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, ValueError) as e:
        report.error = f"{type(e).__name__}: {e}"
        return report

    # Run all detectors
    all_findings = []
    all_findings.extend(find_unused_imports(tree))
    all_findings.extend(find_unused_functions(
        tree, cross_module_names=cross_module_names,
        is_wholesale_imported=is_wholesale_imported,
        is_test_file=_is_test_file(filepath),
    ))
    all_findings.extend(find_unused_variables(tree))
    all_findings.extend(find_unused_arguments(tree))
    all_findings.extend(find_unreachable_code(tree))

    # Set file path on all findings
    for f in all_findings:
        f.file = filepath

    report.findings = all_findings
    return report


# --- File Discovery ---

EXCLUDED_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", "egg-info", ".eggs",
    "site-packages",
}


def find_python_files(path: str) -> list[str]:
    """Recursively find all Python files, excluding common non-source dirs."""
    files = []
    path_obj = Path(path)

    if path_obj.is_file():
        if path_obj.suffix == ".py":
            return [str(path_obj)]
        return []

    for root, dirs, filenames in os.walk(path):
        # Filter out excluded directories in-place
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIRS
            and not d.startswith(".")
            and "site-packages" not in d
            and not (d.startswith(".venv") or d.endswith("_env") or d.endswith("_venv"))
        ]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))

    return sorted(files)


# --- Output Formatting ---

COLORS = {
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "gray": "\033[90m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}

SEVERITY_COLORS = {
    "high": "red",
    "medium": "yellow",
    "low": "blue",
    "info": "gray",
}


def format_severity(severity: str) -> str:
    """Format severity with color."""
    color = SEVERITY_COLORS.get(severity, "gray")
    return f"{COLORS[color]}{severity.upper()}{COLORS['reset']}"


def format_finding(finding: Finding) -> str:
    """Format a single finding for terminal output."""
    sev = format_severity(finding.severity)
    cat = f"{COLORS['cyan']}{finding.category}{COLORS['reset']}"
    loc = f"{COLORS['gray']}{finding.file}:{finding.line}{COLORS['reset']}"
    return f"  {sev:<24s} {cat:<40s} {finding.message}  {loc}"


def print_summary(result: ScanResult):
    """Print summary statistics."""
    print(f"\n{COLORS['bold']}Dead Code Summary{COLORS['reset']}")
    print(f"  Files scanned: {result.files_scanned}")
    print(f"  Total findings: {result.total_findings}")

    if result.by_category:
        print(f"\n  {COLORS['bold']}By category:{COLORS['reset']}")
        for cat, count in sorted(result.by_category.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count}")

    if result.by_severity:
        print(f"\n  {COLORS['bold']}By severity:{COLORS['reset']}")
        for sev in ["high", "medium", "low", "info"]:
            if sev in result.by_severity:
                print(f"    {format_severity(sev)}: {result.by_severity[sev]}")


# --- Main ---

def _should_auto_cross_module(path: str) -> bool:
    """Determine if cross-module analysis should be auto-enabled.

    Returns True when scanning a directory with multiple Python files.
    Single-file scans don't benefit from cross-module analysis.
    """
    if not os.path.isdir(path):
        return False
    # Quick check: are there at least 2 Python files?
    count = 0
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.endswith(".py"):
                count += 1
                if count >= 2:
                    return True
    return False


def scan(path: str, category: Optional[str] = None,
         ignore_patterns: Optional[list[str]] = None,
         min_severity: str = "info",
         cross_module: Optional[bool] = None) -> ScanResult:
    """Scan a path for dead code.

    Args:
        cross_module: Enable cross-module analysis. When None (default),
            auto-enabled for directory scans with 2+ Python files.
            Set explicitly to True/False to override.
    """
    if cross_module is None:
        cross_module = _should_auto_cross_module(path)
    result = ScanResult()
    files = find_python_files(path)
    result.files_scanned = len(files)

    min_sev_order = SEVERITY_ORDER.get(min_severity, 0)

    # Build cross-module reference map if enabled
    imported_names: dict[str, set[str]] = {}
    wholesale_imported: set[str] = set()
    if cross_module and len(files) > 1:
        root = str(Path(path).resolve()) if os.path.isdir(path) else str(Path(path).parent.resolve())
        imported_names, wholesale_imported = build_cross_module_refs(files, root)

    for filepath in files:
        # Check ignore patterns
        if ignore_patterns:
            basename = os.path.basename(filepath)
            if any(pattern in basename or pattern in filepath for pattern in ignore_patterns):
                continue

        report = analyze_file(
            filepath,
            cross_module_names=imported_names.get(filepath) if cross_module else None,
            is_wholesale_imported=filepath in wholesale_imported if cross_module else False,
        )

        # Filter by category
        if category:
            report.findings = [f for f in report.findings if f.category == category]

        # Filter by severity
        report.findings = [
            f for f in report.findings
            if SEVERITY_ORDER.get(f.severity, 0) >= min_sev_order
        ]

        if report.findings or report.error:
            result.reports.append(report)
            result.total_findings += len(report.findings)
            for f in report.findings:
                result.by_category[f.category] = result.by_category.get(f.category, 0) + 1
                result.by_severity[f.severity] = result.by_severity.get(f.severity, 0) + 1

    return result


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ai-dead-code",
        description="AST-based Python dead code detector",
    )
    parser.add_argument("path", nargs="?", default=".",
                        help="Path to scan (default: current directory)")
    parser.add_argument("-f", "--file", help="Scan a single file")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output in JSON format")
    parser.add_argument("--category", choices=list(CATEGORIES.keys()),
                        help="Filter by category")
    parser.add_argument("--severity", default="info",
                        choices=["high", "medium", "low", "info"],
                        help="Minimum severity to report (default: info)")
    parser.add_argument("--ignore", nargs="*", default=[],
                        help="File patterns to ignore")
    parser.add_argument("--cross-module", action="store_true", default=None,
                        help="Force cross-module analysis (auto-enabled for directory scans)")
    parser.add_argument("--no-cross-module", action="store_true",
                        help="Disable cross-module analysis even for directory scans")
    parser.add_argument("--score", action="store_true",
                        help="Show project-level dead code health score")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)

    scan_path = args.file if args.file else args.path
    # Determine cross-module mode: explicit flags override auto-detection
    if args.no_cross_module:
        cross_module = False
    elif args.cross_module:
        cross_module = True
    else:
        cross_module = None  # auto-detect
    result = scan(scan_path, category=args.category,
                  ignore_patterns=args.ignore if args.ignore else None,
                  min_severity=args.severity,
                  cross_module=cross_module)

    # Score mode: compute and display project health score
    if args.score:
        stats = compute_project_stats(result)
        if args.json_output:
            print(format_score_json(stats))
        else:
            print(format_score_text(stats))
        return 0 if stats.score >= 60 else 1

    if args.json_output:
        output = {
            "version": __version__,
            "files_scanned": result.files_scanned,
            "total_findings": result.total_findings,
            "by_category": result.by_category,
            "by_severity": result.by_severity,
            "findings": [asdict(f) for r in result.reports for f in r.findings],
            "errors": [{"file": r.file, "error": r.error}
                       for r in result.reports if r.error],
        }
        print(json.dumps(output, indent=2))
    else:
        if result.total_findings == 0:
            print(f"{COLORS['bold']}No dead code found in {result.files_scanned} files.{COLORS['reset']}")
        else:
            for report in result.reports:
                if report.error:
                    print(f"{COLORS['red']}ERROR{COLORS['reset']} {report.file}: {report.error}")
                for finding in sorted(report.findings,
                                      key=lambda f: (SEVERITY_ORDER.get(f.severity, 0) * -1, f.line)):
                    print(format_finding(finding))
            print_summary(result)

    # Exit code: 1 if medium+ findings, 0 otherwise
    has_actionable = any(
        SEVERITY_ORDER.get(f.severity, 0) >= SEVERITY_ORDER["medium"]
        for r in result.reports for f in r.findings
    )
    return 1 if has_actionable else 0


if __name__ == "__main__":
    sys.exit(main())
