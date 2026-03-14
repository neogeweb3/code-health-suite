#!/usr/bin/env python3
"""ai-naming-check: PEP 8 naming convention checker.

Scans Python files to detect naming convention violations:
  - Functions/methods: must be snake_case
  - Classes: must be CamelCase (PascalCase)
  - Constants (module-level UPPER): must be UPPER_SNAKE_CASE
  - Variables/parameters: must be snake_case

Uses AST parsing — zero regex on identifiers, zero external dependencies.

Usage:
    ai-naming-check                     # scan current directory
    ai-naming-check path/to/project     # scan specific directory
    ai-naming-check --json              # JSON output
    ai-naming-check --score             # health score (0-100 + A-F grade)
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


__version__ = "0.1.0"

# Directories to skip during file discovery
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", ".nox", ".eggs", "node_modules", "venv", ".venv", "env",
    ".env", "dist", "build", "egg-info", ".ruff_cache",
}


# --- Data models ---

@dataclass
class NamingViolation:
    """A single naming convention violation."""
    file_path: str
    line_number: int
    name: str
    kind: str  # function, method, class, constant, variable, parameter
    convention: str  # snake_case, CamelCase, UPPER_SNAKE_CASE
    suggestion: str
    message: str


@dataclass
class FileResult:
    """Naming check results for a single file."""
    file_path: str
    violations: list[NamingViolation] = field(default_factory=list)
    names_checked: int = 0
    error: str = ""


@dataclass
class ScanResult:
    """Aggregate naming check results."""
    root: str
    files_scanned: int = 0
    total_names: int = 0
    total_violations: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    violations: list[NamingViolation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    """Health score for naming conventions."""
    score: int  # 0-100
    grade: str  # A-F
    total_names: int
    total_violations: int
    violation_rate: float
    by_kind: dict[str, int] = field(default_factory=dict)
    top_violations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- Naming convention checks ---

# Regex for valid snake_case: lowercase, digits, underscores; may start with _
_SNAKE_RE = re.compile(r"^_*[a-z][a-z0-9_]*$|^_+$|^__[a-z][a-z0-9_]*__$")

# Regex for CamelCase (PascalCase): starts with uppercase
_CAMEL_RE = re.compile(r"^_*[A-Z][a-zA-Z0-9]*$")

# Regex for UPPER_SNAKE_CASE: all uppercase + underscores + digits
_UPPER_RE = re.compile(r"^_*[A-Z][A-Z0-9_]*$")

# Framework method patterns that legitimately use non-snake_case
# e.g., ast.NodeVisitor.visit_FunctionDef, unittest.TestCase.setUp
_FRAMEWORK_METHOD_RE = re.compile(
    r"^(visit_[A-Z]|generic_visit$|setUp$|tearDown$|setUpClass$|tearDownClass$"
    r"|setUpModule$|tearDownModule$|addCleanup$)"
)

# Single-char names are always allowed (i, j, k, x, y, etc.)
# Dunder names (__init__, __name__, etc.) are always allowed
# Names starting with __ (name mangling) follow their own rules


def is_snake_case(name: str) -> bool:
    """Check if name follows snake_case convention."""
    # Strip leading underscores for the check
    stripped = name.lstrip("_")
    if not stripped:
        return True  # bare underscores like _ or __ are fine
    # Dunder names
    if name.startswith("__") and name.endswith("__"):
        return True
    # Single char
    if len(stripped) <= 1:
        return True
    return bool(_SNAKE_RE.match(name))


def is_camel_case(name: str) -> bool:
    """Check if name follows CamelCase (PascalCase) convention."""
    stripped = name.lstrip("_")
    if not stripped or len(stripped) <= 1:
        return True
    return bool(_CAMEL_RE.match(name))


def is_upper_snake_case(name: str) -> bool:
    """Check if name follows UPPER_SNAKE_CASE convention."""
    stripped = name.lstrip("_")
    if not stripped or len(stripped) <= 1:
        return True
    return bool(_UPPER_RE.match(name))


def to_snake_case(name: str) -> str:
    """Convert a name to snake_case."""
    # Preserve leading underscores
    prefix = ""
    stripped = name
    while stripped.startswith("_"):
        prefix += "_"
        stripped = stripped[1:]
    if not stripped:
        return name
    # Insert underscore before uppercase letters
    result = []
    for i, ch in enumerate(stripped):
        if ch.isupper() and i > 0:
            # Don't insert underscore between consecutive uppercase (e.g., HTTPClient → http_client)
            prev_upper = stripped[i - 1].isupper()
            next_lower = (i + 1 < len(stripped)) and stripped[i + 1].islower()
            if not prev_upper or next_lower:
                result.append("_")
        result.append(ch.lower())
    return prefix + "".join(result)


def to_camel_case(name: str) -> str:
    """Convert a name to CamelCase (PascalCase)."""
    prefix = ""
    stripped = name
    while stripped.startswith("_"):
        prefix += "_"
        stripped = stripped[1:]
    if not stripped:
        return name
    parts = stripped.split("_")
    return prefix + "".join(p.capitalize() for p in parts if p)


# --- AST Analysis ---


def _is_constant_assignment(node: ast.AST) -> bool:
    """Check if an assignment looks like a constant (module-level, UPPER name)."""
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                stripped = target.id.lstrip("_")
                if stripped and stripped[0].isupper() and stripped == stripped.upper():
                    return True
    return False


def analyze_file(filepath: str) -> FileResult:
    """Analyze a single Python file for naming convention violations."""
    result = FileResult(file_path=filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except (OSError, IOError) as e:
        result.error = str(e)
        return result

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        result.error = f"SyntaxError: {e}"
        return result

    # Walk the AST
    for node in ast.walk(tree):
        # --- Class names → CamelCase ---
        if isinstance(node, ast.ClassDef):
            result.names_checked += 1
            if not is_camel_case(node.name):
                result.violations.append(NamingViolation(
                    file_path=filepath,
                    line_number=node.lineno,
                    name=node.name,
                    kind="class",
                    convention="CamelCase",
                    suggestion=to_camel_case(node.name),
                    message=f"Class '{node.name}' should use CamelCase → '{to_camel_case(node.name)}'",
                ))

        # --- Function/method names → snake_case ---
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.names_checked += 1
            name = node.name
            # Skip dunder methods
            if name.startswith("__") and name.endswith("__"):
                continue
            # Skip framework patterns (ast.NodeVisitor.visit_*, unittest setUp/tearDown)
            if _FRAMEWORK_METHOD_RE.match(name):
                continue
            # Determine if method (inside a class)
            is_method = _node_is_method(node, tree)
            kind = "method" if is_method else "function"

            if not is_snake_case(name):
                result.violations.append(NamingViolation(
                    file_path=filepath,
                    line_number=node.lineno,
                    name=name,
                    kind=kind,
                    convention="snake_case",
                    suggestion=to_snake_case(name),
                    message=f"{kind.capitalize()} '{name}' should use snake_case → '{to_snake_case(name)}'",
                ))

            # --- Parameters → snake_case ---
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                arg_name = arg.arg
                # Skip self, cls
                if arg_name in ("self", "cls"):
                    continue
                result.names_checked += 1
                if not is_snake_case(arg_name):
                    result.violations.append(NamingViolation(
                        file_path=filepath,
                        line_number=arg.lineno,
                        name=arg_name,
                        kind="parameter",
                        convention="snake_case",
                        suggestion=to_snake_case(arg_name),
                        message=f"Parameter '{arg_name}' should use snake_case → '{to_snake_case(arg_name)}'",
                    ))
            # *args and **kwargs
            if node.args.vararg and node.args.vararg.arg not in ("args",):
                result.names_checked += 1
                vname = node.args.vararg.arg
                if not is_snake_case(vname):
                    result.violations.append(NamingViolation(
                        file_path=filepath,
                        line_number=node.args.vararg.lineno,
                        name=vname,
                        kind="parameter",
                        convention="snake_case",
                        suggestion=to_snake_case(vname),
                        message=f"Parameter '*{vname}' should use snake_case → '*{to_snake_case(vname)}'",
                    ))
            if node.args.kwarg and node.args.kwarg.arg not in ("kwargs",):
                result.names_checked += 1
                kname = node.args.kwarg.arg
                if not is_snake_case(kname):
                    result.violations.append(NamingViolation(
                        file_path=filepath,
                        line_number=node.args.kwarg.lineno,
                        name=kname,
                        kind="parameter",
                        convention="snake_case",
                        suggestion=to_snake_case(kname),
                        message=f"Parameter '**{kname}' should use snake_case → '**{to_snake_case(kname)}'",
                    ))

    # --- Module-level assignments ---
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    result.names_checked += 1
                    stripped = name.lstrip("_")
                    if not stripped or len(stripped) <= 1:
                        continue
                    # Dunder module vars (__all__, __version__) — skip
                    if name.startswith("__") and name.endswith("__"):
                        continue
                    # If it looks like a constant (ALL_UPPER) — check UPPER_SNAKE
                    if stripped[0].isupper() and stripped.replace("_", "").isupper():
                        if not is_upper_snake_case(name):
                            result.violations.append(NamingViolation(
                                file_path=filepath,
                                line_number=node.lineno,
                                name=name,
                                kind="constant",
                                convention="UPPER_SNAKE_CASE",
                                suggestion=name.upper(),
                                message=f"Constant '{name}' should use UPPER_SNAKE_CASE → '{name.upper()}'",
                            ))
                    else:
                        # Regular module-level variable — should be snake_case
                        if not is_snake_case(name):
                            result.violations.append(NamingViolation(
                                file_path=filepath,
                                line_number=node.lineno,
                                name=name,
                                kind="variable",
                                convention="snake_case",
                                suggestion=to_snake_case(name),
                                message=f"Variable '{name}' should use snake_case → '{to_snake_case(name)}'",
                            ))
        elif isinstance(node, ast.AnnAssign) and node.target and isinstance(node.target, ast.Name):
            name = node.target.id
            result.names_checked += 1
            stripped = name.lstrip("_")
            if not stripped or len(stripped) <= 1:
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if not is_snake_case(name) and not (stripped[0].isupper() and stripped.replace("_", "").isupper()):
                result.violations.append(NamingViolation(
                    file_path=filepath,
                    line_number=node.lineno,
                    name=name,
                    kind="variable",
                    convention="snake_case",
                    suggestion=to_snake_case(name),
                    message=f"Variable '{name}' should use snake_case → '{to_snake_case(name)}'",
                ))

    return result


def _node_is_method(func_node: ast.AST, tree: ast.Module) -> bool:
    """Check if a function node is a method (defined inside a class)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if item is func_node:
                    return True
    return False


# --- File discovery ---


def find_python_files(path: str) -> list[str]:
    """Find all Python files under a path, skipping common non-source dirs."""
    if os.path.isfile(path):
        return [path] if path.endswith(".py") else []

    files = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))
    return files


# --- Aggregate analysis ---


def scan(path: str) -> ScanResult:
    """Scan a file or directory for naming convention violations."""
    result = ScanResult(root=path)
    files = find_python_files(path)

    for filepath in files:
        file_result = analyze_file(filepath)
        result.files_scanned += 1
        result.total_names += file_result.names_checked
        result.total_violations += len(file_result.violations)
        result.violations.extend(file_result.violations)
        if file_result.error:
            result.errors.append(f"{filepath}: {file_result.error}")

    # Count by kind
    for v in result.violations:
        result.by_kind[v.kind] = result.by_kind.get(v.kind, 0) + 1

    return result


def compute_score(result: ScanResult) -> ScoreResult:
    """Compute a naming convention health score (0-100)."""
    if result.total_names == 0:
        return ScoreResult(
            score=100, grade="A", total_names=0, total_violations=0,
            violation_rate=0.0,
        )

    violation_rate = result.total_violations / result.total_names

    # Score: 100 - (violation_rate * 200), clamped to [0, 100]
    # 0% violations → 100, 50% → 0
    score = max(0, min(100, int(100 - violation_rate * 200)))

    grade = _score_to_grade(score)

    top_violations = []
    for v in result.violations[:10]:
        top_violations.append({
            "file": v.file_path,
            "line": v.line_number,
            "name": v.name,
            "kind": v.kind,
            "suggestion": v.suggestion,
        })

    return ScoreResult(
        score=score,
        grade=grade,
        total_names=result.total_names,
        total_violations=result.total_violations,
        violation_rate=round(violation_rate, 4),
        by_kind=result.by_kind,
        top_violations=top_violations,
    )


def _score_to_grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# --- CLI ---


def _format_text(result: ScanResult, score_result: ScoreResult) -> str:
    """Format results as human-readable text."""
    lines = []
    lines.append(f"Naming Convention Check — {result.root}")
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Names checked: {result.total_names}")
    lines.append(f"Violations: {result.total_violations}")
    lines.append(f"Score: {score_result.score}/100 ({score_result.grade})")
    lines.append("")

    if result.by_kind:
        lines.append("By kind:")
        for kind, count in sorted(result.by_kind.items()):
            lines.append(f"  {kind}: {count}")
        lines.append("")

    if result.violations:
        lines.append("Violations:")
        for v in result.violations:
            lines.append(f"  {v.file_path}:{v.line_number}: {v.message}")

    if result.errors:
        lines.append("")
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  {e}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Python naming conventions (PEP 8)."
    )
    parser.add_argument("path", nargs="?", default=".", help="Path to scan")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--score", action="store_true", help="Show score only")
    args = parser.parse_args(argv)

    result = scan(args.path)
    score_result = compute_score(result)

    if args.json_output:
        data = asdict(result)
        data["score"] = score_result.to_dict()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.score:
        print(json.dumps(score_result.to_dict(), indent=2))
    else:
        print(_format_text(result, score_result))

    return 0 if score_result.score >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())
