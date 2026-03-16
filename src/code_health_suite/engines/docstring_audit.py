#!/usr/bin/env python3
"""ai-docstring-audit: Docstring coverage and quality checker.

Scans Python files to measure:
  - Docstring presence on public functions, methods, and classes
  - Docstring quality (length, structure, parameter documentation)
  - Coverage percentage per file and overall

Uses AST parsing — zero external dependencies.

Usage:
    ai-docstring-audit                     # scan current directory
    ai-docstring-audit path/to/project     # scan specific directory
    ai-docstring-audit --json              # JSON output
    ai-docstring-audit --score             # health score (0-100 + A-F grade)
"""
from __future__ import annotations

import ast
import json
import os
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
class DocstringIssue:
    """A single docstring issue (missing or low quality)."""
    file_path: str
    line_number: int
    name: str
    kind: str  # function, method, class, module
    issue_type: str  # missing, too_short, no_params, no_returns, no_summary
    message: str
    severity: str = "medium"  # low, medium, high


@dataclass
class EntityInfo:
    """Info about a documentable entity."""
    name: str
    kind: str  # function, method, class, module
    line_number: int
    has_docstring: bool
    docstring_lines: int = 0
    has_params_doc: bool = False
    has_returns_doc: bool = False
    has_raises_doc: bool = False
    param_count: int = 0
    is_public: bool = True


@dataclass
class FileResult:
    """Docstring audit results for a single file."""
    file_path: str
    entities: list[EntityInfo] = field(default_factory=list)
    issues: list[DocstringIssue] = field(default_factory=list)
    total_public: int = 0
    documented: int = 0
    coverage: float = 0.0
    error: str = ""


@dataclass
class ScanResult:
    """Aggregate docstring audit results."""
    root: str
    files_scanned: int = 0
    total_public: int = 0
    total_documented: int = 0
    total_issues: int = 0
    coverage: float = 0.0
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    issues: list[DocstringIssue] = field(default_factory=list)
    file_results: list[FileResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    """Health score for docstring coverage."""
    score: int  # 0-100
    grade: str  # A-F
    total_public: int
    total_documented: int
    coverage: float
    total_issues: int
    by_kind: dict[str, dict[str, int]] = field(default_factory=dict)
    worst_files: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- File discovery ---

def find_python_files(path: str) -> list[str]:
    """Find Python files, skipping common non-project directories."""
    target = Path(path)
    if target.is_file():
        if target.suffix == ".py":
            return [str(target)]
        return []

    results = []
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for f in sorted(files):
            if f.endswith(".py"):
                results.append(os.path.join(root, f))
    return results


# --- AST Analysis ---

def _is_public(name: str) -> bool:
    """Check if a name is public (doesn't start with underscore)."""
    if name.startswith("__") and name.endswith("__"):
        return True  # dunder methods are public
    return not name.startswith("_")


def _count_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count non-self/cls parameters."""
    args = node.args
    all_args = args.args + args.posonlyargs + args.kwonlyargs
    count = len(all_args)
    # Subtract self/cls
    if all_args and all_args[0].arg in ("self", "cls"):
        count -= 1
    if args.vararg:
        count += 1
    if args.kwarg:
        count += 1
    return count


def _get_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from an AST node."""
    if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and node.body):
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return first.value.value
    return None


def _check_docstring_quality(docstring: str, param_count: int) -> list[str]:
    """Check docstring quality, return list of issue types."""
    issues = []
    lines = docstring.strip().splitlines()

    # Too short (single word or less)
    stripped = docstring.strip()
    if len(stripped) < 10:
        issues.append("too_short")

    # Check for parameter documentation if function has params
    if param_count > 0:
        lower = docstring.lower()
        has_params = any(marker in lower for marker in [
            ":param ", "args:", "arguments:", "parameters:",
            "param ", ":type ", "keyword arguments",
        ])
        # Also check for Google/NumPy style
        if not has_params:
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith(("Args:", "Parameters:", "Keyword Args:")):
                    has_params = True
                    break
        if not has_params:
            issues.append("no_params")

    return issues


def analyze_file(filepath: str) -> FileResult:
    """Analyze a single Python file for docstring coverage."""
    result = FileResult(file_path=filepath)

    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        result.error = str(exc)
        return result

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        result.error = f"SyntaxError: {exc}"
        return result

    entities: list[EntityInfo] = []

    # Check module docstring
    mod_doc = _get_docstring(tree)
    entities.append(EntityInfo(
        name=Path(filepath).stem,
        kind="module",
        line_number=1,
        has_docstring=mod_doc is not None,
        docstring_lines=len(mod_doc.splitlines()) if mod_doc else 0,
        is_public=not Path(filepath).stem.startswith("_"),
    ))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_pub = _is_public(node.name)
            doc = _get_docstring(node)
            entities.append(EntityInfo(
                name=node.name,
                kind="class",
                line_number=node.lineno,
                has_docstring=doc is not None,
                docstring_lines=len(doc.splitlines()) if doc else 0,
                is_public=is_pub,
            ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_pub = _is_public(node.name)
            doc = _get_docstring(node)
            param_count = _count_params(node)

            # Determine if method or function
            kind = "function"
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.iter_child_nodes(parent):
                        if child is node:
                            kind = "method"
                            break

            entity = EntityInfo(
                name=node.name,
                kind=kind,
                line_number=node.lineno,
                has_docstring=doc is not None,
                docstring_lines=len(doc.splitlines()) if doc else 0,
                param_count=param_count,
                is_public=is_pub,
            )

            if doc:
                lower = doc.lower()
                entity.has_params_doc = any(m in lower for m in [
                    ":param ", "args:", "arguments:", "parameters:",
                ])
                entity.has_returns_doc = any(m in lower for m in [
                    ":returns:", ":return:", "returns:", "return:",
                ])
                entity.has_raises_doc = any(m in lower for m in [
                    ":raises:", "raises:",
                ])

            entities.append(entity)

    result.entities = entities

    # Count public entities and documented ones
    for ent in entities:
        if ent.is_public:
            result.total_public += 1
            if ent.has_docstring:
                result.documented += 1

    if result.total_public > 0:
        result.coverage = result.documented / result.total_public

    # Generate issues
    for ent in entities:
        if not ent.is_public:
            continue

        if not ent.has_docstring:
            severity = "high" if ent.kind == "class" else "medium"
            result.issues.append(DocstringIssue(
                file_path=filepath,
                line_number=ent.line_number,
                name=ent.name,
                kind=ent.kind,
                issue_type="missing",
                message=f"Public {ent.kind} '{ent.name}' has no docstring",
                severity=severity,
            ))
        else:
            doc = _get_docstring_for_entity(tree, ent)
            if doc:
                quality_issues = _check_docstring_quality(doc, ent.param_count)
                for issue_type in quality_issues:
                    msgs = {
                        "too_short": f"Docstring for '{ent.name}' is too short (<10 chars)",
                        "no_params": f"Docstring for '{ent.name}' doesn't document {ent.param_count} parameter(s)",
                    }
                    result.issues.append(DocstringIssue(
                        file_path=filepath,
                        line_number=ent.line_number,
                        name=ent.name,
                        kind=ent.kind,
                        issue_type=issue_type,
                        message=msgs.get(issue_type, issue_type),
                        severity="low",
                    ))

    result.total_public = result.total_public  # already set
    return result


def _get_docstring_for_entity(tree: ast.Module, entity: EntityInfo) -> Optional[str]:
    """Find docstring for a specific entity by matching name and line."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == entity.name and node.lineno == entity.line_number:
                return _get_docstring(node)
        elif isinstance(node, ast.Module) and entity.kind == "module":
            return _get_docstring(node)
    return None


# --- Directory scanning ---

def scan(path: str) -> ScanResult:
    """Scan a file or directory for docstring coverage."""
    result = ScanResult(root=path)
    files = find_python_files(path)

    by_kind: dict[str, dict[str, int]] = {}

    for filepath in files:
        file_result = analyze_file(filepath)
        result.files_scanned += 1

        if file_result.error:
            result.errors.append(f"{filepath}: {file_result.error}")
            continue

        result.total_public += file_result.total_public
        result.total_documented += file_result.documented
        result.total_issues += len(file_result.issues)
        result.issues.extend(file_result.issues)
        result.file_results.append(file_result)

        for ent in file_result.entities:
            if ent.is_public:
                kind = ent.kind
                if kind not in by_kind:
                    by_kind[kind] = {"total": 0, "documented": 0}
                by_kind[kind]["total"] += 1
                if ent.has_docstring:
                    by_kind[kind]["documented"] += 1

    result.by_kind = by_kind
    if result.total_public > 0:
        result.coverage = result.total_documented / result.total_public

    return result


# --- Scoring ---

def _score_to_grade(score: int) -> str:
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


def compute_score(scan_result: ScanResult) -> ScoreResult:
    """Compute a health score (0-100) for docstring coverage."""
    if scan_result.total_public == 0:
        return ScoreResult(
            score=100, grade="A", total_public=0, total_documented=0,
            coverage=1.0, total_issues=0,
        )

    coverage = scan_result.coverage

    # Base score: coverage percentage (0-100)
    base_score = int(coverage * 100)

    # Quality penalty: deduct for issues beyond missing docstrings
    quality_issues = sum(1 for i in scan_result.issues if i.issue_type != "missing")
    quality_penalty = min(10, quality_issues)  # cap at 10 points

    score = max(0, min(100, base_score - quality_penalty))
    grade = _score_to_grade(score)

    # Worst files
    worst = sorted(
        [fr for fr in scan_result.file_results if fr.total_public > 0],
        key=lambda f: f.coverage,
    )[:5]
    worst_files = [
        {
            "file": fr.file_path,
            "coverage": round(fr.coverage, 3),
            "public": fr.total_public,
            "documented": fr.documented,
            "issues": len(fr.issues),
        }
        for fr in worst
    ]

    return ScoreResult(
        score=score,
        grade=grade,
        total_public=scan_result.total_public,
        total_documented=scan_result.total_documented,
        coverage=round(coverage, 4),
        total_issues=scan_result.total_issues,
        by_kind=scan_result.by_kind,
        worst_files=worst_files,
    )


# --- CLI ---

def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = argv if argv is not None else sys.argv[1:]

    path = "."
    json_mode = False
    score_mode = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--json", "-j"):
            json_mode = True
        elif arg in ("--score", "-s"):
            score_mode = True
        elif arg in ("--help", "-h"):
            print(__doc__ or "ai-docstring-audit")
            return 0
        elif arg in ("--version", "-V"):
            print(f"ai-docstring-audit {__version__}")
            return 0
        elif not arg.startswith("-"):
            path = arg
        i += 1

    if not os.path.exists(path):
        print(f"Error: path not found: {path}", file=sys.stderr)
        return 1

    result = scan(path)

    if score_mode:
        score = compute_score(result)
        if json_mode:
            print(json.dumps(score.to_dict(), indent=2))
        else:
            print(f"Docstring Coverage Score: {score.score}/100 (Grade: {score.grade})")
            print(f"  Public entities: {score.total_public}")
            print(f"  Documented: {score.total_documented}")
            print(f"  Coverage: {score.coverage:.1%}")
            if score.worst_files:
                print(f"\n  Worst files:")
                for wf in score.worst_files:
                    print(f"    {wf['file']}: {wf['coverage']:.0%} ({wf['documented']}/{wf['public']})")
        return 0

    if json_mode:
        data = asdict(result)
        # Remove bulky file_results for cleaner output
        del data["file_results"]
        print(json.dumps(data, indent=2))
        return 0

    # Human-readable output
    print(f"Docstring Audit: {result.root}")
    print(f"  Files scanned: {result.files_scanned}")
    print(f"  Public entities: {result.total_public}")
    print(f"  Documented: {result.total_documented} ({result.coverage:.1%})")
    print(f"  Issues: {result.total_issues}")

    if result.by_kind:
        print(f"\n  By kind:")
        for kind, counts in sorted(result.by_kind.items()):
            total = counts["total"]
            documented = counts["documented"]
            pct = documented / total * 100 if total else 0
            print(f"    {kind}: {documented}/{total} ({pct:.0f}%)")

    if result.issues:
        print(f"\n  Top issues (showing up to 20):")
        for issue in result.issues[:20]:
            print(f"    {issue.file_path}:{issue.line_number} [{issue.severity}] {issue.message}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"    {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
