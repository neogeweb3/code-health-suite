#!/usr/bin/env python3
"""ai-todo-scanner: Technical debt comment tracker.

Scans source files to find TODO, FIXME, HACK, XXX, NOTE, and NOQA comments.
Optionally enriches with git blame metadata (author, age).
Computes a health score based on density and severity of debt markers.

Uses regex matching — zero AST dependency, works with any text file.
Focused on Python but supports any language with # or // comments.

Usage:
    ai-todo-scanner                     # scan current directory
    ai-todo-scanner path/to/project     # scan specific directory
    ai-todo-scanner --json              # JSON output
    ai-todo-scanner --score             # health score (0-100 + A-F grade)
    ai-todo-scanner --blame             # enrich with git blame metadata
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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

# File extensions to scan
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".sql", ".r", ".R",
}

# Tag patterns sorted by severity (highest first)
TAG_SEVERITY = {
    "FIXME": "high",
    "HACK": "high",
    "BUG": "high",
    "XXX": "medium",
    "TODO": "medium",
    "NOQA": "low",
    "NOTE": "low",
    "CHANGED": "low",
    "OPTIMIZE": "medium",
    "REFACTOR": "medium",
}

# Severity weights for scoring
SEVERITY_WEIGHTS = {
    "high": 3,
    "medium": 2,
    "low": 1,
}

# Regex to match TODO-style tags in comments
# Matches: TODO, FIXME, HACK, XXX, NOTE, NOQA, BUG, OPTIMIZE, REFACTOR, CHANGED
# With optional colon, parenthesized author, and trailing message
_TAG_PATTERN = re.compile(
    r"(?:#|//|/\*|\*|--|;)\s*"  # comment prefix
    r"(?P<tag>" + "|".join(re.escape(t) for t in TAG_SEVERITY) + r")"  # tag
    r"(?:\((?P<inline_author>[^)]+)\))?"  # optional (author)
    r"\s*:?\s*"  # optional colon + whitespace
    r"(?P<message>.+?)?\s*$",  # trailing message
    re.IGNORECASE,
)


# --- Data models ---

@dataclass
class TodoItem:
    """A single TODO/FIXME/HACK comment found in source code."""
    file_path: str
    line_number: int
    tag: str          # TODO, FIXME, HACK, etc.
    severity: str     # high, medium, low
    message: str
    inline_author: str = ""
    # Git blame fields (populated when --blame is used)
    blame_author: str = ""
    blame_date: str = ""
    blame_age_days: int = -1


@dataclass
class FileResult:
    """TODO scan results for a single file."""
    file_path: str
    items: list[TodoItem] = field(default_factory=list)
    lines_scanned: int = 0
    error: str = ""


@dataclass
class ScanResult:
    """Aggregate TODO scan results."""
    root: str
    files_scanned: int = 0
    total_lines: int = 0
    total_items: int = 0
    by_tag: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_file: dict[str, int] = field(default_factory=dict)
    items: list[TodoItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ScoreResult:
    """Health score for technical debt markers."""
    score: int       # 0-100
    grade: str       # A-F
    total_items: int
    total_lines: int
    density: float   # items per 1000 lines
    weighted_debt: int
    by_tag: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    hotspot_files: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- File discovery ---

def find_source_files(path: str, extensions: set[str] | None = None) -> list[str]:
    """Find source files under a path, skipping common non-source dirs."""
    exts = extensions or SOURCE_EXTENSIONS

    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        return [path] if ext in exts else []

    files = []
    for root, dirs, filenames in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fname in sorted(filenames):
            _, ext = os.path.splitext(fname)
            if ext in exts:
                files.append(os.path.join(root, fname))
    return files


# --- Parsing ---

def analyze_file(filepath: str) -> FileResult:
    """Scan a single file for TODO-style comments."""
    result = FileResult(file_path=filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, IOError) as e:
        result.error = str(e)
        return result

    result.lines_scanned = len(lines)

    for i, line in enumerate(lines, start=1):
        match = _TAG_PATTERN.search(line)
        if match:
            tag = match.group("tag").upper()
            severity = TAG_SEVERITY.get(tag, "low")
            message = (match.group("message") or "").strip()
            inline_author = (match.group("inline_author") or "").strip()

            result.items.append(TodoItem(
                file_path=filepath,
                line_number=i,
                tag=tag,
                severity=severity,
                message=message,
                inline_author=inline_author,
            ))

    return result


# --- Git blame enrichment ---

def _git_blame_line(filepath: str, line_number: int) -> tuple[str, str]:
    """Get git blame info for a specific line. Returns (author, date)."""
    try:
        output = subprocess.run(
            ["git", "blame", "-L", f"{line_number},{line_number}",
             "--porcelain", filepath],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.abspath(filepath)) or ".",
        )
        if output.returncode != 0:
            return "", ""

        author = ""
        date = ""
        for bl in output.stdout.splitlines():
            if bl.startswith("author "):
                author = bl[7:]
            elif bl.startswith("author-time "):
                import datetime
                ts = int(bl[12:])
                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        return author, date
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return "", ""


def enrich_with_blame(items: list[TodoItem]) -> None:
    """Enrich TODO items with git blame metadata in-place."""
    import datetime

    today = datetime.date.today()

    for item in items:
        author, date_str = _git_blame_line(item.file_path, item.line_number)
        item.blame_author = author
        item.blame_date = date_str
        if date_str:
            try:
                d = datetime.date.fromisoformat(date_str)
                item.blame_age_days = (today - d).days
            except ValueError:
                pass


# --- Aggregate analysis ---

def scan(path: str, extensions: set[str] | None = None) -> ScanResult:
    """Scan a file or directory for TODO-style comments."""
    result = ScanResult(root=path)
    files = find_source_files(path, extensions)

    for filepath in files:
        file_result = analyze_file(filepath)
        result.files_scanned += 1
        result.total_lines += file_result.lines_scanned
        result.total_items += len(file_result.items)
        result.items.extend(file_result.items)

        if file_result.items:
            result.by_file[filepath] = len(file_result.items)

        if file_result.error:
            result.errors.append(f"{filepath}: {file_result.error}")

    # Count by tag and severity
    for item in result.items:
        result.by_tag[item.tag] = result.by_tag.get(item.tag, 0) + 1
        result.by_severity[item.severity] = result.by_severity.get(item.severity, 0) + 1

    return result


def compute_score(result: ScanResult) -> ScoreResult:
    """Compute a technical debt health score (0-100).

    Scoring logic:
    - Base: 100 points
    - Penalty: weighted_debt / total_lines * 10000
    - Each high-severity item counts 3x, medium 2x, low 1x
    - Clamped to [0, 100]
    """
    weighted_debt = sum(
        SEVERITY_WEIGHTS.get(item.severity, 1)
        for item in result.items
    )

    if result.total_lines == 0:
        return ScoreResult(
            score=100, grade="A", total_items=0, total_lines=0,
            density=0.0, weighted_debt=0,
        )

    density = (result.total_items / result.total_lines) * 1000  # per 1K lines

    # Penalty: weighted_debt per 1000 lines * scaling factor
    penalty = (weighted_debt / result.total_lines) * 1000 * 5
    score = max(0, min(100, int(100 - penalty)))

    grade = _score_to_grade(score)

    # Top hotspot files
    hotspots = sorted(result.by_file.items(), key=lambda x: x[1], reverse=True)[:10]
    hotspot_files = [{"file": f, "count": c} for f, c in hotspots]

    return ScoreResult(
        score=score,
        grade=grade,
        total_items=result.total_items,
        total_lines=result.total_lines,
        density=round(density, 2),
        weighted_debt=weighted_debt,
        by_tag=result.by_tag,
        by_severity=result.by_severity,
        hotspot_files=hotspot_files,
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
    lines.append(f"TODO/FIXME Scanner — {result.root}")
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Lines scanned: {result.total_lines}")
    lines.append(f"Debt markers found: {result.total_items}")
    lines.append(f"Density: {score_result.density}/1K lines")
    lines.append(f"Score: {score_result.score}/100 ({score_result.grade})")
    lines.append("")

    if result.by_tag:
        lines.append("By tag:")
        for tag, count in sorted(result.by_tag.items(), key=lambda x: x[1], reverse=True):
            sev = TAG_SEVERITY.get(tag, "low")
            lines.append(f"  {tag}: {count} ({sev})")
        lines.append("")

    if result.by_severity:
        lines.append("By severity:")
        for sev in ("high", "medium", "low"):
            if sev in result.by_severity:
                lines.append(f"  {sev}: {result.by_severity[sev]}")
        lines.append("")

    if score_result.hotspot_files:
        lines.append("Hotspot files:")
        for h in score_result.hotspot_files:
            lines.append(f"  {h['file']}: {h['count']} items")
        lines.append("")

    if result.items:
        lines.append("Items:")
        for item in result.items:
            blame = ""
            if item.blame_author:
                blame = f" [{item.blame_author} {item.blame_date}]"
            author = f" ({item.inline_author})" if item.inline_author else ""
            msg = f": {item.message}" if item.message else ""
            lines.append(f"  {item.file_path}:{item.line_number}: {item.tag}{author}{msg}{blame}")

    if result.errors:
        lines.append("")
        lines.append("Errors:")
        for e in result.errors:
            lines.append(f"  {e}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan source code for TODO/FIXME/HACK technical debt markers."
    )
    parser.add_argument("path", nargs="?", default=".", help="Path to scan")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--score", action="store_true", help="Show score only")
    parser.add_argument("--blame", action="store_true", help="Enrich with git blame metadata")
    parser.add_argument("--tag", type=str, default=None,
                        help="Filter by tag (e.g., TODO, FIXME, HACK)")
    parser.add_argument("--severity", type=str, default=None,
                        choices=["high", "medium", "low"],
                        help="Filter by minimum severity")
    args = parser.parse_args(argv)

    result = scan(args.path)

    # Filter by tag
    if args.tag:
        tag_upper = args.tag.upper()
        result.items = [i for i in result.items if i.tag == tag_upper]
        result.total_items = len(result.items)

    # Filter by severity
    if args.severity:
        sev_order = {"high": 3, "medium": 2, "low": 1}
        min_sev = sev_order.get(args.severity, 0)
        result.items = [i for i in result.items if sev_order.get(i.severity, 0) >= min_sev]
        result.total_items = len(result.items)

    # Enrich with blame
    if args.blame:
        enrich_with_blame(result.items)

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
