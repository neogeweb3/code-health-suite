#!/usr/bin/env python3
"""ai-git-audit: Automated git commit audit tool.

Analyzes recent git commits by extracting changed files and running static
analysis (security scan, complexity) on each. Produces structured audit
reports with per-commit grades. Zero external dependencies beyond sibling
ai-* tools (gracefully degrades if unavailable).

Usage:
    ai-git-audit                           # audit last 10 commits in cwd
    ai-git-audit /path/to/repo             # audit specific repo
    ai-git-audit --since "3 days ago"      # commits since date
    ai-git-audit --commits 5               # last N commits
    ai-git-audit --json                    # JSON output
    ai-git-audit --author "Neo"            # filter by author
    ai-git-audit --severity medium         # min severity for security findings
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


__version__ = "0.1.0"

# --- Grade thresholds ---

GRADE_THRESHOLDS = [
    (95, "A+"), (90, "A"), (85, "A-"),
    (80, "B+"), (75, "B"), (70, "B-"),
    (60, "C+"), (50, "C"), (40, "C-"),
    (0, "D"),
]

SEVERITY_WEIGHT = {"critical": 20, "high": 10, "medium": 3, "low": 1, "info": 0}


# --- Data models ---

@dataclass
class FileChange:
    path: str
    status: str  # A=added, M=modified, D=deleted, R=renamed
    additions: int = 0
    deletions: int = 0


@dataclass
class SecurityFinding:
    file: str
    line: int
    rule: str
    severity: str
    message: str
    cwe: str = ""


@dataclass
class ComplexityFinding:
    file: str
    function: str
    line: int
    cyclomatic: int
    cognitive: int = 0
    rating: str = ""


@dataclass
class CommitAudit:
    sha: str
    short_sha: str
    author: str
    date: str
    message: str
    files: list[FileChange] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    security_findings: list[SecurityFinding] = field(default_factory=list)
    complexity_findings: list[ComplexityFinding] = field(default_factory=list)
    grade: str = "A+"
    score: int = 100
    python_files_changed: int = 0


@dataclass
class AuditReport:
    repo_path: str
    repo_name: str
    commits_analyzed: int = 0
    total_files_changed: int = 0
    total_additions: int = 0
    total_deletions: int = 0
    total_security_findings: int = 0
    security_by_severity: dict = field(default_factory=dict)
    total_high_complexity: int = 0
    overall_grade: str = "A+"
    overall_score: int = 100
    commits: list[CommitAudit] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)
    tools_unavailable: list[str] = field(default_factory=list)


# --- Git operations ---

def run_git(args: list[str], cwd: str) -> tuple[int, str]:
    """Run a git command and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, str(e)


def get_commits(
    repo_path: str,
    num_commits: int = 10,
    since: str | None = None,
    author: str | None = None,
) -> list[dict]:
    """Get recent commits from git log."""
    args = [
        "log",
        f"--max-count={num_commits}",
        "--format=%H|%h|%an|%ai|%s",
        "--no-merges",
    ]
    if since:
        args.append(f"--since={since}")
    if author:
        args.append(f"--author={author}")

    rc, output = run_git(args, repo_path)
    if rc != 0 or not output:
        return []

    commits = []
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            commits.append({
                "sha": parts[0],
                "short_sha": parts[1],
                "author": parts[2],
                "date": parts[3],
                "message": parts[4],
            })
    return commits


def get_changed_files(repo_path: str, sha: str) -> list[FileChange]:
    """Get files changed in a specific commit."""
    rc, output = run_git(
        ["diff-tree", "--root", "--no-commit-id", "-r", "--numstat", "-M", sha],
        repo_path,
    )
    if rc != 0 or not output:
        return []

    # Also get status letters
    rc2, status_output = run_git(
        ["diff-tree", "--root", "--no-commit-id", "-r", "--name-status", "-M", sha],
        repo_path,
    )
    status_map = {}
    if rc2 == 0 and status_output:
        for line in status_output.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                status_map[parts[-1]] = parts[0][0]  # First char of status

    files = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            adds = int(parts[0]) if parts[0] != "-" else 0
            dels = int(parts[1]) if parts[1] != "-" else 0
            path = parts[-1]  # Use last part (handles renames)
            status = status_map.get(path, "M")
            files.append(FileChange(
                path=path, status=status, additions=adds, deletions=dels,
            ))
    return files


def get_file_at_commit(repo_path: str, sha: str, file_path: str) -> str | None:
    """Get file contents at a specific commit."""
    rc, output = run_git(["show", f"{sha}:{file_path}"], repo_path)
    if rc != 0:
        return None
    return output


# --- Tool runners ---

def find_tool(tool_name: str) -> str | None:
    """Find a sibling ai-* tool by name."""
    workspace = Path(__file__).parent.parent
    candidates = [
        workspace / tool_name / f"{tool_name.replace('-', '_')}.py",
        workspace / tool_name / f"{tool_name}.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def run_security_scan(file_path: str, severity: str = "low") -> list[SecurityFinding]:
    """Run ai-security-scan on a file."""
    tool = find_tool("ai-security-scan")
    if not tool:
        return []

    try:
        result = subprocess.run(
            [sys.executable, tool, "-f", file_path, "--json", "--severity", severity],
            capture_output=True, text=True, timeout=30,
        )
        # Exit codes vary: 0=clean, 1/2=findings found
        if result.stdout:
            data = json.loads(result.stdout)
        else:
            return []
        findings = data.get("findings", [])
        return [
            SecurityFinding(
                file=f.get("file", file_path),
                line=f.get("line", 0),
                rule=f.get("rule", ""),
                severity=f.get("severity", "info"),
                message=f.get("message", ""),
                cwe=f.get("cwe", ""),
            )
            for f in findings
        ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        return []


def run_complexity_analysis(
    file_path: str, threshold: int = 10,
) -> list[ComplexityFinding]:
    """Run ai-complexity on a file."""
    tool = find_tool("ai-complexity")
    if not tool:
        return []

    try:
        result = subprocess.run(
            [
                sys.executable, tool, "-f", file_path,
                "--json", "--threshold", str(threshold),
            ],
            capture_output=True, text=True, timeout=30,
        )
        # Exit codes: 0=clean, 1=above threshold
        if result.stdout:
            data = json.loads(result.stdout)
        else:
            return []
        # Functions are nested under modules[].functions[]
        functions = []
        for module in data.get("modules", []):
            functions.extend(module.get("functions", []))
        # Fallback: top-level functions key
        if not functions:
            functions = data.get("functions", [])
        return [
            ComplexityFinding(
                file=f.get("file", file_path),
                function=f.get("name", f.get("function", "")),
                line=f.get("line", 0),
                cyclomatic=f.get("cyclomatic", f.get("complexity", 0)),
                cognitive=f.get("cognitive", 0),
                rating=f.get("grade", ""),
            )
            for f in functions
            if f.get("cyclomatic", f.get("complexity", 0)) >= threshold
        ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        return []


def check_tools() -> tuple[list[str], list[str]]:
    """Check which sibling tools are available."""
    tools = ["ai-security-scan", "ai-complexity"]
    available = []
    unavailable = []
    for tool in tools:
        if find_tool(tool):
            available.append(tool)
        else:
            unavailable.append(tool)
    return available, unavailable


# --- Scoring ---

def compute_grade(score: int) -> str:
    """Convert numeric score to letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "D"


def score_commit(audit: CommitAudit) -> tuple[int, str]:
    """Score a commit based on findings."""
    score = 100

    # Deduct for security findings
    for f in audit.security_findings:
        score -= SEVERITY_WEIGHT.get(f.severity, 1)

    # Deduct for high-complexity functions
    for f in audit.complexity_findings:
        if f.cyclomatic >= 20:
            score -= 5
        elif f.cyclomatic >= 15:
            score -= 3
        elif f.cyclomatic >= 10:
            score -= 1

    score = max(0, min(100, score))
    return score, compute_grade(score)


# --- Analysis pipeline ---

def audit_commit(
    repo_path: str,
    commit: dict,
    severity: str = "low",
    complexity_threshold: int = 10,
    available_tools: list[str] | None = None,
) -> CommitAudit:
    """Audit a single commit."""
    if available_tools is None:
        available_tools = []

    files = get_changed_files(repo_path, commit["sha"])

    audit = CommitAudit(
        sha=commit["sha"],
        short_sha=commit["short_sha"],
        author=commit["author"],
        date=commit["date"],
        message=commit["message"],
        files=files,
        total_additions=sum(f.additions for f in files),
        total_deletions=sum(f.deletions for f in files),
    )

    # Analyze changed Python files (non-deleted)
    python_files = [
        f for f in files
        if f.path.endswith(".py") and f.status != "D"
    ]
    audit.python_files_changed = len(python_files)

    import tempfile

    for file_change in python_files:
        # Get file contents at this commit
        content = get_file_at_commit(repo_path, commit["sha"], file_change.path)
        if content is None:
            continue

        # Write to temp file for analysis
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if "ai-security-scan" in available_tools:
                findings = run_security_scan(tmp_path, severity)
                # Fix file paths in findings
                for f in findings:
                    f.file = file_change.path
                audit.security_findings.extend(findings)

            if "ai-complexity" in available_tools:
                findings = run_complexity_analysis(tmp_path, complexity_threshold)
                for f in findings:
                    f.file = file_change.path
                audit.complexity_findings.extend(findings)
        finally:
            os.unlink(tmp_path)

    audit.score, audit.grade = score_commit(audit)
    return audit


def run_audit(
    repo_path: str,
    num_commits: int = 10,
    since: str | None = None,
    author: str | None = None,
    severity: str = "low",
    complexity_threshold: int = 10,
) -> AuditReport:
    """Run full audit on a repository."""
    repo_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(repo_path)

    available, unavailable = check_tools()

    commits = get_commits(repo_path, num_commits, since, author)

    report = AuditReport(
        repo_path=repo_path,
        repo_name=repo_name,
        commits_analyzed=len(commits),
        tools_available=available,
        tools_unavailable=unavailable,
    )

    for commit in commits:
        audit = audit_commit(
            repo_path, commit, severity, complexity_threshold, available,
        )
        report.commits.append(audit)

    # Aggregate stats
    report.total_files_changed = sum(len(c.files) for c in report.commits)
    report.total_additions = sum(c.total_additions for c in report.commits)
    report.total_deletions = sum(c.total_deletions for c in report.commits)
    report.total_security_findings = sum(
        len(c.security_findings) for c in report.commits
    )
    report.total_high_complexity = sum(
        len(c.complexity_findings) for c in report.commits
    )

    # Security by severity
    sev_counts: dict[str, int] = {}
    for c in report.commits:
        for f in c.security_findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
    report.security_by_severity = sev_counts

    # Overall score = average of commit scores
    if report.commits:
        avg = sum(c.score for c in report.commits) / len(report.commits)
        report.overall_score = int(avg)
        report.overall_grade = compute_grade(report.overall_score)

    return report


# --- Output formatting ---

def format_terminal(report: AuditReport) -> str:
    """Format report for terminal display."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Git Audit Report: {report.repo_name}")
    lines.append("=" * 60)
    lines.append(f"  Commits analyzed: {report.commits_analyzed}")
    lines.append(
        f"  Total changes: {report.total_files_changed} files "
        f"(+{report.total_additions}/-{report.total_deletions})"
    )
    lines.append(f"  Tools: {', '.join(report.tools_available) or 'none'}")
    if report.tools_unavailable:
        lines.append(
            f"  Unavailable: {', '.join(report.tools_unavailable)}"
        )
    lines.append("")

    for audit in report.commits:
        grade_marker = audit.grade
        lines.append(f"  [{grade_marker}] {audit.short_sha} {audit.message}")
        lines.append(f"      Author: {audit.author} | {audit.date}")
        lines.append(
            f"      Files: {len(audit.files)} "
            f"(+{audit.total_additions}/-{audit.total_deletions})"
            f" | Python: {audit.python_files_changed}"
        )

        if audit.security_findings:
            sev_summary = {}
            for f in audit.security_findings:
                sev_summary[f.severity] = sev_summary.get(f.severity, 0) + 1
            sev_str = ", ".join(
                f"{count} {sev}" for sev, count in sorted(
                    sev_summary.items(),
                    key=lambda x: SEVERITY_WEIGHT.get(x[0], 0),
                    reverse=True,
                )
            )
            lines.append(f"      Security: {len(audit.security_findings)} ({sev_str})")
            for f in audit.security_findings:
                lines.append(
                    f"        - {f.file}:{f.line} [{f.severity}] "
                    f"{f.rule}: {f.message}"
                )

        if audit.complexity_findings:
            lines.append(
                f"      Complexity: {len(audit.complexity_findings)} "
                f"above threshold"
            )
            for f in audit.complexity_findings:
                lines.append(
                    f"        - {f.file}:{f.line} {f.function} "
                    f"CC={f.cyclomatic}"
                )

        if not audit.security_findings and not audit.complexity_findings:
            lines.append("      No issues found")

        lines.append("")

    # Summary
    lines.append("-" * 60)
    lines.append("  Summary")
    lines.append("-" * 60)
    lines.append(
        f"  Security findings: {report.total_security_findings}"
    )
    if report.security_by_severity:
        sev_str = ", ".join(
            f"{count} {sev}"
            for sev, count in sorted(
                report.security_by_severity.items(),
                key=lambda x: SEVERITY_WEIGHT.get(x[0], 0),
                reverse=True,
            )
        )
        lines.append(f"    Breakdown: {sev_str}")
    lines.append(f"  High-complexity functions: {report.total_high_complexity}")
    lines.append(f"  Overall grade: {report.overall_grade} ({report.overall_score}/100)")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_json(report: AuditReport) -> str:
    """Format report as JSON."""
    data = asdict(report)
    return json.dumps(data, indent=2, default=str)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-git-audit",
        description="Automated git commit audit with static analysis",
    )
    parser.add_argument(
        "repo", nargs="?", default=".",
        help="Repository path (default: current directory)",
    )
    parser.add_argument(
        "--commits", "-n", type=int, default=10,
        help="Number of commits to audit (default: 10)",
    )
    parser.add_argument(
        "--since", "-s",
        help='Audit commits since date (e.g., "3 days ago", "2026-03-01")',
    )
    parser.add_argument(
        "--author", "-a",
        help="Filter commits by author",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--severity", default="low",
        choices=["critical", "high", "medium", "low", "info"],
        help="Minimum security finding severity (default: low)",
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=10,
        help="Complexity threshold for flagging functions (default: 10)",
    )
    parser.add_argument(
        "--version", "-V", action="version",
        version=f"ai-git-audit {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate repo path
    repo_path = os.path.abspath(args.repo)
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a directory", file=sys.stderr)
        return 2

    # Check it's a git repo
    rc, _ = run_git(["rev-parse", "--git-dir"], repo_path)
    if rc != 0:
        print(f"Error: {repo_path} is not a git repository", file=sys.stderr)
        return 2

    report = run_audit(
        repo_path=repo_path,
        num_commits=args.commits,
        since=args.since,
        author=args.author,
        severity=args.severity,
        complexity_threshold=args.threshold,
    )

    if args.json_output:
        print(format_json(report))
    else:
        print(format_terminal(report))

    # Exit code: 0=clean, 1=findings found
    if report.total_security_findings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
