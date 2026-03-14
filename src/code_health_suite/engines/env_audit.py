#!/usr/bin/env python3
"""ai-env-audit: Environment variable usage auditor.

Scans .env files and source code to find mismatches, missing variables,
potential secrets in templates, and other env management issues.

Supports Python (os.environ, os.getenv), JavaScript/TypeScript (process.env),
and shell scripts ($VAR, ${VAR}).

Usage:
    ai-env-audit                     # scan current directory
    ai-env-audit path/to/project     # scan specific directory
    ai-env-audit --json              # JSON output
    ai-env-audit --score             # health score (0-100 + A-F grade)
"""
from __future__ import annotations

import argparse
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
class EnvVar:
    """An environment variable found in a .env file."""
    name: str
    value: str = ""
    source_file: str = ""
    line_number: int = 0
    has_value: bool = True


@dataclass
class EnvReference:
    """A reference to an environment variable in source code."""
    name: str
    file_path: str
    line_number: int
    context: str = ""  # the matching line
    language: str = ""  # python, javascript, shell


@dataclass
class Finding:
    """An audit finding."""
    check: str  # undefined, unused, missing_template, secret_in_template, duplicate, empty
    severity: str  # HIGH, MODERATE, LOW, INFO
    variable: str
    message: str
    files: list = field(default_factory=list)


# --- Secret patterns ---

SECRET_PATTERNS = [
    re.compile(r"(password|passwd|pwd)", re.IGNORECASE),
    re.compile(r"(secret|token|api.?key)", re.IGNORECASE),
    re.compile(r"(private.?key|priv.?key)", re.IGNORECASE),
    re.compile(r"(auth|credential|cred)", re.IGNORECASE),
    re.compile(r"(database.?url|db.?url|connection.?string)", re.IGNORECASE),
    re.compile(r"(access.?key|aws.?key)", re.IGNORECASE),
]

PLACEHOLDER_PATTERNS = [
    re.compile(r"^(your[_-]|change[_-]me|xxx|placeholder|TODO|CHANGEME|REPLACE)", re.IGNORECASE),
    re.compile(r"^<.*>$"),
    re.compile(r"^\.\.\.$"),
    re.compile(r"^$"),
]


def is_secret_name(name: str) -> bool:
    """Check if a variable name looks like it holds a secret."""
    return any(p.search(name) for p in SECRET_PATTERNS)


def is_placeholder_value(value: str) -> bool:
    """Check if a value looks like a placeholder (safe for templates)."""
    return any(p.match(value.strip()) for p in PLACEHOLDER_PATTERNS)


# --- .env file parser ---

def parse_env_file(path: Path) -> list[EnvVar]:
    """Parse a .env file into a list of EnvVar objects."""
    results = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return results

    for line_num, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Handle export prefix
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()

        # Parse KEY=VALUE or KEY (no value)
        if "=" in stripped:
            key, _, val = stripped.partition("=")
            key = key.strip()
            val = val.strip()
            # Remove surrounding quotes
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            if key and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                results.append(EnvVar(
                    name=key,
                    value=val,
                    source_file=str(path),
                    line_number=line_num,
                    has_value=bool(val),
                ))
        else:
            # Bare variable name (no =)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped):
                results.append(EnvVar(
                    name=stripped,
                    value="",
                    source_file=str(path),
                    line_number=line_num,
                    has_value=False,
                ))
    return results


# --- Source code scanner ---

# Patterns for env var references in different languages
PYTHON_PATTERNS = [
    # os.environ["KEY"], os.environ.get("KEY"), os.environ['KEY']
    re.compile(r"""os\.environ(?:\.get)?\s*\[\s*['"]([\w]+)['"]\s*\]"""),
    re.compile(r"""os\.environ\.get\s*\(\s*['"]([\w]+)['"]"""),
    re.compile(r"""os\.getenv\s*\(\s*['"]([\w]+)['"]"""),
    # dotenv style: config("KEY"), env("KEY")
    re.compile(r"""(?:config|env)\s*\(\s*['"]([\w]+)['"]"""),
]

JS_PATTERNS = [
    # process.env.KEY or process.env["KEY"] or process.env['KEY']
    re.compile(r"""process\.env\.([A-Z_][A-Z0-9_]*)"""),
    re.compile(r"""process\.env\[\s*['"]([\w]+)['"]\s*\]"""),
    # Vite/Next.js: import.meta.env.VITE_KEY
    re.compile(r"""import\.meta\.env\.([A-Z_][A-Z0-9_]*)"""),
]

SHELL_PATTERNS = [
    # $VAR or ${VAR} (but not $1, $@, $$, etc.)
    re.compile(r"""\$\{([A-Za-z_][A-Za-z0-9_]*)\}"""),
    re.compile(r"""\$([A-Z_][A-Z0-9_]{2,})"""),  # Only caps, min 3 chars to reduce noise
]

# File extensions to scan
SCAN_EXTENSIONS = {
    ".py": ("python", PYTHON_PATTERNS),
    ".js": ("javascript", JS_PATTERNS),
    ".jsx": ("javascript", JS_PATTERNS),
    ".ts": ("javascript", JS_PATTERNS),
    ".tsx": ("javascript", JS_PATTERNS),
    ".mjs": ("javascript", JS_PATTERNS),
    ".cjs": ("javascript", JS_PATTERNS),
    ".sh": ("shell", SHELL_PATTERNS),
    ".bash": ("shell", SHELL_PATTERNS),
    ".zsh": ("shell", SHELL_PATTERNS),
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".next", ".nuxt",
    "dist", "build", ".venv", "venv", "env", ".tox",
    ".mypy_cache", ".pytest_cache", "coverage", ".cache",
    "dist-local", ".turbo",
}

# Well-known system env vars to ignore
SYSTEM_VARS = {
    "HOME", "PATH", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
    "PWD", "OLDPWD", "HOSTNAME", "LOGNAME", "DISPLAY",
    "EDITOR", "VISUAL", "PAGER", "TMPDIR", "TMP", "TEMP",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
    "CI", "GITHUB_ACTIONS", "VERCEL", "NETLIFY", "HEROKU",
    "NODE_ENV", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
    "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "npm_package_version",
}


def scan_source_files(root: Path, max_files: int = 5000) -> list[EnvReference]:
    """Scan source files for environment variable references."""
    refs = []
    files_scanned = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in filenames:
            if files_scanned >= max_files:
                return refs

            ext = Path(fname).suffix.lower()
            if ext not in SCAN_EXTENSIONS:
                continue

            fpath = Path(dirpath) / fname
            language, patterns = SCAN_EXTENSIONS[ext]

            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            files_scanned += 1

            for line_num, line in enumerate(text.splitlines(), 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue

                for pattern in patterns:
                    for match in pattern.finditer(line):
                        var_name = match.group(1)
                        if var_name not in SYSTEM_VARS:
                            refs.append(EnvReference(
                                name=var_name,
                                file_path=str(fpath),
                                line_number=line_num,
                                context=stripped[:120],
                                language=language,
                            ))
    return refs


# --- .env file discovery ---

ENV_FILE_PATTERNS = [
    ".env", ".env.local", ".env.development", ".env.production",
    ".env.staging", ".env.test", ".env.example", ".env.sample",
    ".env.template", ".env.defaults",
]


def find_env_files(root: Path) -> dict[str, list[EnvVar]]:
    """Find and parse all .env files in the project root."""
    env_files = {}
    for pattern in ENV_FILE_PATTERNS:
        fpath = root / pattern
        if fpath.is_file():
            vars_list = parse_env_file(fpath)
            env_files[pattern] = vars_list
    return env_files


def classify_env_file(name: str) -> str:
    """Classify an env file as 'template' or 'actual'."""
    if any(t in name for t in ("example", "sample", "template", "defaults")):
        return "template"
    return "actual"


# --- Audit engine ---

def run_audit(root: Path) -> list[Finding]:
    """Run the full environment variable audit."""
    findings = []

    # 1. Parse all .env files
    env_files = find_env_files(root)
    if not env_files:
        # No .env files at all — check if code uses env vars
        refs = scan_source_files(root)
        if refs:
            unique_vars = {r.name for r in refs}
            findings.append(Finding(
                check="no_env_files",
                severity="MODERATE",
                variable="*",
                message=f"No .env files found, but code references {len(unique_vars)} environment variables",
                files=sorted(unique_vars)[:20],
            ))
        return findings

    # Separate actual vs template files
    actual_vars: dict[str, list[EnvVar]] = {}  # var_name -> list of definitions
    template_vars: dict[str, list[EnvVar]] = {}

    for fname, vars_list in env_files.items():
        classification = classify_env_file(fname)
        target = actual_vars if classification == "actual" else template_vars
        for var in vars_list:
            target.setdefault(var.name, []).append(var)

    # 2. Scan source code for references
    refs = scan_source_files(root)
    code_vars: dict[str, list[EnvReference]] = {}
    for ref in refs:
        code_vars.setdefault(ref.name, []).append(ref)

    all_defined = set(actual_vars.keys()) | set(template_vars.keys())
    all_referenced = set(code_vars.keys())

    # 3. Check: variables used in code but not defined in any .env file
    for var_name in sorted(all_referenced - all_defined):
        ref_files = sorted({r.file_path for r in code_vars[var_name]})
        findings.append(Finding(
            check="undefined",
            severity="HIGH",
            variable=var_name,
            message=f"Referenced in code but not defined in any .env file",
            files=ref_files[:5],
        ))

    # 4. Check: variables in .env but never referenced in code
    for var_name in sorted(all_defined - all_referenced):
        src_files = []
        for v in actual_vars.get(var_name, []) + template_vars.get(var_name, []):
            src_files.append(v.source_file)
        findings.append(Finding(
            check="unused",
            severity="LOW",
            variable=var_name,
            message=f"Defined in .env but never referenced in scanned source code",
            files=sorted(set(src_files)),
        ))

    # 5. Check: variables in actual .env but missing from template
    if template_vars:
        for var_name in sorted(set(actual_vars.keys()) - set(template_vars.keys())):
            if var_name in all_referenced:  # Only flag if actually used
                findings.append(Finding(
                    check="missing_template",
                    severity="MODERATE",
                    variable=var_name,
                    message=f"In .env but missing from template — new developers won't know about it",
                    files=[v.source_file for v in actual_vars[var_name]],
                ))

    # 6. Check: template vars missing from actual .env
    if actual_vars:
        for var_name in sorted(set(template_vars.keys()) - set(actual_vars.keys())):
            findings.append(Finding(
                check="missing_actual",
                severity="MODERATE",
                variable=var_name,
                message=f"In template but missing from .env — may cause runtime errors",
                files=[v.source_file for v in template_vars[var_name]],
            ))

    # 7. Check: secrets with real values in template files
    for var_name, var_list in template_vars.items():
        if is_secret_name(var_name):
            for var in var_list:
                if var.has_value and not is_placeholder_value(var.value):
                    findings.append(Finding(
                        check="secret_in_template",
                        severity="HIGH",
                        variable=var_name,
                        message=f"Secret variable has a real-looking value in template file — should use placeholder",
                        files=[var.source_file],
                    ))

    # 8. Check: duplicate definitions in same file
    for fname, vars_list in env_files.items():
        seen: dict[str, int] = {}
        for var in vars_list:
            if var.name in seen:
                findings.append(Finding(
                    check="duplicate",
                    severity="LOW",
                    variable=var.name,
                    message=f"Defined multiple times in {fname} (lines {seen[var.name]} and {var.line_number})",
                    files=[str(root / fname)],
                ))
            else:
                seen[var.name] = var.line_number

    # 9. Check: empty values in actual .env files
    for var_name, var_list in actual_vars.items():
        for var in var_list:
            if not var.has_value and classify_env_file(Path(var.source_file).name) == "actual":
                findings.append(Finding(
                    check="empty",
                    severity="INFO",
                    variable=var_name,
                    message=f"Empty value in {Path(var.source_file).name}",
                    files=[var.source_file],
                ))

    return findings


# --- Scoring ---

SEVERITY_WEIGHTS = {"HIGH": 10, "MODERATE": 5, "LOW": 2, "INFO": 1}


def calculate_score(findings: list[Finding], env_files: dict, code_vars: dict) -> tuple[int, str]:
    """Calculate a health score (0-100) and grade (A-F)."""
    if not env_files and not code_vars:
        return 100, "A"  # Nothing to audit

    penalty = sum(SEVERITY_WEIGHTS.get(f.severity, 1) for f in findings)

    # Scale penalty relative to total variables
    total_vars = max(len(set(v.name for vl in env_files.values() for v in vl) | set(code_vars.keys())), 1)
    # Each var can contribute max ~10 penalty points
    max_penalty = total_vars * 10
    normalized = min(penalty / max(max_penalty, 1), 1.0)

    score = max(0, int(100 * (1 - normalized)))

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return score, grade


def classify_profile(findings: list[Finding]) -> str:
    """Classify the type of env management issues."""
    checks = [f.check for f in findings]
    if not checks:
        return "clean"

    undefined_count = checks.count("undefined")
    unused_count = checks.count("unused")
    secret_count = checks.count("secret_in_template")
    template_count = checks.count("missing_template") + checks.count("missing_actual")

    if secret_count > 0 and secret_count >= len(checks) // 3:
        return "secret_exposure"
    if undefined_count > unused_count and undefined_count >= len(checks) // 3:
        return "missing_config"
    if unused_count > undefined_count and unused_count >= len(checks) // 3:
        return "config_bloat"
    if template_count >= len(checks) // 3:
        return "template_drift"
    if len(checks) > 5:
        return "mixed"
    return "minor_issues"


# --- Output formatting ---

def format_text(findings: list[Finding], root: Path, env_files: dict, code_refs: list) -> str:
    """Format findings as human-readable text."""
    lines = [f"ai-env-audit v{__version__} — {root}"]
    lines.append("=" * 60)

    # Summary
    env_count = len(env_files)
    all_env_vars = set()
    for vl in env_files.values():
        for v in vl:
            all_env_vars.add(v.name)
    code_var_count = len({r.name for r in code_refs})

    lines.append(f"Env files found: {env_count}")
    lines.append(f"Variables in .env files: {len(all_env_vars)}")
    lines.append(f"Variables referenced in code: {code_var_count}")
    lines.append("")

    if not findings:
        lines.append("No issues found. Environment configuration looks clean.")
        return "\n".join(lines)

    # Group by severity
    by_severity = {}
    for f in findings:
        by_severity.setdefault(f.severity, []).append(f)

    for sev in ["HIGH", "MODERATE", "LOW", "INFO"]:
        if sev not in by_severity:
            continue
        lines.append(f"--- {sev} ---")
        for f in by_severity[sev]:
            lines.append(f"  [{f.check}] {f.variable}: {f.message}")
            if f.files:
                for fp in f.files[:3]:
                    lines.append(f"    → {fp}")
        lines.append("")

    # Counts
    lines.append(f"Total: {len(findings)} findings "
                 f"({len(by_severity.get('HIGH', []))} high, "
                 f"{len(by_severity.get('MODERATE', []))} moderate, "
                 f"{len(by_severity.get('LOW', []))} low, "
                 f"{len(by_severity.get('INFO', []))} info)")

    return "\n".join(lines)


def format_score(score: int, grade: str, profile: str, findings: list[Finding]) -> str:
    """Format score output."""
    lines = [
        f"Score: {score}/100 (Grade: {grade})",
        f"Profile: {profile}",
        f"Findings: {len(findings)}",
    ]
    by_sev = {}
    for f in findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    breakdown = ", ".join(f"{s}={c}" for s, c in sorted(by_sev.items()))
    if breakdown:
        lines.append(f"Breakdown: {breakdown}")
    return "\n".join(lines)


def format_json(findings: list[Finding], root: Path, env_files: dict,
                code_refs: list, show_score: bool = False) -> str:
    """Format findings as JSON."""
    result = {
        "version": __version__,
        "root": str(root),
        "env_files": list(env_files.keys()),
        "findings": [asdict(f) for f in findings],
        "summary": {
            "total": len(findings),
            "high": sum(1 for f in findings if f.severity == "HIGH"),
            "moderate": sum(1 for f in findings if f.severity == "MODERATE"),
            "low": sum(1 for f in findings if f.severity == "LOW"),
            "info": sum(1 for f in findings if f.severity == "INFO"),
        },
    }
    if show_score:
        code_vars = {}
        for r in code_refs:
            code_vars.setdefault(r.name, []).append(r)
        score, grade = calculate_score(findings, env_files, code_vars)
        result["score"] = score
        result["grade"] = grade
        result["profile"] = classify_profile(findings)
    return json.dumps(result, indent=2)


# --- CLI ---

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-env-audit",
        description="Audit environment variable usage and .env file hygiene",
    )
    parser.add_argument("path", nargs="?", default=".", help="Project directory to scan")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--score", action="store_true", help="Show health score (0-100)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    # Run audit
    findings = run_audit(root)
    env_files = find_env_files(root)
    code_refs = scan_source_files(root)

    if args.json:
        print(format_json(findings, root, env_files, code_refs, show_score=args.score))
    elif args.score:
        code_vars = {}
        for r in code_refs:
            code_vars.setdefault(r.name, []).append(r)
        score, grade = calculate_score(findings, env_files, code_vars)
        profile = classify_profile(findings)
        print(format_score(score, grade, profile, findings))
    else:
        print(format_text(findings, root, env_files, code_refs))

    # Exit code: 1 if any HIGH findings
    return 1 if any(f.severity == "HIGH" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
