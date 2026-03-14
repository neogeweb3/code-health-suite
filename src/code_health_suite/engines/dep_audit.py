#!/usr/bin/env python3
"""ai-dep-audit: Python dependency health checker.

Scans pyproject.toml or requirements.txt, checks PyPI for outdated packages,
and queries Google OSV for known vulnerabilities.

Usage:
    ai-dep-audit                     # scan current directory
    ai-dep-audit path/to/project     # scan specific directory
    ai-dep-audit -f requirements.txt # scan specific file
    ai-dep-audit --json              # JSON output
    ai-dep-audit --severity high     # filter by minimum severity
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


__version__ = "0.3.0"


# --- Data models ---

@dataclass
class Dependency:
    name: str
    specified_version: Optional[str] = None
    constraint: str = ""  # e.g. ">=", "==", "~="
    source_file: str = ""
    extras: str = ""  # e.g. "[security]"


@dataclass
class StaticFinding:
    """A finding from offline static analysis (no network required)."""
    check: str  # unpinned, wildcard, duplicate, conflict, hygiene
    severity: str  # HIGH, MODERATE, LOW, INFO
    package: str
    message: str
    source_files: list = field(default_factory=list)


@dataclass
class VersionInfo:
    current: Optional[str] = None
    latest: Optional[str] = None
    update_type: Optional[str] = None  # major, minor, patch, up-to-date


@dataclass
class Vulnerability:
    id: str
    summary: str
    severity: str = "UNKNOWN"  # CRITICAL, HIGH, MODERATE, LOW, UNKNOWN
    fixed_version: Optional[str] = None
    url: str = ""


@dataclass
class AuditResult:
    dependency: Dependency
    version_info: VersionInfo = field(default_factory=VersionInfo)
    vulnerabilities: list[Vulnerability] = field(default_factory=list)

    @property
    def is_outdated(self) -> bool:
        return self.version_info.update_type not in (None, "up-to-date")

    @property
    def has_vulns(self) -> bool:
        return len(self.vulnerabilities) > 0

    @property
    def max_severity(self) -> str:
        if not self.vulnerabilities:
            return "NONE"
        order = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1, "UNKNOWN": 0}
        return max(self.vulnerabilities, key=lambda v: order.get(v.severity, 0)).severity


# --- Parsers ---

def parse_requirements_txt(filepath: str) -> list[Dependency]:
    """Parse requirements.txt into dependency list."""
    deps = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Handle: package==1.0, package>=1.0, package~=1.0, package
            # Extras like [security] are skipped between name and constraint
            match = re.match(
                r"^([a-zA-Z0-9_][a-zA-Z0-9._-]*)(?:\[[^\]]*\])?\s*([><=~!]+)?\s*([\d][a-zA-Z0-9.*-]*)?",
                line,
            )
            if match:
                name = match.group(1).strip()
                constraint = match.group(2) or ""
                version = match.group(3) or None
                deps.append(Dependency(
                    name=name,
                    specified_version=version,
                    constraint=constraint,
                    source_file=filepath,
                ))
    return deps


def parse_pyproject_toml(filepath: str) -> list[Dependency]:
    """Parse pyproject.toml [project].dependencies into dependency list.

    Minimal TOML parser — handles the dependencies array without a full TOML library.
    """
    deps = []
    with open(filepath) as f:
        content = f.read()

    # Find dependencies array start, then bracket-depth parse to handle
    # extras like "pkg[extra]>=1.0" whose ] would break a naive regex
    start_match = re.search(r'(?:^|\n)\s*dependencies\s*=\s*\[', content)
    if not start_match:
        return deps

    start = start_match.end()
    depth = 1
    in_string = None
    i = start
    while i < len(content) and depth > 0:
        c = content[i]
        if in_string:
            if c == in_string and (i == 0 or content[i - 1] != '\\'):
                in_string = None
        else:
            if c == '"':
                in_string = '"'
            elif c == "'":
                in_string = "'"
            elif c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
        i += 1
    dep_block = content[start:i - 1]

    # Extract quoted strings
    for item in re.findall(r'"([^"]+)"|\'([^\']+)\'', dep_block):
        spec = item[0] or item[1]
        # Skip extras between name and constraint
        match = re.match(
            r"^([a-zA-Z0-9_][a-zA-Z0-9._-]*)(?:\[[^\]]*\])?\s*([><=~!]+)?\s*([\d][a-zA-Z0-9.*-]*)?",
            spec,
        )
        if match:
            deps.append(Dependency(
                name=match.group(1).strip(),
                specified_version=match.group(3) or None,
                constraint=match.group(2) or "",
                source_file=filepath,
            ))
    return deps


def find_and_parse_deps(target: str) -> list[Dependency]:
    """Auto-detect and parse dependency files from target path."""
    path = Path(target)

    if path.is_file():
        if path.name == "pyproject.toml":
            return parse_pyproject_toml(str(path))
        elif path.name in ("requirements.txt", "requirements-dev.txt"):
            return parse_requirements_txt(str(path))
        else:
            raise ValueError(f"Unsupported file: {path.name}")

    if path.is_dir():
        deps = []
        pyproject = path / "pyproject.toml"
        reqs = path / "requirements.txt"
        found_file = False
        if pyproject.exists():
            found_file = True
            deps.extend(parse_pyproject_toml(str(pyproject)))
        if reqs.exists():
            found_file = True
            deps.extend(parse_requirements_txt(str(reqs)))
        if not found_file:
            raise FileNotFoundError(
                f"No pyproject.toml or requirements.txt found in {path}"
            )
        return deps

    raise FileNotFoundError(f"Path not found: {target}")


# --- PyPI version checker ---

def query_pypi(package_name: str, timeout: int = 5) -> Optional[str]:
    """Get latest version of a package from PyPI JSON API."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("info", {}).get("version")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def parse_version_tuple(version_str: str) -> tuple[int, ...]:
    """Parse version string into comparable tuple."""
    parts = []
    for part in version_str.split("."):
        match = re.match(r"(\d+)", part)
        if match:
            parts.append(int(match.group(1)))
    return tuple(parts) if parts else (0,)


def classify_update(current: str, latest: str) -> str:
    """Classify update as major, minor, or patch."""
    cur = parse_version_tuple(current)
    lat = parse_version_tuple(latest)

    if cur >= lat:
        return "up-to-date"

    # Pad to same length
    max_len = max(len(cur), len(lat))
    cur_padded = cur + (0,) * (max_len - len(cur))
    lat_padded = lat + (0,) * (max_len - len(lat))

    if lat_padded[0] > cur_padded[0]:
        return "major"
    if len(lat_padded) > 1 and len(cur_padded) > 1 and lat_padded[1] > cur_padded[1]:
        return "minor"
    return "patch"


def check_version(dep: Dependency, timeout: int = 5) -> VersionInfo:
    """Check if a dependency is outdated against PyPI."""
    latest = query_pypi(dep.name, timeout=timeout)
    if latest is None:
        return VersionInfo()  # Can't determine

    if dep.specified_version:
        update_type = classify_update(dep.specified_version, latest)
    else:
        update_type = None  # No version pinned — can't compare

    return VersionInfo(
        current=dep.specified_version,
        latest=latest,
        update_type=update_type,
    )


# --- OSV vulnerability checker ---

def query_osv(package_name: str, version: Optional[str] = None, timeout: int = 10) -> list[Vulnerability]:
    """Query Google OSV API for known vulnerabilities."""
    url = "https://api.osv.dev/v1/query"
    payload: dict = {
        "package": {
            "name": package_name,
            "ecosystem": "PyPI",
        }
    }
    if version:
        payload["version"] = version

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError):
        return []

    vulns = []
    for vuln_data in result.get("vulns", []):
        vuln_id = vuln_data.get("id", "UNKNOWN")
        summary = vuln_data.get("summary", "No description available")

        # Extract severity
        severity = "UNKNOWN"
        for sev_entry in vuln_data.get("severity", []):
            score_str = sev_entry.get("score", "")
            # CVSS v3 score extraction
            cvss_match = re.search(r"CVSS:[\d.]+/AV:\w+.*", score_str)
            if cvss_match:
                # Try to extract base score from database_specific
                pass
        # Check database_specific for severity
        db_specific = vuln_data.get("database_specific", {})
        if "severity" in db_specific:
            severity = db_specific["severity"].upper()

        # Extract fixed version
        fixed_version = None
        for affected in vuln_data.get("affected", []):
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if "fixed" in event:
                        fixed_version = event["fixed"]

        # Build reference URL
        url_ref = ""
        for ref in vuln_data.get("references", []):
            if ref.get("type") == "ADVISORY":
                url_ref = ref.get("url", "")
                break
        if not url_ref:
            url_ref = f"https://osv.dev/vulnerability/{vuln_id}"

        vulns.append(Vulnerability(
            id=vuln_id,
            summary=summary[:200],
            severity=severity,
            fixed_version=fixed_version,
            url=url_ref,
        ))

    return vulns


# --- Static analysis checks (offline, no network) ---

def normalize_package_name(name: str) -> str:
    """Normalize package name per PEP 503: lowercase, replace [-_.] with -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def check_unpinned(deps: list[Dependency]) -> list[StaticFinding]:
    """Find dependencies with no version constraint at all."""
    findings = []
    for dep in deps:
        if not dep.constraint and dep.specified_version is None:
            findings.append(StaticFinding(
                check="unpinned",
                severity="MODERATE",
                package=dep.name,
                message=f"'{dep.name}' has no version constraint — builds are non-reproducible",
                source_files=[dep.source_file],
            ))
    return findings


def check_wildcard(deps: list[Dependency]) -> list[StaticFinding]:
    """Find dependencies with overly broad version constraints."""
    findings = []
    broad_constraints = {">=", ">", "!="}
    for dep in deps:
        if dep.constraint in broad_constraints:
            findings.append(StaticFinding(
                check="wildcard",
                severity="LOW",
                package=dep.name,
                message=f"'{dep.name}{dep.constraint}{dep.specified_version}' uses broad constraint '{dep.constraint}' — may pull breaking changes. Prefer '~=' or '=='",
                source_files=[dep.source_file],
            ))
        elif dep.specified_version and "*" in dep.specified_version:
            findings.append(StaticFinding(
                check="wildcard",
                severity="MODERATE",
                package=dep.name,
                message=f"'{dep.name}=={dep.specified_version}' uses wildcard — non-deterministic resolution",
                source_files=[dep.source_file],
            ))
    return findings


def check_duplicates(deps: list[Dependency]) -> list[StaticFinding]:
    """Find same package declared in multiple files."""
    findings = []
    by_name: dict[str, list[Dependency]] = {}
    for dep in deps:
        norm = normalize_package_name(dep.name)
        by_name.setdefault(norm, []).append(dep)

    for norm_name, group in by_name.items():
        source_files = list({d.source_file for d in group})
        if len(source_files) > 1:
            findings.append(StaticFinding(
                check="duplicate",
                severity="MODERATE",
                package=group[0].name,
                message=f"'{group[0].name}' declared in {len(source_files)} files: {', '.join(source_files)}",
                source_files=source_files,
            ))
    return findings


def check_conflicts(deps: list[Dependency]) -> list[StaticFinding]:
    """Find same package pinned to different versions."""
    findings = []
    by_name: dict[str, list[Dependency]] = {}
    for dep in deps:
        if dep.specified_version and dep.constraint == "==":
            norm = normalize_package_name(dep.name)
            by_name.setdefault(norm, []).append(dep)

    for norm_name, group in by_name.items():
        versions = {d.specified_version for d in group}
        if len(versions) > 1:
            ver_str = ", ".join(f"{d.specified_version} ({d.source_file})" for d in group)
            findings.append(StaticFinding(
                check="conflict",
                severity="HIGH",
                package=group[0].name,
                message=f"'{group[0].name}' pinned to conflicting versions: {ver_str}",
                source_files=list({d.source_file for d in group}),
            ))
    return findings


def check_hygiene(deps: list[Dependency]) -> list[StaticFinding]:
    """Check for common dependency anti-patterns."""
    findings = []
    # Check for dev-only packages in main dependencies
    dev_indicators = {
        "pytest", "pytest-cov", "pytest-mock", "pytest-asyncio", "pytest-xdist",
        "mypy", "black", "ruff", "flake8", "isort", "pylint",
        "coverage", "tox", "nox", "pre-commit", "sphinx", "mkdocs",
        "ipython", "ipdb", "pdb", "debugpy",
    }
    for dep in deps:
        norm = normalize_package_name(dep.name)
        if norm in dev_indicators:
            # Only flag if it's from a non-dev file
            if "dev" not in dep.source_file.lower() and "test" not in dep.source_file.lower():
                findings.append(StaticFinding(
                    check="hygiene",
                    severity="INFO",
                    package=dep.name,
                    message=f"'{dep.name}' looks like a dev dependency but is in '{dep.source_file}' — consider moving to dev/test dependencies",
                    source_files=[dep.source_file],
                ))
    return findings


def run_static_checks(deps: list[Dependency]) -> list[StaticFinding]:
    """Run all static analysis checks."""
    findings = []
    findings.extend(check_unpinned(deps))
    findings.extend(check_wildcard(deps))
    findings.extend(check_duplicates(deps))
    findings.extend(check_conflicts(deps))
    findings.extend(check_hygiene(deps))
    return findings


# --- Audit orchestrator ---

def audit_dependencies(
    deps: list[Dependency],
    check_versions: bool = True,
    check_vulns: bool = True,
    timeout: int = 5,
) -> list[AuditResult]:
    """Run full audit on a list of dependencies."""
    results = []
    for dep in deps:
        result = AuditResult(dependency=dep)

        if check_versions:
            result.version_info = check_version(dep, timeout=timeout)

        if check_vulns:
            result.vulnerabilities = query_osv(
                dep.name, dep.specified_version, timeout=timeout
            )

        results.append(result)

    return results


# --- Output formatters ---

SEVERITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1, "UNKNOWN": 0, "NONE": -1}


def severity_passes_filter(severity: str, min_severity: str) -> bool:
    """Check if severity meets the minimum threshold."""
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(min_severity, 0)


def format_terminal(
    results: list[AuditResult],
    min_severity: str = "LOW",
    static_findings: list[StaticFinding] | None = None,
) -> str:
    """Format results for terminal output."""
    lines = []
    lines.append(f"ai-dep-audit v{__version__} — Dependency Health Report")
    lines.append("=" * 60)

    # Summary counts
    total = len(results)
    outdated = sum(1 for r in results if r.is_outdated)
    vulnerable = sum(1 for r in results if r.has_vulns)
    lines.append(f"Scanned {total} dependencies: {outdated} outdated, {vulnerable} with known vulnerabilities")
    lines.append("")

    # Vulnerabilities section
    vuln_results = [r for r in results if r.has_vulns and severity_passes_filter(r.max_severity, min_severity)]
    if vuln_results:
        lines.append("VULNERABILITIES")
        lines.append("-" * 40)
        for r in sorted(vuln_results, key=lambda x: SEVERITY_ORDER.get(x.max_severity, 0), reverse=True):
            for v in r.vulnerabilities:
                if severity_passes_filter(v.severity, min_severity):
                    icon = {"CRITICAL": "!!!", "HIGH": "!!", "MODERATE": "!", "LOW": "~"}.get(v.severity, "?")
                    fix_str = f" (fix: >={v.fixed_version})" if v.fixed_version else ""
                    ver_str = f"@{r.dependency.specified_version}" if r.dependency.specified_version else ""
                    lines.append(f"  [{icon}] {v.severity}: {r.dependency.name}{ver_str} — {v.id}")
                    lines.append(f"       {v.summary}")
                    if fix_str:
                        lines.append(f"       Fix: upgrade to {v.fixed_version}+")
                    lines.append("")

    # Outdated section
    outdated_results = [r for r in results if r.is_outdated]
    if outdated_results:
        lines.append("OUTDATED PACKAGES")
        lines.append("-" * 40)
        # Sort: major > minor > patch
        type_order = {"major": 3, "minor": 2, "patch": 1}
        for r in sorted(outdated_results, key=lambda x: type_order.get(x.version_info.update_type or "", 0), reverse=True):
            vi = r.version_info
            tag = (vi.update_type or "").upper()
            lines.append(f"  [{tag}] {r.dependency.name}: {vi.current} -> {vi.latest}")

    # Static analysis findings
    if static_findings:
        sf_filtered = [sf for sf in static_findings if severity_passes_filter(sf.severity, min_severity)]
        if sf_filtered:
            lines.append("")
            lines.append("STATIC ANALYSIS")
            lines.append("-" * 40)
            sev_order = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "INFO": 0}
            for sf in sorted(sf_filtered, key=lambda x: sev_order.get(x.severity, 0), reverse=True):
                icon = {"HIGH": "!!", "MODERATE": "!", "LOW": "~", "INFO": "i"}.get(sf.severity, "?")
                lines.append(f"  [{icon}] {sf.severity} ({sf.check}): {sf.message}")

    # Up-to-date
    uptodate = [r for r in results if not r.is_outdated and not r.has_vulns]
    if uptodate:
        lines.append("")
        lines.append(f"{len(uptodate)} dependencies are up-to-date and vulnerability-free.")

    return "\n".join(lines)


def format_json(
    results: list[AuditResult],
    static_findings: list[StaticFinding] | None = None,
) -> str:
    """Format results as JSON."""
    output = {
        "version": __version__,
        "summary": {
            "total": len(results),
            "outdated": sum(1 for r in results if r.is_outdated),
            "vulnerable": sum(1 for r in results if r.has_vulns),
            "static_issues": len(static_findings) if static_findings else 0,
        },
        "results": [],
        "static_findings": [],
    }
    for r in results:
        entry = {
            "name": r.dependency.name,
            "specified_version": r.dependency.specified_version,
            "constraint": r.dependency.constraint,
            "source_file": r.dependency.source_file,
            "latest_version": r.version_info.latest,
            "update_type": r.version_info.update_type,
            "vulnerabilities": [asdict(v) for v in r.vulnerabilities],
        }
        output["results"].append(entry)
    if static_findings:
        for sf in static_findings:
            output["static_findings"].append(asdict(sf))
    return json.dumps(output, indent=2)


# --- Scoring & classification ---

SEVERITY_WEIGHTS = {"CRITICAL": 25, "HIGH": 15, "MODERATE": 8, "LOW": 3, "UNKNOWN": 5}
OUTDATED_WEIGHTS = {"major": 5, "minor": 2, "patch": 1}
STATIC_WEIGHTS = {"HIGH": 10, "MODERATE": 5, "LOW": 2, "INFO": 0.5}
GRADE_THRESHOLDS = [(90, "A"), (80, "B"), (70, "C"), (50, "D"), (0, "F")]


@dataclass
class DepHealthScore:
    score: int
    grade: str
    total_deps: int
    vulnerable_count: int
    outdated_count: int
    static_issue_count: int
    vuln_penalty: float
    outdated_penalty: float
    static_penalty: float
    severity_counts: dict
    profile: str
    profile_detail: str


def compute_dep_health(
    results: list[AuditResult],
    static_findings: list[StaticFinding],
) -> DepHealthScore:
    """Compute dependency health score (0-100) from audit results."""
    total_deps = len(results)
    if total_deps == 0:
        return DepHealthScore(
            score=100, grade="A", total_deps=0, vulnerable_count=0,
            outdated_count=0, static_issue_count=0, vuln_penalty=0,
            outdated_penalty=0, static_penalty=0, severity_counts={},
            profile="clean", profile_detail="No dependencies to audit",
        )

    # Vulnerability penalty
    vuln_penalty = 0.0
    severity_counts: dict[str, int] = {}
    for r in results:
        for v in r.vulnerabilities:
            sev = v.severity.upper()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            vuln_penalty += SEVERITY_WEIGHTS.get(sev, 5)

    # Outdated penalty
    outdated_penalty = 0.0
    for r in results:
        ut = r.version_info.update_type
        if ut and ut != "up-to-date":
            outdated_penalty += OUTDATED_WEIGHTS.get(ut, 1)

    # Static finding penalty
    static_penalty = 0.0
    for sf in static_findings:
        static_penalty += STATIC_WEIGHTS.get(sf.severity, 1)

    # Density-normalize by dep count (larger projects tolerate more issues)
    total_penalty = (vuln_penalty + outdated_penalty + static_penalty) / max(total_deps, 1) * 10
    raw_score = max(0, min(100, round(100 - total_penalty)))

    grade = "F"
    for threshold, g in GRADE_THRESHOLDS:
        if raw_score >= threshold:
            grade = g
            break

    vulnerable_count = sum(1 for r in results if r.has_vulns)
    outdated_count = sum(1 for r in results if r.is_outdated)

    profile, profile_detail = classify_dep_profile(
        results, static_findings, vuln_penalty, outdated_penalty, static_penalty,
    )

    return DepHealthScore(
        score=raw_score, grade=grade, total_deps=total_deps,
        vulnerable_count=vulnerable_count, outdated_count=outdated_count,
        static_issue_count=len(static_findings), vuln_penalty=vuln_penalty,
        outdated_penalty=outdated_penalty, static_penalty=static_penalty,
        severity_counts=severity_counts, profile=profile,
        profile_detail=profile_detail,
    )


def classify_dep_profile(
    results: list[AuditResult],
    static_findings: list[StaticFinding],
    vuln_penalty: float,
    outdated_penalty: float,
    static_penalty: float,
) -> tuple[str, str]:
    """Classify the project's dependency health profile.

    Returns (profile_name, description) tuple.
    Profiles: clean, critical_risk, vulnerability_heavy, outdated_heavy,
              hygiene_issues, mixed.
    """
    has_vulns = any(r.has_vulns for r in results)
    has_outdated = any(r.is_outdated for r in results)
    has_static = len(static_findings) > 0
    total_penalty = vuln_penalty + outdated_penalty + static_penalty

    if total_penalty == 0:
        return "clean", "No dependency issues detected"

    # Critical risk: any CRITICAL vulnerability
    has_critical = any(
        v.severity.upper() == "CRITICAL"
        for r in results for v in r.vulnerabilities
    )
    if has_critical:
        crit_count = sum(
            1 for r in results for v in r.vulnerabilities
            if v.severity.upper() == "CRITICAL"
        )
        return "critical_risk", f"{crit_count} critical vulnerability(ies) — immediate patching required"

    # Dominant category (>50% of total penalty)
    if total_penalty > 0:
        if vuln_penalty / total_penalty > 0.5:
            vuln_count = sum(len(r.vulnerabilities) for r in results)
            return "vulnerability_heavy", f"{vuln_count} vulnerabilities dominate risk — prioritize patching"
        if outdated_penalty / total_penalty > 0.5:
            out_count = sum(1 for r in results if r.is_outdated)
            return "outdated_heavy", f"{out_count} outdated dependencies dominate — update schedule needed"
        if static_penalty / total_penalty > 0.5:
            return "hygiene_issues", f"{len(static_findings)} static issues dominate — pinning/cleanup needed"

    # Mixed: multiple issue types, none dominant
    sources = []
    if has_vulns:
        sources.append("vulnerabilities")
    if has_outdated:
        sources.append("outdated deps")
    if has_static:
        sources.append("static issues")
    return "mixed", f"Multiple issue types: {', '.join(sources)}"


def format_score_text(score: DepHealthScore, no_color: bool = False) -> str:
    """Format health score as human-readable text."""
    lines = []
    lines.append(f"Dependency Health Score: {score.score}/100 ({score.grade})")
    lines.append(f"Profile: {score.profile}")
    lines.append(f"  {score.profile_detail}")
    lines.append(f"Dependencies: {score.total_deps} total, {score.vulnerable_count} vulnerable, {score.outdated_count} outdated, {score.static_issue_count} static issues")
    if score.severity_counts:
        sev_str = ", ".join(f"{k}={v}" for k, v in sorted(score.severity_counts.items()))
        lines.append(f"Vulnerability severities: {sev_str}")
    lines.append(f"Penalty breakdown: vuln={score.vuln_penalty:.1f}, outdated={score.outdated_penalty:.1f}, static={score.static_penalty:.1f}")
    return "\n".join(lines)


def format_score_json(score: DepHealthScore) -> str:
    """Format health score as JSON."""
    return json.dumps(asdict(score), indent=2)


# --- CLI ---

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-dep-audit",
        description="Python dependency health checker — outdated packages + known CVEs",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Directory or file to scan (default: current directory)",
    )
    parser.add_argument(
        "-f", "--file",
        help="Specific dependency file to scan",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--severity",
        choices=["low", "moderate", "high", "critical"],
        default="low",
        help="Minimum severity to report (default: low)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode — static analysis only, no network calls",
    )
    parser.add_argument(
        "--no-version-check",
        action="store_true",
        help="Skip PyPI version checking",
    )
    parser.add_argument(
        "--no-vuln-check",
        action="store_true",
        help="Skip OSV vulnerability checking",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="HTTP request timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Show dependency health score (0-100) with profile classification",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target = args.file if args.file else args.target

    try:
        deps = find_and_parse_deps(target)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not deps:
        print("No dependencies found.", file=sys.stderr)
        return 0

    # Always run static analysis (offline, fast)
    static_findings = run_static_checks(deps)

    # Offline mode skips all network checks
    skip_versions = args.no_version_check or args.offline
    skip_vulns = args.no_vuln_check or args.offline

    results = audit_dependencies(
        deps,
        check_versions=not skip_versions,
        check_vulns=not skip_vulns,
        timeout=args.timeout,
    )

    if args.score:
        health = compute_dep_health(results, static_findings)
        if args.json_output:
            print(format_score_json(health))
        else:
            print(format_score_text(health))
        return 0

    if args.json_output:
        print(format_json(results, static_findings=static_findings))
    else:
        print(format_terminal(results, min_severity=args.severity.upper(), static_findings=static_findings))

    # Exit code: 2 if vulnerabilities found, 1 if HIGH static issues, 0 otherwise
    has_vulns = any(r.has_vulns for r in results)
    has_high_static = any(sf.severity == "HIGH" for sf in static_findings)
    if has_vulns:
        return 2
    if has_high_static:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
