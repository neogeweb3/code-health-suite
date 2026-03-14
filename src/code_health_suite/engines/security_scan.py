#!/usr/bin/env python3
"""ai-security-scan: AST-based Python static security scanner.

Detects common security vulnerabilities in Python code using abstract syntax
tree analysis for high accuracy with zero external dependencies.

Usage:
    ai-security-scan                      # scan current directory
    ai-security-scan path/to/project      # scan specific directory
    ai-security-scan -f specific_file.py  # scan single file
    ai-security-scan --json               # JSON output
    ai-security-scan --severity high      # filter by minimum severity
    ai-security-scan --ignore CWE-78      # ignore specific CWE
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


__version__ = "0.2.0"

# --- Severity & CWE mappings ---

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

# Common Weakness Enumeration references
CWE_MAP = {
    "command-injection": "CWE-78",
    "sql-injection": "CWE-89",
    "eval-exec": "CWE-95",
    "code-injection": "CWE-94",
    "hardcoded-secret": "CWE-798",
    "insecure-deserialization": "CWE-502",
    "path-traversal": "CWE-22",
    "insecure-random": "CWE-330",
    "insecure-temp": "CWE-377",
    "debug-enabled": "CWE-489",
    "insecure-yaml": "CWE-20",
    "ssrf": "CWE-918",
    "weak-crypto": "CWE-327",
    "hardcoded-ip": "CWE-1051",
}

# --- Data models ---


@dataclass
class Finding:
    file: str
    line: int
    col: int
    rule: str
    cwe: str
    severity: str
    message: str
    snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    files_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "errors": self.errors,
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts


# --- Security scoring ---

SEVERITY_WEIGHTS = {"critical": 25, "high": 15, "medium": 8, "low": 3, "info": 1}

GRADE_THRESHOLDS = [
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (40, "D"),
    (0, "F"),
]

# Rule categories for profile classification
RULE_CATEGORIES = {
    "injection": {"command-injection", "sql-injection", "code-injection", "eval-exec"},
    "secrets": {"hardcoded-secret", "hardcoded-ip"},
    "deserialization": {"insecure-deserialization", "insecure-yaml"},
    "crypto": {"weak-crypto", "insecure-random"},
    "config": {"debug-enabled", "insecure-temp"},
    "network": {"ssrf", "path-traversal"},
}


@dataclass
class SecurityScore:
    score: int
    grade: str
    files_scanned: int
    total_findings: int
    severity_counts: dict[str, int]
    profile: str
    profile_detail: str
    findings_per_file: float
    top_rules: list[tuple[str, int]]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "grade": self.grade,
            "files_scanned": self.files_scanned,
            "total_findings": self.total_findings,
            "severity_counts": self.severity_counts,
            "profile": self.profile,
            "profile_detail": self.profile_detail,
            "findings_per_file": round(self.findings_per_file, 2),
            "top_rules": [{"rule": r, "count": c} for r, c in self.top_rules],
        }


def compute_security_score(result: ScanResult) -> SecurityScore:
    """Compute a 0-100 security health score from scan results."""
    if result.files_scanned == 0:
        return SecurityScore(
            score=100, grade="A", files_scanned=0, total_findings=0,
            severity_counts={}, profile="clean",
            profile_detail="No files scanned", findings_per_file=0.0,
            top_rules=[],
        )

    total_penalty = 0
    for f in result.findings:
        total_penalty += SEVERITY_WEIGHTS.get(f.severity, 1)

    # Normalize penalty by file count to avoid penalizing large projects unfairly
    density_penalty = total_penalty / max(result.files_scanned, 1)
    # Scale: 1 critical per file = score ~75, 2 criticals/file = ~50
    raw_score = max(0, min(100, 100 - int(density_penalty * 4)))

    grade = "F"
    for threshold, g in GRADE_THRESHOLDS:
        if raw_score >= threshold:
            grade = g
            break

    severity_counts = result.summary()
    findings_per_file = len(result.findings) / max(result.files_scanned, 1)

    # Top rules by frequency
    rule_counts: dict[str, int] = {}
    for f in result.findings:
        rule_counts[f.rule] = rule_counts.get(f.rule, 0) + 1
    top_rules = sorted(rule_counts.items(), key=lambda x: -x[1])[:5]

    profile, profile_detail = classify_security_profile(result.findings)

    return SecurityScore(
        score=raw_score,
        grade=grade,
        files_scanned=result.files_scanned,
        total_findings=len(result.findings),
        severity_counts=severity_counts,
        profile=profile,
        profile_detail=profile_detail,
        findings_per_file=findings_per_file,
        top_rules=top_rules,
    )


def classify_security_profile(findings: list[Finding]) -> tuple[str, str]:
    """Classify the project's security vulnerability profile.

    Returns (profile_name, description) tuple.
    Profiles: clean, injection_prone, secrets_heavy, deserialization_risk,
              crypto_weak, config_issues, network_exposed, mixed.
    """
    if not findings:
        return "clean", "No security findings detected"

    # Count findings per category
    cat_counts: dict[str, int] = {}
    for f in findings:
        for cat, rules in RULE_CATEGORIES.items():
            if f.rule in rules:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                break

    total_categorized = sum(cat_counts.values())
    if total_categorized == 0:
        return "clean", "No categorizable findings"

    # Find dominant category (>50% of categorized findings)
    dominant_cat = max(cat_counts, key=cat_counts.get)  # type: ignore[arg-type]
    dominant_pct = cat_counts[dominant_cat] / total_categorized

    profile_map = {
        "injection": ("injection_prone", "Majority of findings are injection vulnerabilities (command/SQL/code injection)"),
        "secrets": ("secrets_heavy", "Majority of findings are hardcoded secrets or credentials"),
        "deserialization": ("deserialization_risk", "Majority of findings involve unsafe deserialization"),
        "crypto": ("crypto_weak", "Majority of findings are weak cryptography or insecure random usage"),
        "config": ("config_issues", "Majority of findings are configuration/debug issues"),
        "network": ("network_exposed", "Majority of findings are network-related (SSRF/path traversal)"),
    }

    if dominant_pct > 0.5 and dominant_cat in profile_map:
        return profile_map[dominant_cat]

    # Build mixed description from top 2 categories
    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:2]
    parts = [f"{cat}({count})" for cat, count in sorted_cats]
    return "mixed", f"Multiple vulnerability types: {', '.join(parts)}"


def format_score_text(score: SecurityScore, no_color: bool = False) -> str:
    """Format security score for terminal display."""
    if no_color:
        c = {k: "" for k in COLORS}
    else:
        c = COLORS

    grade_colors = {"A": "\033[32m", "B": "\033[32m", "C": "\033[33m",
                    "D": "\033[31m", "F": "\033[1;31m"}
    gc = grade_colors.get(score.grade, "") if not no_color else ""

    lines = [
        f"\n{c['bold']}Security Health Score{c['reset']}",
        f"  Score: {gc}{score.score}/100 ({score.grade}){c['reset']}",
        f"  Profile: {score.profile} — {score.profile_detail}",
        f"  Files: {score.files_scanned}  Findings: {score.total_findings}  "
        f"Density: {score.findings_per_file:.2f}/file",
    ]

    if score.severity_counts:
        sev_parts = []
        for sev in ("critical", "high", "medium", "low", "info"):
            cnt = score.severity_counts.get(sev, 0)
            if cnt:
                sc = c.get(sev, "")
                sev_parts.append(f"{sc}{cnt} {sev}{c['reset']}")
        lines.append(f"  Severity: {', '.join(sev_parts)}")

    if score.top_rules:
        rules_str = ", ".join(f"{r}({n})" for r, n in score.top_rules)
        lines.append(f"  Top rules: {rules_str}")

    return "\n".join(lines)


def format_score_json(score: SecurityScore) -> str:
    """Format security score as JSON string."""
    return json.dumps(score.to_dict(), indent=2)


# --- Secret detection patterns ---

SECRET_VAR_PATTERNS = re.compile(
    r"(?:^|_)("
    r"password|passwd|secret|api_key|apikey|api_secret|apisecret|"
    r"access_key|accesskey|private_key|privatekey|token|auth_token|"
    r"authtoken|secret_key|secretkey|db_pass|db_password|"
    r"encryption_key|signing_key|jwt_secret|webhook_secret"
    r")(?:$|_)",
    re.IGNORECASE,
)

# High-entropy string heuristic: long alphanumeric strings likely to be secrets
SECRET_VALUE_PATTERN = re.compile(
    r"^[A-Za-z0-9+/=_\-]{20,}$"
)

# Known safe assignments that look like secrets but aren't
SAFE_SECRET_VALUES = frozenset({
    "None", "none", "True", "true", "False", "false",
    "", "changeme", "placeholder", "CHANGEME", "xxx",
    "your-api-key-here", "your-secret-here", "TODO",
    "test", "testing", "dummy", "example",
})

# --- Hardcoded IP/URL patterns ---

PRIVATE_IP_PATTERN = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3})\b"
)

HARDCODED_URL_PATTERN = re.compile(
    r"https?://(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3})"
)


# --- SQL detection ---

SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION)\b",
    re.IGNORECASE,
)


# --- AST Visitor ---


class SecurityVisitor(ast.NodeVisitor):
    """Walks the AST looking for security issues."""

    def __init__(self, filepath: str, source_lines: list[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        # Track imports for context-aware analysis
        self._imports: dict[str, str] = {}  # alias -> module
        # Track variables assigned from constant URL strings (for SSRF FP reduction)
        self._const_url_vars: set[str] = set()
        # Track variables assigned from hardcoded iterables (for SQL FP reduction)
        self._hardcoded_iter_vars: set[str] = set()

    def _add(self, node: ast.AST, rule: str, severity: str, message: str) -> None:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        snippet = ""
        if 0 < line <= len(self.source_lines):
            snippet = self.source_lines[line - 1].rstrip()
        self.findings.append(Finding(
            file=self.filepath,
            line=line,
            col=col,
            rule=rule,
            cwe=CWE_MAP.get(rule, ""),
            severity=severity,
            message=message,
            snippet=snippet,
        ))

    # --- Import tracking ---

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            self._imports[name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # --- Rule: eval/exec ---

    def visit_Call(self, node: ast.Call) -> None:
        self._check_eval_exec(node)
        self._check_command_injection(node)
        self._check_insecure_deserialization(node)
        self._check_insecure_yaml(node)
        self._check_insecure_temp(node)
        self._check_sql_injection(node)
        self._check_weak_crypto(node)
        self._check_ssrf(node)
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the full dotted call name, resolving imports."""
        if isinstance(node.func, ast.Name):
            bare = node.func.id
            # Resolve 'from X import Y' style: bare name → full module path
            return self._imports.get(bare, bare)
        if isinstance(node.func, ast.Attribute):
            parts = []
            obj = node.func
            while isinstance(obj, ast.Attribute):
                parts.append(obj.attr)
                obj = obj.value
            if isinstance(obj, ast.Name):
                # BUG-46: Resolve import alias for the base name.
                # 'import subprocess as sp' → sp.call → subprocess.call
                # Without this, aliased imports bypass all rule checks.
                resolved = self._imports.get(obj.id, obj.id)
                parts.append(resolved)
            return ".".join(reversed(parts))
        return ""

    def _check_eval_exec(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        if name in ("eval", "exec"):
            self._add(node, "eval-exec", "critical",
                      f"Use of {name}() is dangerous — can execute arbitrary code")
        elif name == "compile" and node.args:
            # compile() is fine for literal strings, flag only if dynamic
            if not isinstance(node.args[0], ast.Constant):
                self._add(node, "code-injection", "high",
                          "compile() with dynamic input can enable code injection")

    # --- Rule: command injection ---

    def _check_command_injection(self, node: ast.Call) -> None:
        name = self._get_call_name(node)

        # os.system(), os.popen()
        if name in ("os.system", "os.popen"):
            self._add(node, "command-injection", "critical",
                      f"{name}() executes shell commands — use subprocess with shell=False")
            return

        # subprocess.* with shell=True
        if name.startswith("subprocess."):
            for kw in node.keywords:
                if kw.arg == "shell":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        self._add(node, "command-injection", "high",
                                  f"{name}() with shell=True — use shell=False with args list")
                        return

    # --- Rule: insecure deserialization ---

    def _check_insecure_deserialization(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        if name in ("pickle.loads", "pickle.load",
                     "cPickle.loads", "cPickle.load"):
            self._add(node, "insecure-deserialization", "high",
                      f"{name}() can execute arbitrary code during deserialization")
        elif name in ("marshal.loads", "marshal.load"):
            self._add(node, "insecure-deserialization", "medium",
                      f"{name}() is not safe for untrusted data")
        elif name == "shelve.open":
            self._add(node, "insecure-deserialization", "medium",
                      "shelve uses pickle internally — unsafe for untrusted data")

    # --- Rule: insecure YAML ---

    def _check_insecure_yaml(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        if name == "yaml.load":
            # Check if Loader kwarg is specified
            has_loader = any(kw.arg == "Loader" for kw in node.keywords)
            # Also check for positional second arg
            has_positional_loader = len(node.args) >= 2
            if not has_loader and not has_positional_loader:
                self._add(node, "insecure-yaml", "high",
                          "yaml.load() without Loader parameter — use yaml.safe_load() or specify Loader=yaml.SafeLoader")

    # --- Rule: insecure temp files ---

    def _check_insecure_temp(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        if name == "tempfile.mktemp":
            self._add(node, "insecure-temp", "medium",
                      "tempfile.mktemp() has a race condition — use tempfile.mkstemp() or NamedTemporaryFile")

    # --- Rule: SQL injection ---

    def _check_sql_injection(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        # cursor.execute() / connection.execute() with string formatting
        if name.endswith(".execute") or name.endswith(".executemany"):
            if node.args:
                first_arg = node.args[0]
                if self._is_formatted_string(first_arg):
                    if self._contains_sql(first_arg):
                        # If .execute() has a second argument (params), the developer
                        # is using parameterized queries for values and only f-string
                        # for query structure (e.g. dynamic WHERE clause). This is a
                        # much lower risk pattern — downgrade from critical to low.
                        has_params = len(node.args) >= 2 or any(
                            kw.arg in ("parameters", "params") for kw in node.keywords
                        )
                        if has_params:
                            self._add(node, "sql-injection", "low",
                                      "SQL query structure built with string formatting "
                                      "but values are parameterized — verify dynamic "
                                      "parts are not user-controlled")
                        else:
                            self._add(node, "sql-injection", "critical",
                                      "SQL query built with string formatting — use parameterized queries")

    def _is_formatted_string(self, node: ast.AST) -> bool:
        """Check if a node is a dynamically formatted string."""
        # f-string
        if isinstance(node, ast.JoinedStr):
            return True
        # "..." % args
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return True
        # "...".format()
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return True
        # "..." + variable
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self._has_string_part(node)
        return False

    def _has_string_part(self, node: ast.BinOp) -> bool:
        """Check if a BinOp concatenation has a string constant part."""
        return isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) or \
               isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)

    def _contains_sql(self, node: ast.AST) -> bool:
        """Check if a node contains SQL keywords."""
        sql_text = self._extract_string_value(node)
        return bool(SQL_KEYWORDS.search(sql_text))

    def _extract_string_value(self, node: ast.AST) -> str:
        """Extract string constant from various AST node types."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
            return "".join(parts)
        if isinstance(node, ast.BinOp):
            return self._extract_string_value(node.left) + self._extract_string_value(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "format":
                return self._extract_string_value(node.func.value)
        return ""

    # --- Rule: weak crypto ---

    def _check_weak_crypto(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        if name in ("hashlib.md5", "hashlib.sha1"):
            self._add(node, "weak-crypto", "medium",
                      f"{name}() is cryptographically weak — use SHA-256+ for security purposes")

    # --- Rule: SSRF ---

    def _check_ssrf(self, node: ast.Call) -> None:
        name = self._get_call_name(node)
        # requests.get/post/put with dynamic URL from user input
        if name in ("requests.get", "requests.post", "requests.put",
                     "requests.delete", "requests.patch", "requests.head",
                     "urllib.request.urlopen"):
            if not node.args:
                return
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                return  # constant URL — safe
            # Check if variable was assigned from a constant URL
            if isinstance(arg, ast.Name) and arg.id in self._const_url_vars:
                return  # variable traces to constant URL — safe
            # Check if inline f-string with constant URL prefix
            if isinstance(arg, ast.JoinedStr) and arg.values:
                first = arg.values[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if first.value.startswith(("http://", "https://")):
                        return  # f-string with constant URL base — safe
                # f"{BASE_URL}/path" where BASE_URL is tracked const URL
                if isinstance(first, ast.FormattedValue):
                    inner = first.value
                    if isinstance(inner, ast.Name) and inner.id in self._const_url_vars:
                        return  # f-string with tracked const URL var — safe
            self._add(node, "ssrf", "medium",
                      f"{name}() with dynamic URL — validate/allowlist URLs to prevent SSRF")

    # --- Variable origin tracking (for FP reduction) ---

    def _track_const_url_assign(self, node: ast.Assign) -> None:
        """Track variables assigned from constant URL strings or f-strings
        with constant URL prefixes. Used to suppress SSRF false positives."""
        var_name = self._get_assign_name(node)
        if not var_name:
            return
        val = node.value
        # Case 1: url = "https://api.example.com/..."
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            if val.value.startswith(("http://", "https://")):
                self._const_url_vars.add(var_name)
                return
        # Case 2: url = f"https://api.example.com/{param}/..."
        # Case 2b: url = f"{BASE_URL}/path" where BASE_URL is a tracked const URL
        if isinstance(val, ast.JoinedStr) and val.values:
            first = val.values[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if first.value.startswith(("http://", "https://")):
                    self._const_url_vars.add(var_name)
                    return
            if isinstance(first, ast.FormattedValue):
                inner = first.value
                if isinstance(inner, ast.Name) and inner.id in self._const_url_vars:
                    self._const_url_vars.add(var_name)
                    return
        # Case 3: url = BASE_URL + path (where BASE_URL is a tracked const URL)
        if isinstance(val, ast.BinOp) and isinstance(val.op, ast.Add):
            if isinstance(val.left, ast.Name) and val.left.id in self._const_url_vars:
                self._const_url_vars.add(var_name)
                return
        # Case 4: req = urllib.request.Request(url, ...) where url is a const URL
        if isinstance(val, ast.Call):
            call_name = self._get_call_name(val)
            if call_name in ("urllib.request.Request", "Request"):
                if val.args:
                    url_arg = val.args[0]
                    if isinstance(url_arg, ast.Constant) and isinstance(url_arg.value, str):
                        if url_arg.value.startswith(("http://", "https://")):
                            self._const_url_vars.add(var_name)
                            return
                    if isinstance(url_arg, ast.Name) and url_arg.id in self._const_url_vars:
                        self._const_url_vars.add(var_name)
                        return
                    if isinstance(url_arg, ast.JoinedStr) and url_arg.values:
                        first = url_arg.values[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            if first.value.startswith(("http://", "https://")):
                                self._const_url_vars.add(var_name)
                                return
                        if isinstance(first, ast.FormattedValue):
                            inner = first.value
                            if isinstance(inner, ast.Name) and inner.id in self._const_url_vars:
                                self._const_url_vars.add(var_name)
                                return
        # If reassigned to something non-constant, remove tracking
        self._const_url_vars.discard(var_name)

    # --- Rule: hardcoded secrets (at assignment level) ---

    def visit_Assign(self, node: ast.Assign) -> None:
        self._track_const_url_assign(node)
        self._check_hardcoded_secret(node)
        self._check_insecure_random_assign(node)
        self._check_debug_enabled(node)
        self._check_hardcoded_ip_assign(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments like `password: str = "secret123"`."""
        if node.value is not None:
            # Wrap as Assign so existing checks can reuse unchanged
            fake = ast.Assign(targets=[node.target], value=node.value)
            fake.lineno = node.lineno  # type: ignore[attr-defined]
            fake.col_offset = node.col_offset  # type: ignore[attr-defined]
            self._track_const_url_assign(fake)
            self._check_hardcoded_secret(fake)
            self._check_insecure_random_assign(fake)
            self._check_debug_enabled(fake)
            self._check_hardcoded_ip_assign(fake)
        self.generic_visit(node)

    def _get_assign_name(self, node: ast.Assign) -> str:
        """Get variable name from assignment."""
        if node.targets and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        if node.targets and isinstance(node.targets[0], ast.Attribute):
            return node.targets[0].attr
        return ""

    def _check_hardcoded_secret(self, node: ast.Assign) -> None:
        var_name = self._get_assign_name(node)
        if not var_name or not SECRET_VAR_PATTERNS.search(var_name):
            return

        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            str_val = value.value
            if str_val in SAFE_SECRET_VALUES or len(str_val) < 4:
                return
            # Check for environment variable pattern (safe)
            # os.environ.get("KEY") or os.getenv("KEY") are safe
            self._add(node, "hardcoded-secret", "high",
                      f"Hardcoded secret in '{var_name}' — use environment variables or secret manager")

    def _check_insecure_random_assign(self, node: ast.Assign) -> None:
        """Flag random.* assigned to security-sensitive variables."""
        var_name = self._get_assign_name(node)
        if not var_name:
            return
        is_security = bool(SECRET_VAR_PATTERNS.search(var_name)) or \
                       any(k in var_name.lower() for k in ("nonce", "salt", "iv", "otp"))

        if is_security and isinstance(node.value, ast.Call):
            name = self._get_call_name(node.value)
            if name.startswith("random.") and not name.startswith("random.System"):
                self._add(node, "insecure-random", "high",
                          f"Insecure random for security value '{var_name}' — use secrets module")

    def _check_debug_enabled(self, node: ast.Assign) -> None:
        var_name = self._get_assign_name(node)
        if var_name.upper() in ("DEBUG", "DEBUG_MODE"):
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                self._add(node, "debug-enabled", "medium",
                          "Debug mode enabled in source code — ensure disabled in production")

    def _check_hardcoded_ip_assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            val = node.value.value
            if PRIVATE_IP_PATTERN.search(val) or HARDCODED_URL_PATTERN.search(val):
                var_name = self._get_assign_name(node)
                self._add(node, "hardcoded-ip", "low",
                          f"Hardcoded private IP/URL in '{var_name}' — use configuration or env vars")


# --- File scanner ---


def scan_file(filepath: str) -> tuple[list[Finding], Optional[str]]:
    """Scan a single Python file for security issues."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except (OSError, IOError) as e:
        return [], f"Cannot read {filepath}: {e}"

    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return [], f"Syntax error in {filepath}: {e}"

    visitor = SecurityVisitor(filepath, source_lines)
    visitor.visit(tree)
    return visitor.findings, None


def find_python_files(target: str, exclude_dirs: set[str] | None = None) -> list[str]:
    """Find all .py files in directory, respecting common excludes."""
    if exclude_dirs is None:
        exclude_dirs = {
            "__pycache__", ".git", ".venv", "venv", "env",
            "node_modules", ".tox", ".mypy_cache", ".pytest_cache",
            "dist", "build", ".eggs", "*.egg-info",
            "site-packages", ".nox",
        }

    if os.path.isfile(target):
        return [target] if target.endswith(".py") else []

    files = []
    for root, dirs, filenames in os.walk(target):
        # Prune excluded directories + virtual environments (any *env* or *venv* pattern)
        dirs[:] = [
            d for d in dirs
            if d not in exclude_dirs
            and not d.endswith(".egg-info")
            and not (d.startswith(".venv") or d.endswith("_env") or d.endswith("-env"))
        ]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))
    return files


def scan_directory(target: str, exclude_dirs: set[str] | None = None) -> ScanResult:
    """Scan all Python files in a directory."""
    result = ScanResult()
    files = find_python_files(target, exclude_dirs)

    for fpath in files:
        findings, error = scan_file(fpath)
        result.files_scanned += 1
        result.findings.extend(findings)
        if error:
            result.errors.append(error)

    # Sort: critical first, then by file+line
    result.findings.sort(
        key=lambda f: (-SEVERITY_ORDER.get(f.severity, 0), f.file, f.line)
    )
    return result


# --- Terminal output ---

COLORS = {
    "critical": "\033[1;31m",  # bold red
    "high": "\033[31m",        # red
    "medium": "\033[33m",      # yellow
    "low": "\033[36m",         # cyan
    "info": "\033[37m",        # white
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
}


def format_terminal(result: ScanResult, min_severity: str = "low",
                    ignore_rules: set[str] | None = None,
                    no_color: bool = False) -> str:
    """Format scan results for terminal display."""
    lines: list[str] = []
    min_level = SEVERITY_ORDER.get(min_severity, 1)

    filtered = [
        f for f in result.findings
        if SEVERITY_ORDER.get(f.severity, 0) >= min_level
        and (ignore_rules is None or f.cwe not in ignore_rules)
    ]

    if no_color:
        c = {k: "" for k in COLORS}
    else:
        c = COLORS

    lines.append(f"\n{c['bold']}ai-security-scan v{__version__}{c['reset']}")
    lines.append(f"Scanned {result.files_scanned} file(s)\n")

    if not filtered:
        lines.append(f"{c['bold']}No security issues found!{c['reset']}")
        return "\n".join(lines)

    # Group by file
    by_file: dict[str, list[Finding]] = {}
    for f in filtered:
        by_file.setdefault(f.file, []).append(f)

    for filepath, findings in by_file.items():
        lines.append(f"{c['bold']}{filepath}{c['reset']}")
        for f in findings:
            sev_color = c.get(f.severity, "")
            cwe_str = f" ({f.cwe})" if f.cwe else ""
            lines.append(
                f"  {c['dim']}L{f.line}:{f.col}{c['reset']}  "
                f"{sev_color}[{f.severity.upper()}]{c['reset']}{cwe_str}  "
                f"{f.message}"
            )
            if f.snippet:
                lines.append(f"    {c['dim']}{f.snippet.strip()}{c['reset']}")
        lines.append("")

    # Summary
    summary = result.summary()
    # Only include severities present in findings
    summary_parts = []
    for sev in ("critical", "high", "medium", "low", "info"):
        if sev in summary:
            sev_color = c.get(sev, "")
            summary_parts.append(f"{sev_color}{summary[sev]} {sev}{c['reset']}")

    lines.append(f"{c['bold']}Total: {len(filtered)} finding(s){c['reset']}  "
                 + "  ".join(summary_parts))

    return "\n".join(lines)


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-security-scan",
        description="AST-based Python static security scanner",
    )
    parser.add_argument("target", nargs="?", default=".",
                        help="Directory or file to scan (default: current dir)")
    parser.add_argument("-f", "--file", help="Scan a specific file")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output results as JSON")
    parser.add_argument("--severity", default="low",
                        choices=["critical", "high", "medium", "low", "info"],
                        help="Minimum severity to report (default: low)")
    parser.add_argument("--ignore", nargs="*", default=[],
                        help="CWE IDs to ignore (e.g. CWE-78 CWE-89)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--score", action="store_true",
                        help="Show security health score (0-100 with grade)")
    parser.add_argument("--version", action="version",
                        version=f"ai-security-scan {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    target = args.file or args.target
    target = os.path.abspath(target)

    if not os.path.exists(target):
        print(f"Error: {target} does not exist", file=sys.stderr)
        return 1

    result = scan_directory(target)

    ignore_rules = set(args.ignore) if args.ignore else None

    if args.score:
        score = compute_security_score(result)
        if args.json_output:
            print(format_score_json(score))
        else:
            print(format_score_text(score, no_color=args.no_color))
    elif args.json_output:
        # Filter before JSON output
        min_level = SEVERITY_ORDER.get(args.severity, 1)
        result.findings = [
            f for f in result.findings
            if SEVERITY_ORDER.get(f.severity, 0) >= min_level
            and (ignore_rules is None or f.cwe not in ignore_rules)
        ]
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_terminal(result, min_severity=args.severity,
                              ignore_rules=ignore_rules,
                              no_color=args.no_color))

    # Exit code: 2 if critical/high findings remain after filtering, 0 otherwise
    min_level = SEVERITY_ORDER.get(args.severity, 1)
    effective_findings = [
        f for f in result.findings
        if SEVERITY_ORDER.get(f.severity, 0) >= min_level
        and (ignore_rules is None or f.cwe not in ignore_rules)
    ]
    has_severe = any(
        f.severity in ("critical", "high") for f in effective_findings
    )
    return 2 if has_severe else 0


if __name__ == "__main__":
    sys.exit(main())
