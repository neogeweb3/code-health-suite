"""Tests for the security scan engine."""
import ast
import os
import textwrap
import json

import pytest

from code_health_suite.engines import security_scan
from code_health_suite.engines.security_scan import (
    Finding, ScanResult, SecurityScore,
    SecurityVisitor,
    compute_security_score, classify_security_profile,
    format_score_text, format_score_json,
    format_terminal,
    scan_file, find_python_files, scan_directory,
    build_parser, main,
    SECRET_VAR_PATTERNS, SECRET_VALUE_PATTERN, SAFE_SECRET_VALUES,
    PRIVATE_IP_PATTERN, HARDCODED_URL_PATTERN, SQL_KEYWORDS,
    SEVERITY_ORDER, SEVERITY_WEIGHTS, GRADE_THRESHOLDS,
    CWE_MAP, RULE_CATEGORIES, COLORS,
)


# --- Helpers ---

def _scan_code(code: str, filename: str = "test.py") -> list[Finding]:
    """Parse code string and return findings from SecurityVisitor."""
    source = textwrap.dedent(code)
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=filename)
    visitor = SecurityVisitor(filename, source_lines)
    visitor.visit(tree)
    return visitor.findings


def _write_py(tmp_path, code: str, name: str = "sample.py") -> str:
    """Write a Python source file and return its path."""
    filepath = tmp_path / name
    filepath.write_text(textwrap.dedent(code))
    return str(filepath)


def _finding_rules(findings: list[Finding]) -> list[str]:
    """Extract rule names from findings."""
    return [f.rule for f in findings]


# =============================================================
# Data models
# =============================================================

class TestFinding:
    def test_to_dict(self):
        f = Finding(file="a.py", line=1, col=0, rule="eval-exec",
                    cwe="CWE-95", severity="critical",
                    message="dangerous eval", snippet="eval(x)")
        d = f.to_dict()
        assert d["file"] == "a.py"
        assert d["line"] == 1
        assert d["rule"] == "eval-exec"
        assert d["cwe"] == "CWE-95"
        assert d["severity"] == "critical"
        assert d["snippet"] == "eval(x)"

    def test_default_snippet(self):
        f = Finding(file="a.py", line=1, col=0, rule="r", cwe="", severity="low", message="m")
        assert f.snippet == ""


class TestScanResult:
    def test_empty(self):
        r = ScanResult()
        assert r.files_scanned == 0
        assert r.findings == []
        assert r.errors == []

    def test_to_dict(self):
        f = Finding("a.py", 1, 0, "eval-exec", "CWE-95", "critical", "msg")
        r = ScanResult(files_scanned=2, findings=[f], errors=["err1"])
        d = r.to_dict()
        assert d["files_scanned"] == 2
        assert d["finding_count"] == 1
        assert len(d["findings"]) == 1
        assert d["errors"] == ["err1"]
        assert d["summary"] == {"critical": 1}

    def test_summary_counts(self):
        findings = [
            Finding("a.py", 1, 0, "r1", "", "critical", "m"),
            Finding("a.py", 2, 0, "r2", "", "critical", "m"),
            Finding("a.py", 3, 0, "r3", "", "high", "m"),
            Finding("a.py", 4, 0, "r4", "", "low", "m"),
        ]
        r = ScanResult(findings=findings)
        s = r.summary()
        assert s == {"critical": 2, "high": 1, "low": 1}


class TestSecurityScore:
    def test_to_dict(self):
        s = SecurityScore(
            score=85, grade="B", files_scanned=10, total_findings=3,
            severity_counts={"high": 2, "low": 1}, profile="injection_prone",
            profile_detail="desc", findings_per_file=0.3,
            top_rules=[("eval-exec", 2), ("hardcoded-secret", 1)],
        )
        d = s.to_dict()
        assert d["score"] == 85
        assert d["grade"] == "B"
        assert d["findings_per_file"] == 0.3
        assert d["top_rules"] == [
            {"rule": "eval-exec", "count": 2},
            {"rule": "hardcoded-secret", "count": 1},
        ]


# =============================================================
# Security scoring
# =============================================================

class TestComputeSecurityScore:
    def test_no_files(self):
        r = ScanResult(files_scanned=0)
        s = compute_security_score(r)
        assert s.score == 100
        assert s.grade == "A"
        assert s.profile == "clean"
        assert s.total_findings == 0

    def test_clean_scan(self):
        r = ScanResult(files_scanned=5)
        s = compute_security_score(r)
        assert s.score == 100
        assert s.grade == "A"

    def test_one_critical(self):
        f = Finding("a.py", 1, 0, "eval-exec", "CWE-95", "critical", "msg")
        r = ScanResult(files_scanned=1, findings=[f])
        s = compute_security_score(r)
        # penalty = 25, density = 25, raw = 100 - 100 = 0
        assert s.score == 0
        assert s.grade == "F"

    def test_one_critical_many_files(self):
        """One critical in 10 files should not be grade F."""
        f = Finding("a.py", 1, 0, "eval-exec", "CWE-95", "critical", "msg")
        r = ScanResult(files_scanned=10, findings=[f])
        s = compute_security_score(r)
        # penalty=25, density=2.5, raw=100-10=90
        assert s.score == 90
        assert s.grade == "A"

    def test_grade_boundaries(self):
        """Test all grade boundaries."""
        # Score 90 -> A
        assert compute_security_score(ScanResult(files_scanned=1)).grade == "A"

        # Create findings to hit different scores
        # 1 medium in 1 file: penalty=8, density=8, raw=100-32=68 -> C
        f_med = Finding("a.py", 1, 0, "r", "", "medium", "m")
        r = ScanResult(files_scanned=1, findings=[f_med])
        s = compute_security_score(r)
        assert s.grade == "C"

    def test_top_rules_limited_to_5(self):
        findings = [
            Finding("a.py", i, 0, f"rule-{i}", "", "low", "m")
            for i in range(10)
        ]
        r = ScanResult(files_scanned=1, findings=findings)
        s = compute_security_score(r)
        assert len(s.top_rules) <= 5

    def test_findings_per_file(self):
        findings = [Finding("a.py", i, 0, "r", "", "low", "m") for i in range(6)]
        r = ScanResult(files_scanned=3, findings=findings)
        s = compute_security_score(r)
        assert s.findings_per_file == 2.0

    def test_score_clamped_to_0(self):
        """Score should not go below 0."""
        findings = [
            Finding("a.py", i, 0, "eval-exec", "CWE-95", "critical", "m")
            for i in range(20)
        ]
        r = ScanResult(files_scanned=1, findings=findings)
        s = compute_security_score(r)
        assert s.score == 0


class TestClassifySecurityProfile:
    def test_no_findings(self):
        profile, detail = classify_security_profile([])
        assert profile == "clean"

    def test_injection_prone(self):
        findings = [
            Finding("a.py", 1, 0, "command-injection", "", "critical", "m"),
            Finding("a.py", 2, 0, "sql-injection", "", "critical", "m"),
            Finding("a.py", 3, 0, "eval-exec", "", "critical", "m"),
        ]
        profile, _ = classify_security_profile(findings)
        assert profile == "injection_prone"

    def test_secrets_heavy(self):
        findings = [
            Finding("a.py", 1, 0, "hardcoded-secret", "", "high", "m"),
            Finding("a.py", 2, 0, "hardcoded-secret", "", "high", "m"),
            Finding("a.py", 3, 0, "hardcoded-ip", "", "low", "m"),
        ]
        profile, _ = classify_security_profile(findings)
        assert profile == "secrets_heavy"

    def test_deserialization_risk(self):
        findings = [
            Finding("a.py", 1, 0, "insecure-deserialization", "", "high", "m"),
            Finding("a.py", 2, 0, "insecure-deserialization", "", "high", "m"),
        ]
        profile, _ = classify_security_profile(findings)
        assert profile == "deserialization_risk"

    def test_crypto_weak(self):
        findings = [
            Finding("a.py", 1, 0, "weak-crypto", "", "medium", "m"),
            Finding("a.py", 2, 0, "insecure-random", "", "high", "m"),
        ]
        profile, _ = classify_security_profile(findings)
        assert profile == "crypto_weak"

    def test_config_issues(self):
        findings = [
            Finding("a.py", 1, 0, "debug-enabled", "", "medium", "m"),
            Finding("a.py", 2, 0, "insecure-temp", "", "medium", "m"),
        ]
        profile, _ = classify_security_profile(findings)
        assert profile == "config_issues"

    def test_network_exposed(self):
        findings = [
            Finding("a.py", 1, 0, "ssrf", "", "medium", "m"),
            Finding("a.py", 2, 0, "path-traversal", "", "high", "m"),
        ]
        profile, _ = classify_security_profile(findings)
        assert profile == "network_exposed"

    def test_mixed_profile(self):
        findings = [
            Finding("a.py", 1, 0, "command-injection", "", "critical", "m"),
            Finding("a.py", 2, 0, "hardcoded-secret", "", "high", "m"),
            Finding("a.py", 3, 0, "weak-crypto", "", "medium", "m"),
        ]
        profile, detail = classify_security_profile(findings)
        assert profile == "mixed"
        assert "Multiple vulnerability types" in detail

    def test_uncategorizable_findings(self):
        """Findings with rules not in RULE_CATEGORIES."""
        findings = [Finding("a.py", 1, 0, "unknown-rule", "", "low", "m")]
        profile, detail = classify_security_profile(findings)
        assert profile == "clean"
        assert "No categorizable" in detail


# =============================================================
# Regex patterns
# =============================================================

class TestSecretVarPatterns:
    @pytest.mark.parametrize("name", [
        "password", "db_password", "API_KEY", "api_secret",
        "access_key", "private_key", "auth_token", "secret_key",
        "jwt_secret", "webhook_secret", "ENCRYPTION_KEY",
    ])
    def test_matches_secret_names(self, name):
        assert SECRET_VAR_PATTERNS.search(name)

    @pytest.mark.parametrize("name", [
        "username", "email", "count", "filepath",
        "description", "message", "result",
    ])
    def test_no_match_normal_names(self, name):
        assert not SECRET_VAR_PATTERNS.search(name)


class TestSecretValuePattern:
    def test_matches_long_alphanumeric(self):
        assert SECRET_VALUE_PATTERN.match("A" * 20)
        assert SECRET_VALUE_PATTERN.match("abc123DEF456GHI789JKL0")

    def test_no_match_short(self):
        assert not SECRET_VALUE_PATTERN.match("short")

    def test_no_match_with_spaces(self):
        assert not SECRET_VALUE_PATTERN.match("has spaces in it long enough")


class TestPrivateIpPattern:
    @pytest.mark.parametrize("ip", [
        "10.0.0.1", "10.255.255.255",
        "172.16.0.1", "172.31.255.255",
        "192.168.0.1", "192.168.1.100",
    ])
    def test_matches_private_ips(self, ip):
        assert PRIVATE_IP_PATTERN.search(ip)

    @pytest.mark.parametrize("ip", [
        "8.8.8.8", "1.1.1.1", "172.15.0.1", "172.32.0.1",
    ])
    def test_no_match_public_ips(self, ip):
        assert not PRIVATE_IP_PATTERN.search(ip)


class TestSqlKeywords:
    @pytest.mark.parametrize("kw", [
        "SELECT * FROM", "INSERT INTO", "UPDATE table", "DELETE FROM",
        "DROP TABLE", "CREATE TABLE", "UNION SELECT",
    ])
    def test_matches_sql(self, kw):
        assert SQL_KEYWORDS.search(kw)

    def test_no_match_normal_text(self):
        assert not SQL_KEYWORDS.search("hello world")


# =============================================================
# SecurityVisitor — eval/exec
# =============================================================

class TestEvalExec:
    def test_eval_detected(self):
        findings = _scan_code("eval(user_input)")
        assert "eval-exec" in _finding_rules(findings)
        assert findings[0].severity == "critical"

    def test_exec_detected(self):
        findings = _scan_code("exec(code_string)")
        assert "eval-exec" in _finding_rules(findings)

    def test_compile_dynamic(self):
        findings = _scan_code("compile(user_input, '<string>', 'exec')")
        assert "code-injection" in _finding_rules(findings)

    def test_compile_literal_safe(self):
        findings = _scan_code("compile('print(1)', '<string>', 'exec')")
        assert "code-injection" not in _finding_rules(findings)

    def test_eval_not_builtin_name(self):
        """eval as method name should not trigger if not bare name."""
        findings = _scan_code("obj.eval(x)")
        assert "eval-exec" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — command injection
# =============================================================

class TestCommandInjection:
    def test_os_system(self):
        findings = _scan_code("import os\nos.system(cmd)")
        assert "command-injection" in _finding_rules(findings)
        assert any(f.severity == "critical" for f in findings if f.rule == "command-injection")

    def test_os_popen(self):
        findings = _scan_code("import os\nos.popen(cmd)")
        assert "command-injection" in _finding_rules(findings)

    def test_subprocess_shell_true(self):
        findings = _scan_code("""\
            import subprocess
            subprocess.run(cmd, shell=True)
        """)
        assert "command-injection" in _finding_rules(findings)
        cmd_findings = [f for f in findings if f.rule == "command-injection"]
        assert cmd_findings[0].severity == "high"

    def test_subprocess_shell_false_safe(self):
        findings = _scan_code("""\
            import subprocess
            subprocess.run(["ls", "-la"], shell=False)
        """)
        assert "command-injection" not in _finding_rules(findings)

    def test_subprocess_no_shell_safe(self):
        findings = _scan_code("""\
            import subprocess
            subprocess.run(["ls", "-la"])
        """)
        assert "command-injection" not in _finding_rules(findings)

    def test_subprocess_alias(self):
        """Import alias should be resolved."""
        findings = _scan_code("""\
            import subprocess as sp
            sp.call(cmd, shell=True)
        """)
        assert "command-injection" in _finding_rules(findings)


# =============================================================
# SecurityVisitor — insecure deserialization
# =============================================================

class TestInsecureDeserialization:
    def test_pickle_load(self):
        findings = _scan_code("import pickle\npickle.load(f)")
        assert "insecure-deserialization" in _finding_rules(findings)

    def test_pickle_loads(self):
        findings = _scan_code("import pickle\npickle.loads(data)")
        assert "insecure-deserialization" in _finding_rules(findings)

    def test_cpickle(self):
        findings = _scan_code("import cPickle\ncPickle.load(f)")
        assert "insecure-deserialization" in _finding_rules(findings)

    def test_marshal_load(self):
        findings = _scan_code("import marshal\nmarshal.load(f)")
        deser = [f for f in findings if f.rule == "insecure-deserialization"]
        assert len(deser) == 1
        assert deser[0].severity == "medium"

    def test_shelve_open(self):
        findings = _scan_code("import shelve\nshelve.open('data')")
        assert "insecure-deserialization" in _finding_rules(findings)


# =============================================================
# SecurityVisitor — insecure YAML
# =============================================================

class TestInsecureYaml:
    def test_yaml_load_no_loader(self):
        findings = _scan_code("import yaml\nyaml.load(data)")
        assert "insecure-yaml" in _finding_rules(findings)

    def test_yaml_load_with_loader_kwarg(self):
        findings = _scan_code("import yaml\nyaml.load(data, Loader=yaml.SafeLoader)")
        assert "insecure-yaml" not in _finding_rules(findings)

    def test_yaml_load_with_positional_loader(self):
        findings = _scan_code("import yaml\nyaml.load(data, yaml.SafeLoader)")
        assert "insecure-yaml" not in _finding_rules(findings)

    def test_yaml_safe_load_ok(self):
        findings = _scan_code("import yaml\nyaml.safe_load(data)")
        assert "insecure-yaml" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — insecure temp
# =============================================================

class TestInsecureTemp:
    def test_mktemp(self):
        findings = _scan_code("import tempfile\ntempfile.mktemp()")
        assert "insecure-temp" in _finding_rules(findings)

    def test_mkstemp_safe(self):
        findings = _scan_code("import tempfile\ntempfile.mkstemp()")
        assert "insecure-temp" not in _finding_rules(findings)

    def test_named_temporary_file_safe(self):
        findings = _scan_code("import tempfile\ntempfile.NamedTemporaryFile()")
        assert "insecure-temp" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — SQL injection
# =============================================================

class TestSqlInjection:
    def test_fstring_sql(self):
        findings = _scan_code("""\
            cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
        """)
        assert "sql-injection" in _finding_rules(findings)
        sql = [f for f in findings if f.rule == "sql-injection"]
        assert sql[0].severity == "critical"

    def test_percent_format_sql(self):
        findings = _scan_code("""\
            cursor.execute("SELECT * FROM users WHERE id=%s" % user_id)
        """)
        assert "sql-injection" in _finding_rules(findings)

    def test_format_method_sql(self):
        findings = _scan_code("""\
            cursor.execute("SELECT * FROM users WHERE id={}".format(user_id))
        """)
        assert "sql-injection" in _finding_rules(findings)

    def test_concat_sql(self):
        findings = _scan_code("""\
            cursor.execute("SELECT * FROM users WHERE id=" + user_id)
        """)
        assert "sql-injection" in _finding_rules(findings)

    def test_parameterized_safe_downgrade(self):
        """Parameterized queries with f-string structure should be low severity."""
        findings = _scan_code("""\
            cursor.execute(f"SELECT * FROM {table}", (value,))
        """)
        sql = [f for f in findings if f.rule == "sql-injection"]
        assert len(sql) == 1
        assert sql[0].severity == "low"

    def test_parameterized_with_kwarg(self):
        findings = _scan_code("""\
            cursor.execute(f"SELECT * FROM {table}", parameters=(val,))
        """)
        sql = [f for f in findings if f.rule == "sql-injection"]
        assert sql[0].severity == "low"

    def test_constant_string_safe(self):
        findings = _scan_code("""\
            cursor.execute("SELECT * FROM users")
        """)
        assert "sql-injection" not in _finding_rules(findings)

    def test_executemany(self):
        findings = _scan_code("""\
            cursor.executemany(f"INSERT INTO {table} VALUES (%s)", data)
        """)
        sql = [f for f in findings if f.rule == "sql-injection"]
        assert len(sql) == 1

    def test_no_sql_keywords_no_finding(self):
        """Formatted string without SQL keywords should not trigger."""
        findings = _scan_code("""\
            cursor.execute(f"hello {world}")
        """)
        assert "sql-injection" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — weak crypto
# =============================================================

class TestWeakCrypto:
    def test_md5(self):
        findings = _scan_code("import hashlib\nhashlib.md5(data)")
        assert "weak-crypto" in _finding_rules(findings)

    def test_sha1(self):
        findings = _scan_code("import hashlib\nhashlib.sha1(data)")
        assert "weak-crypto" in _finding_rules(findings)

    def test_sha256_safe(self):
        findings = _scan_code("import hashlib\nhashlib.sha256(data)")
        assert "weak-crypto" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — SSRF
# =============================================================

class TestSsrf:
    def test_requests_get_dynamic_url(self):
        findings = _scan_code("""\
            import requests
            requests.get(user_url)
        """)
        assert "ssrf" in _finding_rules(findings)

    def test_requests_post_dynamic(self):
        findings = _scan_code("""\
            import requests
            requests.post(url)
        """)
        assert "ssrf" in _finding_rules(findings)

    def test_requests_constant_url_safe(self):
        findings = _scan_code("""\
            import requests
            requests.get("https://api.example.com/data")
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_tracked_const_url_var_safe(self):
        """Variable assigned from constant URL should be safe."""
        findings = _scan_code("""\
            import requests
            url = "https://api.example.com"
            requests.get(url)
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_fstring_with_const_prefix_safe(self):
        findings = _scan_code("""\
            import requests
            requests.get(f"https://api.example.com/{path}")
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_fstring_with_tracked_var_safe(self):
        findings = _scan_code("""\
            import requests
            BASE = "https://api.example.com"
            requests.get(f"{BASE}/endpoint")
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_concatenation_with_tracked_var_safe(self):
        findings = _scan_code("""\
            import requests
            BASE = "https://api.example.com"
            url = BASE + "/endpoint"
            requests.get(url)
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_urllib_urlopen(self):
        findings = _scan_code("""\
            import urllib.request
            urllib.request.urlopen(url)
        """)
        assert "ssrf" in _finding_rules(findings)

    def test_no_args_safe(self):
        """requests.get() with no args should not crash."""
        findings = _scan_code("""\
            import requests
            requests.get()
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_request_object_const_url_safe(self):
        """urllib.request.Request with constant URL should track."""
        findings = _scan_code("""\
            import urllib.request
            req = urllib.request.Request("https://api.example.com")
            urllib.request.urlopen(req)
        """)
        assert "ssrf" not in _finding_rules(findings)

    def test_reassigned_var_loses_tracking(self):
        """If a tracked URL var is reassigned to non-URL, tracking lost."""
        findings = _scan_code("""\
            import requests
            url = "https://api.example.com"
            url = some_function()
            requests.get(url)
        """)
        assert "ssrf" in _finding_rules(findings)


# =============================================================
# SecurityVisitor — hardcoded secrets
# =============================================================

class TestHardcodedSecrets:
    def test_hardcoded_password(self):
        findings = _scan_code('password = "supersecret123"')
        assert "hardcoded-secret" in _finding_rules(findings)

    def test_hardcoded_api_key(self):
        findings = _scan_code('API_KEY = "sk-1234567890abcdef"')
        assert "hardcoded-secret" in _finding_rules(findings)

    def test_safe_values_ignored(self):
        for safe in ("None", "changeme", "placeholder", "test", "TODO"):
            findings = _scan_code(f'password = "{safe}"')
            assert "hardcoded-secret" not in _finding_rules(findings), f"False positive for {safe}"

    def test_short_values_ignored(self):
        findings = _scan_code('password = "abc"')
        assert "hardcoded-secret" not in _finding_rules(findings)

    def test_non_secret_var_ignored(self):
        findings = _scan_code('username = "admin_long_name_here"')
        assert "hardcoded-secret" not in _finding_rules(findings)

    def test_non_string_value_ignored(self):
        findings = _scan_code("password = get_password()")
        assert "hardcoded-secret" not in _finding_rules(findings)

    def test_annotated_assignment(self):
        findings = _scan_code('password: str = "supersecret123"')
        assert "hardcoded-secret" in _finding_rules(findings)

    def test_attribute_assignment(self):
        findings = _scan_code('self.password = "supersecret123"')
        assert "hardcoded-secret" in _finding_rules(findings)


# =============================================================
# SecurityVisitor — insecure random
# =============================================================

class TestInsecureRandom:
    def test_random_for_token(self):
        findings = _scan_code("""\
            import random
            auth_token = random.randint(0, 999999)
        """)
        assert "insecure-random" in _finding_rules(findings)

    def test_random_for_nonce(self):
        findings = _scan_code("""\
            import random
            nonce = random.random()
        """)
        assert "insecure-random" in _finding_rules(findings)

    def test_random_for_salt(self):
        findings = _scan_code("""\
            import random
            salt = random.randint(0, 100)
        """)
        assert "insecure-random" in _finding_rules(findings)

    def test_random_for_normal_var_safe(self):
        findings = _scan_code("""\
            import random
            index = random.randint(0, 10)
        """)
        assert "insecure-random" not in _finding_rules(findings)

    def test_secrets_module_safe(self):
        """random.SystemRandom is safe."""
        findings = _scan_code("""\
            import random
            token = random.SystemRandom().randint(0, 999999)
        """)
        # This won't match random.SystemRandom... as the call name
        # because the call is on the result of SystemRandom()
        assert "insecure-random" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — debug enabled
# =============================================================

class TestDebugEnabled:
    def test_debug_true(self):
        findings = _scan_code("DEBUG = True")
        assert "debug-enabled" in _finding_rules(findings)

    def test_debug_mode_true(self):
        findings = _scan_code("DEBUG_MODE = True")
        assert "debug-enabled" in _finding_rules(findings)

    def test_debug_false_safe(self):
        findings = _scan_code("DEBUG = False")
        assert "debug-enabled" not in _finding_rules(findings)

    def test_debug_lowercase_true(self):
        findings = _scan_code("debug = True")
        assert "debug-enabled" in _finding_rules(findings)

    def test_unrelated_var_safe(self):
        findings = _scan_code("VERBOSE = True")
        assert "debug-enabled" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — hardcoded IP
# =============================================================

class TestHardcodedIp:
    def test_private_ip(self):
        findings = _scan_code('host = "192.168.1.100"')
        assert "hardcoded-ip" in _finding_rules(findings)

    def test_private_ip_10(self):
        findings = _scan_code('server = "10.0.0.1"')
        assert "hardcoded-ip" in _finding_rules(findings)

    def test_private_url(self):
        findings = _scan_code('url = "http://192.168.1.1:8080/api"')
        assert "hardcoded-ip" in _finding_rules(findings)

    def test_public_ip_safe(self):
        findings = _scan_code('dns = "8.8.8.8"')
        assert "hardcoded-ip" not in _finding_rules(findings)

    def test_non_string_safe(self):
        findings = _scan_code("host = get_host()")
        assert "hardcoded-ip" not in _finding_rules(findings)


# =============================================================
# SecurityVisitor — import tracking
# =============================================================

class TestImportTracking:
    def test_import_alias_resolved(self):
        """Import aliases should be resolved for rule matching."""
        findings = _scan_code("""\
            import os as myos
            myos.system(cmd)
        """)
        assert "command-injection" in _finding_rules(findings)

    def test_from_import_resolved(self):
        findings = _scan_code("""\
            from pickle import loads
            loads(data)
        """)
        assert "insecure-deserialization" in _finding_rules(findings)

    def test_from_import_alias(self):
        findings = _scan_code("""\
            from pickle import loads as pload
            pload(data)
        """)
        assert "insecure-deserialization" in _finding_rules(findings)


# =============================================================
# SecurityVisitor — _get_call_name
# =============================================================

class TestGetCallName:
    def test_simple_name(self):
        code = "eval(x)"
        tree = ast.parse(code)
        visitor = SecurityVisitor("test.py", [code])
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert visitor._get_call_name(call) == "eval"

    def test_dotted_name(self):
        code = "os.path.join(a, b)"
        tree = ast.parse(code)
        visitor = SecurityVisitor("test.py", [code])
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert visitor._get_call_name(call) == "os.path.join"

    def test_import_alias_resolution(self):
        code = "import subprocess as sp\nsp.call(cmd)"
        tree = ast.parse(code)
        visitor = SecurityVisitor("test.py", code.splitlines())
        visitor.visit(tree.body[0])  # process import
        call = tree.body[1].value  # type: ignore[attr-defined]
        assert visitor._get_call_name(call) == "subprocess.call"

    def test_non_name_func(self):
        """Lambda or complex expression calls return empty string."""
        code = "(lambda: None)()"
        tree = ast.parse(code)
        visitor = SecurityVisitor("test.py", [code])
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert visitor._get_call_name(call) == ""


# =============================================================
# SecurityVisitor — string analysis helpers
# =============================================================

class TestStringHelpers:
    def test_is_formatted_fstring(self):
        code = 'f"hello {x}"'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert visitor._is_formatted_string(tree.body)

    def test_is_formatted_percent(self):
        code = '"hello %s" % x'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert visitor._is_formatted_string(tree.body)

    def test_is_formatted_format_method(self):
        code = '"hello {}".format(x)'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert visitor._is_formatted_string(tree.body)

    def test_is_formatted_concat(self):
        code = '"SELECT " + var'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert visitor._is_formatted_string(tree.body)

    def test_not_formatted_literal(self):
        code = '"hello"'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert not visitor._is_formatted_string(tree.body)

    def test_extract_string_from_constant(self):
        code = '"SELECT * FROM users"'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert visitor._extract_string_value(tree.body) == "SELECT * FROM users"

    def test_extract_string_from_fstring(self):
        code = 'f"SELECT * FROM {table}"'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        result = visitor._extract_string_value(tree.body)
        assert "SELECT" in result

    def test_extract_string_from_binop(self):
        code = '"SELECT " + var'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        result = visitor._extract_string_value(tree.body)
        assert "SELECT" in result

    def test_contains_sql_true(self):
        code = '"SELECT * FROM users"'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert visitor._contains_sql(tree.body)

    def test_contains_sql_false(self):
        code = '"hello world"'
        tree = ast.parse(code, mode="eval")
        visitor = SecurityVisitor("test.py", [])
        assert not visitor._contains_sql(tree.body)


# =============================================================
# SecurityVisitor — annotated assignment
# =============================================================

class TestAnnotatedAssignment:
    def test_ann_assign_secret(self):
        findings = _scan_code('password: str = "mysecretpass"')
        assert "hardcoded-secret" in _finding_rules(findings)

    def test_ann_assign_debug(self):
        findings = _scan_code("DEBUG: bool = True")
        assert "debug-enabled" in _finding_rules(findings)

    def test_ann_assign_ip(self):
        findings = _scan_code('host: str = "192.168.1.1"')
        assert "hardcoded-ip" in _finding_rules(findings)

    def test_ann_assign_no_value_safe(self):
        findings = _scan_code("password: str")
        assert findings == []

    def test_ann_assign_url_tracking(self):
        """Annotated URL assignments should be tracked for SSRF FP reduction."""
        findings = _scan_code("""\
            import requests
            BASE_URL: str = "https://api.example.com"
            requests.get(BASE_URL)
        """)
        assert "ssrf" not in _finding_rules(findings)


# =============================================================
# File scanning
# =============================================================

class TestScanFile:
    def test_scan_clean_file(self, tmp_path):
        path = _write_py(tmp_path, "x = 1\ny = 2\n")
        findings, error = scan_file(path)
        assert findings == []
        assert error is None

    def test_scan_file_with_finding(self, tmp_path):
        path = _write_py(tmp_path, "eval(input())")
        findings, error = scan_file(path)
        assert len(findings) == 1
        assert findings[0].rule == "eval-exec"
        assert error is None

    def test_scan_nonexistent_file(self):
        findings, error = scan_file("/nonexistent/file.py")
        assert findings == []
        assert error is not None
        assert "Cannot read" in error

    def test_scan_syntax_error(self, tmp_path):
        path = _write_py(tmp_path, "def broken(:\n  pass\n")
        findings, error = scan_file(path)
        assert findings == []
        assert error is not None
        assert "Syntax error" in error

    def test_snippet_captured(self, tmp_path):
        path = _write_py(tmp_path, "eval(x)\n")
        findings, _ = scan_file(path)
        assert findings[0].snippet == "eval(x)"


class TestFindPythonFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        (tmp_path / "c.txt").write_text("z=3")
        files = find_python_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_excludes_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_excludes_venv(self, tmp_path):
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_single_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x=1")
        files = find_python_files(str(f))
        assert files == [str(f)]

    def test_single_non_py_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("x=1")
        files = find_python_files(str(f))
        assert files == []

    def test_custom_excludes(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        (tmp_path / "mydir" / "a.py").write_text("x=1")
        files = find_python_files(str(tmp_path), exclude_dirs={"mydir"})
        assert len(files) == 0

    def test_egg_info_excluded(self, tmp_path):
        (tmp_path / "pkg.egg-info").mkdir()
        (tmp_path / "pkg.egg-info" / "a.py").write_text("x=1")
        (tmp_path / "b.py").write_text("y=2")
        files = find_python_files(str(tmp_path))
        assert len(files) == 1

    def test_env_suffix_excluded(self, tmp_path):
        (tmp_path / "myproject_env").mkdir()
        (tmp_path / "myproject_env" / "a.py").write_text("x=1")
        files = find_python_files(str(tmp_path))
        assert len(files) == 0


class TestScanDirectory:
    def test_scan_dir(self, tmp_path):
        _write_py(tmp_path, "eval(x)", "bad.py")
        _write_py(tmp_path, "x = 1", "good.py")
        result = scan_directory(str(tmp_path))
        assert result.files_scanned == 2
        assert len(result.findings) == 1

    def test_findings_sorted_by_severity(self, tmp_path):
        code = 'eval(x)\npassword = "secret1234"\n'
        _write_py(tmp_path, code, "mixed.py")
        result = scan_directory(str(tmp_path))
        # Critical should come first
        assert result.findings[0].severity == "critical"


# =============================================================
# Terminal output
# =============================================================

class TestFormatTerminal:
    def test_no_findings(self):
        r = ScanResult(files_scanned=3)
        output = format_terminal(r, no_color=True)
        assert "No security issues found" in output
        assert "3 file(s)" in output

    def test_with_findings(self):
        f = Finding("a.py", 10, 5, "eval-exec", "CWE-95", "critical",
                     "eval is dangerous", "eval(x)")
        r = ScanResult(files_scanned=1, findings=[f])
        output = format_terminal(r, no_color=True)
        assert "a.py" in output
        assert "CRITICAL" in output
        assert "CWE-95" in output
        assert "eval is dangerous" in output
        assert "eval(x)" in output

    def test_severity_filter(self):
        findings = [
            Finding("a.py", 1, 0, "r1", "", "low", "low msg"),
            Finding("a.py", 2, 0, "r2", "", "critical", "crit msg"),
        ]
        r = ScanResult(files_scanned=1, findings=findings)
        output = format_terminal(r, min_severity="high", no_color=True)
        assert "crit msg" in output
        assert "low msg" not in output

    def test_ignore_rules(self):
        f = Finding("a.py", 1, 0, "eval-exec", "CWE-95", "critical", "msg")
        r = ScanResult(files_scanned=1, findings=[f])
        output = format_terminal(r, ignore_rules={"CWE-95"}, no_color=True)
        assert "No security issues found" in output

    def test_grouped_by_file(self):
        findings = [
            Finding("a.py", 1, 0, "r1", "", "high", "msg1"),
            Finding("b.py", 2, 0, "r2", "", "high", "msg2"),
        ]
        r = ScanResult(files_scanned=2, findings=findings)
        output = format_terminal(r, no_color=True)
        assert "a.py" in output
        assert "b.py" in output


class TestFormatScoreText:
    def test_basic_output(self):
        s = SecurityScore(
            score=85, grade="B", files_scanned=10, total_findings=3,
            severity_counts={"high": 2, "low": 1}, profile="mixed",
            profile_detail="Multiple types", findings_per_file=0.3,
            top_rules=[("eval-exec", 2)],
        )
        output = format_score_text(s, no_color=True)
        assert "85/100" in output
        assert "(B)" in output
        assert "mixed" in output
        assert "eval-exec(2)" in output

    def test_no_severity_counts(self):
        s = SecurityScore(
            score=100, grade="A", files_scanned=5, total_findings=0,
            severity_counts={}, profile="clean",
            profile_detail="No findings", findings_per_file=0.0,
            top_rules=[],
        )
        output = format_score_text(s, no_color=True)
        assert "100/100" in output
        assert "Severity" not in output  # no severity line when empty


class TestFormatScoreJson:
    def test_valid_json(self):
        s = SecurityScore(
            score=50, grade="C", files_scanned=5, total_findings=2,
            severity_counts={"high": 2}, profile="injection_prone",
            profile_detail="desc", findings_per_file=0.4,
            top_rules=[("eval-exec", 2)],
        )
        result = format_score_json(s)
        data = json.loads(result)
        assert data["score"] == 50
        assert data["grade"] == "C"


# =============================================================
# CLI
# =============================================================

class TestCli:
    def test_main_clean_dir(self, tmp_path):
        _write_py(tmp_path, "x = 1", "clean.py")
        exit_code = main([str(tmp_path)])
        assert exit_code == 0

    def test_main_with_findings(self, tmp_path):
        _write_py(tmp_path, "eval(input())", "bad.py")
        exit_code = main([str(tmp_path)])
        assert exit_code == 2  # critical findings

    def test_main_json_output(self, tmp_path, capsys):
        _write_py(tmp_path, "eval(x)", "bad.py")
        main([str(tmp_path), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["finding_count"] >= 1

    def test_main_score_flag(self, tmp_path, capsys):
        _write_py(tmp_path, "x = 1", "clean.py")
        main([str(tmp_path), "--score"])
        captured = capsys.readouterr()
        assert "Score" in captured.out or "100/100" in captured.out

    def test_main_score_json(self, tmp_path, capsys):
        _write_py(tmp_path, "x = 1", "clean.py")
        main([str(tmp_path), "--score", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["score"] == 100

    def test_main_file_flag(self, tmp_path):
        path = _write_py(tmp_path, "eval(x)", "single.py")
        exit_code = main(["-f", path])
        assert exit_code == 2

    def test_main_severity_filter(self, tmp_path):
        _write_py(tmp_path, 'host = "192.168.1.1"', "low.py")
        exit_code = main([str(tmp_path), "--severity", "high"])
        assert exit_code == 0  # hardcoded-ip is low severity

    def test_main_ignore_rule(self, tmp_path):
        _write_py(tmp_path, "eval(x)", "bad.py")
        exit_code = main([str(tmp_path), "--ignore", "CWE-95"])
        assert exit_code == 0

    def test_main_nonexistent_target(self, tmp_path, capsys):
        exit_code = main(["/nonexistent/path/xyz123"])
        assert exit_code == 1

    def test_main_no_color(self, tmp_path, capsys):
        _write_py(tmp_path, "eval(x)", "bad.py")
        main([str(tmp_path), "--no-color"])
        captured = capsys.readouterr()
        assert "\033[" not in captured.out

    def test_build_parser(self):
        parser = build_parser()
        args = parser.parse_args(["mydir", "--json", "--severity", "high"])
        assert args.target == "mydir"
        assert args.json_output is True
        assert args.severity == "high"


# =============================================================
# Edge cases & integration
# =============================================================

class TestEdgeCases:
    def test_empty_file(self, tmp_path):
        path = _write_py(tmp_path, "", "empty.py")
        findings, error = scan_file(path)
        assert findings == []
        assert error is None

    def test_multiple_rules_same_file(self):
        """Multiple different rules should all fire."""
        findings = _scan_code("""\
            import os
            import pickle
            eval(x)
            os.system(cmd)
            pickle.loads(data)
        """)
        rules = set(_finding_rules(findings))
        assert "eval-exec" in rules
        assert "command-injection" in rules
        assert "insecure-deserialization" in rules

    def test_nested_function_calls(self):
        findings = _scan_code("eval(eval(x))")
        eval_findings = [f for f in findings if f.rule == "eval-exec"]
        assert len(eval_findings) == 2

    def test_cwe_mapping_complete(self):
        """All rule names used in RULE_CATEGORIES should have CWE mappings."""
        for rules in RULE_CATEGORIES.values():
            for rule in rules:
                assert rule in CWE_MAP, f"Rule {rule} missing CWE mapping"

    def test_severity_order_complete(self):
        """All severities used in weights should be in SEVERITY_ORDER."""
        for sev in SEVERITY_WEIGHTS:
            assert sev in SEVERITY_ORDER

    def test_finding_line_and_col(self):
        """Finding should have correct line and column numbers."""
        findings = _scan_code("x = 1\neval(y)\n")
        assert findings[0].line == 2

    def test_class_method_detection(self):
        """Security issues inside class methods should be found."""
        findings = _scan_code("""\
            class MyClass:
                def process(self):
                    eval(self.data)
        """)
        assert "eval-exec" in _finding_rules(findings)

    def test_no_false_positive_on_safe_code(self):
        """Common safe patterns should not trigger findings."""
        findings = _scan_code("""\
            import json
            import hashlib
            data = json.loads(text)
            h = hashlib.sha256(b"data")
            result = subprocess.run(["ls", "-la"])
        """)
        # json.loads, sha256, subprocess without shell=True are all safe
        assert "insecure-deserialization" not in _finding_rules(findings)
        assert "weak-crypto" not in _finding_rules(findings)
        assert "command-injection" not in _finding_rules(findings)

    def test_from_import_star_handling(self):
        """from module import * should not crash."""
        findings = _scan_code("from os import *\n")
        assert isinstance(findings, list)  # Just verify no crash
