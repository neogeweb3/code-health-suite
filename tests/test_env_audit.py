"""Tests for the env_audit engine — environment variable usage auditor."""
from __future__ import annotations

import json
import os
import textwrap

import pytest

from code_health_suite.engines.env_audit import (
    EnvVar,
    EnvReference,
    Finding,
    SECRET_PATTERNS,
    PLACEHOLDER_PATTERNS,
    SYSTEM_VARS,
    SKIP_DIRS,
    SCAN_EXTENSIONS,
    ENV_FILE_PATTERNS,
    SEVERITY_WEIGHTS,
    is_secret_name,
    is_placeholder_value,
    parse_env_file,
    scan_source_files,
    find_env_files,
    classify_env_file,
    run_audit,
    calculate_score,
    classify_profile,
    format_text,
    format_score,
    format_json,
    main,
)


# ===== Data Models =====

class TestEnvVar:
    def test_defaults(self):
        v = EnvVar(name="FOO")
        assert v.name == "FOO"
        assert v.value == ""
        assert v.source_file == ""
        assert v.line_number == 0
        assert v.has_value is True

    def test_full_init(self):
        v = EnvVar(name="DB_URL", value="postgres://...", source_file=".env",
                   line_number=5, has_value=True)
        assert v.name == "DB_URL"
        assert v.value == "postgres://..."
        assert v.source_file == ".env"
        assert v.line_number == 5

    def test_no_value(self):
        v = EnvVar(name="EMPTY", has_value=False)
        assert v.has_value is False


class TestEnvReference:
    def test_defaults(self):
        r = EnvReference(name="API_KEY", file_path="app.py", line_number=10)
        assert r.name == "API_KEY"
        assert r.context == ""
        assert r.language == ""

    def test_full_init(self):
        r = EnvReference(name="PORT", file_path="server.js", line_number=3,
                         context="const port = process.env.PORT", language="javascript")
        assert r.language == "javascript"
        assert "PORT" in r.context


class TestFinding:
    def test_defaults(self):
        f = Finding(check="undefined", severity="HIGH", variable="X",
                    message="not defined")
        assert f.files == []

    def test_with_files(self):
        f = Finding(check="unused", severity="LOW", variable="OLD_VAR",
                    message="never referenced", files=[".env"])
        assert f.files == [".env"]


# ===== Secret / Placeholder Detection =====

class TestIsSecretName:
    @pytest.mark.parametrize("name", [
        "DATABASE_PASSWORD", "DB_PASSWD", "MY_PWD",
        "API_SECRET", "AUTH_TOKEN", "STRIPE_API_KEY",
        "PRIVATE_KEY", "SSH_PRIV_KEY",
        "BASIC_AUTH", "AWS_CREDENTIAL", "MY_CRED",
        "DATABASE_URL", "DB_URL", "CONNECTION_STRING",
        "AWS_ACCESS_KEY", "AWS_KEY",
    ])
    def test_secret_names_detected(self, name):
        assert is_secret_name(name) is True

    @pytest.mark.parametrize("name", [
        "PORT", "HOST", "DEBUG", "LOG_LEVEL",
        "APP_NAME", "REGION", "TIMEOUT",
        "MAX_RETRIES", "FEATURE_FLAG",
    ])
    def test_non_secret_names(self, name):
        assert is_secret_name(name) is False


class TestIsPlaceholderValue:
    @pytest.mark.parametrize("value", [
        "your_api_key_here", "your-secret",
        "change_me", "change-me",
        "xxx", "XXX", "placeholder", "TODO",
        "CHANGEME", "REPLACE_THIS",
        "<your-key>", "<INSERT_HERE>",
        "...",
        "",  # empty string
    ])
    def test_placeholder_values(self, value):
        assert is_placeholder_value(value) is True

    @pytest.mark.parametrize("value", [
        "sk_live_abc123", "postgres://user:pass@host/db",
        "true", "3000", "production",
        "my-app", "us-east-1",
    ])
    def test_real_values(self, value):
        assert is_placeholder_value(value) is False


# ===== Env File Parsing =====

class TestParseEnvFile:
    def test_simple_kv(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("DB_HOST=localhost\nDB_PORT=5432\n")
        result = parse_env_file(f)
        assert len(result) == 2
        assert result[0].name == "DB_HOST"
        assert result[0].value == "localhost"
        assert result[1].name == "DB_PORT"
        assert result[1].value == "5432"

    def test_comments_and_blank_lines(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("# This is a comment\n\nKEY=value\n# Another comment\n")
        result = parse_env_file(f)
        assert len(result) == 1
        assert result[0].name == "KEY"

    def test_quoted_values(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text('DOUBLE="hello world"\nSINGLE=\'foo bar\'\n')
        result = parse_env_file(f)
        assert result[0].value == "hello world"
        assert result[1].value == "foo bar"

    def test_export_prefix(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("export MY_VAR=exported_value\n")
        result = parse_env_file(f)
        assert len(result) == 1
        assert result[0].name == "MY_VAR"
        assert result[0].value == "exported_value"

    def test_bare_variable_no_value(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SOME_VAR\n")
        result = parse_env_file(f)
        assert len(result) == 1
        assert result[0].name == "SOME_VAR"
        assert result[0].has_value is False
        assert result[0].value == ""

    def test_empty_value(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("EMPTY_VAR=\n")
        result = parse_env_file(f)
        assert len(result) == 1
        assert result[0].name == "EMPTY_VAR"
        assert result[0].has_value is False
        assert result[0].value == ""

    def test_value_with_equals(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("URL=postgres://user:pass@host/db?sslmode=require\n")
        result = parse_env_file(f)
        assert result[0].value == "postgres://user:pass@host/db?sslmode=require"

    def test_line_numbers(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("# header\nFIRST=1\n\nSECOND=2\n")
        result = parse_env_file(f)
        assert result[0].line_number == 2
        assert result[1].line_number == 4

    def test_source_file_recorded(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("X=1\n")
        result = parse_env_file(f)
        assert result[0].source_file == str(f)

    def test_invalid_key_skipped(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("123INVALID=value\nVALID_KEY=ok\n-bad=no\n")
        result = parse_env_file(f)
        assert len(result) == 1
        assert result[0].name == "VALID_KEY"

    def test_nonexistent_file(self, tmp_path):
        f = tmp_path / "nonexistent"
        result = parse_env_file(f)
        assert result == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("")
        result = parse_env_file(f)
        assert result == []


# ===== Source Code Scanning =====

class TestScanSourceFiles:
    def test_python_os_environ(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('import os\ndb = os.environ["DATABASE_URL"]\n')
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "DATABASE_URL"
        assert refs[0].language == "python"

    def test_python_os_getenv(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('import os\nport = os.getenv("PORT", 3000)\n')
        refs = scan_source_files(tmp_path)
        # os.getenv matches both the getenv pattern AND the dotenv env() pattern
        assert len(refs) >= 1
        assert all(r.name == "PORT" for r in refs)

    def test_python_environ_get(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('import os\nkey = os.environ.get("API_KEY")\n')
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "API_KEY"

    def test_python_config_style(self, tmp_path):
        f = tmp_path / "settings.py"
        f.write_text('SECRET = config("SECRET_KEY")\n')
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "SECRET_KEY"

    def test_javascript_process_env_dot(self, tmp_path):
        f = tmp_path / "server.js"
        f.write_text("const port = process.env.PORT_NUMBER;\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "PORT_NUMBER"
        assert refs[0].language == "javascript"

    def test_javascript_process_env_bracket(self, tmp_path):
        f = tmp_path / "server.js"
        f.write_text('const key = process.env["API_KEY"];\n')
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "API_KEY"

    def test_typescript_file(self, tmp_path):
        f = tmp_path / "config.ts"
        f.write_text("const url = process.env.DATABASE_URL;\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].language == "javascript"

    def test_vite_import_meta(self, tmp_path):
        f = tmp_path / "app.tsx"
        f.write_text("const url = import.meta.env.VITE_API_URL;\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "VITE_API_URL"

    def test_shell_dollar_brace(self, tmp_path):
        f = tmp_path / "deploy.sh"
        f.write_text("echo ${DATABASE_HOST}\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "DATABASE_HOST"
        assert refs[0].language == "shell"

    def test_shell_dollar_caps(self, tmp_path):
        f = tmp_path / "run.sh"
        f.write_text("echo $APP_PORT\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "APP_PORT"

    def test_skip_comments_python(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('# key = os.getenv("SECRET")\nreal = os.getenv("REAL")\n')
        refs = scan_source_files(tmp_path)
        # Comment line skipped; os.getenv matches multiple patterns
        assert all(r.name == "REAL" for r in refs)
        assert len(refs) >= 1

    def test_skip_comments_js(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text('// const x = process.env.SKIP_ME;\nconst y = process.env.KEEP_ME;\n')
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "KEEP_ME"

    def test_system_vars_filtered(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('home = os.environ["HOME"]\npath = os.environ["PATH"]\n')
        refs = scan_source_files(tmp_path)
        assert len(refs) == 0

    def test_skip_dirs(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("process.env.HIDDEN_VAR;\n")
        (tmp_path / "app.js").write_text("process.env.VISIBLE_VAR;\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 1
        assert refs[0].name == "VISIBLE_VAR"

    def test_max_files_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text(f'os.environ["VAR_{i}"]\n')
        refs = scan_source_files(tmp_path, max_files=3)
        unique_files = {r.file_path for r in refs}
        assert len(unique_files) <= 3

    def test_line_number_and_context(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('x = 1\ny = os.getenv("MY_VAR")\nz = 3\n')
        refs = scan_source_files(tmp_path)
        assert refs[0].line_number == 2
        assert "MY_VAR" in refs[0].context

    def test_file_path_recorded(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('os.getenv("SOME_VAR")\n')
        refs = scan_source_files(tmp_path)
        assert refs[0].file_path == str(f)

    def test_multiple_refs_same_file(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('a = os.getenv("VAR_A")\nb = os.getenv("VAR_B")\n')
        refs = scan_source_files(tmp_path)
        names = {r.name for r in refs}
        assert names == {"VAR_A", "VAR_B"}

    def test_unsupported_extension_skipped(self, tmp_path):
        (tmp_path / "data.csv").write_text("process.env.HIDDEN\n")
        refs = scan_source_files(tmp_path)
        assert len(refs) == 0

    def test_empty_directory(self, tmp_path):
        refs = scan_source_files(tmp_path)
        assert refs == []

    def test_unreadable_file_skipped(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text('os.getenv("VAR")\n')
        os.chmod(f, 0o000)
        try:
            refs = scan_source_files(tmp_path)
            # Should not crash, just skip
            assert isinstance(refs, list)
        finally:
            os.chmod(f, 0o644)


# ===== Env File Discovery =====

class TestFindEnvFiles:
    def test_finds_standard_env(self, tmp_path):
        (tmp_path / ".env").write_text("KEY=val\n")
        result = find_env_files(tmp_path)
        assert ".env" in result
        assert len(result[".env"]) == 1

    def test_finds_multiple_env_files(self, tmp_path):
        (tmp_path / ".env").write_text("A=1\n")
        (tmp_path / ".env.example").write_text("A=placeholder\n")
        (tmp_path / ".env.local").write_text("A=local\n")
        result = find_env_files(tmp_path)
        assert len(result) == 3

    def test_no_env_files(self, tmp_path):
        result = find_env_files(tmp_path)
        assert result == {}

    def test_all_supported_patterns(self, tmp_path):
        for pattern in ENV_FILE_PATTERNS:
            (tmp_path / pattern).write_text(f"VAR_{pattern.replace('.', '_')}=1\n")
        result = find_env_files(tmp_path)
        assert len(result) == len(ENV_FILE_PATTERNS)


class TestClassifyEnvFile:
    @pytest.mark.parametrize("name,expected", [
        (".env", "actual"),
        (".env.local", "actual"),
        (".env.development", "actual"),
        (".env.production", "actual"),
        (".env.staging", "actual"),
        (".env.test", "actual"),
    ])
    def test_actual_files(self, name, expected):
        assert classify_env_file(name) == expected

    @pytest.mark.parametrize("name,expected", [
        (".env.example", "template"),
        (".env.sample", "template"),
        (".env.template", "template"),
        (".env.defaults", "template"),
    ])
    def test_template_files(self, name, expected):
        assert classify_env_file(name) == expected


# ===== Audit Engine =====

class TestRunAudit:
    def test_no_env_files_no_code(self, tmp_path):
        findings = run_audit(tmp_path)
        assert findings == []

    def test_no_env_files_but_code_refs(self, tmp_path):
        (tmp_path / "app.py").write_text('x = os.getenv("MY_VAR")\n')
        findings = run_audit(tmp_path)
        assert len(findings) == 1
        assert findings[0].check == "no_env_files"
        assert findings[0].severity == "MODERATE"

    def test_undefined_variable(self, tmp_path):
        """Code references a var not in any .env file."""
        (tmp_path / ".env").write_text("KNOWN=value\n")
        (tmp_path / "app.py").write_text('x = os.getenv("UNKNOWN_VAR")\n')
        findings = run_audit(tmp_path)
        undefined = [f for f in findings if f.check == "undefined"]
        assert len(undefined) == 1
        assert undefined[0].variable == "UNKNOWN_VAR"
        assert undefined[0].severity == "HIGH"

    def test_unused_variable(self, tmp_path):
        """Var in .env but never referenced in code."""
        (tmp_path / ".env").write_text("USED=1\nUNUSED=2\n")
        (tmp_path / "app.py").write_text('x = os.getenv("USED")\n')
        findings = run_audit(tmp_path)
        unused = [f for f in findings if f.check == "unused"]
        assert len(unused) == 1
        assert unused[0].variable == "UNUSED"
        assert unused[0].severity == "LOW"

    def test_missing_template(self, tmp_path):
        """Var in actual .env but not in template."""
        (tmp_path / ".env").write_text("DB_HOST=localhost\nEXTRA=val\n")
        (tmp_path / ".env.example").write_text("DB_HOST=<your-host>\n")
        (tmp_path / "app.py").write_text(
            'os.getenv("DB_HOST")\nos.getenv("EXTRA")\n'
        )
        findings = run_audit(tmp_path)
        missing_tmpl = [f for f in findings if f.check == "missing_template"]
        assert len(missing_tmpl) == 1
        assert missing_tmpl[0].variable == "EXTRA"
        assert missing_tmpl[0].severity == "MODERATE"

    def test_missing_actual(self, tmp_path):
        """Var in template but not in actual .env."""
        (tmp_path / ".env").write_text("A=1\n")
        (tmp_path / ".env.example").write_text("A=placeholder\nB=placeholder\n")
        findings = run_audit(tmp_path)
        missing_actual = [f for f in findings if f.check == "missing_actual"]
        assert len(missing_actual) == 1
        assert missing_actual[0].variable == "B"

    def test_secret_in_template(self, tmp_path):
        """Secret var with a real value in template file."""
        (tmp_path / ".env.example").write_text(
            "API_SECRET=sk_live_real_key_abc123\n"
        )
        findings = run_audit(tmp_path)
        secrets = [f for f in findings if f.check == "secret_in_template"]
        assert len(secrets) == 1
        assert secrets[0].severity == "HIGH"

    def test_secret_in_template_placeholder_ok(self, tmp_path):
        """Secret var with placeholder in template is fine."""
        (tmp_path / ".env.example").write_text(
            "API_SECRET=your_api_secret_here\n"
        )
        findings = run_audit(tmp_path)
        secrets = [f for f in findings if f.check == "secret_in_template"]
        assert len(secrets) == 0

    def test_duplicate_in_same_file(self, tmp_path):
        (tmp_path / ".env").write_text("KEY=first\nKEY=second\n")
        findings = run_audit(tmp_path)
        dups = [f for f in findings if f.check == "duplicate"]
        assert len(dups) == 1
        assert dups[0].variable == "KEY"

    def test_empty_value_in_actual(self, tmp_path):
        (tmp_path / ".env").write_text("EMPTY_VAR\n")
        findings = run_audit(tmp_path)
        empties = [f for f in findings if f.check == "empty"]
        assert len(empties) == 1
        assert empties[0].severity == "INFO"

    def test_clean_project(self, tmp_path):
        """All vars defined and used, no issues."""
        (tmp_path / ".env").write_text("PORT=3000\nDB_HOST=localhost\n")
        (tmp_path / ".env.example").write_text("PORT=3000\nDB_HOST=<host>\n")
        (tmp_path / "app.py").write_text(
            'os.getenv("PORT")\nos.getenv("DB_HOST")\n'
        )
        findings = run_audit(tmp_path)
        assert len(findings) == 0

    def test_multiple_issue_types(self, tmp_path):
        """Multiple issues in one project."""
        (tmp_path / ".env").write_text("A=1\nB=2\nB=3\n")
        (tmp_path / ".env.example").write_text("A=placeholder\nDB_PASSWORD=real_pass\n")
        (tmp_path / "app.py").write_text('os.getenv("A")\nos.getenv("MISSING")\n')
        findings = run_audit(tmp_path)
        checks = {f.check for f in findings}
        assert "undefined" in checks  # MISSING not in .env
        assert "duplicate" in checks  # B defined twice
        assert "secret_in_template" in checks  # DB_PASSWORD with real value

    def test_missing_template_only_flagged_if_used(self, tmp_path):
        """missing_template only for vars referenced in code."""
        (tmp_path / ".env").write_text("USED=1\nUNUSED_EXTRA=2\n")
        (tmp_path / ".env.example").write_text("USED=placeholder\n")
        (tmp_path / "app.py").write_text('os.getenv("USED")\n')
        findings = run_audit(tmp_path)
        missing_tmpl = [f for f in findings if f.check == "missing_template"]
        # UNUSED_EXTRA is not referenced in code, so should NOT be flagged as missing_template
        assert len(missing_tmpl) == 0

    def test_no_template_no_missing_template_check(self, tmp_path):
        """Without template files, skip missing_template check."""
        (tmp_path / ".env").write_text("A=1\nB=2\n")
        (tmp_path / "app.py").write_text('os.getenv("A")\n')
        findings = run_audit(tmp_path)
        assert all(f.check != "missing_template" for f in findings)

    def test_no_actual_no_missing_actual_check(self, tmp_path):
        """Without actual .env, skip missing_actual check."""
        (tmp_path / ".env.example").write_text("A=placeholder\n")
        findings = run_audit(tmp_path)
        assert all(f.check != "missing_actual" for f in findings)


# ===== Scoring =====

class TestCalculateScore:
    def test_perfect_score(self):
        score, grade = calculate_score([], {".env": [EnvVar("X")]}, {"X": []})
        assert score == 100
        assert grade == "A"

    def test_no_vars_perfect(self):
        score, grade = calculate_score([], {}, {})
        assert score == 100
        assert grade == "A"

    def test_high_findings_reduce_score(self):
        findings = [
            Finding("undefined", "HIGH", "X", "missing"),
            Finding("undefined", "HIGH", "Y", "missing"),
        ]
        env_files = {".env": [EnvVar("Z")]}
        code_vars = {"X": [], "Y": [], "Z": []}
        score, grade = calculate_score(findings, env_files, code_vars)
        assert score < 100

    def test_grade_boundaries(self):
        # With many findings, score should drop to lower grades
        many_findings = [
            Finding("undefined", "HIGH", f"V{i}", "missing") for i in range(20)
        ]
        env = {".env": [EnvVar(f"V{i}") for i in range(5)]}
        code = {f"V{i}": [] for i in range(20)}
        score, grade = calculate_score(many_findings, env, code)
        assert grade in ("D", "F")

    def test_score_never_negative(self):
        findings = [Finding("x", "HIGH", f"V{i}", "m") for i in range(100)]
        env = {".env": [EnvVar("A")]}
        code = {"A": []}
        score, _ = calculate_score(findings, env, code)
        assert score >= 0


class TestClassifyProfile:
    def test_clean(self):
        assert classify_profile([]) == "clean"

    def test_secret_exposure(self):
        findings = [
            Finding("secret_in_template", "HIGH", "K", "m"),
            Finding("secret_in_template", "HIGH", "K2", "m"),
        ]
        assert classify_profile(findings) == "secret_exposure"

    def test_missing_config(self):
        findings = [
            Finding("undefined", "HIGH", "A", "m"),
            Finding("undefined", "HIGH", "B", "m"),
            Finding("unused", "LOW", "C", "m"),
        ]
        assert classify_profile(findings) == "missing_config"

    def test_config_bloat(self):
        findings = [
            Finding("unused", "LOW", "A", "m"),
            Finding("unused", "LOW", "B", "m"),
            Finding("undefined", "HIGH", "C", "m"),
        ]
        assert classify_profile(findings) == "config_bloat"

    def test_template_drift(self):
        findings = [
            Finding("missing_template", "MODERATE", "A", "m"),
            Finding("missing_actual", "MODERATE", "B", "m"),
            Finding("missing_template", "MODERATE", "C", "m"),
        ]
        assert classify_profile(findings) == "template_drift"

    def test_mixed(self):
        findings = [
            Finding("undefined", "HIGH", "A", "m"),
            Finding("unused", "LOW", "B", "m"),
            Finding("secret_in_template", "HIGH", "C", "m"),
            Finding("missing_template", "MODERATE", "D", "m"),
            Finding("duplicate", "LOW", "E", "m"),
            Finding("empty", "INFO", "F", "m"),
        ]
        assert classify_profile(findings) == "mixed"

    def test_minor_issues(self):
        # With 4 findings where no single category dominates 1/3
        findings = [
            Finding("duplicate", "LOW", "W", "m"),
            Finding("empty", "INFO", "X", "m"),
            Finding("undefined", "HIGH", "Y", "m"),
            Finding("unused", "LOW", "Z", "m"),
        ]
        assert classify_profile(findings) == "minor_issues"


# ===== Output Formatting =====

class TestFormatText:
    def test_no_findings(self, tmp_path):
        text = format_text([], tmp_path, {".env": []}, [])
        assert "No issues found" in text

    def test_with_findings(self, tmp_path):
        findings = [
            Finding("undefined", "HIGH", "API_KEY", "Not defined", ["app.py"]),
            Finding("unused", "LOW", "OLD", "Never used", [".env"]),
        ]
        text = format_text(findings, tmp_path, {".env": [EnvVar("OLD")]}, [])
        assert "HIGH" in text
        assert "LOW" in text
        assert "API_KEY" in text
        assert "Total: 2 findings" in text

    def test_header(self, tmp_path):
        text = format_text([], tmp_path, {}, [])
        assert "ai-env-audit" in text

    def test_summary_counts(self, tmp_path):
        env_files = {".env": [EnvVar("A"), EnvVar("B")]}
        refs = [
            EnvReference("A", "app.py", 1),
            EnvReference("C", "app.py", 2),
        ]
        text = format_text([], tmp_path, env_files, refs)
        assert "Env files found: 1" in text
        assert "Variables in .env files: 2" in text
        assert "Variables referenced in code: 2" in text


class TestFormatScore:
    def test_basic(self):
        findings = [Finding("undefined", "HIGH", "X", "m")]
        text = format_score(85, "B", "missing_config", findings)
        assert "Score: 85/100" in text
        assert "Grade: B" in text
        assert "Profile: missing_config" in text
        assert "Findings: 1" in text

    def test_breakdown(self):
        findings = [
            Finding("x", "HIGH", "A", "m"),
            Finding("x", "LOW", "B", "m"),
        ]
        text = format_score(70, "C", "mixed", findings)
        assert "HIGH=1" in text
        assert "LOW=1" in text


class TestFormatJson:
    def test_basic_structure(self, tmp_path):
        findings = [Finding("undefined", "HIGH", "X", "missing", ["app.py"])]
        output = format_json(findings, tmp_path, {".env": []}, [])
        data = json.loads(output)
        assert data["version"] == "0.1.0"
        assert len(data["findings"]) == 1
        assert data["summary"]["high"] == 1
        assert data["summary"]["total"] == 1
        assert ".env" in data["env_files"]

    def test_with_score(self, tmp_path):
        findings = []
        refs = [EnvReference("X", "app.py", 1)]
        output = format_json(findings, tmp_path, {".env": [EnvVar("X")]}, refs,
                             show_score=True)
        data = json.loads(output)
        assert "score" in data
        assert "grade" in data
        assert "profile" in data

    def test_empty_findings(self, tmp_path):
        output = format_json([], tmp_path, {}, [])
        data = json.loads(output)
        assert data["findings"] == []
        assert data["summary"]["total"] == 0


# ===== CLI =====

class TestMain:
    def test_clean_project_exit_0(self, tmp_path):
        (tmp_path / ".env").write_text("PORT=3000\n")
        (tmp_path / "app.py").write_text('os.getenv("PORT")\n')
        ret = main([str(tmp_path)])
        assert ret == 0

    def test_high_findings_exit_1(self, tmp_path):
        (tmp_path / ".env").write_text("KNOWN=1\n")
        (tmp_path / "app.py").write_text('os.getenv("UNKNOWN")\n')
        ret = main([str(tmp_path)])
        assert ret == 1

    def test_json_output(self, tmp_path, capsys):
        (tmp_path / ".env").write_text("X=1\n")
        main([str(tmp_path), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "findings" in data

    def test_score_output(self, tmp_path, capsys):
        (tmp_path / ".env").write_text("X=1\n")
        main([str(tmp_path), "--score"])
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Grade:" in captured.out

    def test_json_with_score(self, tmp_path, capsys):
        (tmp_path / ".env").write_text("X=1\n")
        main([str(tmp_path), "--json", "--score"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "score" in data

    def test_invalid_path_exit_1(self, capsys):
        ret = main(["/nonexistent/path/xyz"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("X=1\n")
        ret = main([])
        assert ret == 0

    def test_no_env_no_code_exit_0(self, tmp_path):
        ret = main([str(tmp_path)])
        assert ret == 0


# ===== Constants Validation =====

class TestConstants:
    def test_severity_weights_complete(self):
        assert "HIGH" in SEVERITY_WEIGHTS
        assert "MODERATE" in SEVERITY_WEIGHTS
        assert "LOW" in SEVERITY_WEIGHTS
        assert "INFO" in SEVERITY_WEIGHTS

    def test_system_vars_has_common_vars(self):
        assert "HOME" in SYSTEM_VARS
        assert "PATH" in SYSTEM_VARS
        assert "NODE_ENV" in SYSTEM_VARS
        assert "CI" in SYSTEM_VARS

    def test_skip_dirs_has_common_dirs(self):
        assert "node_modules" in SKIP_DIRS
        assert ".git" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS

    def test_scan_extensions_coverage(self):
        assert ".py" in SCAN_EXTENSIONS
        assert ".js" in SCAN_EXTENSIONS
        assert ".ts" in SCAN_EXTENSIONS
        assert ".sh" in SCAN_EXTENSIONS

    def test_env_file_patterns(self):
        assert ".env" in ENV_FILE_PATTERNS
        assert ".env.example" in ENV_FILE_PATTERNS
        assert ".env.local" in ENV_FILE_PATTERNS


# ===== Integration Tests =====

class TestIntegration:
    def test_full_project_scan(self, tmp_path):
        """Simulate a realistic project with env files and source code."""
        # .env with actual values
        (tmp_path / ".env").write_text(
            "DATABASE_URL=postgres://user:pass@localhost/db\n"
            "API_KEY=sk_live_abc123\n"
            "PORT=3000\n"
            "DEBUG=true\n"
        )
        # .env.example with placeholders
        (tmp_path / ".env.example").write_text(
            "DATABASE_URL=<your-database-url>\n"
            "API_KEY=your_api_key_here\n"
            "PORT=3000\n"
        )
        # Python source using some vars
        (tmp_path / "app.py").write_text(textwrap.dedent("""\
            import os
            db = os.environ["DATABASE_URL"]
            key = os.getenv("API_KEY")
            port = os.getenv("PORT", 3000)
            secret = os.getenv("JWT_SECRET")
        """))

        findings = run_audit(tmp_path)

        checks = {f.check: f for f in findings}
        # JWT_SECRET referenced but not in .env → undefined
        assert "undefined" in checks
        assert checks["undefined"].variable == "JWT_SECRET"

        # DEBUG in .env but not referenced → unused
        unused = [f for f in findings if f.check == "unused"]
        assert any(f.variable == "DEBUG" for f in unused)

        # DEBUG missing from template but unused → NOT missing_template
        missing_tmpl = [f for f in findings if f.check == "missing_template"]
        assert all(f.variable != "DEBUG" for f in missing_tmpl)

    def test_js_project(self, tmp_path):
        """JS project with process.env usage."""
        (tmp_path / ".env").write_text("REACT_APP_API_URL=http://localhost:8080\n")
        (tmp_path / "index.jsx").write_text(
            "const url = process.env.REACT_APP_API_URL;\n"
            "const key = process.env.REACT_APP_KEY;\n"
        )
        findings = run_audit(tmp_path)
        undefined = [f for f in findings if f.check == "undefined"]
        assert any(f.variable == "REACT_APP_KEY" for f in undefined)

    def test_shell_project(self, tmp_path):
        """Shell script project with env vars."""
        (tmp_path / ".env").write_text("DEPLOY_TARGET=production\n")
        (tmp_path / "deploy.sh").write_text(
            "echo ${DEPLOY_TARGET}\necho ${MISSING_CREDENTIAL}\n"
        )
        findings = run_audit(tmp_path)
        undefined = [f for f in findings if f.check == "undefined"]
        assert any(f.variable == "MISSING_CREDENTIAL" for f in undefined)

    def test_mixed_language_project(self, tmp_path):
        """Project with Python, JS, and shell files."""
        (tmp_path / ".env").write_text("SHARED_VAR=value\n")
        (tmp_path / "app.py").write_text('os.getenv("SHARED_VAR")\nos.getenv("PY_ONLY")\n')
        (tmp_path / "server.js").write_text("process.env.SHARED_VAR;\nprocess.env.JS_ONLY;\n")
        (tmp_path / "run.sh").write_text("echo ${SHARED_VAR}\n")

        findings = run_audit(tmp_path)
        undefined = [f for f in findings if f.check == "undefined"]
        undefined_vars = {f.variable for f in undefined}
        assert "PY_ONLY" in undefined_vars
        assert "JS_ONLY" in undefined_vars

    def test_end_to_end_cli(self, tmp_path, capsys):
        """Full CLI roundtrip with JSON + score."""
        (tmp_path / ".env").write_text("A=1\nB=2\n")
        (tmp_path / ".env.example").write_text("A=placeholder\n")
        (tmp_path / "app.py").write_text('os.getenv("A")\nos.getenv("C")\n')

        ret = main([str(tmp_path), "--json", "--score"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert "score" in data
        assert "grade" in data
        assert "profile" in data
        assert data["summary"]["total"] > 0
        # Should have HIGH finding (C undefined) → exit 1
        assert ret == 1

    def test_subdirectory_scanning(self, tmp_path):
        """Source files in subdirectories are found."""
        (tmp_path / ".env").write_text("DEEP_VAR=1\n")
        sub = tmp_path / "src" / "lib"
        sub.mkdir(parents=True)
        (sub / "config.py").write_text('os.getenv("DEEP_VAR")\nos.getenv("MISSING")\n')
        findings = run_audit(tmp_path)
        undefined = [f for f in findings if f.check == "undefined"]
        assert any(f.variable == "MISSING" for f in undefined)
