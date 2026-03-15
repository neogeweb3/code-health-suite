"""Tests for the dep_audit engine — Python dependency health checker."""
from __future__ import annotations

import json
import os
import textwrap

import pytest

from code_health_suite.engines.dep_audit import (
    Dependency,
    StaticFinding,
    VersionInfo,
    Vulnerability,
    AuditResult,
    DepHealthScore,
    SEVERITY_ORDER,
    SEVERITY_WEIGHTS,
    OUTDATED_WEIGHTS,
    STATIC_WEIGHTS,
    GRADE_THRESHOLDS,
    __version__,
    parse_requirements_txt,
    parse_pyproject_toml,
    find_and_parse_deps,
    query_pypi,
    parse_version_tuple,
    classify_update,
    check_version,
    query_osv,
    normalize_package_name,
    check_unpinned,
    check_wildcard,
    check_duplicates,
    check_conflicts,
    check_hygiene,
    run_static_checks,
    audit_dependencies,
    severity_passes_filter,
    format_terminal,
    format_json,
    compute_dep_health,
    classify_dep_profile,
    format_score_text,
    format_score_json,
    build_parser,
    main,
)


# ===== Data Models =====


class TestDependency:
    def test_defaults(self):
        d = Dependency(name="requests")
        assert d.name == "requests"
        assert d.specified_version is None
        assert d.constraint == ""
        assert d.source_file == ""
        assert d.extras == ""

    def test_full_init(self):
        d = Dependency(
            name="flask",
            specified_version="2.3.0",
            constraint="==",
            source_file="requirements.txt",
            extras="[async]",
        )
        assert d.name == "flask"
        assert d.specified_version == "2.3.0"
        assert d.constraint == "=="
        assert d.source_file == "requirements.txt"
        assert d.extras == "[async]"


class TestStaticFinding:
    def test_defaults(self):
        sf = StaticFinding(
            check="unpinned", severity="MODERATE", package="foo", message="msg"
        )
        assert sf.check == "unpinned"
        assert sf.severity == "MODERATE"
        assert sf.package == "foo"
        assert sf.message == "msg"
        assert sf.source_files == []

    def test_with_source_files(self):
        sf = StaticFinding(
            check="duplicate",
            severity="HIGH",
            package="bar",
            message="dup",
            source_files=["a.txt", "b.txt"],
        )
        assert sf.source_files == ["a.txt", "b.txt"]


class TestVersionInfo:
    def test_defaults(self):
        vi = VersionInfo()
        assert vi.current is None
        assert vi.latest is None
        assert vi.update_type is None

    def test_full_init(self):
        vi = VersionInfo(current="1.0.0", latest="2.0.0", update_type="major")
        assert vi.current == "1.0.0"
        assert vi.latest == "2.0.0"
        assert vi.update_type == "major"


class TestVulnerability:
    def test_defaults(self):
        v = Vulnerability(id="CVE-2024-1234", summary="Bad thing")
        assert v.id == "CVE-2024-1234"
        assert v.summary == "Bad thing"
        assert v.severity == "UNKNOWN"
        assert v.fixed_version is None
        assert v.url == ""

    def test_full_init(self):
        v = Vulnerability(
            id="GHSA-abc",
            summary="XSS",
            severity="HIGH",
            fixed_version="1.2.3",
            url="https://osv.dev/vulnerability/GHSA-abc",
        )
        assert v.severity == "HIGH"
        assert v.fixed_version == "1.2.3"


class TestAuditResult:
    def test_defaults(self):
        dep = Dependency(name="pkg")
        r = AuditResult(dependency=dep)
        assert r.dependency.name == "pkg"
        assert r.version_info.current is None
        assert r.vulnerabilities == []

    def test_is_outdated_true(self):
        dep = Dependency(name="pkg", specified_version="1.0")
        r = AuditResult(
            dependency=dep,
            version_info=VersionInfo(current="1.0", latest="2.0", update_type="major"),
        )
        assert r.is_outdated is True

    def test_is_outdated_false_uptodate(self):
        dep = Dependency(name="pkg", specified_version="2.0")
        r = AuditResult(
            dependency=dep,
            version_info=VersionInfo(
                current="2.0", latest="2.0", update_type="up-to-date"
            ),
        )
        assert r.is_outdated is False

    def test_is_outdated_false_none(self):
        dep = Dependency(name="pkg")
        r = AuditResult(dependency=dep)
        assert r.is_outdated is False

    def test_has_vulns_true(self):
        dep = Dependency(name="pkg")
        r = AuditResult(
            dependency=dep,
            vulnerabilities=[Vulnerability(id="CVE-1", summary="bad")],
        )
        assert r.has_vulns is True

    def test_has_vulns_false(self):
        dep = Dependency(name="pkg")
        r = AuditResult(dependency=dep)
        assert r.has_vulns is False

    def test_max_severity_none(self):
        dep = Dependency(name="pkg")
        r = AuditResult(dependency=dep)
        assert r.max_severity == "NONE"

    def test_max_severity_picks_highest(self):
        dep = Dependency(name="pkg")
        r = AuditResult(
            dependency=dep,
            vulnerabilities=[
                Vulnerability(id="V1", summary="low", severity="LOW"),
                Vulnerability(id="V2", summary="crit", severity="CRITICAL"),
                Vulnerability(id="V3", summary="high", severity="HIGH"),
            ],
        )
        assert r.max_severity == "CRITICAL"

    def test_max_severity_single(self):
        dep = Dependency(name="pkg")
        r = AuditResult(
            dependency=dep,
            vulnerabilities=[
                Vulnerability(id="V1", summary="mod", severity="MODERATE"),
            ],
        )
        assert r.max_severity == "MODERATE"


# ===== Constants =====


class TestConstants:
    def test_severity_order_hierarchy(self):
        assert SEVERITY_ORDER["CRITICAL"] > SEVERITY_ORDER["HIGH"]
        assert SEVERITY_ORDER["HIGH"] > SEVERITY_ORDER["MODERATE"]
        assert SEVERITY_ORDER["MODERATE"] > SEVERITY_ORDER["LOW"]
        assert SEVERITY_ORDER["LOW"] > SEVERITY_ORDER["UNKNOWN"]
        assert SEVERITY_ORDER["UNKNOWN"] > SEVERITY_ORDER["NONE"]

    def test_grade_thresholds_descending(self):
        thresholds = [t for t, _ in GRADE_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_grade_threshold_covers_zero(self):
        assert GRADE_THRESHOLDS[-1][0] == 0

    def test_severity_weights_exist(self):
        for key in ("CRITICAL", "HIGH", "MODERATE", "LOW", "UNKNOWN"):
            assert key in SEVERITY_WEIGHTS

    def test_outdated_weights_exist(self):
        for key in ("major", "minor", "patch"):
            assert key in OUTDATED_WEIGHTS

    def test_static_weights_exist(self):
        for key in ("HIGH", "MODERATE", "LOW", "INFO"):
            assert key in STATIC_WEIGHTS

    def test_version_string(self):
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3


# ===== Parsers =====


class TestParseRequirementsTxt:
    def test_pinned_versions(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests==2.31.0\nflask==3.0.0\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 2
        assert deps[0].name == "requests"
        assert deps[0].specified_version == "2.31.0"
        assert deps[0].constraint == "=="
        assert deps[1].name == "flask"

    def test_no_version(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests\nflask\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 2
        assert deps[0].specified_version is None
        assert deps[0].constraint == ""

    def test_greater_than(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests>=2.20.0\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 1
        assert deps[0].constraint == ">="
        assert deps[0].specified_version == "2.20.0"

    def test_compatible_release(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests~=2.31.0\n")
        deps = parse_requirements_txt(str(f))
        assert deps[0].constraint == "~="

    def test_comments_and_blanks(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("# comment\n\nrequests==1.0\n# another\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_options_lines_skipped(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("-r base.txt\n--index-url https://pypi.org/simple/\nrequests==1.0\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 1

    def test_extras_ignored(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests[security]==2.31.0\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 1
        assert deps[0].name == "requests"
        assert deps[0].specified_version == "2.31.0"

    def test_source_file_set(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("foo==1.0\n")
        deps = parse_requirements_txt(str(f))
        assert deps[0].source_file == str(f)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("")
        deps = parse_requirements_txt(str(f))
        assert deps == []

    def test_dashes_underscores_dots_in_name(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("my-package==1.0\nmy_other.pkg==2.0\n")
        deps = parse_requirements_txt(str(f))
        assert len(deps) == 2
        assert deps[0].name == "my-package"
        assert deps[1].name == "my_other.pkg"


class TestParsePyprojectToml:
    def test_basic(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(textwrap.dedent("""\
            [project]
            name = "myproject"
            dependencies = [
                "requests==2.31.0",
                "flask>=3.0.0",
            ]
        """))
        deps = parse_pyproject_toml(str(f))
        assert len(deps) == 2
        assert deps[0].name == "requests"
        assert deps[0].specified_version == "2.31.0"
        assert deps[0].constraint == "=="
        assert deps[1].name == "flask"
        assert deps[1].constraint == ">="

    def test_no_dependencies_key(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(textwrap.dedent("""\
            [project]
            name = "myproject"
        """))
        deps = parse_pyproject_toml(str(f))
        assert deps == []

    def test_extras_in_dependency(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(textwrap.dedent("""\
            [project]
            dependencies = [
                "requests[security]>=2.20.0",
            ]
        """))
        deps = parse_pyproject_toml(str(f))
        assert len(deps) == 1
        assert deps[0].name == "requests"
        assert deps[0].constraint == ">="

    def test_no_version(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(textwrap.dedent("""\
            [project]
            dependencies = [
                "requests",
            ]
        """))
        deps = parse_pyproject_toml(str(f))
        assert len(deps) == 1
        assert deps[0].specified_version is None

    def test_single_quoted_strings(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text(textwrap.dedent("""\
            [project]
            dependencies = [
                'requests==2.31.0',
            ]
        """))
        deps = parse_pyproject_toml(str(f))
        assert len(deps) == 1
        assert deps[0].name == "requests"

    def test_source_file_set(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text('[project]\ndependencies = ["foo==1.0"]\n')
        deps = parse_pyproject_toml(str(f))
        assert deps[0].source_file == str(f)

    def test_empty_dependencies(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text('[project]\ndependencies = []\n')
        deps = parse_pyproject_toml(str(f))
        assert deps == []


class TestFindAndParseDeps:
    def test_directory_with_pyproject(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text('[project]\ndependencies = ["requests==2.31.0"]\n')
        deps = find_and_parse_deps(str(tmp_path))
        assert len(deps) == 1

    def test_directory_with_requirements(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests==2.31.0\n")
        deps = find_and_parse_deps(str(tmp_path))
        assert len(deps) == 1

    def test_directory_with_both(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["flask==3.0.0"]\n'
        )
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        deps = find_and_parse_deps(str(tmp_path))
        assert len(deps) == 2

    def test_file_path_pyproject(self, tmp_path):
        f = tmp_path / "pyproject.toml"
        f.write_text('[project]\ndependencies = ["requests==2.31.0"]\n')
        deps = find_and_parse_deps(str(f))
        assert len(deps) == 1

    def test_file_path_requirements(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests==2.31.0\n")
        deps = find_and_parse_deps(str(f))
        assert len(deps) == 1

    def test_unsupported_file(self, tmp_path):
        f = tmp_path / "setup.cfg"
        f.write_text("")
        with pytest.raises(ValueError, match="Unsupported file"):
            find_and_parse_deps(str(f))

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_and_parse_deps(str(tmp_path / "nonexistent"))

    def test_directory_no_dep_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No pyproject.toml"):
            find_and_parse_deps(str(tmp_path))

    def test_requirements_dev_txt(self, tmp_path):
        f = tmp_path / "requirements-dev.txt"
        f.write_text("pytest==7.0.0\n")
        deps = find_and_parse_deps(str(f))
        assert len(deps) == 1


# ===== Version Checking =====


class TestParseVersionTuple:
    def test_simple(self):
        assert parse_version_tuple("1.2.3") == (1, 2, 3)

    def test_two_parts(self):
        assert parse_version_tuple("1.0") == (1, 0)

    def test_single(self):
        assert parse_version_tuple("5") == (5,)

    def test_prerelease(self):
        # Should extract leading digits
        assert parse_version_tuple("1.2.3rc1") == (1, 2, 3)

    def test_post_release(self):
        # "post1" part starts with 'p', so regex skips it
        assert parse_version_tuple("1.2.3.post1") == (1, 2, 3)

    def test_empty_fallback(self):
        # Edge case: no digits at all
        assert parse_version_tuple("abc") == (0,)


class TestClassifyUpdate:
    def test_up_to_date_same(self):
        assert classify_update("2.0.0", "2.0.0") == "up-to-date"

    def test_up_to_date_ahead(self):
        assert classify_update("3.0.0", "2.0.0") == "up-to-date"

    def test_major(self):
        assert classify_update("1.0.0", "2.0.0") == "major"

    def test_minor(self):
        assert classify_update("1.0.0", "1.1.0") == "minor"

    def test_patch(self):
        assert classify_update("1.0.0", "1.0.1") == "patch"

    def test_major_with_different_lengths(self):
        assert classify_update("1.0", "2.0.0") == "major"

    def test_minor_two_part(self):
        assert classify_update("1.0", "1.1") == "minor"

    def test_patch_when_minor_same(self):
        assert classify_update("1.2.0", "1.2.1") == "patch"


class TestCheckVersion:
    def test_with_mock_pypi(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "2.31.0",
        )
        dep = Dependency(name="requests", specified_version="2.28.0", constraint="==")
        vi = check_version(dep)
        assert vi.latest == "2.31.0"
        assert vi.current == "2.28.0"
        assert vi.update_type == "minor"

    def test_pypi_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: None,
        )
        dep = Dependency(name="nonexistent", specified_version="1.0")
        vi = check_version(dep)
        assert vi.current is None
        assert vi.latest is None

    def test_no_specified_version(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "3.0.0",
        )
        dep = Dependency(name="requests")
        vi = check_version(dep)
        assert vi.latest == "3.0.0"
        assert vi.update_type is None  # Can't compare without pinned version

    def test_up_to_date(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "2.31.0",
        )
        dep = Dependency(name="requests", specified_version="2.31.0", constraint="==")
        vi = check_version(dep)
        assert vi.update_type == "up-to-date"


# ===== OSV Vulnerability Checker =====


class TestQueryOsv:
    def test_with_mock(self, monkeypatch):
        mock_response = json.dumps({
            "vulns": [
                {
                    "id": "PYSEC-2024-001",
                    "summary": "RCE in foo",
                    "database_specific": {"severity": "HIGH"},
                    "affected": [
                        {
                            "ranges": [
                                {"events": [{"introduced": "0"}, {"fixed": "1.2.3"}]}
                            ]
                        }
                    ],
                    "references": [
                        {"type": "ADVISORY", "url": "https://example.com/advisory"}
                    ],
                }
            ]
        }).encode()

        class FakeResp:
            def read(self):
                return mock_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=10: FakeResp()
        )
        vulns = query_osv("foo", "1.0.0")
        assert len(vulns) == 1
        assert vulns[0].id == "PYSEC-2024-001"
        assert vulns[0].severity == "HIGH"
        assert vulns[0].fixed_version == "1.2.3"
        assert vulns[0].url == "https://example.com/advisory"

    def test_no_vulns(self, monkeypatch):
        mock_response = json.dumps({"vulns": []}).encode()

        class FakeResp:
            def read(self):
                return mock_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=10: FakeResp()
        )
        vulns = query_osv("safe-pkg", "1.0.0")
        assert vulns == []

    def test_network_error(self, monkeypatch):
        import urllib.error

        def raise_err(req, timeout=10):
            raise urllib.error.URLError("Connection refused")

        monkeypatch.setattr("urllib.request.urlopen", raise_err)
        vulns = query_osv("foo", "1.0.0")
        assert vulns == []

    def test_fallback_url(self, monkeypatch):
        """When no ADVISORY reference, falls back to osv.dev URL."""
        mock_response = json.dumps({
            "vulns": [
                {
                    "id": "GHSA-xyz",
                    "summary": "bad",
                    "references": [{"type": "WEB", "url": "https://example.com"}],
                }
            ]
        }).encode()

        class FakeResp:
            def read(self):
                return mock_response
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=10: FakeResp()
        )
        vulns = query_osv("foo")
        assert vulns[0].url == "https://osv.dev/vulnerability/GHSA-xyz"


# ===== Static Analysis Checks =====


class TestNormalizePackageName:
    def test_lowercase(self):
        assert normalize_package_name("Flask") == "flask"

    def test_dashes(self):
        assert normalize_package_name("my-package") == "my-package"

    def test_underscores(self):
        assert normalize_package_name("my_package") == "my-package"

    def test_dots(self):
        assert normalize_package_name("my.package") == "my-package"

    def test_mixed(self):
        assert normalize_package_name("My_Package.Name-v2") == "my-package-name-v2"

    def test_consecutive_separators(self):
        assert normalize_package_name("a__b--c..d") == "a-b-c-d"


class TestCheckUnpinned:
    def test_unpinned_detected(self):
        deps = [Dependency(name="requests")]
        findings = check_unpinned(deps)
        assert len(findings) == 1
        assert findings[0].check == "unpinned"
        assert findings[0].severity == "MODERATE"
        assert "requests" in findings[0].message

    def test_pinned_not_flagged(self):
        deps = [Dependency(name="requests", specified_version="2.31.0", constraint="==")]
        findings = check_unpinned(deps)
        assert findings == []

    def test_constraint_only(self):
        deps = [Dependency(name="requests", constraint=">=", specified_version="1.0")]
        findings = check_unpinned(deps)
        assert findings == []

    def test_multiple_mixed(self):
        deps = [
            Dependency(name="pinned", specified_version="1.0", constraint="=="),
            Dependency(name="unpinned1"),
            Dependency(name="unpinned2"),
        ]
        findings = check_unpinned(deps)
        assert len(findings) == 2


class TestCheckWildcard:
    def test_greater_than_or_equal(self):
        deps = [Dependency(name="foo", constraint=">=", specified_version="1.0")]
        findings = check_wildcard(deps)
        assert len(findings) == 1
        assert findings[0].check == "wildcard"
        assert findings[0].severity == "LOW"

    def test_greater_than(self):
        deps = [Dependency(name="foo", constraint=">", specified_version="1.0")]
        findings = check_wildcard(deps)
        assert len(findings) == 1

    def test_not_equal(self):
        deps = [Dependency(name="foo", constraint="!=", specified_version="1.0")]
        findings = check_wildcard(deps)
        assert len(findings) == 1

    def test_wildcard_version(self):
        deps = [Dependency(name="foo", constraint="==", specified_version="1.*")]
        findings = check_wildcard(deps)
        assert len(findings) == 1
        assert findings[0].severity == "MODERATE"

    def test_exact_pin_not_flagged(self):
        deps = [Dependency(name="foo", constraint="==", specified_version="1.0.0")]
        findings = check_wildcard(deps)
        assert findings == []

    def test_compatible_release_not_flagged(self):
        deps = [Dependency(name="foo", constraint="~=", specified_version="1.0")]
        findings = check_wildcard(deps)
        assert findings == []


class TestCheckDuplicates:
    def test_duplicate_across_files(self):
        deps = [
            Dependency(name="requests", source_file="requirements.txt"),
            Dependency(name="requests", source_file="pyproject.toml"),
        ]
        findings = check_duplicates(deps)
        assert len(findings) == 1
        assert findings[0].check == "duplicate"
        assert findings[0].severity == "MODERATE"

    def test_no_duplicate_same_file(self):
        deps = [
            Dependency(name="requests", source_file="requirements.txt"),
            Dependency(name="requests", source_file="requirements.txt"),
        ]
        findings = check_duplicates(deps)
        assert findings == []  # Same file doesn't count

    def test_normalized_names(self):
        deps = [
            Dependency(name="my-package", source_file="a.txt"),
            Dependency(name="my_package", source_file="b.txt"),
        ]
        findings = check_duplicates(deps)
        assert len(findings) == 1

    def test_no_duplicates(self):
        deps = [
            Dependency(name="requests", source_file="a.txt"),
            Dependency(name="flask", source_file="b.txt"),
        ]
        findings = check_duplicates(deps)
        assert findings == []


class TestCheckConflicts:
    def test_conflicting_versions(self):
        deps = [
            Dependency(name="requests", specified_version="2.28.0", constraint="=="),
            Dependency(name="requests", specified_version="2.31.0", constraint="=="),
        ]
        findings = check_conflicts(deps)
        assert len(findings) == 1
        assert findings[0].check == "conflict"
        assert findings[0].severity == "HIGH"

    def test_same_version_no_conflict(self):
        deps = [
            Dependency(name="requests", specified_version="2.31.0", constraint="=="),
            Dependency(name="requests", specified_version="2.31.0", constraint="=="),
        ]
        findings = check_conflicts(deps)
        assert findings == []

    def test_non_pinned_not_checked(self):
        deps = [
            Dependency(name="requests", specified_version="2.28.0", constraint=">="),
            Dependency(name="requests", specified_version="2.31.0", constraint=">="),
        ]
        findings = check_conflicts(deps)
        assert findings == []  # Only == pins are checked

    def test_normalized_names(self):
        deps = [
            Dependency(name="my-pkg", specified_version="1.0", constraint="=="),
            Dependency(name="my_pkg", specified_version="2.0", constraint="=="),
        ]
        findings = check_conflicts(deps)
        assert len(findings) == 1


class TestCheckHygiene:
    def test_dev_dep_in_main_requirements(self):
        deps = [
            Dependency(name="pytest", source_file="requirements.txt"),
        ]
        findings = check_hygiene(deps)
        assert len(findings) == 1
        assert findings[0].check == "hygiene"
        assert findings[0].severity == "INFO"

    def test_dev_dep_in_dev_requirements(self):
        deps = [
            Dependency(name="pytest", source_file="requirements-dev.txt"),
        ]
        findings = check_hygiene(deps)
        assert findings == []

    def test_dev_dep_in_test_requirements(self):
        deps = [
            Dependency(name="pytest", source_file="test-requirements.txt"),
        ]
        findings = check_hygiene(deps)
        assert findings == []

    def test_non_dev_dep_not_flagged(self):
        deps = [
            Dependency(name="requests", source_file="requirements.txt"),
        ]
        findings = check_hygiene(deps)
        assert findings == []

    def test_multiple_dev_deps(self):
        deps = [
            Dependency(name="pytest", source_file="requirements.txt"),
            Dependency(name="mypy", source_file="requirements.txt"),
            Dependency(name="black", source_file="requirements.txt"),
        ]
        findings = check_hygiene(deps)
        assert len(findings) == 3

    def test_normalized_name_check(self):
        deps = [
            Dependency(name="pytest-cov", source_file="requirements.txt"),
        ]
        findings = check_hygiene(deps)
        assert len(findings) == 1


class TestRunStaticChecks:
    def test_combines_all_checks(self):
        deps = [
            Dependency(name="unpinned_pkg"),
            Dependency(name="broad_pkg", constraint=">=", specified_version="1.0"),
            Dependency(name="dup_pkg", source_file="a.txt"),
            Dependency(name="dup_pkg", source_file="b.txt"),
            Dependency(name="foo", specified_version="1.0", constraint="=="),
            Dependency(name="foo", specified_version="2.0", constraint="=="),
            Dependency(name="pytest", source_file="requirements.txt"),
        ]
        findings = run_static_checks(deps)
        checks = {f.check for f in findings}
        assert "unpinned" in checks
        assert "wildcard" in checks
        assert "duplicate" in checks
        assert "conflict" in checks
        assert "hygiene" in checks

    def test_empty_deps(self):
        findings = run_static_checks([])
        assert findings == []


# ===== Audit Orchestrator =====


class TestAuditDependencies:
    def test_full_audit(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "2.31.0",
        )
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_osv",
            lambda name, version=None, timeout=10: [],
        )
        deps = [Dependency(name="requests", specified_version="2.28.0", constraint="==")]
        results = audit_dependencies(deps, timeout=5)
        assert len(results) == 1
        assert results[0].version_info.latest == "2.31.0"
        assert results[0].vulnerabilities == []

    def test_skip_versions(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: called.append("pypi") or "2.0",
        )
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_osv",
            lambda name, version=None, timeout=10: [],
        )
        deps = [Dependency(name="foo", specified_version="1.0")]
        results = audit_dependencies(deps, check_versions=False, timeout=5)
        assert "pypi" not in called
        assert results[0].version_info.current is None

    def test_skip_vulns(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "2.0",
        )
        called = []
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_osv",
            lambda name, version=None, timeout=10: called.append("osv") or [],
        )
        deps = [Dependency(name="foo", specified_version="1.0")]
        results = audit_dependencies(deps, check_vulns=False, timeout=5)
        assert "osv" not in called

    def test_multiple_deps(self, monkeypatch):
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "1.0",
        )
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_osv",
            lambda name, version=None, timeout=10: [],
        )
        deps = [
            Dependency(name="a", specified_version="1.0"),
            Dependency(name="b", specified_version="1.0"),
            Dependency(name="c", specified_version="1.0"),
        ]
        results = audit_dependencies(deps, timeout=5)
        assert len(results) == 3


# ===== Severity Filter =====


class TestSeverityPassesFilter:
    def test_critical_passes_low(self):
        assert severity_passes_filter("CRITICAL", "LOW") is True

    def test_low_fails_high(self):
        assert severity_passes_filter("LOW", "HIGH") is False

    def test_same_passes(self):
        assert severity_passes_filter("MODERATE", "MODERATE") is True

    def test_unknown_severity(self):
        assert severity_passes_filter("UNKNOWN", "LOW") is False

    def test_none_always_fails(self):
        assert severity_passes_filter("NONE", "LOW") is False


# ===== Output Formatters =====


class TestFormatTerminal:
    def _make_result(self, name, version="1.0", latest="2.0", update_type="major",
                     vulns=None):
        dep = Dependency(name=name, specified_version=version, constraint="==")
        vi = VersionInfo(current=version, latest=latest, update_type=update_type)
        return AuditResult(dependency=dep, version_info=vi,
                           vulnerabilities=vulns or [])

    def test_header(self):
        output = format_terminal([], min_severity="LOW")
        assert "ai-dep-audit" in output
        assert __version__ in output

    def test_summary_line(self):
        results = [self._make_result("foo")]
        output = format_terminal(results)
        assert "1 dependencies" in output or "Scanned 1" in output
        assert "1 outdated" in output

    def test_outdated_section(self):
        results = [self._make_result("foo", "1.0", "2.0", "major")]
        output = format_terminal(results)
        assert "OUTDATED PACKAGES" in output
        assert "foo" in output
        assert "MAJOR" in output

    def test_vulnerability_section(self):
        vuln = Vulnerability(id="CVE-1", summary="bad", severity="HIGH")
        results = [self._make_result("foo", vulns=[vuln])]
        output = format_terminal(results)
        assert "VULNERABILITIES" in output
        assert "CVE-1" in output

    def test_severity_filter(self):
        vuln = Vulnerability(id="CVE-1", summary="minor", severity="LOW")
        results = [self._make_result("foo", vulns=[vuln])]
        output = format_terminal(results, min_severity="HIGH")
        assert "CVE-1" not in output

    def test_static_findings(self):
        sf = [StaticFinding(
            check="unpinned", severity="MODERATE", package="bar",
            message="bar has no version",
        )]
        output = format_terminal([], static_findings=sf)
        assert "STATIC ANALYSIS" in output
        assert "bar" in output

    def test_up_to_date_section(self):
        results = [self._make_result("foo", "2.0", "2.0", "up-to-date")]
        output = format_terminal(results)
        assert "up-to-date" in output

    def test_fix_version_shown(self):
        vuln = Vulnerability(
            id="CVE-1", summary="bad", severity="HIGH", fixed_version="1.2.3"
        )
        results = [self._make_result("foo", vulns=[vuln])]
        output = format_terminal(results)
        assert "1.2.3" in output


class TestFormatJson:
    def test_basic_structure(self):
        dep = Dependency(name="foo", specified_version="1.0", constraint="==")
        vi = VersionInfo(current="1.0", latest="2.0", update_type="major")
        results = [AuditResult(dependency=dep, version_info=vi)]
        output = json.loads(format_json(results))
        assert output["version"] == __version__
        assert output["summary"]["total"] == 1
        assert output["summary"]["outdated"] == 1
        assert output["summary"]["vulnerable"] == 0
        assert len(output["results"]) == 1
        assert output["results"][0]["name"] == "foo"

    def test_with_static_findings(self):
        results = []
        sf = [StaticFinding(
            check="unpinned", severity="MODERATE", package="bar", message="msg"
        )]
        output = json.loads(format_json(results, static_findings=sf))
        assert output["summary"]["static_issues"] == 1
        assert len(output["static_findings"]) == 1

    def test_with_vulnerabilities(self):
        dep = Dependency(name="foo", specified_version="1.0")
        vuln = Vulnerability(id="CVE-1", summary="bad", severity="HIGH")
        results = [AuditResult(dependency=dep, vulnerabilities=[vuln])]
        output = json.loads(format_json(results))
        assert output["summary"]["vulnerable"] == 1
        assert output["results"][0]["vulnerabilities"][0]["id"] == "CVE-1"

    def test_empty_results(self):
        output = json.loads(format_json([]))
        assert output["summary"]["total"] == 0


# ===== Scoring & Classification =====


class TestComputeDepHealth:
    def test_empty_deps(self):
        score = compute_dep_health([], [])
        assert score.score == 100
        assert score.grade == "A"
        assert score.profile == "clean"

    def test_clean_deps(self):
        dep = Dependency(name="foo", specified_version="1.0")
        results = [
            AuditResult(
                dependency=dep,
                version_info=VersionInfo(current="1.0", latest="1.0", update_type="up-to-date"),
            )
        ]
        score = compute_dep_health(results, [])
        assert score.score == 100
        assert score.grade == "A"
        assert score.profile == "clean"

    def test_vulnerable_dep(self):
        dep = Dependency(name="foo", specified_version="1.0")
        results = [
            AuditResult(
                dependency=dep,
                vulnerabilities=[
                    Vulnerability(id="V1", summary="bad", severity="CRITICAL"),
                ],
            )
        ]
        score = compute_dep_health(results, [])
        assert score.score < 100
        assert score.vulnerable_count == 1
        assert score.profile == "critical_risk"

    def test_outdated_dep(self):
        dep = Dependency(name="foo", specified_version="1.0")
        results = [
            AuditResult(
                dependency=dep,
                version_info=VersionInfo(current="1.0", latest="2.0", update_type="major"),
            )
        ]
        score = compute_dep_health(results, [])
        assert score.outdated_count == 1
        assert score.outdated_penalty > 0

    def test_static_findings_penalty(self):
        dep = Dependency(name="foo", specified_version="1.0")
        results = [AuditResult(dependency=dep)]
        static = [
            StaticFinding(check="unpinned", severity="MODERATE", package="foo", message="msg"),
        ]
        score = compute_dep_health(results, static)
        assert score.static_issue_count == 1
        assert score.static_penalty > 0

    def test_grade_assignment(self):
        # Verify grade thresholds
        dep = Dependency(name="foo", specified_version="1.0")
        results = [AuditResult(dependency=dep)]
        score = compute_dep_health(results, [])
        assert score.grade in ("A", "B", "C", "D", "F")

    def test_severity_counts(self):
        dep = Dependency(name="foo", specified_version="1.0")
        results = [
            AuditResult(
                dependency=dep,
                vulnerabilities=[
                    Vulnerability(id="V1", summary="a", severity="HIGH"),
                    Vulnerability(id="V2", summary="b", severity="HIGH"),
                    Vulnerability(id="V3", summary="c", severity="LOW"),
                ],
            )
        ]
        score = compute_dep_health(results, [])
        assert score.severity_counts.get("HIGH") == 2
        assert score.severity_counts.get("LOW") == 1

    def test_density_normalization(self):
        """More deps = lower per-dep penalty."""
        dep = Dependency(name="foo", specified_version="1.0")
        vuln = Vulnerability(id="V1", summary="bad", severity="HIGH")
        result_vuln = AuditResult(dependency=dep, vulnerabilities=[vuln])
        # Single dep with vuln
        score1 = compute_dep_health([result_vuln], [])
        # Same vuln but with 10 clean deps alongside
        clean = AuditResult(
            dependency=Dependency(name="clean"),
            version_info=VersionInfo(update_type="up-to-date"),
        )
        score10 = compute_dep_health([result_vuln] + [clean] * 9, [])
        assert score10.score > score1.score


class TestClassifyDepProfile:
    def _make_results(self, vulns=None, outdated=False):
        dep = Dependency(name="pkg", specified_version="1.0")
        vi = VersionInfo(
            current="1.0",
            latest="2.0" if outdated else "1.0",
            update_type="major" if outdated else "up-to-date",
        )
        return [AuditResult(dependency=dep, version_info=vi,
                            vulnerabilities=vulns or [])]

    def test_clean(self):
        results = self._make_results()
        profile, detail = classify_dep_profile(results, [], 0, 0, 0)
        assert profile == "clean"

    def test_critical_risk(self):
        vuln = Vulnerability(id="V1", summary="crit", severity="CRITICAL")
        results = self._make_results(vulns=[vuln])
        profile, detail = classify_dep_profile(results, [], 25, 0, 0)
        assert profile == "critical_risk"
        assert "1" in detail

    def test_vulnerability_heavy(self):
        vuln = Vulnerability(id="V1", summary="high", severity="HIGH")
        results = self._make_results(vulns=[vuln])
        profile, detail = classify_dep_profile(results, [], 15, 2, 1)
        assert profile == "vulnerability_heavy"

    def test_outdated_heavy(self):
        results = self._make_results(outdated=True)
        profile, detail = classify_dep_profile(results, [], 0, 10, 2)
        assert profile == "outdated_heavy"

    def test_hygiene_issues(self):
        sf = [StaticFinding(check="unpinned", severity="HIGH", package="x", message="m")]
        results = self._make_results()
        profile, detail = classify_dep_profile(results, sf, 0, 1, 10)
        assert profile == "hygiene_issues"

    def test_mixed(self):
        vuln = Vulnerability(id="V1", summary="high", severity="HIGH")
        results = self._make_results(vulns=[vuln], outdated=True)
        sf = [StaticFinding(check="unpinned", severity="MODERATE", package="x", message="m")]
        # Equal-ish penalties so no single dominates
        profile, detail = classify_dep_profile(results, sf, 5, 5, 5)
        assert profile == "mixed"


class TestFormatScoreText:
    def test_basic(self):
        score = DepHealthScore(
            score=85, grade="B", total_deps=10, vulnerable_count=1,
            outdated_count=2, static_issue_count=1, vuln_penalty=15.0,
            outdated_penalty=7.0, static_penalty=5.0,
            severity_counts={"HIGH": 1}, profile="vulnerability_heavy",
            profile_detail="1 vulns dominate",
        )
        output = format_score_text(score)
        assert "85/100" in output
        assert "(B)" in output
        assert "vulnerability_heavy" in output
        assert "HIGH=1" in output

    def test_clean_score(self):
        score = DepHealthScore(
            score=100, grade="A", total_deps=5, vulnerable_count=0,
            outdated_count=0, static_issue_count=0, vuln_penalty=0,
            outdated_penalty=0, static_penalty=0,
            severity_counts={}, profile="clean",
            profile_detail="No issues",
        )
        output = format_score_text(score)
        assert "100/100" in output
        assert "(A)" in output


class TestFormatScoreJson:
    def test_valid_json(self):
        score = DepHealthScore(
            score=90, grade="A", total_deps=3, vulnerable_count=0,
            outdated_count=0, static_issue_count=0, vuln_penalty=0,
            outdated_penalty=0, static_penalty=0,
            severity_counts={}, profile="clean", profile_detail="No issues",
        )
        parsed = json.loads(format_score_json(score))
        assert parsed["score"] == 90
        assert parsed["grade"] == "A"
        assert parsed["profile"] == "clean"


# ===== CLI =====


class TestBuildParser:
    def test_default_target(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.target == "."

    def test_custom_target(self):
        parser = build_parser()
        args = parser.parse_args(["/some/path"])
        assert args.target == "/some/path"

    def test_file_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-f", "requirements.txt"])
        assert args.file == "requirements.txt"

    def test_json_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--json"])
        assert args.json_output is True

    def test_severity_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--severity", "high"])
        assert args.severity == "high"

    def test_offline_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--offline"])
        assert args.offline is True

    def test_no_version_check(self):
        parser = build_parser()
        args = parser.parse_args(["--no-version-check"])
        assert args.no_version_check is True

    def test_no_vuln_check(self):
        parser = build_parser()
        args = parser.parse_args(["--no-vuln-check"])
        assert args.no_vuln_check is True

    def test_timeout(self):
        parser = build_parser()
        args = parser.parse_args(["--timeout", "10"])
        assert args.timeout == 10

    def test_score_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--score"])
        assert args.score is True


class TestMain:
    def test_no_deps_found(self, tmp_path):
        """Empty project directory."""
        rc = main([str(tmp_path)])
        assert rc == 1  # FileNotFoundError path

    def test_empty_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("")
        rc = main([str(tmp_path)])
        assert rc == 0  # No deps = clean exit

    def test_offline_mode(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        rc = main([str(tmp_path), "--offline"])
        assert rc == 0

    def test_json_output(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        rc = main([str(tmp_path), "--offline", "--json"])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["summary"]["total"] == 1
        assert rc == 0

    def test_score_mode(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        rc = main([str(tmp_path), "--offline", "--score"])
        captured = capsys.readouterr()
        assert "Health Score" in captured.out
        assert rc == 0

    def test_score_json_mode(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        rc = main([str(tmp_path), "--offline", "--score", "--json"])
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "score" in output
        assert rc == 0

    def test_file_flag(self, tmp_path):
        f = tmp_path / "requirements.txt"
        f.write_text("requests==2.31.0\n")
        rc = main(["-f", str(f), "--offline"])
        assert rc == 0

    def test_exit_code_2_for_vulns(self, tmp_path, monkeypatch):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n")
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_pypi",
            lambda name, timeout=5: "2.31.0",
        )
        monkeypatch.setattr(
            "code_health_suite.engines.dep_audit.query_osv",
            lambda name, version=None, timeout=10: [
                Vulnerability(id="CVE-1", summary="bad", severity="HIGH")
            ],
        )
        rc = main([str(tmp_path)])
        assert rc == 2

    def test_exit_code_1_for_high_static(self, tmp_path):
        """Conflicting versions produce HIGH static finding."""
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["foo==1.0", "foo==2.0"]\n'
        )
        rc = main([str(tmp_path), "--offline"])
        assert rc == 1

    def test_severity_filter(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text("requests\n")
        rc = main([str(tmp_path), "--offline", "--severity", "high"])
        captured = capsys.readouterr()
        # MODERATE unpinned finding should be filtered out
        assert "unpinned" not in captured.out.lower() or "MODERATE" not in captured.out


# ===== Integration Tests =====


class TestIntegration:
    def test_full_offline_pipeline(self, tmp_path):
        """End-to-end: parse → static checks → format."""
        (tmp_path / "requirements.txt").write_text(textwrap.dedent("""\
            requests==2.31.0
            flask>=3.0.0
            unpinned-pkg
            pytest==7.4.0
        """))
        deps = find_and_parse_deps(str(tmp_path))
        assert len(deps) == 4

        static = run_static_checks(deps)
        assert len(static) >= 2  # unpinned + wildcard + hygiene

        results = audit_dependencies(deps, check_versions=False, check_vulns=False)
        assert len(results) == 4

        text = format_terminal(results, static_findings=static)
        assert "STATIC ANALYSIS" in text

        health = compute_dep_health(results, static)
        assert 0 <= health.score <= 100
        assert health.grade in ("A", "B", "C", "D", "F")

    def test_pyproject_pipeline(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(textwrap.dedent("""\
            [project]
            name = "test"
            dependencies = [
                "click==8.1.7",
                "rich~=13.0",
            ]
        """))
        deps = find_and_parse_deps(str(tmp_path))
        assert len(deps) == 2
        assert deps[0].name == "click"
        assert deps[1].name == "rich"

        static = run_static_checks(deps)
        results = audit_dependencies(deps, check_versions=False, check_vulns=False)

        json_out = json.loads(format_json(results, static_findings=static))
        assert json_out["summary"]["total"] == 2

    def test_dual_file_duplicate_detection(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["requests==2.31.0"]\n'
        )
        (tmp_path / "requirements.txt").write_text("requests==2.28.0\n")
        deps = find_and_parse_deps(str(tmp_path))
        static = run_static_checks(deps)
        dup_findings = [f for f in static if f.check == "duplicate"]
        conflict_findings = [f for f in static if f.check == "conflict"]
        assert len(dup_findings) == 1
        assert len(conflict_findings) == 1

    def test_main_cli_integration(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text(
            "requests==2.31.0\nflask\n"
        )
        rc = main([str(tmp_path), "--offline"])
        captured = capsys.readouterr()
        assert "ai-dep-audit" in captured.out
        assert rc == 0

    def test_score_pipeline(self, tmp_path, capsys):
        (tmp_path / "requirements.txt").write_text(
            "requests==2.31.0\nflask>=3.0\npytest==7.4.0\n"
        )
        rc = main([str(tmp_path), "--offline", "--score"])
        captured = capsys.readouterr()
        assert "Health Score" in captured.out
        assert "/100" in captured.out
        assert rc == 0

    def test_empty_project_error(self, tmp_path, capsys):
        rc = main([str(tmp_path)])
        captured = capsys.readouterr()
        assert "Error" in captured.err or rc == 1

    def test_nonexistent_path(self, capsys):
        rc = main(["/nonexistent/path/to/project"])
        assert rc == 1
