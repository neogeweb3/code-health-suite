"""Tests for the todo_scanner engine — technical debt comment tracker."""
import os
import subprocess
import textwrap
from unittest.mock import patch, MagicMock

import pytest

from code_health_suite.engines import todo_scanner


# --- Helper ---

def _write_file(tmp_path, code: str, name: str = "sample.py") -> str:
    """Write a source file and return its path."""
    filepath = tmp_path / name
    filepath.write_text(textwrap.dedent(code))
    return str(filepath)


# === TAG_SEVERITY mapping ===

class TestTagSeverity:
    def test_fixme_is_high(self):
        assert todo_scanner.TAG_SEVERITY["FIXME"] == "high"

    def test_hack_is_high(self):
        assert todo_scanner.TAG_SEVERITY["HACK"] == "high"

    def test_bug_is_high(self):
        assert todo_scanner.TAG_SEVERITY["BUG"] == "high"

    def test_todo_is_medium(self):
        assert todo_scanner.TAG_SEVERITY["TODO"] == "medium"

    def test_xxx_is_medium(self):
        assert todo_scanner.TAG_SEVERITY["XXX"] == "medium"

    def test_optimize_is_medium(self):
        assert todo_scanner.TAG_SEVERITY["OPTIMIZE"] == "medium"

    def test_refactor_is_medium(self):
        assert todo_scanner.TAG_SEVERITY["REFACTOR"] == "medium"

    def test_note_is_low(self):
        assert todo_scanner.TAG_SEVERITY["NOTE"] == "low"

    def test_noqa_is_low(self):
        assert todo_scanner.TAG_SEVERITY["NOQA"] == "low"

    def test_changed_is_low(self):
        assert todo_scanner.TAG_SEVERITY["CHANGED"] == "low"


# === Regex pattern matching ===

class TestTagPattern:
    """Test _TAG_PATTERN regex on individual lines."""

    def test_simple_todo(self):
        m = todo_scanner._TAG_PATTERN.search("# TODO: fix this bug")
        assert m is not None
        assert m.group("tag").upper() == "TODO"
        assert m.group("message").strip() == "fix this bug"

    def test_fixme_no_colon(self):
        m = todo_scanner._TAG_PATTERN.search("# FIXME remove hardcoded value")
        assert m is not None
        assert m.group("tag").upper() == "FIXME"

    def test_hack_with_author(self):
        m = todo_scanner._TAG_PATTERN.search("# HACK(john): workaround for API")
        assert m is not None
        assert m.group("tag").upper() == "HACK"
        assert m.group("inline_author") == "john"
        assert "workaround" in m.group("message")

    def test_xxx_marker(self):
        m = todo_scanner._TAG_PATTERN.search("# XXX: dangerous code path")
        assert m is not None
        assert m.group("tag").upper() == "XXX"

    def test_note_tag(self):
        m = todo_scanner._TAG_PATTERN.search("# NOTE: see RFC 1234 for details")
        assert m is not None
        assert m.group("tag").upper() == "NOTE"

    def test_noqa(self):
        m = todo_scanner._TAG_PATTERN.search("import os  # noqa: E402")
        assert m is not None
        assert m.group("tag").upper() == "NOQA"

    def test_js_style_comment(self):
        m = todo_scanner._TAG_PATTERN.search("// TODO: implement feature")
        assert m is not None
        assert m.group("tag").upper() == "TODO"

    def test_c_style_comment(self):
        m = todo_scanner._TAG_PATTERN.search("/* FIXME: memory leak */")
        assert m is not None
        assert m.group("tag").upper() == "FIXME"

    def test_star_prefix(self):
        m = todo_scanner._TAG_PATTERN.search(" * TODO: add validation")
        assert m is not None
        assert m.group("tag").upper() == "TODO"

    def test_case_insensitive(self):
        m = todo_scanner._TAG_PATTERN.search("# todo: lowercase")
        assert m is not None
        assert m.group("tag").upper() == "TODO"

    def test_no_match_in_string(self):
        # Plain text without comment prefix should not match
        m = todo_scanner._TAG_PATTERN.search('message = "TODO: fix this"')
        assert m is None

    def test_no_match_plain_code(self):
        m = todo_scanner._TAG_PATTERN.search("x = 42")
        assert m is None

    def test_optimize_tag(self):
        m = todo_scanner._TAG_PATTERN.search("# OPTIMIZE: use vectorized ops")
        assert m is not None
        assert m.group("tag").upper() == "OPTIMIZE"

    def test_refactor_tag(self):
        m = todo_scanner._TAG_PATTERN.search("# REFACTOR: extract method")
        assert m is not None
        assert m.group("tag").upper() == "REFACTOR"

    def test_bug_tag(self):
        m = todo_scanner._TAG_PATTERN.search("# BUG: off by one")
        assert m is not None
        assert m.group("tag").upper() == "BUG"

    def test_semicolon_comment(self):
        m = todo_scanner._TAG_PATTERN.search("; TODO: lisp-style")
        assert m is not None
        assert m.group("tag").upper() == "TODO"

    def test_tag_without_message(self):
        m = todo_scanner._TAG_PATTERN.search("# TODO")
        assert m is not None
        assert m.group("tag").upper() == "TODO"
        assert m.group("message") is None or m.group("message").strip() == ""


# === File analysis ===

class TestAnalyzeFile:
    def test_empty_file(self, tmp_path):
        path = _write_file(tmp_path, "", "empty.py")
        result = todo_scanner.analyze_file(path)
        assert result.lines_scanned == 0
        assert len(result.items) == 0
        assert result.error == ""

    def test_no_todos(self, tmp_path):
        path = _write_file(tmp_path, """\
            def hello():
                return "world"
        """)
        result = todo_scanner.analyze_file(path)
        assert result.lines_scanned > 0
        assert len(result.items) == 0

    def test_single_todo(self, tmp_path):
        path = _write_file(tmp_path, """\
            # TODO: add error handling
            def hello():
                return "world"
        """)
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1
        item = result.items[0]
        assert item.tag == "TODO"
        assert item.severity == "medium"
        assert "error handling" in item.message
        assert item.line_number == 1

    def test_multiple_tags(self, tmp_path):
        path = _write_file(tmp_path, """\
            # TODO: first task
            def hello():
                # FIXME: broken logic
                return "world"
            # HACK: temp workaround
        """)
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 3
        tags = {item.tag for item in result.items}
        assert tags == {"TODO", "FIXME", "HACK"}

    def test_inline_author(self, tmp_path):
        path = _write_file(tmp_path, """\
            # TODO(alice): review this section
        """)
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1
        assert result.items[0].inline_author == "alice"

    def test_severity_assignment(self, tmp_path):
        path = _write_file(tmp_path, """\
            # FIXME: high severity
            # TODO: medium severity
            # NOTE: low severity
        """)
        result = todo_scanner.analyze_file(path)
        sevs = {item.tag: item.severity for item in result.items}
        assert sevs["FIXME"] == "high"
        assert sevs["TODO"] == "medium"
        assert sevs["NOTE"] == "low"

    def test_nonexistent_file(self):
        result = todo_scanner.analyze_file("/nonexistent/file.py")
        assert result.error != ""
        assert len(result.items) == 0

    def test_file_path_in_items(self, tmp_path):
        path = _write_file(tmp_path, "# TODO: test\n")
        result = todo_scanner.analyze_file(path)
        assert result.items[0].file_path == path

    def test_line_numbers_accurate(self, tmp_path):
        path = _write_file(tmp_path, """\
            line 1
            line 2
            # TODO: on line 3
            line 4
            # FIXME: on line 5
        """)
        result = todo_scanner.analyze_file(path)
        lines = sorted(item.line_number for item in result.items)
        assert lines == [3, 5]


# === File discovery ===

class TestFindSourceFiles:
    def test_single_python_file(self, tmp_path):
        _write_file(tmp_path, "pass", "test.py")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("test.py")

    def test_skips_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        _write_file(tmp_path, "pass", ".git/config.py")
        _write_file(tmp_path, "pass", "main.py")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 1

    def test_skips_venv(self, tmp_path):
        venv = tmp_path / "venv"
        venv.mkdir()
        (venv / "lib.py").write_text("pass")
        _write_file(tmp_path, "pass", "main.py")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 1

    def test_skips_egg_info(self, tmp_path):
        egg = tmp_path / "pkg.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO.py").write_text("pass")
        _write_file(tmp_path, "pass", "main.py")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 1

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "index.js").write_text("pass")
        _write_file(tmp_path, "pass", "main.py")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 1

    def test_finds_multiple_extensions(self, tmp_path):
        _write_file(tmp_path, "pass", "main.py")
        _write_file(tmp_path, "pass", "app.js")
        _write_file(tmp_path, "pass", "lib.ts")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 3

    def test_single_file_input(self, tmp_path):
        path = _write_file(tmp_path, "pass", "main.py")
        files = todo_scanner.find_source_files(path)
        assert len(files) == 1
        assert files[0] == path

    def test_non_source_file_ignored(self, tmp_path):
        _write_file(tmp_path, "data", "readme.txt")
        _write_file(tmp_path, "data", "image.png")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 0

    def test_custom_extensions(self, tmp_path):
        _write_file(tmp_path, "pass", "main.py")
        _write_file(tmp_path, "pass", "data.txt")
        files = todo_scanner.find_source_files(str(tmp_path), extensions={".txt"})
        assert len(files) == 1
        assert files[0].endswith("data.txt")

    def test_nested_directories(self, tmp_path):
        sub = tmp_path / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "module.py").write_text("# TODO: nested\n")
        files = todo_scanner.find_source_files(str(tmp_path))
        assert len(files) == 1


# === Aggregate scan ===

class TestScan:
    def test_empty_directory(self, tmp_path):
        result = todo_scanner.scan(str(tmp_path))
        assert result.files_scanned == 0
        assert result.total_items == 0
        assert result.total_lines == 0

    def test_counts_by_tag(self, tmp_path):
        _write_file(tmp_path, """\
            # TODO: first
            # TODO: second
            # FIXME: third
        """)
        result = todo_scanner.scan(str(tmp_path))
        assert result.by_tag["TODO"] == 2
        assert result.by_tag["FIXME"] == 1

    def test_counts_by_severity(self, tmp_path):
        _write_file(tmp_path, """\
            # FIXME: high
            # HACK: also high
            # TODO: medium
            # NOTE: low
        """)
        result = todo_scanner.scan(str(tmp_path))
        assert result.by_severity["high"] == 2
        assert result.by_severity["medium"] == 1
        assert result.by_severity["low"] == 1

    def test_by_file_tracking(self, tmp_path):
        _write_file(tmp_path, "# TODO: a\n# FIXME: b\n", "a.py")
        _write_file(tmp_path, "# TODO: c\n", "b.py")
        result = todo_scanner.scan(str(tmp_path))
        assert len(result.by_file) == 2
        assert result.total_items == 3

    def test_errors_tracked(self, tmp_path):
        # Create a file that can't be read
        bad = tmp_path / "bad.py"
        bad.mkdir()  # Directory, not file — will cause error
        _write_file(tmp_path, "# TODO: ok\n", "good.py")
        # scan should handle the error gracefully
        result = todo_scanner.scan(str(tmp_path))
        assert result.files_scanned >= 1

    def test_total_lines(self, tmp_path):
        _write_file(tmp_path, "line1\nline2\nline3\n", "a.py")
        _write_file(tmp_path, "line1\nline2\n", "b.py")
        result = todo_scanner.scan(str(tmp_path))
        assert result.total_lines == 5

    def test_scan_single_file(self, tmp_path):
        path = _write_file(tmp_path, "# TODO: single file test\n")
        result = todo_scanner.scan(path)
        assert result.files_scanned == 1
        assert result.total_items == 1


# === Score computation ===

class TestComputeScore:
    def test_perfect_score_no_items(self, tmp_path):
        result = todo_scanner.ScanResult(root=str(tmp_path))
        score = todo_scanner.compute_score(result)
        assert score.score == 100
        assert score.grade == "A"

    def test_perfect_score_clean_code(self, tmp_path):
        _write_file(tmp_path, "x = 1\ny = 2\nz = 3\n" * 100)
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        assert score.score == 100
        assert score.grade == "A"

    def test_density_calculation(self, tmp_path):
        # 1 TODO in 100 lines = 10 per 1K lines
        lines = "x = 1\n" * 99 + "# TODO: one item\n"
        _write_file(tmp_path, lines)
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        assert score.density == 10.0

    def test_weighted_debt_high_severity(self, tmp_path):
        _write_file(tmp_path, "# FIXME: critical\n" + "x = 1\n" * 99)
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        # FIXME = high = weight 3, so weighted_debt = 3
        assert score.weighted_debt == 3

    def test_weighted_debt_mixed(self, tmp_path):
        _write_file(tmp_path, """\
            # FIXME: high (3)
            # TODO: medium (2)
            # NOTE: low (1)
        """ + "x = 1\n" * 97)
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        assert score.weighted_debt == 6  # 3 + 2 + 1

    def test_grade_a(self):
        assert todo_scanner._score_to_grade(95) == "A"

    def test_grade_b(self):
        assert todo_scanner._score_to_grade(80) == "B"

    def test_grade_c(self):
        assert todo_scanner._score_to_grade(65) == "C"

    def test_grade_d(self):
        assert todo_scanner._score_to_grade(45) == "D"

    def test_grade_f(self):
        assert todo_scanner._score_to_grade(30) == "F"

    def test_hotspot_files(self, tmp_path):
        _write_file(tmp_path, "# TODO: a\n# TODO: b\n# TODO: c\n", "hot.py")
        _write_file(tmp_path, "# TODO: d\n", "cool.py")
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        assert len(score.hotspot_files) >= 1
        # Hottest file should be first
        assert score.hotspot_files[0]["count"] >= score.hotspot_files[-1]["count"]

    def test_score_decreases_with_more_debt(self, tmp_path):
        # Few items
        _write_file(tmp_path, "# TODO: one\n" + "x = 1\n" * 499, "a.py")
        result1 = todo_scanner.scan(str(tmp_path))
        score1 = todo_scanner.compute_score(result1)

        # Many items
        _write_file(tmp_path, ("# FIXME: critical\n" * 50) + "x = 1\n" * 450, "a.py")
        result2 = todo_scanner.scan(str(tmp_path))
        score2 = todo_scanner.compute_score(result2)

        assert score1.score > score2.score


# === Git blame enrichment ===

class TestBlameEnrichment:
    def test_blame_with_no_git(self, tmp_path):
        """Blame should gracefully handle non-git directories."""
        path = _write_file(tmp_path, "# TODO: test\n")
        result = todo_scanner.analyze_file(path)
        todo_scanner.enrich_with_blame(result.items)
        # Should not crash, blame fields remain empty
        assert result.items[0].blame_author == ""
        assert result.items[0].blame_date == ""

    def test_blame_timeout_handling(self, tmp_path):
        """Blame should handle subprocess timeout."""
        path = _write_file(tmp_path, "# TODO: test\n")
        result = todo_scanner.analyze_file(path)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 10)):
            todo_scanner.enrich_with_blame(result.items)

        assert result.items[0].blame_author == ""

    def test_blame_success(self, tmp_path):
        """Test successful blame parsing."""
        path = _write_file(tmp_path, "# TODO: test\n")
        result = todo_scanner.analyze_file(path)

        mock_output = MagicMock()
        mock_output.returncode = 0
        mock_output.stdout = (
            "abc123 1 1 1\n"
            "author Test User\n"
            "author-mail <test@example.com>\n"
            "author-time 1700000000\n"
            "author-tz +0000\n"
            "\t# TODO: test\n"
        )

        with patch("subprocess.run", return_value=mock_output):
            todo_scanner.enrich_with_blame(result.items)

        assert result.items[0].blame_author == "Test User"
        assert result.items[0].blame_date != ""

    def test_blame_age_days(self, tmp_path):
        """Test that blame_age_days is computed correctly."""
        import datetime

        path = _write_file(tmp_path, "# TODO: test\n")
        result = todo_scanner.analyze_file(path)

        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        ts = int(datetime.datetime.combine(yesterday, datetime.time()).timestamp())

        mock_output = MagicMock()
        mock_output.returncode = 0
        mock_output.stdout = f"abc123 1 1 1\nauthor Dev\nauthor-time {ts}\n"

        with patch("subprocess.run", return_value=mock_output):
            todo_scanner.enrich_with_blame(result.items)

        assert result.items[0].blame_age_days == 1

    def test_blame_git_failure(self, tmp_path):
        """Test blame with git returning non-zero exit."""
        path = _write_file(tmp_path, "# TODO: test\n")
        result = todo_scanner.analyze_file(path)

        mock_output = MagicMock()
        mock_output.returncode = 128

        with patch("subprocess.run", return_value=mock_output):
            todo_scanner.enrich_with_blame(result.items)

        assert result.items[0].blame_author == ""


# === Multi-language support ===

class TestMultiLanguage:
    def test_javascript_todos(self, tmp_path):
        path = _write_file(tmp_path, """\
            // TODO: add tests
            function hello() {
                // FIXME: broken
                return "world";
            }
        """, "app.js")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 2

    def test_c_style_comment(self, tmp_path):
        path = _write_file(tmp_path, """\
            /* TODO: implement this */
            int main() {
                return 0;
            }
        """, "main.c")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1
        assert result.items[0].tag == "TODO"

    def test_jsdoc_star(self, tmp_path):
        path = _write_file(tmp_path, """\
            /**
             * TODO: document params
             */
            function hello() {}
        """, "lib.ts")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1

    def test_shell_comment(self, tmp_path):
        path = _write_file(tmp_path, """\
            #!/bin/bash
            # TODO: add error handling
            echo "hello"
        """, "script.sh")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1

    def test_yaml_comment(self, tmp_path):
        path = _write_file(tmp_path, """\
            # TODO: configure properly
            key: value
        """, "config.yaml")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1


# === CLI ===

class TestCLI:
    def test_main_returns_0_for_clean(self, tmp_path):
        _write_file(tmp_path, "x = 1\n" * 10)
        ret = todo_scanner.main([str(tmp_path)])
        assert ret == 0

    def test_main_json_output(self, tmp_path, capsys):
        _write_file(tmp_path, "# TODO: test item\nx = 1\n")
        todo_scanner.main([str(tmp_path), "--json"])
        captured = capsys.readouterr()
        data = __import__("json").loads(captured.out)
        assert "items" in data
        assert "score" in data
        assert data["total_items"] >= 1

    def test_main_score_output(self, tmp_path, capsys):
        _write_file(tmp_path, "x = 1\n" * 100)
        todo_scanner.main([str(tmp_path), "--score"])
        captured = capsys.readouterr()
        data = __import__("json").loads(captured.out)
        assert "score" in data
        assert "grade" in data

    def test_main_text_output(self, tmp_path, capsys):
        _write_file(tmp_path, "# TODO: visible item\nx = 1\n")
        todo_scanner.main([str(tmp_path)])
        captured = capsys.readouterr()
        assert "TODO/FIXME Scanner" in captured.out
        assert "Score:" in captured.out

    def test_main_tag_filter(self, tmp_path, capsys):
        _write_file(tmp_path, "# TODO: keep\n# FIXME: filter out\n")
        todo_scanner.main([str(tmp_path), "--tag", "TODO", "--json"])
        captured = capsys.readouterr()
        data = __import__("json").loads(captured.out)
        for item in data["items"]:
            assert item["tag"] == "TODO"

    def test_main_severity_filter(self, tmp_path, capsys):
        _write_file(tmp_path, "# FIXME: high\n# TODO: medium\n# NOTE: low\n")
        todo_scanner.main([str(tmp_path), "--severity", "high", "--json"])
        captured = capsys.readouterr()
        data = __import__("json").loads(captured.out)
        for item in data["items"]:
            assert item["severity"] == "high"

    def test_main_returns_1_for_heavy_debt(self, tmp_path):
        # Many FIXMEs in few lines → score < 60
        code = "# FIXME: critical\n" * 50 + "x = 1\n" * 50
        _write_file(tmp_path, code)
        ret = todo_scanner.main([str(tmp_path)])
        assert ret == 1


# === Data models ===

class TestDataModels:
    def test_todo_item_defaults(self):
        item = todo_scanner.TodoItem(
            file_path="test.py", line_number=1,
            tag="TODO", severity="medium", message="test"
        )
        assert item.inline_author == ""
        assert item.blame_author == ""
        assert item.blame_date == ""
        assert item.blame_age_days == -1

    def test_scan_result_defaults(self):
        result = todo_scanner.ScanResult(root=".")
        assert result.files_scanned == 0
        assert result.total_items == 0
        assert result.by_tag == {}
        assert result.by_severity == {}

    def test_score_result_to_dict(self):
        score = todo_scanner.ScoreResult(
            score=85, grade="B", total_items=5, total_lines=1000,
            density=5.0, weighted_debt=12,
        )
        d = score.to_dict()
        assert d["score"] == 85
        assert d["grade"] == "B"
        assert d["density"] == 5.0

    def test_file_result_defaults(self):
        result = todo_scanner.FileResult(file_path="test.py")
        assert result.items == []
        assert result.lines_scanned == 0
        assert result.error == ""


# === Edge cases ===

class TestEdgeCases:
    def test_binary_file_handling(self, tmp_path):
        """Binary files should be handled gracefully."""
        path = tmp_path / "binary.py"
        path.write_bytes(b"\x00\x01\x02\x03# TODO: test\n")
        result = todo_scanner.analyze_file(str(path))
        # Should not crash, may or may not find TODO in binary content

    def test_unicode_content(self, tmp_path):
        path = _write_file(tmp_path, "# TODO: 修复中文处理\nx = '你好'\n")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1
        assert "中文" in result.items[0].message

    def test_very_long_line(self, tmp_path):
        path = _write_file(tmp_path, "# TODO: " + "x" * 10000 + "\n")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1

    def test_empty_message(self, tmp_path):
        path = _write_file(tmp_path, "# TODO\n")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1
        assert result.items[0].message == ""

    def test_multiple_tags_same_line(self, tmp_path):
        # Only first match should be captured
        path = _write_file(tmp_path, "# TODO: FIXME: both tags\n")
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 1
        assert result.items[0].tag == "TODO"

    def test_tag_in_string_not_matched(self, tmp_path):
        path = _write_file(tmp_path, """\
            message = "TODO: this is not a comment"
            x = 1
        """)
        result = todo_scanner.analyze_file(path)
        assert len(result.items) == 0

    def test_deeply_nested_directory(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("# TODO: deep file\n")
        result = todo_scanner.scan(str(tmp_path))
        assert result.total_items == 1

    def test_score_clamped_at_zero(self, tmp_path):
        """Score should never go below 0."""
        # Extreme case: all FIXME, very few lines
        code = "# FIXME: critical\n" * 100
        _write_file(tmp_path, code)
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        assert score.score >= 0

    def test_score_clamped_at_100(self):
        """Score should never exceed 100."""
        result = todo_scanner.ScanResult(root=".", total_lines=10000)
        score = todo_scanner.compute_score(result)
        assert score.score <= 100


# === Format output ===

class TestFormatText:
    def test_text_header(self, tmp_path):
        result = todo_scanner.ScanResult(root=str(tmp_path))
        score = todo_scanner.compute_score(result)
        text = todo_scanner._format_text(result, score)
        assert "TODO/FIXME Scanner" in text

    def test_text_includes_score(self, tmp_path):
        result = todo_scanner.ScanResult(root=str(tmp_path))
        score = todo_scanner.compute_score(result)
        text = todo_scanner._format_text(result, score)
        assert "Score:" in text

    def test_text_includes_items(self, tmp_path):
        _write_file(tmp_path, "# TODO: show me\n")
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        text = todo_scanner._format_text(result, score)
        assert "TODO" in text
        assert "show me" in text

    def test_text_includes_severity_breakdown(self, tmp_path):
        _write_file(tmp_path, "# FIXME: high\n# TODO: medium\n")
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        text = todo_scanner._format_text(result, score)
        assert "By severity:" in text

    def test_text_includes_tag_breakdown(self, tmp_path):
        _write_file(tmp_path, "# FIXME: a\n# TODO: b\n")
        result = todo_scanner.scan(str(tmp_path))
        score = todo_scanner.compute_score(result)
        text = todo_scanner._format_text(result, score)
        assert "By tag:" in text

    def test_text_blame_info(self, tmp_path):
        item = todo_scanner.TodoItem(
            file_path="test.py", line_number=1,
            tag="TODO", severity="medium", message="test",
            blame_author="Alice", blame_date="2025-01-01",
        )
        result = todo_scanner.ScanResult(root=str(tmp_path), items=[item], total_items=1)
        score = todo_scanner.compute_score(result)
        text = todo_scanner._format_text(result, score)
        assert "Alice" in text
        assert "2025-01-01" in text
