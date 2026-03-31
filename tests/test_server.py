"""Tests for server.py path confinement (CHS-01 fix)."""
import os
import tempfile

import pytest

from code_health_suite.server import _check_path, _init_allowed_roots, _ALLOWED_ROOTS
import code_health_suite.server as server_mod


@pytest.fixture(autouse=True)
def _reset_allowed_roots():
    """Reset _ALLOWED_ROOTS before and after each test."""
    server_mod._ALLOWED_ROOTS = []
    yield
    server_mod._ALLOWED_ROOTS = []


class TestCheckPathNoConfinement:
    """When no allowed roots are configured, backward-compatible behavior."""

    def test_existing_path_allowed(self, tmp_path):
        assert _check_path(str(tmp_path)) is None

    def test_nonexistent_path_rejected(self):
        assert _check_path("/nonexistent/path/xyz") is not None
        assert "not found" in _check_path("/nonexistent/path/xyz").lower()


class TestCheckPathWithConfinement:
    """When CODE_HEALTH_ALLOWED_ROOTS is set, paths outside roots are rejected."""

    def test_path_inside_root_allowed(self, tmp_path):
        server_mod._ALLOWED_ROOTS = [str(tmp_path)]
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        assert _check_path(str(subdir)) is None

    def test_exact_root_allowed(self, tmp_path):
        server_mod._ALLOWED_ROOTS = [str(tmp_path)]
        assert _check_path(str(tmp_path)) is None

    def test_path_outside_root_rejected(self, tmp_path):
        server_mod._ALLOWED_ROOTS = [str(tmp_path)]
        result = _check_path("/etc")
        assert result is not None
        assert "not allowed" in result.lower()

    def test_dotdot_traversal_blocked(self, tmp_path):
        """../.. traversal that escapes the allowed root is blocked."""
        server_mod._ALLOWED_ROOTS = [str(tmp_path)]
        traversal = os.path.join(str(tmp_path), "..", "..")
        # The traversal resolves outside tmp_path
        if os.path.realpath(traversal) != os.path.realpath(str(tmp_path)):
            result = _check_path(traversal)
            assert result is not None
            assert "not allowed" in result.lower()

    def test_symlink_escape_blocked(self, tmp_path):
        """Symlink pointing outside allowed root is blocked."""
        server_mod._ALLOWED_ROOTS = [str(tmp_path)]
        link = tmp_path / "escape_link"
        target = tempfile.gettempdir()
        # Only test if target is outside tmp_path
        if not os.path.realpath(target).startswith(str(tmp_path)):
            os.symlink(target, link)
            result = _check_path(str(link))
            assert result is not None
            assert "not allowed" in result.lower()

    def test_multiple_roots(self, tmp_path):
        """Multiple allowed roots — path in any root is allowed."""
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        server_mod._ALLOWED_ROOTS = [str(root_a), str(root_b)]

        assert _check_path(str(root_a)) is None
        assert _check_path(str(root_b)) is None

        # Outside both roots
        outside = tmp_path / "c"
        outside.mkdir()
        result = _check_path(str(outside))
        assert result is not None
        assert "not allowed" in result.lower()


class TestInitAllowedRoots:
    """Test _init_allowed_roots reads from env var."""

    def test_unset_env_defaults_to_cwd(self, monkeypatch):
        """When env var is unset, defaults to cwd (secure-by-default)."""
        monkeypatch.delenv("CODE_HEALTH_ALLOWED_ROOTS", raising=False)
        _init_allowed_roots()
        cwd = os.path.realpath(os.getcwd())
        if cwd == "/":
            assert server_mod._ALLOWED_ROOTS == []
        else:
            assert server_mod._ALLOWED_ROOTS == [cwd]

    def test_unset_env_root_cwd_warns(self, monkeypatch, capsys):
        """When env var is unset and cwd is /, emit warning and leave empty."""
        monkeypatch.delenv("CODE_HEALTH_ALLOWED_ROOTS", raising=False)
        monkeypatch.chdir("/")
        _init_allowed_roots()
        assert server_mod._ALLOWED_ROOTS == []
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "cwd is /" in captured.err

    def test_single_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODE_HEALTH_ALLOWED_ROOTS", str(tmp_path))
        _init_allowed_roots()
        assert len(server_mod._ALLOWED_ROOTS) == 1
        assert server_mod._ALLOWED_ROOTS[0] == os.path.realpath(str(tmp_path))

    def test_multiple_roots_separated(self, monkeypatch, tmp_path):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        monkeypatch.setenv(
            "CODE_HEALTH_ALLOWED_ROOTS",
            f"{root_a}{os.pathsep}{root_b}",
        )
        _init_allowed_roots()
        assert len(server_mod._ALLOWED_ROOTS) == 2

    def test_empty_segments_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "CODE_HEALTH_ALLOWED_ROOTS",
            f"{tmp_path}{os.pathsep}{os.pathsep}",
        )
        _init_allowed_roots()
        assert len(server_mod._ALLOWED_ROOTS) == 1

    def test_startup_logs_allowed_roots(self, monkeypatch, tmp_path, capsys):
        """Startup should log the actual allowed roots to stderr."""
        monkeypatch.setenv("CODE_HEALTH_ALLOWED_ROOTS", str(tmp_path))
        _init_allowed_roots()
        captured = capsys.readouterr()
        assert "allowed roots" in captured.err
        assert str(os.path.realpath(str(tmp_path))) in captured.err
