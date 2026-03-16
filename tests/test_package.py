"""Tests for the packaged Code Health Suite."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


# --- Package import tests ---

def test_package_importable():
    import code_health_suite
    assert hasattr(code_health_suite, '__version__')
    assert code_health_suite.__version__ == "0.7.0"


def test_server_importable():
    from code_health_suite.server import main, run_stdio, handle_message
    assert callable(main)
    assert callable(run_stdio)
    assert callable(handle_message)


def test_all_engines_importable():
    from code_health_suite.engines import complexity
    from code_health_suite.engines import dead_code
    from code_health_suite.engines import security_scan
    from code_health_suite.engines import import_graph
    from code_health_suite.engines import clone_detect
    from code_health_suite.engines import test_quality
    from code_health_suite.engines import hotspot
    from code_health_suite.engines import dep_audit
    from code_health_suite.engines import change_impact
    from code_health_suite.engines import type_audit
    from code_health_suite.engines import env_audit
    from code_health_suite.engines import git_audit
    from code_health_suite.engines import bug_detect
    assert hasattr(complexity, 'analyze')
    assert hasattr(dead_code, 'scan')
    assert hasattr(security_scan, 'scan_directory')
    assert hasattr(import_graph, 'analyze')
    assert hasattr(clone_detect, 'scan_directory')
    assert hasattr(test_quality, 'analyze_suite')


def test_ast_utils_importable():
    from code_health_suite.ast_utils import walk_scope, walk_scope_bfs
    assert callable(walk_scope)
    assert callable(walk_scope_bfs)


# --- MCP protocol tests ---

def _send_mcp(messages: list[dict]) -> list[dict]:
    """Send MCP messages via subprocess and parse responses."""
    input_text = "\n".join(json.dumps(m) for m in messages) + "\n"
    result = subprocess.run(
        [sys.executable, "-m", "code_health_suite"],
        input=input_text, capture_output=True, text=True, timeout=30,
    )
    responses = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            responses.append(json.loads(line))
    return responses


def test_mcp_initialize():
    responses = _send_mcp([{
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1.0"}},
    }])
    assert len(responses) == 1
    assert responses[0]["result"]["serverInfo"]["name"] == "code-health-suite"
    assert responses[0]["result"]["serverInfo"]["version"] == "0.8.0"


def test_mcp_tools_list():
    responses = _send_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ])
    tools_resp = [r for r in responses if r.get("id") == 2][0]
    tools = tools_resp["result"]["tools"]
    assert len(tools) == 28
    tool_names = {t["name"] for t in tools}
    expected = {
        "analyze_complexity", "get_complexity_score", "find_dead_code",
        "security_scan", "get_security_score", "analyze_imports",
        "get_import_health", "find_clones", "analyze_test_quality",
        "find_hotspots", "get_hotspot_score", "audit_dependencies",
        "analyze_change_impact", "analyze_coupling", "analyze_type_coverage",
        "get_type_score", "audit_env_vars", "audit_git_commits",
        "get_git_audit_score", "check_naming", "get_naming_score",
        "scan_todos", "get_todo_score",
        "detect_bugs", "get_bug_score",
        "audit_docstrings", "get_docstring_score",
        "full_health_check",
    }
    assert tool_names == expected


# --- Tool execution tests ---

@pytest.fixture
def sample_py(tmp_path):
    """Create a sample Python file for testing."""
    code = tmp_path / "sample.py"
    code.write_text(textwrap.dedent("""\
        import os
        import sys
        import json  # unused

        def hello(name: str) -> str:
            return f"Hello, {name}!"

        def complex_func(x, y, z):
            if x > 0:
                if y > 0:
                    if z > 0:
                        return x + y + z
                    else:
                        return x + y
                else:
                    return x
            else:
                return 0
    """))
    return str(code)


def test_tool_analyze_complexity(sample_py):
    responses = _send_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "analyze_complexity", "arguments": {"path": sample_py}}},
    ])
    tool_resp = [r for r in responses if r.get("id") == 2][0]
    assert "isError" not in tool_resp["result"] or not tool_resp["result"]["isError"]
    content = json.loads(tool_resp["result"]["content"][0]["text"])
    assert "functions" in content
    assert len(content["functions"]) == 2  # hello + complex_func


def test_tool_find_dead_code(sample_py):
    responses = _send_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "find_dead_code", "arguments": {"path": sample_py}}},
    ])
    tool_resp = [r for r in responses if r.get("id") == 2][0]
    content = json.loads(tool_resp["result"]["content"][0]["text"])
    assert "findings" in content
    # json is unused
    unused_imports = [f for f in content["findings"] if f.get("category") == "unused-import"]
    assert len(unused_imports) >= 1


def test_tool_security_scan(sample_py):
    responses = _send_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "security_scan", "arguments": {"path": sample_py}}},
    ])
    tool_resp = [r for r in responses if r.get("id") == 2][0]
    assert "isError" not in tool_resp["result"] or not tool_resp["result"]["isError"]


def test_tool_get_security_score(sample_py):
    responses = _send_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "get_security_score", "arguments": {"path": sample_py}}},
    ])
    tool_resp = [r for r in responses if r.get("id") == 2][0]
    content = json.loads(tool_resp["result"]["content"][0]["text"])
    assert "score" in content
    assert "grade" in content
    assert 0 <= content["score"] <= 100


def test_tool_unknown_returns_error():
    responses = _send_mcp([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "nonexistent_tool", "arguments": {}}},
    ])
    tool_resp = [r for r in responses if r.get("id") == 2][0]
    assert "error" in tool_resp
    assert tool_resp["error"]["code"] == -32602


def test_module_entry_point():
    """Test python -m code_health_suite works."""
    result = subprocess.run(
        [sys.executable, "-m", "code_health_suite"],
        input='{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}\n',
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    resp = json.loads(result.stdout.strip())
    assert resp["id"] == 1
    assert "result" in resp
