#!/usr/bin/env python3
"""MCP server: Code Health Suite v0.7.0 — 15 analysis engines, 26 tools.

Bundles complexity, dead code, security, import graph, clone detection,
test quality, hotspot, dependency audit, change impact, type coverage,
environment audit, and git commit audit into a single MCP server for AI assistants.
Zero external dependencies. Uses JSON-RPC 2.0 over stdio (NDJSON transport).

Usage:
    python mcp_server.py                    # start MCP server (stdio)

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "code-health": {
          "command": "python3",
          "args": ["/path/to/ai-code-health/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any

# Engine imports via package

from code_health_suite.engines import complexity
from code_health_suite.engines import dead_code
from code_health_suite.engines import security_scan as security
from code_health_suite.engines import import_graph
from code_health_suite.engines import clone_detect
from code_health_suite.engines import test_quality
from code_health_suite.engines import hotspot
from code_health_suite.engines import dep_audit
from code_health_suite.engines import change_impact
from code_health_suite.engines import type_audit
from code_health_suite.engines import env_audit
from code_health_suite.engines import git_audit
from code_health_suite.engines import naming_check
from code_health_suite.engines import todo_scanner
from code_health_suite.engines import bug_detect

# --- Constants ---

SERVER_NAME = "code-health-suite"
SERVER_VERSION = "0.7.0"
PROTOCOL_VERSION = "2024-11-05"

# --- Tool Definitions ---

TOOLS = [
    # --- Complexity tools ---
    {
        "name": "analyze_complexity",
        "description": (
            "Analyze Python code complexity for a file or directory. "
            "Returns per-function metrics: cyclomatic complexity (McCabe), "
            "cognitive complexity, nesting depth, function length, and letter grades (A-F)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to analyze.",
                },
                "threshold": {
                    "type": "integer",
                    "description": "Cyclomatic complexity threshold (default: 10).",
                    "default": 10,
                },
                "top": {
                    "type": "integer",
                    "description": "Return only top N functions by complexity. 0 = all.",
                    "default": 20,
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["complexity", "cognitive", "length", "nesting"],
                    "description": "Sort metric (default: complexity).",
                    "default": "complexity",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_complexity_score",
        "description": (
            "Get an overall complexity health score (0-100) with letter grade (A-F), "
            "complexity profile classification, and top offenders. Quick project health check."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to analyze.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Dead code tools ---
    {
        "name": "find_dead_code",
        "description": (
            "Detect unused imports, functions, variables, and arguments in Python code. "
            "Supports cross-module analysis to reduce false positives."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to scan.",
                },
                "category": {
                    "type": "string",
                    "enum": ["unused-import", "unused-function", "unused-variable", "unused-argument"],
                    "description": "Filter by category. Omit for all categories.",
                },
                "min_severity": {
                    "type": "string",
                    "enum": ["info", "low", "medium", "high", "critical"],
                    "description": "Minimum severity level (default: info).",
                    "default": "info",
                },
            },
            "required": ["path"],
        },
    },
    # --- Security tools ---
    {
        "name": "security_scan",
        "description": (
            "Scan Python code for security vulnerabilities: command injection, SQL injection, "
            "path traversal, hardcoded secrets, unsafe deserialization, XSS, and more. "
            "Maps findings to CWE identifiers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to scan.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_security_score",
        "description": (
            "Get an overall security health score (0-100) with grade, profile, "
            "and top vulnerability rules. Quick security posture assessment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to scan.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Import graph tools ---
    {
        "name": "analyze_imports",
        "description": (
            "Analyze Python import dependency graph. Returns module metrics, "
            "circular dependencies (cycles), orphan modules, hub modules, "
            "and instability scores. Use to understand project architecture."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root directory of the Python project to analyze.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_import_health",
        "description": (
            "Get an import graph health score (0-100) with grade. "
            "Penalizes circular dependencies, orphan modules, unstable modules, "
            "and hub concentration. Quick architecture health check."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root directory to analyze.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Clone detection tools ---
    {
        "name": "find_clones",
        "description": (
            "Detect code clones (duplicated code blocks) in a Python project. "
            "Finds Type-1 (exact), Type-2 (renamed), and Type-3 (near-miss) clones. "
            "Returns clone pairs with similarity scores and cluster analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to scan for clones.",
                },
                "min_lines": {
                    "type": "integer",
                    "description": "Minimum function length to consider (default: 5).",
                    "default": 5,
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold 0.0-1.0 (default: 0.8).",
                    "default": 0.8,
                },
            },
            "required": ["path"],
        },
    },
    # --- Test quality tools ---
    {
        "name": "analyze_test_quality",
        "description": (
            "Analyze test suite quality: assertion density, test length, "
            "naming conventions, magic numbers, and more. Returns per-file and "
            "per-test metrics with a quality score (0-100) and grade."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Root directory containing test files.",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Max test function length before flagging (default: 50).",
                    "default": 50,
                },
            },
            "required": ["path"],
        },
    },
    # --- Hotspot tools ---
    {
        "name": "find_hotspots",
        "description": (
            "Find code hotspots — files with high git churn AND high complexity. "
            "These are the riskiest files in a project: frequently changed AND hard to understand. "
            "Requires a git repository. Returns hotspot scores, risk levels, and churn/complexity breakdown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Git repository root path.",
                },
                "top": {
                    "type": "integer",
                    "description": "Return top N hotspots (default: 20).",
                    "default": 20,
                },
                "since_days": {
                    "type": "integer",
                    "description": "Look back N days for churn data (default: 180).",
                    "default": 180,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_hotspot_score",
        "description": (
            "Get overall hotspot health score (0-100) with grade. "
            "Measures concentration of risk (churn x complexity). "
            "Lower scores indicate more hotspots that need attention."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Git repository root path.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Dependency audit tools ---
    {
        "name": "audit_dependencies",
        "description": (
            "Audit Python project dependencies for outdated packages and known vulnerabilities. "
            "Reads requirements.txt and/or pyproject.toml. Returns per-dependency status, "
            "latest versions, and CVE/vulnerability details."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project root directory containing requirements.txt or pyproject.toml.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Change impact tools ---
    {
        "name": "analyze_change_impact",
        "description": (
            "Analyze the ripple effect of changing specific files. "
            "Shows direct dependents, transitive impact, affected tests, "
            "and an impact score (fraction of project affected). "
            "Use before refactoring to understand blast radius."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project root directory.",
                },
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths (relative to root) that are being changed.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Max transitive dependency depth (default: 10).",
                    "default": 10,
                },
            },
            "required": ["path", "changed_files"],
        },
    },
    {
        "name": "analyze_coupling",
        "description": (
            "Analyze module coupling metrics: afferent coupling (Ca), efferent coupling (Ce), "
            "instability (Ce/(Ca+Ce)), and hub scores. Identifies tightly-coupled modules "
            "that may resist change."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project root directory.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Type coverage tools ---
    {
        "name": "analyze_type_coverage",
        "description": (
            "Analyze Python type annotation coverage: function signatures, parameters, "
            "return types, Any usage, and type: ignore comments. "
            "Returns per-file metrics and coverage percentages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to analyze.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_type_score",
        "description": (
            "Get overall type coverage health score (0-100) with grade. "
            "Measures annotation completeness, Any usage, and type: ignore density."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to analyze.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Environment audit tools ---
    {
        "name": "audit_env_vars",
        "description": (
            "Audit environment variable usage: find undefined vars referenced in code, "
            "unused vars in .env files, secrets in templates, and missing .env.example entries. "
            "Supports Python, JavaScript, and shell scripts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project root directory.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Git audit tools ---
    {
        "name": "audit_git_commits",
        "description": (
            "Audit recent git commits in a repository. Extracts changed files "
            "and runs static analysis (security scan, complexity) on each commit. "
            "Returns per-commit grades, security findings, and complexity violations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Path to the git repository to audit.",
                },
                "commits": {
                    "type": "integer",
                    "description": "Number of recent commits to audit (default: 10).",
                    "default": 10,
                },
                "since": {
                    "type": "string",
                    "description": "Audit commits since date (e.g., '3 days ago', '2026-03-01').",
                },
                "author": {
                    "type": "string",
                    "description": "Filter commits by author name.",
                },
                "severity": {
                    "type": "string",
                    "description": "Minimum security finding severity (default: low).",
                    "enum": ["critical", "high", "medium", "low", "info"],
                    "default": "low",
                },
                "threshold": {
                    "type": "integer",
                    "description": "Complexity threshold for flagging functions (default: 10).",
                    "default": 10,
                },
            },
            "required": ["repo"],
        },
    },
    {
        "name": "get_git_audit_score",
        "description": (
            "Get a quick overall grade and score for recent commits in a repo. "
            "Returns the aggregate score, grade, commit count, and security summary."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Path to the git repository.",
                },
                "commits": {
                    "type": "integer",
                    "description": "Number of recent commits to audit (default: 10).",
                    "default": 10,
                },
            },
            "required": ["repo"],
        },
    },
    # --- Naming convention tools ---
    {
        "name": "check_naming",
        "description": (
            "Check Python naming conventions (PEP 8). Detects violations: "
            "functions/methods must be snake_case, classes must be CamelCase, "
            "constants must be UPPER_SNAKE_CASE. Returns violations with suggestions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to check.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_naming_score",
        "description": (
            "Get a naming convention health score (0-100) with grade. "
            "Measures PEP 8 naming compliance: snake_case functions, "
            "CamelCase classes, UPPER_SNAKE_CASE constants."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to check.",
                },
            },
            "required": ["path"],
        },
    },
    # --- TODO scanner tools ---
    {
        "name": "scan_todos",
        "description": (
            "Scan source code for technical debt markers: TODO, FIXME, HACK, XXX, "
            "BUG, NOTE, OPTIMIZE, REFACTOR comments. Returns items with file, line, "
            "tag, severity, and message. Optionally enriches with git blame metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to scan.",
                },
                "blame": {
                    "type": "boolean",
                    "description": "Enrich with git blame metadata (author, date). Slower.",
                    "default": False,
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by specific tag (e.g., TODO, FIXME, HACK).",
                },
                "severity": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Filter by minimum severity level.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_todo_score",
        "description": (
            "Get a technical debt health score (0-100) with grade based on "
            "density and severity of TODO/FIXME/HACK markers. Shows hotspot files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to scan.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Bug detection tools ---
    {
        "name": "detect_bugs",
        "description": (
            "Detect common Python semantic bugs using AST analysis. Finds 8 categories: "
            "missing f-strings, mutable class variables, late-binding closures, "
            "call-expression defaults (datetime.now()), mutable default arguments, "
            "assert-on-tuple, unreachable code, and unreachable exception handlers. "
            "Every finding indicates a likely real bug, not a style violation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to scan.",
                },
                "rules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by specific rules. Options: missing-fstring, mutable-class-var, "
                        "late-binding-closure, call-default, mutable-default-arg, assert-tuple, "
                        "unreachable-code, unreachable-except. Omit for all rules."
                    ),
                },
                "min_severity": {
                    "type": "string",
                    "enum": ["error", "warning", "info"],
                    "description": "Minimum severity to report (default: info).",
                    "default": "info",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_bug_score",
        "description": (
            "Get a bug detection health score (0-100) with grade (A-F), "
            "bug profile classification (clean/fstring_heavy/closure_heavy/etc.), "
            "and breakdown by rule and severity. Quick bug health check."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to scan.",
                },
            },
            "required": ["path"],
        },
    },
    # --- Combined ---
    {
        "name": "full_health_check",
        "description": (
            "Run all analyses (complexity + dead code + security + imports + "
            "clones + test quality + type coverage + env audit + naming + TODO debt + bug detection) on a Python project "
            "and return a combined health report with scores, grades, and top issues. "
            "Note: hotspot, dependency, and change impact require additional context "
            "(git repo, requirements files, changed files) so are excluded from this scan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to analyze.",
                },
            },
            "required": ["path"],
        },
    },
]


# --- Tool Handlers ---


def _check_path(path: str) -> str | None:
    """Return error message if path doesn't exist, else None."""
    if not os.path.exists(path):
        return f"Path not found: {path}"
    return None


def handle_analyze_complexity(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    threshold = args.get("threshold", complexity.DEFAULT_CYCLOMATIC_THRESHOLD)
    top_n = args.get("top", 20)
    sort_by = args.get("sort_by", "complexity")

    result = complexity.analyze(path)
    data = result.to_dict()

    sort_key = {
        "complexity": lambda f: f.cyclomatic,
        "cognitive": lambda f: f.cognitive,
        "length": lambda f: f.length,
        "nesting": lambda f: f.max_nesting,
    }.get(sort_by, lambda f: f.cyclomatic)

    sorted_fns = sorted(result.all_functions, key=sort_key, reverse=True)
    if top_n > 0:
        sorted_fns = sorted_fns[:top_n]

    violations = [f for f in result.all_functions if f.cyclomatic >= threshold]

    return {
        "files_analyzed": data["files_analyzed"],
        "total_functions": data["total_functions"],
        "summary": data["summary"],
        "violations_count": len(violations),
        "threshold": threshold,
        "functions": [f.to_dict() for f in sorted_fns],
        "errors": data["errors"],
    }


def handle_get_complexity_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = complexity.analyze(path)
    score = complexity.compute_complexity_score(result)
    return score.to_dict()


def handle_find_dead_code(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    category = args.get("category")
    min_severity = args.get("min_severity", "info")

    result = dead_code.scan(path, category=category, min_severity=min_severity)
    data = asdict(result)

    # Flatten findings from reports
    all_findings = []
    for report in data.get("reports", []):
        for finding in report.get("findings", []):
            all_findings.append(finding)

    return {
        "files_scanned": data["files_scanned"],
        "total_findings": data["total_findings"],
        "by_category": data["by_category"],
        "by_severity": data["by_severity"],
        "findings": all_findings,
    }


def handle_security_scan(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    if os.path.isfile(path):
        findings, error = security.scan_file(path)
        result = security.ScanResult(
            files_scanned=1,
            findings=findings,
            errors=[error] if error else [],
        )
    else:
        result = security.scan_directory(path)

    return result.to_dict()


def handle_get_security_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    if os.path.isfile(path):
        findings, error = security.scan_file(path)
        result = security.ScanResult(
            files_scanned=1,
            findings=findings,
            errors=[error] if error else [],
        )
    else:
        result = security.scan_directory(path)

    score = security.compute_security_score(result)
    return score.to_dict()


def handle_analyze_imports(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = import_graph.analyze(path)
    data = asdict(result)

    return {
        "root": data["root"],
        "total_modules": data["total_modules"],
        "total_edges": data["total_edges"],
        "internal_edges": data["internal_edges"],
        "external_packages": data["external_packages"],
        "cycles": data["cycles"],
        "orphans": data["orphans"],
        "hub_modules": data["hub_modules"][:10],
        "unstable_modules": data["unstable_modules"][:10],
    }


def handle_get_import_health(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = import_graph.analyze(path)
    score = import_graph.compute_import_health(result)
    return asdict(score)


def handle_find_clones(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    min_lines = args.get("min_lines", 5)
    threshold = args.get("threshold", 0.8)

    result = clone_detect.scan_directory(path, min_lines=min_lines, threshold=threshold)
    data = asdict(result)

    # Summarize clone pairs (full source can be large)
    pairs_summary = []
    for pair in data.get("clone_pairs", [])[:20]:
        pairs_summary.append({
            "block_a": f"{pair['block_a']['filepath']}:{pair['block_a']['start_line']}-{pair['block_a']['end_line']} ({pair['block_a']['name']})",
            "block_b": f"{pair['block_b']['filepath']}:{pair['block_b']['start_line']}-{pair['block_b']['end_line']} ({pair['block_b']['name']})",
            "similarity": round(pair["similarity"], 3),
            "clone_type": pair["clone_type"],
        })

    return {
        "files_scanned": data["files_scanned"],
        "blocks_extracted": data["blocks_extracted"],
        "clone_pairs_count": len(data.get("clone_pairs", [])),
        "clone_score": data["clone_score"],
        "clusters_count": len(data.get("clusters", [])),
        "clone_pairs": pairs_summary,
    }


def handle_analyze_test_quality(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    max_length = args.get("max_length", 50)

    result = test_quality.analyze_suite(path, max_length=max_length)
    data = asdict(result)

    # Summarize file reports (omit full test details for brevity)
    file_summaries = []
    for fr in data.get("files", []):
        file_summaries.append({
            "file": fr["file"],
            "test_count": fr["test_count"],
            "total_assertions": fr["total_assertions"],
            "assertion_density": round(fr["assertion_density"], 2),
            "issues_count": len(fr.get("issues", [])),
        })

    return {
        "files_analyzed": data["files_analyzed"],
        "total_tests": data["total_tests"],
        "total_assertions": data["total_assertions"],
        "total_issues": data["total_issues"],
        "issues_by_severity": data["issues_by_severity"],
        "issues_by_check": data["issues_by_check"],
        "score": data["score"],
        "grade": data["grade"],
        "files": file_summaries,
    }


def handle_full_health_check(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    # Complexity
    cx_result = complexity.analyze(path)
    cx_score = complexity.compute_complexity_score(cx_result)

    # Dead code
    dc_result = dead_code.scan(path)
    dc_data = asdict(dc_result)

    # Security
    if os.path.isfile(path):
        sec_findings, sec_error = security.scan_file(path)
        sec_result = security.ScanResult(
            files_scanned=1,
            findings=sec_findings,
            errors=[sec_error] if sec_error else [],
        )
    else:
        sec_result = security.scan_directory(path)
    sec_score = security.compute_security_score(sec_result)

    # Import graph
    ig_result = import_graph.analyze(path)
    ig_score = import_graph.compute_import_health(ig_result)

    # Clone detection
    cd_result = clone_detect.scan_directory(path)

    # Test quality
    tq_result = test_quality.analyze_suite(path)

    # Type coverage
    type_file_results = []
    for fpath in type_audit.find_python_files(path):
        type_file_results.append(type_audit.analyze_file(fpath))
    ta_result = type_audit.aggregate_results(path, type_file_results)
    ta_score = type_audit.compute_score(ta_result)
    ta_grade = type_audit.score_to_grade(ta_score)

    # Env audit
    from pathlib import Path as _Path
    env_findings = env_audit.run_audit(_Path(path))
    env_files = env_audit.find_env_files(_Path(path))
    env_code_refs = env_audit.scan_source_files(_Path(path))
    env_code_vars = {}
    for ref in env_code_refs:
        env_code_vars[ref.name] = env_code_vars.get(ref.name, 0) + 1
    env_score_val, env_grade = env_audit.calculate_score(env_findings, env_files, env_code_vars)

    # Naming conventions
    nc_result = naming_check.scan(path)
    nc_score = naming_check.compute_score(nc_result)

    # TODO/FIXME scanner
    td_result = todo_scanner.scan(path)
    td_score = todo_scanner.compute_score(td_result)

    # Bug detection
    bd_result = bug_detect.scan(path)
    bd_score = bug_detect.compute_score(bd_result)

    grades = [cx_score.grade, sec_score.grade, ig_score.grade, tq_result.grade, ta_grade, env_grade, nc_score.grade, td_score.grade, bd_score.grade]

    return {
        "path": path,
        "complexity": {
            "score": cx_score.score,
            "grade": cx_score.grade,
            "profile": cx_score.profile,
            "total_functions": cx_score.total_functions,
            "violations": cx_score.violations_count,
            "top_offenders": cx_score.top_offenders[:5],
        },
        "dead_code": {
            "files_scanned": dc_data["files_scanned"],
            "total_findings": dc_data["total_findings"],
            "by_category": dc_data["by_category"],
        },
        "security": {
            "score": sec_score.score,
            "grade": sec_score.grade,
            "profile": sec_score.profile,
            "total_findings": sec_score.total_findings,
            "top_rules": sec_score.top_rules[:5],
        },
        "imports": {
            "score": ig_score.score,
            "grade": ig_score.grade,
            "total_modules": ig_score.total_modules,
            "cycle_count": ig_score.cycle_count,
            "orphan_count": ig_score.orphan_count,
            "profile": ig_score.profile,
        },
        "clones": {
            "clone_score": cd_result.clone_score,
            "clone_pairs": len(cd_result.clone_pairs),
            "clusters": len(cd_result.clusters),
        },
        "test_quality": {
            "score": tq_result.score,
            "grade": tq_result.grade,
            "total_tests": tq_result.total_tests,
            "total_issues": tq_result.total_issues,
        },
        "type_coverage": {
            "score": ta_score,
            "grade": ta_grade,
            "function_coverage": round(ta_result.function_coverage, 3),
            "param_coverage": round(ta_result.param_coverage, 3),
        },
        "env_audit": {
            "score": env_score_val,
            "grade": env_grade,
            "total_findings": len(env_findings),
        },
        "naming": {
            "score": nc_score.score,
            "grade": nc_score.grade,
            "total_violations": nc_score.total_violations,
            "violation_rate": nc_score.violation_rate,
        },
        "todo_debt": {
            "score": td_score.score,
            "grade": td_score.grade,
            "total_items": td_score.total_items,
            "density": td_score.density,
            "by_severity": td_score.by_severity,
        },
        "bug_detect": {
            "score": bd_score.score,
            "grade": bd_score.grade,
            "profile": bd_score.profile,
            "total_findings": bd_score.total_findings,
            "density": bd_score.density,
            "by_rule": bd_score.by_rule,
        },
        "overall_grade": _combined_grade(grades),
    }


def _combined_grade(grades: list[str]) -> str:
    """Compute combined grade (worst of all grades)."""
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    worst = max(grade_order.get(g, 4) for g in grades)
    return {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}[worst]


def handle_find_hotspots(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    top_n = args.get("top", 20)
    since_days = args.get("since_days", 180)

    result = hotspot.analyze(path, top_n=top_n, since_days=since_days)
    hotspots = []
    for h in result.hotspots[:top_n]:
        hotspots.append({
            "filepath": h.filepath,
            "hotspot_score": round(h.hotspot_score, 2),
            "risk_level": h.risk_level,
            "commits": h.churn.commits,
            "lines_added": h.churn.lines_added,
            "lines_deleted": h.churn.lines_deleted,
            "max_cyclomatic": h.complexity.max_cc,
            "num_functions": h.complexity.num_functions,
        })

    return {
        "repo_path": result.repo_path,
        "total_files_analyzed": result.total_files_analyzed,
        "total_python_files": result.total_python_files,
        "since_days": result.since_days,
        "hotspots": hotspots,
        "errors": result.errors,
    }


def handle_get_hotspot_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = hotspot.analyze(path)
    stats = hotspot.compute_project_stats(result)
    return asdict(stats)


def handle_audit_dependencies(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    from pathlib import Path
    root = Path(path)
    deps: list = []

    req_txt = root / "requirements.txt"
    if req_txt.exists():
        deps.extend(dep_audit.parse_requirements_txt(str(req_txt)))

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        deps.extend(dep_audit.parse_pyproject_toml(str(pyproject)))

    if not deps:
        return {"error": "No requirements.txt or pyproject.toml found", "dependencies": []}

    results = dep_audit.audit_dependencies(deps, check_versions=True, check_vulns=True)

    dep_list = []
    for r in results:
        entry = {
            "name": r.dependency.name,
            "specified": r.dependency.specified_version or "",
            "source": r.dependency.source_file,
        }
        if r.version_info:
            entry["latest"] = r.version_info.latest or ""
            entry["update_type"] = r.version_info.update_type
        if r.vulnerabilities:
            entry["vulnerabilities"] = [
                {"id": v.id, "severity": v.severity, "summary": v.summary}
                for v in r.vulnerabilities
            ]
        dep_list.append(entry)

    vulnerable = [r for r in results if r.has_vulns]
    outdated = [r for r in results if r.is_outdated]

    return {
        "total_dependencies": len(results),
        "vulnerable_count": len(vulnerable),
        "outdated_count": len(outdated),
        "dependencies": dep_list,
    }


def handle_analyze_change_impact(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    changed_files = args.get("changed_files", [])
    max_depth = args.get("max_depth", 10)

    result = change_impact.analyze(path, changed_files, max_depth=max_depth)

    direct = [{"module": m.module, "path": m.path, "depth": m.depth, "is_test": m.is_test}
              for m in result.direct_impact]
    transitive = [{"module": m.module, "path": m.path, "depth": m.depth, "is_test": m.is_test}
                  for m in result.transitive_impact]

    return {
        "changed_files": result.changed_files,
        "changed_modules": result.changed_modules,
        "total_project_modules": result.total_project_modules,
        "direct_impact": direct,
        "transitive_impact": transitive,
        "affected_tests": result.affected_tests,
        "impact_score": round(result.impact_score, 3),
        "risk_level": result.risk_level,
        "summary": result.summary,
    }


def handle_analyze_coupling(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = change_impact.compute_coupling_metrics(path)
    modules = []
    for m in result.modules[:20]:
        modules.append(asdict(m))

    return {
        "root": result.root,
        "total_modules": result.total_modules,
        "modules": modules,
    }


def handle_analyze_type_coverage(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    file_results = [type_audit.analyze_file(f) for f in type_audit.find_python_files(path)]
    result = type_audit.aggregate_results(path, file_results)

    return {
        "target": result.target,
        "files_scanned": result.files_scanned,
        "total_functions": result.total_functions,
        "fully_typed": result.fully_typed_functions,
        "partially_typed": result.partially_typed_functions,
        "untyped": result.untyped_functions,
        "function_coverage": round(result.function_coverage, 3),
        "param_coverage": round(result.param_coverage, 3),
        "return_coverage": round(result.return_coverage, 3),
        "any_count": result.any_count,
        "type_ignore_count": result.type_ignore_count,
    }


def handle_get_type_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    file_results = [type_audit.analyze_file(f) for f in type_audit.find_python_files(path)]
    result = type_audit.aggregate_results(path, file_results)

    score = type_audit.compute_score(result)
    grade = type_audit.score_to_grade(score)
    profile = type_audit.classify_profile(result)

    return {
        "score": score,
        "grade": grade,
        "profile": profile,
        "function_coverage": round(result.function_coverage, 3),
        "param_coverage": round(result.param_coverage, 3),
        "any_count": result.any_count,
        "type_ignore_count": result.type_ignore_count,
    }


def handle_audit_env_vars(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    from pathlib import Path
    findings = env_audit.run_audit(Path(path))

    findings_list = []
    for f in findings:
        findings_list.append({
            "check": f.check,
            "severity": f.severity,
            "variable": f.variable,
            "message": f.message,
            "files": f.files,
        })

    env_files = env_audit.find_env_files(Path(path))
    code_refs = env_audit.scan_source_files(Path(path))
    code_vars = {}
    for ref in code_refs:
        code_vars[ref.name] = code_vars.get(ref.name, 0) + 1

    score, grade = env_audit.calculate_score(findings, env_files, code_vars)

    sev_counts: dict[str, int] = {}
    for f in findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1

    return {
        "total_findings": len(findings),
        "by_severity": sev_counts,
        "score": score,
        "grade": grade,
        "findings": findings_list,
    }


def handle_audit_git_commits(args: dict[str, Any]) -> dict[str, Any]:
    repo = args["repo"]
    if err := _check_path(repo):
        return {"error": err}

    num_commits = args.get("commits", 10)
    since = args.get("since")
    author = args.get("author")
    severity = args.get("severity", "low")
    threshold = args.get("threshold", 10)

    report = git_audit.run_audit(
        repo_path=repo,
        num_commits=num_commits,
        since=since,
        author=author,
        severity=severity,
        complexity_threshold=threshold,
    )

    data = asdict(report)

    # Summarize commits for output
    commits_summary = []
    for c in data.get("commits", []):
        commits_summary.append({
            "sha": c["short_sha"],
            "author": c["author"],
            "date": c["date"],
            "message": c["message"],
            "files_changed": len(c["files"]),
            "additions": c["total_additions"],
            "deletions": c["total_deletions"],
            "security_findings": len(c["security_findings"]),
            "complexity_findings": len(c["complexity_findings"]),
            "grade": c["grade"],
            "score": c["score"],
        })

    return {
        "repo_name": data["repo_name"],
        "commits_analyzed": data["commits_analyzed"],
        "total_files_changed": data["total_files_changed"],
        "total_additions": data["total_additions"],
        "total_deletions": data["total_deletions"],
        "total_security_findings": data["total_security_findings"],
        "security_by_severity": data["security_by_severity"],
        "total_high_complexity": data["total_high_complexity"],
        "overall_grade": data["overall_grade"],
        "overall_score": data["overall_score"],
        "tools_available": data["tools_available"],
        "commits": commits_summary,
    }


def handle_get_git_audit_score(args: dict[str, Any]) -> dict[str, Any]:
    repo = args["repo"]
    if err := _check_path(repo):
        return {"error": err}

    num_commits = args.get("commits", 10)

    report = git_audit.run_audit(repo_path=repo, num_commits=num_commits)

    return {
        "repo_name": report.repo_name,
        "commits_analyzed": report.commits_analyzed,
        "overall_score": report.overall_score,
        "overall_grade": report.overall_grade,
        "total_security_findings": report.total_security_findings,
        "security_by_severity": report.security_by_severity,
        "total_high_complexity": report.total_high_complexity,
    }


def handle_check_naming(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = naming_check.scan(path)

    violations_list = []
    for v in result.violations:
        violations_list.append({
            "file": v.file_path,
            "line": v.line_number,
            "name": v.name,
            "kind": v.kind,
            "convention": v.convention,
            "suggestion": v.suggestion,
            "message": v.message,
        })

    return {
        "files_scanned": result.files_scanned,
        "names_checked": result.total_names,
        "total_violations": result.total_violations,
        "by_kind": result.by_kind,
        "violations": violations_list,
        "errors": result.errors,
    }


def handle_get_naming_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = naming_check.scan(path)
    score = naming_check.compute_score(result)
    return score.to_dict()


def handle_scan_todos(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = todo_scanner.scan(path)

    # Filter by tag
    tag_filter = args.get("tag")
    if tag_filter:
        tag_upper = tag_filter.upper()
        result.items = [i for i in result.items if i.tag == tag_upper]
        result.total_items = len(result.items)

    # Filter by severity
    sev_filter = args.get("severity")
    if sev_filter:
        sev_order = {"high": 3, "medium": 2, "low": 1}
        min_sev = sev_order.get(sev_filter, 0)
        result.items = [i for i in result.items if sev_order.get(i.severity, 0) >= min_sev]
        result.total_items = len(result.items)

    # Enrich with blame if requested
    if args.get("blame"):
        todo_scanner.enrich_with_blame(result.items)

    items_list = []
    for item in result.items:
        d = {
            "file": item.file_path,
            "line": item.line_number,
            "tag": item.tag,
            "severity": item.severity,
            "message": item.message,
        }
        if item.inline_author:
            d["inline_author"] = item.inline_author
        if item.blame_author:
            d["blame_author"] = item.blame_author
            d["blame_date"] = item.blame_date
            d["blame_age_days"] = item.blame_age_days
        items_list.append(d)

    return {
        "files_scanned": result.files_scanned,
        "total_lines": result.total_lines,
        "total_items": result.total_items,
        "by_tag": result.by_tag,
        "by_severity": result.by_severity,
        "items": items_list,
        "errors": result.errors,
    }


def handle_get_todo_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = todo_scanner.scan(path)
    score = todo_scanner.compute_score(result)
    return score.to_dict()


def handle_detect_bugs(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    rules = args.get("rules")
    min_severity = args.get("min_severity", "info")
    result = bug_detect.scan(path, rules=rules, min_severity=min_severity)
    return result.to_dict()


def handle_get_bug_score(args: dict[str, Any]) -> dict[str, Any]:
    path = args["path"]
    if err := _check_path(path):
        return {"error": err}

    result = bug_detect.scan(path)
    score = bug_detect.compute_score(result)
    return score.to_dict()


TOOL_HANDLERS = {
    "analyze_complexity": handle_analyze_complexity,
    "get_complexity_score": handle_get_complexity_score,
    "find_dead_code": handle_find_dead_code,
    "security_scan": handle_security_scan,
    "get_security_score": handle_get_security_score,
    "analyze_imports": handle_analyze_imports,
    "get_import_health": handle_get_import_health,
    "find_clones": handle_find_clones,
    "analyze_test_quality": handle_analyze_test_quality,
    "find_hotspots": handle_find_hotspots,
    "get_hotspot_score": handle_get_hotspot_score,
    "audit_dependencies": handle_audit_dependencies,
    "analyze_change_impact": handle_analyze_change_impact,
    "analyze_coupling": handle_analyze_coupling,
    "analyze_type_coverage": handle_analyze_type_coverage,
    "get_type_score": handle_get_type_score,
    "audit_env_vars": handle_audit_env_vars,
    "audit_git_commits": handle_audit_git_commits,
    "get_git_audit_score": handle_get_git_audit_score,
    "check_naming": handle_check_naming,
    "get_naming_score": handle_get_naming_score,
    "scan_todos": handle_scan_todos,
    "get_todo_score": handle_get_todo_score,
    "detect_bugs": handle_detect_bugs,
    "get_bug_score": handle_get_bug_score,
    "full_health_check": handle_full_health_check,
}


# --- JSON-RPC / MCP Protocol ---


def make_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def make_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", "")
    id = msg.get("id")
    params = msg.get("params", {})

    if id is None:
        return None  # notification, no response

    if method == "initialize":
        return make_response(id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method == "tools/list":
        return make_response(id, {"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return make_error(id, -32602, f"Unknown tool: {tool_name}")
        try:
            result = handler(arguments)
            text = json.dumps(result, indent=2, ensure_ascii=False)
            return make_response(id, {"content": [{"type": "text", "text": text}]})
        except Exception as e:
            return make_response(id, {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            })

    if method == "ping":
        return make_response(id, {})

    return make_error(id, -32601, f"Method not found: {method}")


def run_stdio():
    """Run MCP server on stdio (NDJSON)."""
    sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1, closefd=False)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(make_error(None, -32700, "Parse error")) + "\n")
            sys.stdout.flush()
            continue

        response = handle_message(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


def main():
    """Entry point for the code-health-suite command."""
    run_stdio()


if __name__ == "__main__":
    main()
