# Code Health Suite

16 analysis engines, 28 MCP tools for Python code quality. Zero external dependencies.

<!-- mcp-name: io.github.neogeweb3/code-health-suite -->

## Quick Start

### As MCP Server (Claude Desktop / Claude Code)

```json
{
  "mcpServers": {
    "code-health": {
      "command": "code-health-suite"
    }
  }
}
```

### Install from PyPI

```bash
pip install code-health-suite
```

Or with uvx (no install needed):

```bash
uvx code-health-suite
```

### Install from GitHub

```bash
pip install git+https://github.com/nge/code-health-suite
```

## Tools

| # | Tool | Engine | What it does |
|---|------|--------|-------------|
| 1 | `analyze_complexity` | complexity | Per-function CC, cognitive complexity, nesting, grades |
| 2 | `get_complexity_score` | complexity | Project health score 0-100 |
| 3 | `find_dead_code` | dead-code | Unused imports, functions, variables, arguments |
| 4 | `security_scan` | security | OWASP vulns, CWE-mapped findings |
| 5 | `get_security_score` | security | Security health score 0-100 |
| 6 | `analyze_imports` | import-graph | Import dependency graph, circular deps |
| 7 | `get_import_health` | import-graph | Import architecture score 0-100 |
| 8 | `find_clones` | clone-detect | Type-1/2/3 code clone detection |
| 9 | `analyze_test_quality` | test-quality | Test suite metrics, anti-patterns |
| 10 | `full_health_check` | all engines | Combined report with overall grade |
| 11 | `find_hotspots` | hotspot | Files with high git churn AND high complexity |
| 12 | `get_hotspot_score` | hotspot | Project churn-complexity score |
| 13 | `audit_dependencies` | dep-audit | Outdated/vulnerable dependency check |
| 14 | `analyze_change_impact` | change-impact | Blast radius of file changes |
| 15 | `get_coupling_score` | change-impact | Module coupling metrics |
| 16 | `analyze_types` | type-audit | Type annotation coverage |
| 17 | `get_type_score` | type-audit | Type coverage score 0-100 |
| 18 | `audit_env` | env-audit | Environment variable audit |
| 19 | `audit_git_commits` | git-audit | Commit quality audit (security + complexity) |
| 20 | `get_git_audit_score` | git-audit | Git commit health score |

## Requirements

- Python 3.10+
- Zero external dependencies (stdlib only)

## License

MIT
