# Code Health Suite — Deployment Guide

## Status (2026-03-15)
- Server: **VERIFIED WORKING** (22 tools, 13 engines, initialize + tools/list + tools/call all pass)
- Package: **BUILD READY** (pyproject.toml + hatchling, zero deps)
- Tests: 4,164+ passing
- GitHub Actions: **READY** (.github/workflows/pypi-publish.yml — trusted publisher OIDC)

## Priority Path: Official MCP Registry (~20 min total)

The **Official MCP Registry** (registry.modelcontextprotocol.io) is the highest-value target.
Linux Foundation-backed industry standard. Preview phase = first-mover advantage window.
API v0.1 is frozen — publishing now won't break on API changes.

### Step 1: Create GitHub Repo (5 min)

```bash
cd ~/automation/workspace/code-health-suite
gh repo create code-health-suite --public --source=. --push
```

### Step 2: Configure PyPI Trusted Publisher (5 min)

1. Go to https://pypi.org/manage/account/publishing/
2. Add new pending publisher:
   - PyPI project name: `code-health-suite`
   - Owner: `nge` (your GitHub username)
   - Repository: `code-health-suite`
   - Workflow: `pypi-publish.yml`
   - Environment: `pypi`

### Step 3: Publish to PyPI (3 min)

```bash
cd ~/automation/workspace/code-health-suite
git tag v0.5.0
git push --tags
# GitHub Actions will auto-publish to PyPI via trusted publisher (OIDC, no API token needed)
```

Verify at: https://pypi.org/project/code-health-suite/

### Step 4: Publish to Official MCP Registry (5 min)

```bash
pip install mcp-publisher
mcp-publisher init          # generates server.json
mcp-publisher login github  # GitHub OAuth
mcp-publisher publish       # publishes to registry.modelcontextprotocol.io
```

Verify at: https://registry.modelcontextprotocol.io

### Post-publish: Check MCP Hive deadline
- MCP Hive founding provider deadline: **May 11, 2026**
- Apply at mcphive.com after registry listing is live

---

## Bonus: Additional Distribution Channels

### MCPize Cloud (~5 min)

MCPize deploys local code to their cloud. No GitHub repo needed.
Revenue split: 85% you / 15% MCPize.

```bash
cd ~/automation/workspace/code-health-suite
npx mcpize login
npx mcpize analyze
npx mcpize deploy
```

### Smithery Registry (~5 min, needs GitHub)

```bash
npx @anthropic-ai/smithery-cli auth login
npx @anthropic-ai/smithery-cli mcp publish https://github.com/nge/code-health-suite -n code-health-suite
```

### PulseMCP (2 min)
Visit https://pulsemcp.com → Submit Server → paste GitHub URL.

---

## Verification Commands

```bash
# Test server locally
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' | python3 -m code_health_suite

# Test tool listing
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}\n{"jsonrpc":"2.0","method":"notifications/initialized"}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n' | python3 -m code_health_suite

# Run test suite
cd ~/automation/workspace/code-health-suite && python -m pytest tests/ -q
```
