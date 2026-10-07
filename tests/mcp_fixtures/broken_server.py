"""Test-only MCP "server": exits immediately instead of speaking the
protocol, so app.mcp.manager's connection-failure and backoff paths have a real
subprocess to fail against, not a mock. Never registered in
app.mcp.builtin.REGISTRY; tests reach it by monkeypatching that registry.
"""

import sys

sys.exit(1)
