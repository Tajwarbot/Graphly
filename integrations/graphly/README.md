# Graphly agent integration

This plugin supplies a plotting skill for Codex and Claude Code. Register the local MCP server separately using [the setup and live-session guide](../../mcp-server/README.md). It intentionally contains no machine-specific executable paths or automatic config writes.

For Claude Code, load this directory with `claude --plugin-dir /absolute/path/to/Graphly/integrations/graphly`. For Codex, install this plugin directory through your supported local-plugin workflow, or copy `skills/graphly` into your personal skills directory. Restart the client after registering MCP tools.

The repository includes both `.codex-plugin/plugin.json` and `.claude-plugin/plugin.json`. A plugin skill alone does not start the viewer or server. Run the updated local frontend and configure `GRAPHLY_BASE_URL` as described in the server guide before testing live sessions.
