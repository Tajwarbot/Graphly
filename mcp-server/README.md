# Graphly MCP Server & External AI Integration

This directory contains the **Model Context Protocol (MCP)** server for Graphly, enabling external AI assistants like **Claude Desktop** and **ChatGPT (via GPT Actions)** to generate interactive graphs.

---

## 1. Architectural Model: Shareable Link vs. Live Session

> [!IMPORTANT]
> **Clickable Link Architecture (Implemented)**:
> External tools call Graphly endpoints or MCP tools to serialize a complete graph specification into a compact, state-encoded URL:
> `https://graphly.netlify.app/?state=<base64-json>`
> When the user clicks the generated link in Claude or ChatGPT, Graphly immediately unpacks the state and renders the exact visualization in their browser.
>
> **Live Session Sync (Future Phase 8)**:
> Drawing live into an already-open browser tab without user navigation would require a real-time WebSocket session-pairing architecture (pairing codes, room coordination, bi-directional socket tunnels). This is flagged as a prospective Phase 8.

---

## 2. Registering with Claude Desktop

To use Graphly tools directly inside Claude Desktop:

1. Open your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
2. Add `graphly` under `mcpServers`:

```json
{
  "mcpServers": {
    "graphly": {
      "command": "node",
      "args": [
        "C:\\Users\\ahmad\\Downloads\\Graphly\\mcp-server\\index.js"
      ],
      "env": {
        "GRAPHLY_BASE_URL": "https://graphly.netlify.app"
      }
    }
  }
}
```

3. Restart Claude Desktop.
4. Claude can now invoke:
   - `plot_function(expression)`
   - `plot_data_table(rows, name)`
   - `plot_3d_surface(expression)`

---

## 3. Registering with ChatGPT (GPT Actions)

1. Go to ChatGPT &rarr; **My GPTs** &rarr; **Create a GPT**.
2. Under the **Configure** tab, scroll down to **Actions** &rarr; **Create new action**.
3. Copy and paste the OpenAPI specification from [`docs/openapi-gpt-action.json`](../docs/openapi-gpt-action.json).
4. Set Authentication to **None** (the link generator is public).
5. ChatGPT will now format graph requests into clickable Graphly links.
