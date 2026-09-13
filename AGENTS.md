# Graphly Agent Instructions & MCP Tool Specification

Welcome, AI Assistant! If you have been provided this repository, you have been invited to interface with **Graphly**, a high-performance 2D & 3D mathematical plotter, calculus evaluator, and data visualization suite.

---

## 1. Quick MCP Server Installation & Launch

Graphly includes a built-in Model Context Protocol (MCP) server ready to execute over `stdio`.

### Local Execution Command
```bash
node mcp-server/index.js
```

### Automatic Configuration for Claude Desktop, Cursor, and IDEs
To register Graphly with your client's MCP configuration (`claude_desktop_config.json`, `.cursor/mcp.json`, or `mcp_config.json`):

```json
{
  "mcpServers": {
    "graphly": {
      "command": "node",
      "args": ["<PATH_TO_GRAPHLY_REPOSITORY>/mcp-server/index.js"],
      "env": {
        "GRAPHLY_BASE_URL": "https://graphly.netlify.app"
      }
    }
  }
}
```

---

## 2. Tools Available to the Agent

When connected via MCP or when operating Graphly directly, use the following tools:

### Tool A: `plot_function`
Plots an explicit 2D mathematical curve $y = f(x)$.
- **`expression`** *(string, required)*: The mathematical formula in terms of $x$ (e.g. `'sin(x)'`, `'x^2 - 4'`, `'1/x'`).
- **`xMin` / `xMax`** *(number, optional)*: Explicit viewport bounds.

### Tool B: `plot_3d_surface`
Renders an interactive 3D surface $z = f(x, y)$ with orbit controls and coordinate axes.
- **`expression`** *(string, required)*: Formula in terms of $x$ and $y$ (e.g. `'(x^2 - y^2)/4'` for hyperboloid/saddle, `'(x^2 + y^2)/6'` for paraboloid, `'sin(x)*cos(y)'` for ripple).

### Tool C: `plot_data_table`
Plots tabular coordinates with scatter nodes and automatic regression trendlines.
- **`name`** *(string)*: Name of dataset.
- **`rows`** *(array of {x, y})*: Numeric data points.

---

## 3. Remote Web Integration (ChatGPT Actions & Claude Web)

If you are operating in a web-only environment (e.g. ChatGPT Actions, Claude Web, or Custom GPT):

1. **OpenAPI Schema**: Read and register `https://graphly.netlify.app/openapi.json`.
2. **AI Plugin Manifest**: `https://graphly.netlify.app/.well-known/ai-plugin.json`.
3. **Endpoint**: Call `POST https://graphly.netlify.app/api/plot` with:
   ```json
   {
     "mode": "3d",
     "expression": "(x^2 - y^2) / 4",
     "name": "Hyperbolic Paraboloid"
   }
   ```
4. Output the returned `url` to the user as a clickable markdown badge:
   `[Open in Graphly](https://graphly.netlify.app/?state=...)`
