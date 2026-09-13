---
name: graphly-plotting
description: Plot 2D mathematical curves, 3D quadric surfaces (hyperboloids, paraboloids, saddles, ripples), and scatter datasets with regression in Graphly.
---

# Graphly Plotting & Analysis Skill

Use this skill whenever the user requests mathematical graphs, function plotting, 3D surface visualizations, or data trendlines.

## How to Execute

### Option 1: Via MCP Server
Run the local MCP server located at `mcp-server/index.js` or call the registered tools:
- `plot_function({ expression: "sin(x)" })`
- `plot_3d_surface({ expression: "(x^2 - y^2)/4" })`
- `plot_data_table({ name: "Experiment", rows: [{ x: 1, y: 2 }, { x: 2, y: 4 }] })`

### Option 2: Generating Direct Graphly Links
Generate a direct URL that opens the live visualization in Graphly:

- **3D Surface**:
  `https://graphly.netlify.app/?mode=3d&fn=<URL_ENCODED_EQUATION>`
  *Example*: `https://graphly.netlify.app/?mode=3d&fn=%28x%5E2+-+y%5E2%29+%2F+4` (Hyperboloid)

- **2D Function**:
  `https://graphly.netlify.app/?mode=function&fn=<URL_ENCODED_EQUATION>`
  *Example*: `https://graphly.netlify.app/?mode=function&fn=sin%28x%29`

- **Full State**:
  Base64-encode JSON `{"mode": "3d", "expression": "..."}` and link to:
  `https://graphly.netlify.app/?state=<BASE64_STATE>`
