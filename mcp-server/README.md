# Graphly MCP server

Dependency-free Node.js (20+) MCP server over stdio. Run `node /absolute/path/to/Graphly/mcp-server/index.js` from an MCP-capable client. Stdout is reserved for JSON-RPC; errors go to stderr.

## Choose the viewer

**Live sessions require the updated Graphly frontend in this repository.** Until this build is deployed, run Graphly locally (`npm install`, `npm run dev -- --host 127.0.0.1`) and set `GRAPHLY_BASE_URL=http://127.0.0.1:5173` in the MCP server's environment. Match the actual Vite port if it changes. The default remains `https://graphly.netlify.app` for existing link tools; its deployed version may not yet support live sessions or implicit surfaces.

Example MCP client configuration (substitute your actual repository path):

```json
{
  "mcpServers": {
    "graphly": {
      "command": "node",
      "args": ["/absolute/path/to/Graphly/mcp-server/index.js"],
      "env": { "GRAPHLY_BASE_URL": "http://127.0.0.1:5173" }
    }
  }
}
```

Client configuration format varies. The command and environment above can also be entered through a client's MCP settings. No API key or dependency installation is needed for this server.

## Live workflow

1. Call `create_graph` with `dimension: "3d"` (or `"2d"`) and optional title. Open the returned URL once.
2. Call `upsert_expression` with its `graph_id` and `equation: "x^2/9+y^2/4+z^2=1"`.
3. Add a plane with another `upsert_expression`, `equation: "z=0"`. **Omit `expression_id` to append.** Existing curves are preserved.
4. To intentionally edit a curve, reuse that curve's ID from the returned graph snapshot.
5. Use `set_view` for bounds (`xMin`, `xMax`, `yMin`, `yMax`, optional `zMin`, `zMax`) or a 3D camera (`position` and `target`, each a mathematical x/y/z triple).
6. Use `get_graph` to inspect state, `remove_expression` to remove one layer, and `export_graph` for a durable snapshot link containing all expressions and the view.

The same browser tab receives complete revisioned snapshots over server-sent events and reconnects automatically. Local MCP sessions stay in memory **only while that server process runs**. Export a snapshot to preserve a graph across restarts; the exported link is a copy, not a live session.

Tool results report `accepted`: the server stored the requested state. This is not a render acknowledgement or proof the equation is valid. The viewer displays evaluation errors. Agent responses must not claim to have visually verified a graph without inspecting the viewer. Manual edits in the viewer are not sent back to the MCP server; later tool updates restore the server's session-owned layers. User layers outside the session are preserved.

## Compatibility tools

`plot_function`, `plot_3d_surface`, and `plot_data_table` still generate standalone one-plot links. For ongoing conversations and multiple equations, prefer live graph tools. The updated 3D viewer accepts full equations, including spheres and ellipsoids; do not split closed surfaces into square-root halves.

## Local connection and privacy

The bridge starts lazily on an ephemeral port bound only to `127.0.0.1`. A random capability token is placed in the viewer URL fragment, which is not sent to the web hosting server. The bridge checks the exact viewer Origin and capability token on every event-stream request. It serves read-only snapshots and no files or mutation endpoint. Keep live links private: they grant access to that process's session snapshots from the configured viewer origin. Exported links contain graph data directly; anyone with an export link can read it.

The viewer must run on the same computer as the MCP process. Browser local-network permissions, remote development environments, CSP, or hosted clients may block loopback connections. Use the local viewer when needed. This implementation provides a browser viewer; it does not claim universal inline MCP Apps support. No external hosting or publishing happens when a tool is called.

## Validation

Run `npm test --prefix mcp-server` (or `node --test mcp-server/session.test.js`) in an environment permitting loopback listeners. Tests cover state isolation, append/edit behavior, validation, Origin/token enforcement, live revisions, and stdio protocol errors. Legacy tools return validation failures as MCP `isError` results rather than silently hanging.
