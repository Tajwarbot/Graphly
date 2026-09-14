---
name: graphly
description: Plot, compare, and refine mathematical equations with the Graphly MCP server and its live browser viewer. Use when the user requests Graphly plotting or edits to an existing Graphly graph.
---

Use the connected Graphly MCP tools. If unavailable, explain that the local MCP server must be registered using the repository's mcp-server/README.md; this skill does not install or start it.

For a new workspace, call create_graph with the appropriate dimension and open its returned viewer URL once. Retain graph_id for follow-up requests. New plots append through upsert_expression without expression_id. Intentional edits reuse the exact expression_id from get_graph; removals use remove_expression. Preserve unrelated layers. Clarify only when the target of an edit is ambiguous.

Pass equations directly: an ellipse is 2D (x^2/9+y^2/4=1), an ellipsoid is 3D (x^2/9+y^2/4+z^2=1), and a paraboloid can be z=x^2+y^2. Do not split closed surfaces into square-root halves. Use set_view for finite increasing bounds or a 3D camera.

Tool acceptance confirms stored state, not successful rendering. Inspect the viewer before claiming visual verification; report any displayed evaluation errors. Implicit rendering samples a finite domain and may miss small features or zero sets without sign changes.

Use export_graph for a durable, multi-expression snapshot link. Live sessions last only while their local MCP process runs. Keep capability-bearing live links private. Manual edits in the viewer are not synchronized back to the server and can be replaced by subsequent server revisions; use MCP edits for session-managed layers.

Legacy plot_function, plot_3d_surface, and plot_data_table return standalone links. Prefer a live session for iterative equation work. Host inline-view support varies; do not promise an embedded graph when the client only supports browser links.
