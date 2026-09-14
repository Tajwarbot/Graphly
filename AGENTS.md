# Graphly agent integration

Graphly plots 2D curves, 3D surfaces, and tabular data. Prefer the local MCP tools for an ongoing graphing conversation; preserve existing layers unless the user asks to edit or remove them.

## Start the MCP server

Run `node /absolute/path/to/Graphly/mcp-server/index.js` over stdio with Node.js 20 or later. The MCP package has no dependencies.

Set `GRAPHLY_BASE_URL` to the frontend running this revision. During development, run `npm run dev -- --host 127.0.0.1` from the repository and use `http://127.0.0.1:5173` (match the actual port). The default is `https://graphly.netlify.app`; live sessions and implicit surfaces require deploying the updated frontend first.

Example client configuration:

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

## Live tools and workflow

- `create_graph`: create a graph with `dimension: "2d"` or `"3d"` and optional `title`. Retain the returned graph ID and open its live URL once.
- `upsert_expression`: send `graph_id` and `equation`. **Omit `expression_id` to append a new layer.** Reuse an existing ID only when intentionally editing that layer. Optional `visible` and six-digit hex `color` control display.
- `get_graph`: inspect current server state, including stable expression IDs and revision.
- `remove_expression`: remove only the specified `expression_id` from `graph_id`.
- `set_view`: set `bounds` with ordered finite `xMin`, `xMax`, `yMin`, `yMax`, optional `zMin`, `zMax`; or set `camera` with `position` and `target` triples in mathematical x/y/z coordinates.
- `export_graph`: return a standalone snapshot link containing every expression, title, and view. It contains no live capability token and survives MCP process restarts.

Use dimension from the user's context. After discussing 3D surfaces, interpret an ambiguous shape in that context or clarify briefly. A 2D ellipse and a 3D ellipsoid are different objects.

## Equations

The updated 3D renderer accepts implicit equations directly:

- Sphere: `x^2+y^2+z^2=9`
- Ellipsoid: `x^2/9+y^2/4+z^2=1`
- Vertical cylinder: `x^2+y^2=9` in a 3D graph
- Paraboloid: `z=x^2+y^2`
- Saddle (hyperbolic paraboloid): `z=(x^2-y^2)/4`
- One-sheet hyperboloid: `x^2+y^2-z^2=1`

Do not split spheres or ellipsoids into positive and negative square-root surfaces. Use full equations. Numerical extraction has finite resolution; singularities and very small features can require tighter bounds. Do not promise arbitrary equations render perfectly.

## Legacy tools

`plot_function` accepts an explicit 2D expression and optional paired `xMin`/`xMax`. `plot_3d_surface` accepts a 3D expression or full implicit equation with the updated viewer. `plot_data_table` accepts optional `name` and `rows: [{x: 1, y: 2}]` with finite numbers. These tools produce one-plot share links; use live tools for incremental multi-expression work.

## Verification and session limits

The server reports **accepted**, meaning it stored state; this does not verify rendering. Inspect the viewer before claiming a graph is displayed correctly. Report expression errors honestly. Never claim to have opened a viewer or changed a browser tab without doing so.

Live sessions use a loopback SSE bridge with an Origin check and capability token. They exist only for the MCP process lifetime, on the same computer as the viewer. Browser local-network restrictions may require local viewing. Keep live URLs private. Exported URLs contain graph data readable by anyone with the link.

The bridge currently sends server state to the viewer, not manual viewer edits back to the server. Later tool updates restore session-owned layers while preserving other user layers. Inline MCP Apps support depends on the host and is not implemented by this bridge. This server does not host or publish the frontend.

See `mcp-server/README.md` for setup details. Run `npm test --prefix mcp-server` with loopback networking permitted to verify protocol, session, bridge, and export behavior.


## Extended equation input

The local viewer also accepts parametric tuples in t (curves) or u,v (3D surfaces), with trailing ranges, such as `(cos(t),sin(t),t/3){0<t<4*pi}`. Use `(3*cos(t),2*sin(t)){0<t<2*pi}` for a 2D ellipse. Use full equations with restrictions (`z=x+y{x>0}`), shaded 2D inequalities (`x^2+y^2<=9`), or boundaries of bounded 3D solids (`x^2+y^2+z^2<=9{z>0}`). Parametric ranges default to 0–1. The local sidebar supports named finite constants such as a=2 and dependent definitions, with basic numeric sliders. Explicit piecewise expressions such as y={x<0:-x,x>=0:x} and z={x<0:sin(y),cos(y)} are supported, including trailing restrictions. The chatbot can simplify, differentiate and solve real numeric-coefficient polynomials of degree at most two; general symbolic solving is not implemented. Piecewise operands are supported in 2D/3D inequalities. Scene tools resolve new and existing parameter definitions atomically; setParameter updates values and slider bounds. Sliders support explicit Play/Pause. Do not claim complete Desmos compatibility.


## Reusable definitions and lists

The local viewer supports reusable function definitions such as `f(t)=t^2` and `g(u,v)=sin(u)+cos(v)`, nested calls, and numeric parameter coefficients. A one-argument function graphs in 2D; a two-argument function graphs as a height surface in 3D. Function definitions remain callable from other equations. Recursion and unbounded expansion are rejected.

Numeric lists such as `a=[1,2,3]`, integer unit ranges `[1...5]`, and one-based indexing `a[2]` are supported. Equations containing lists expand elementwise; combined lists must have equal lengths. Lists are bounded to100 elements. Do not generate nested lists or comprehensions.

Symbolic tools also support multivariable simplification/partial derivatives and polynomial integration through degree8 with symbolic coefficients. Integration returns an antiderivative plus an arbitrary-constant/domain note. This is not unrestricted integration or general equation solving.
