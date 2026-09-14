<p align="center"><img src="public/graphly-logo.png" width="88" alt="Graphly logo" /></p>
<h1 align="center">Graphly</h1>
<p align="center">Equations, data, and interactive 3D graphs—with an assistant that can work alongside you.</p>

Graphly is a React application for plotting mathematical expressions, exploring datasets, and composing multi-layer 3D scenes. Graphs run in your browser; Gemini assistance is optional. A local MCP server lets external assistants create and refine graphs in one live browser tab.

[Website](https://graphly.netlify.app) · [MCP reference](mcp-server/README.md) · [Agent integration](integrations/graphly/README.md) · [Development handoff](context.md)

> This README describes this checkout. The hosted website may run an earlier revision. Use the local viewer to test features before deploying them.

## Contents

- [Quick start](#quick-start)
- [Using Graphly](#using-graphly)
- [Equation examples](#equation-examples)
- [Connect Gemini](#connect-gemini)
- [Connect an assistant through MCP](#connect-an-assistant-through-mcp)
- [Install the optional skill or plugin](#install-the-optional-skill-or-plugin)
- [Production builds and hosting](#production-builds-and-hosting)
- [Development and validation](#development-and-validation)
- [Troubleshooting](#troubleshooting)
- [Scope and limitations](#scope-and-limitations)

## Quick start

Use Node.js **22.12+** and npm. Git is needed to clone the repository. A modern browser with WebGL support is required for 3D. No database, Gemini key, or MCP client is needed for ordinary plotting.

```bash
git clone https://github.com/Tajwarbot/Graphly.git
cd Graphly
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open **http://127.0.0.1:5173** and keep the terminal running. Stop the server with Ctrl+C. If you already have the repository, run these commands from that checkout instead of cloning again. `npm ci` installs the versions recorded in `package-lock.json`; use `npm install` when intentionally changing dependencies.

The fixed port makes MCP configuration predictable. If port 5173 is occupied, choose another port and update `GRAPHLY_BASE_URL` to match it exactly.

## Using Graphly

| Workspace | What to do |
| --- | --- |
| **2D Plotter** | Add equations or datasets in the expressions sidebar; edit data, styles, axes, and trendlines. |
| **3D Surface** | Add expressions as separate layers; change visibility, color, opacity, and wireframe appearance. |
| **Scan & CSV** | Import a table image using Gemini or import comma-separated data. |
| **Ask Graphly** | Describe a graph, request edits to existing layers, or attach supporting files. |

New assistant plots append layers. To replace or remove something, identify the layer explicitly. Work is retained in this browser's local storage; clearing site data removes local drafts and settings. Export important datasets or graph snapshots before clearing browser storage.

In 3D, **z points upward**. The XY ground grid and vertical XZ grid have independent switches in **Settings**. Drag to orbit; touch gestures use one finger to orbit and two fingers to pan or zoom. **Reset View** restores the camera. Appearance and preset sections expand when needed.

## Equation examples

You can enter full equations without rearranging them into `y=...` or `z=...`. Bare expressions mean `y=f(x)` in 2D and `z=f(x,y)` in 3D.

| Type | Mode | Input |
| --- | --- | --- |
| Function | 2D | `sin(x)` |
| Circle | 2D | `x^2+y^2=9` |
| Ellipse | 2D | `x^2/9+y^2/4=1` |
| Restricted region | 2D | `x^2+y^2<=9{x>0}` |
| Piecewise function | 2D | `y={x<0:-x,x>=0:x}` |
| Parametric ellipse | 2D | `(3*cos(t),2*sin(t)){0<t<2*pi}` |
| Ellipsoid | 3D | `x^2/9+y^2/4+z^2=1` |
| Plane | 3D | `x+y+z=2` |
| Paraboloid | 3D | `z=x^2+y^2` |
| Helix | 3D | `(cos(t),sin(t),t/3){0<t<4*pi}` |
| Bounded solid boundary | 3D | `x^2+y^2+z^2<=9{z>0}` |
| Point | 2D / 3D | `(1,2)` / `(1,2,3)` |

A parametric torus in 3D:

```text
((3+cos(v))*cos(u),(3+cos(v))*sin(u),sin(v)){0<u<2*pi}{0<v<2*pi}
```

Reusable definitions belong on separate expression rows. For example, these three rows generate three translated surfaces:

```text
f(t)=t^2
a=[-2,0,2]
z=f(x)/4+a
```

Numeric definitions such as `a=2` expose slider controls with adjustable bounds, step size, and Play/Pause. Numeric lists support `[1,2,3]`, integer unit ranges `[1...5]`, and one-based indexing such as `a[2]`. Combined lists must have equal lengths; the maximum list length is 100.

Parametric domains default to 0–1. Supply explicit ranges for complete circles, helices, or periodic surfaces. Trailing restriction groups intersect.

## Connect Gemini

1. Open **Ask Graphly**, then **Connect AI**, or open **API Key** where available.
2. Follow the link to [Google AI Studio](https://aistudio.google.com/apikey) and obtain an API key for your Google project.
3. Paste the key into Graphly and select **Save key**.
4. Try: “Plot two intersecting planes, keeping my existing layers.”

The current provider uses `gemini-2.5-flash`. Availability, quotas, and billing depend on your Google account. **AI ready** means a key is configured; it does not certify connectivity or remaining quota.

The key is stored in this browser's local storage. Requests go directly from the browser to Google and can include your prompt, graph context, recent conversation, and attachments. Local storage is not an encrypted secret vault. The local MCP server does not need this key; external assistants use their own model connection.

Without Gemini, manual plotting, CSV/TSV data handling, and supported local assistant commands/recipes remain available. Open-ended scene design and image interpretation need the connected provider.

### Attachments

Attach up to **three files** per message:

| Files | Limit per file |
| --- | --- |
| PNG, JPEG, WebP | 5 MB |
| CSV, TSV, TXT, Markdown, JSON | 256 KB |

Images can supply tables or visual references. Other text files can provide context for the assistant. Unsupported formats are rejected; PDF and spreadsheet workbook uploads are not implemented. Generated artwork is a mathematical approximation, not an exact reconstruction guarantee.

### Optional development environment

UI key entry is recommended. For local development only, `.env.example` documents `VITE_GEMINI_API_KEY`. Any `VITE_` variable used by frontend code can be embedded in the browser bundle: **never build or publish with a real key in this variable**. Production users should enter their own keys through the UI. No environment file is required for normal setup.

## Connect an assistant through MCP

There are three independent parts:

| Part | Purpose |
| --- | --- |
| Vite frontend / deployed viewer | Displays and edits graphs. |
| `mcp-server/index.js` | Exposes plotting tools to an MCP client over stdio. |
| Optional Graphly skill/plugin | Teaches the assistant how to use those tools well. |

Start the frontend using the quick-start command. Configure your client to launch the local server; the client owns that process. The server has **no npm dependencies**, needs Node.js 20+, and requires no API key. Using Node.js 22.12+ for both frontend and server avoids separate runtimes.

Replace `/absolute/path/to/Graphly` below with your actual checkout path. Use forward slashes or correctly escaped backslashes in JSON on Windows. If a desktop client cannot find `node`, use its full executable path.

### Codex

```bash
codex mcp add graphly --env GRAPHLY_BASE_URL=http://127.0.0.1:5173 -- node /absolute/path/to/Graphly/mcp-server/index.js
codex mcp list
```

Alternatively, merge this entry into your Codex `config.toml` without replacing other configuration:

```toml
[mcp_servers.graphly]
command = "node"
args = ["/absolute/path/to/Graphly/mcp-server/index.js"]

[mcp_servers.graphly.env]
GRAPHLY_BASE_URL = "http://127.0.0.1:5173"
```

Restart or reload the client after configuration changes. See the [official Codex MCP guide](https://developers.openai.com/codex/mcp) for configuration locations and client-specific controls.

### Claude Code

```bash
claude mcp add --env GRAPHLY_BASE_URL=http://127.0.0.1:5173 --transport stdio graphly -- node /absolute/path/to/Graphly/mcp-server/index.js
claude mcp get graphly
```

Use `/mcp` inside Claude Code to inspect the connection. See the [official Claude Code MCP guide](https://code.claude.com/docs/en/mcp) for user/project scope and configuration management.

### Other MCP clients

For clients accepting an `mcpServers` JSON configuration:

```json
{
  "mcpServers": {
    "graphly": {
      "command": "node",
      "args": ["/absolute/path/to/Graphly/mcp-server/index.js"],
      "env": {
        "GRAPHLY_BASE_URL": "http://127.0.0.1:5173"
      }
    }
  }
}
```

This is a **stdio server**, not a public HTTP MCP endpoint. Do not use the website URL as the server transport URL. Running `npm start --prefix mcp-server` manually starts the protocol process; it does not open the viewer and normally waits for JSON-RPC input.

### Live plotting workflow

Ask your assistant:

> Use Graphly to create a 3D graph containing an ellipsoid and two intersecting planes. Open its live viewer, then keep updating that same graph as we work.

| Tool | Purpose |
| --- | --- |
| `create_graph` | Create a 2D or 3D session and obtain its viewer link. |
| `upsert_expression` | Append an expression, or edit one using its existing ID. |
| `get_graph` | Read server-owned graph state and IDs. |
| `remove_expression` | Remove a specified layer. |
| `set_view` | Set finite bounds or the 3D camera. |
| `export_graph` | Produce a durable snapshot link. |

Omit `expression_id` when adding a new layer. Reuse it only for an intentional edit. Legacy `plot_function`, `plot_3d_surface`, and `plot_data_table` produce standalone links; prefer the live tools for iterative work.

The viewer receives updates over a loopback event stream. Keep both the frontend and the MCP client running. Sessions live only in the server process; export a snapshot before restarting it. A result marked **accepted** confirms storage, not successful rendering—check the viewer for equation errors.

Manual viewer edits do not synchronize back to the MCP server. Later tool updates restore session-owned layers, while unrelated viewer layers are preserved. The browser and MCP process must be on the same computer; remote/cloud clients may not reach the loopback bridge. Universal inline MCP Apps rendering is not implemented.

Keep live session links private. Snapshot links encode graph data and are readable by anyone who has them. See [MCP transport, privacy, and troubleshooting](mcp-server/README.md).

## Install the optional skill or plugin

Register MCP first. The plugin supplies instructions, not a second renderer or an automatically configured server.

### Claude Code plugin

Launch Claude Code with the bundled directory:

```bash
claude --plugin-dir /absolute/path/to/Graphly/integrations/graphly
```

The Claude manifest is `integrations/graphly/.claude-plugin/plugin.json`. This local-directory workflow does not require publishing to a marketplace.

### Codex skill

A portable option is to copy the bundled `integrations/graphly/skills/graphly` directory into your personal `~/.agents/skills` directory. On macOS/Linux, from the Graphly repository:

```bash
mkdir -p ~/.agents/skills
cp -R integrations/graphly/skills/graphly ~/.agents/skills/
```

If a `graphly` skill already exists there, compare it before replacing it. Restart the client and request the Graphly skill. Codex discovers personal skills under `~/.agents/skills`; see [official skill locations](https://developers.openai.com/codex/skills).

The repository also includes `integrations/graphly/.codex-plugin/plugin.json` for clients supporting local plugin installation. Import the **integration directory**, not the repository root, using that client's supported plugin workflow. Installing both the plugin and a separate copy of its skill can produce duplicate entries; choose one. Details are in [the integration guide](integrations/graphly/README.md).

## Production builds and hosting

From the repository root:

```bash
npm ci
npm run build
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

Open **http://127.0.0.1:4173** to inspect the compiled build. `dist/` is generated output and is ignored by Git. `vite preview` is a local build check, not a production hosting service. For MCP testing against this preview, set `GRAPHLY_BASE_URL` to `http://127.0.0.1:4173` and restart its server.

### Netlify

The checked-in `netlify.toml` specifies:

- Build command: `npm run build`
- Publish directory: `dist`
- Function directory: `netlify/functions`
- `/api/plot` route to the compatibility plotting function
- SPA fallback to `index.html`

Use a compatible Node version and leave `VITE_GEMINI_API_KEY` unset in build settings. Deploying the frontend does not host the local stdio MCP server. After deploying this revision, its origin can be used as `GRAPHLY_BASE_URL` when browser loopback permissions permit; the local viewer remains the simplest live-session setup.

For other static hosts, serve `dist` at the site root and configure an SPA fallback. The Netlify compatibility API needs an equivalent backend on other hosts. Subdirectory hosting needs asset/base-path adjustments; it is not configured by default.

## Development and validation

| Command | Result |
| --- | --- |
| `npm ci` | Install the locked dependency tree. |
| `npm run dev` | Start Vite development server. |
| `npm test` | Run application regression tests. |
| `npm run test:mcp` | Run MCP state, bridge, and protocol tests; requires loopback listeners. |
| `npm run lint` | Run ESLint. |
| `npm run build` | Compile production assets into `dist`. |
| `npm run preview` | Serve the compiled assets locally. |

Before committing:

```bash
npm test
npm run test:mcp
npm run lint
npm run build
git diff --check
git status --short
```

GitHub Actions runs installation, both test suites, lint, and the build on pushes and pull requests. Keep environment files, credentials, generated bundles, and `node_modules` out of commits. `.env.example` must contain only empty/example values. Review the diff and new files before staging; local validation does not automatically commit or push anything.

### Repository map

```text
src/components/        Workspaces, equation display, sliders, assistant UI
src/lib/               Expression language, numerical engines, AI tools, state
src/lib/surface.worker.js  Background 3D geometry generation
mcp-server/            Dependency-free stdio MCP server and tests
integrations/graphly/  Codex/Claude manifests and plotting skill
netlify/               Compatibility API function
public/                Logo and public integration assets
.github/workflows/     Continuous integration
AGENTS.md              Repository instructions for coding/plotting agents
context.md             Completed work, verification, and remaining limitations
```

The application uses React 19, Vite, Three.js, Recharts, mathjs, Tailwind CSS, and the Google Generative AI SDK. The provider adapter is in `src/lib/aiProvider.js`; graph operations are separate from provider requests. Additional model providers are not yet exposed in the UI.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `node` or `npm` is missing / build engine error | Install a compatible Node runtime and reopen the terminal or desktop client. |
| Port 5173 is occupied | Stop the other server or choose another port; update the MCP viewer origin too. |
| MCP tools do not appear | Check the absolute path and Node executable, inspect client MCP status, and restart the client. |
| Viewer opens but does not update | Keep the MCP process running; match `GRAPHLY_BASE_URL` exactly, including hostname/port; prefer the local viewer. |
| Old session link stops working | Sessions are process-local. Start a new session or open a previously exported snapshot. |
| 3D renderer fails | Use **Retry rendering**. Reload if the worker could not load; inspect the browser console if it persists. |
| Surface is incomplete or rendering times out | Tighten finite bounds, reduce mesh density, or simplify the expression. Very small features may be missed numerically. |
| AI is configured but requests fail | Check Google key permissions, quota, network access, and the error shown in chat. |
| Hosted app lacks a local feature | Deploy the matching frontend revision; changing MCP configuration does not deploy code. |
| Build reports a large-chunk warning | The current math/3D bundle is large. This advisory is separate from a failed build. |

## Scope and limitations

Graphly accepts many full equations, reusable functions, numeric lists, piecewise expressions, restrictions, parametric geometry, and animated numeric sliders. It is **not a complete Desmos implementation**.

- Numerical curves and meshes have finite sampling budgets; singularities, tiny features, and discontinuities are not universally resolved.
- Numeric list support does not include nested lists or comprehensions.
- Symbolic helpers support multivariable simplification/partial derivatives, polynomial integration through degree 8, and real numeric-coefficient linear/quadratic solving. General symbolic solving and arbitrary integration are not implemented.
- 3D inequalities display solid boundaries within finite bounds; strict and inclusive inequalities share the same boundary appearance.
- Live Gemini text, tool repair, image extraction, and image follow-up were verified during development. Those checks do not guarantee every prompt succeeds.
- Responsive viewport checks have been performed; physical phone/tablet touch hardware still needs release testing.

See [context.md](context.md) for the detailed handoff and current verification history.

## License

[MIT](LICENSE).
