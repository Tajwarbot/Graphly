# Graphly development handoff

Last updated: 2026-09-14, Asia/Dhaka. Read this file and AGENTS.md before continuing.

## Current status — read this before historical notes

Latest local implementation includes the restored animated precision-plotter homepage and conceptual logo; duplicate equation bar removal; accessible MathML with tuple/piecewise cases; parameter definitions and dependencies across both renderers; configurable sliders with Play/Pause; parameter-aware AI scene tools and parameter updates; 2D/3D piecewise inequalities; bounded symbolic chat commands; original butterfly/flower/trefoil examples; recent image attachment followup context; responsive sidebar fixes; and viewport fitting/sampling improvements. All edits remain local and uncommitted on main in Downloads/Graphly. Latest checks: 147 app tests, 4 MCP tests, lint and build pass. Browser verification also caught and fixed a missing2D icon import introduced by JSX cleanup.

Live Gemini is now verified using the supplied key saved only in browser settings: successful tool calls with repair, and synthetic image table extraction with exact four rows/zero tool errors. Actual physical touch hardware remains unavailable. Never include the key in source or documentation. Reusable functions, bounded numeric lists/indexing/broadcasting, multivariable partial derivatives and degree8 polynomial integration are now implemented. Complete Desmos language compatibility remains broader: list comprehensions/nested lists and unrestricted symbolic integration/solving are not implemented; numerical geometry remains finite-resolution. Do not claim unrestricted mathematical art generation or universal parity. There is no remaining requirement to move files or delete a branch. Moving this Codex task into the ChatGPT project Tajwar is unsupported by current app tools.

The sections below retain development history and may describe earlier limitations that later completed sections supersede. Use this current status plus the latest entries, not historical “not implemented” statements, to resume.

## Location, ownership, and non-negotiable instructions

- User owns https://github.com/Tajwarbot/Graphly and published https://graphly.netlify.app.
- Authoritative working codebase: `/home/tajwar/Downloads/Graphly`. It was moved here at the user's explicit request. Do not edit a stale copy under the original Codex workspace.
- DO NOT push code, merge, deploy, or create another PR. User will push to main later. Keep all changes local.
- Local branch is `main`, HEAD `53dc2eb`, ahead of last-fetched `origin/main` by two commits. Remote tracking data is not proof of current GitHub state.
- Local commits: `53dc2eb Expand equation rendering and assistant tools; refine responsive Graphly UI`; `bce8619 Add implicit plotting, live MCP sessions, and additive graph workflows`. Last fetched origin/main: `7cb8ab4`.
- An earlier draft PR #1 was closed at the user's request; temporary branch `fix/graphly-equations-and-workspace` was safely deleted locally and remotely after retaining work on local main. No branch cleanup remains to do.
- The user explicitly permits parallel subagents. Coordinate file ownership to avoid overwriting edits.
- Avoid emojis, generic decorative icons, generic AI-looking marketing layouts, and a generic G logo. Preserve the project's mathematical identity and original brutalist personality while making it consistent.
- Preserve existing graph layers unless the user explicitly asks to replace/remove them. Tool success is not proof of correct rendering.
- Do not claim full Desmos parity, physical touch verification, or live Gemini verification without evidence.
- Filesystem sandbox may need write permission for Downloads; tool default cwd may still be `/home/tajwar/Documents/Codex/2026-09-14/can`. Always set command workdir explicitly.

## What the user is ultimately trying to achieve

A professional, smooth mathematical graphing application with the added AI, image/CSV import, and MCP features. Input should accept full equations, not require solving every expression for y or z. A user should describe an object or mathematical artwork and the chatbot should decompose it into equations, restrictions, colors, and surfaces, then build it incrementally. Codex/Claude Code integration should update the same live graph through local MCP rather than repeatedly opening one-plot URLs. Rendering, orientation, controls, mobile usability, and clarity should approach Desmos quality.

## Latest feedback that is STILL OPEN

1. Replace the generic G with a conceptual, modern mathematical logo (code-native SVG is appropriate).
2. Restore the homepage closer to the original published Graphly design, not the newly introduced generic minimal cards. Original screenshot: centered Graphly badge; large monospace “Precision Coordinate Plotter”; graph-paper grid; visible animated sine-like curves and sample dots; three strong cards for 2D, 3D, Scan/CSV; compact top navigation. Keep copy brief and animation lightweight, visible, and reduced-motion aware.
3. Make buttons consistent in size, border, typography, states and touch targets. The oversized lone hamburger was specifically criticized.
4. Remove the duplicate top 2D “Equation” input form because expressions can already be edited in the sidebar. Keep clear access to the sidebar.
5. Render Active 3D Surfaces equations in readable mathematical/LaTeX-style notation, with accessible plain text and sensible wrapping. Current legend shows raw strings such as `x^2+y^2+z^2<=9{z>0}`. Improve expression previews elsewhere as appropriate.
6. Add a visible chatbot status indicator: distinguish key configured, working, offline, and actual errors. Key presence alone must not be called verified connectivity.
7. Research official Desmos 2D/3D art techniques and test representative art in Graphly. Add missing capabilities that those examples require. Do not merely insert a canned demo and claim arbitrary artwork support.
8. Improve AI mathematical design planning and tool coverage: atomic multi-layer 2D scenes, curves, surfaces, restrictions, coherent colors/bounds/camera, repair after validation errors, preserving prior user data.
9. Implement free parameters/sliders, Desmos-style piecewise notation, and useful symbolic operations. These are currently NOT implemented. Define and communicate the supported subset honestly instead of promising unrestricted symbolic solving.
10. Verify responsive touch behavior on phone/tablet emulation; physical hardware and live Gemini image requests remain unverified. A real working key/device is required for those checks. Do not print/expose secrets.
11. Continue inspecting 3D appearance and orientation against Desmos. Major banding was fixed, but finite meshes, transparency, clipping, sampling and intersections still deserve visual checks.

## Completed work in the current local commits

### Equation input and mathematical rendering

- Full implicit 3D equations: spheres, ellipsoids, cylinders, planes, quadrics, rather than forcing positive/negative square-root halves.
- 2D implicit contouring and explicit functions; vertical relations and parametric ordered pairs.
- Parametric 3D curves in t and surfaces in u,v; ordered triples; trailing parameter ranges/restrictions. Default parameter interval is 0–1.
- Trailing restrictions such as `z=x+y{x>0}`, chained bounds, standard Unicode math normalization.
- Shaded 2D inequalities; 3D inequalities render boundaries of bounded solids with caps within the selected box.
- A bounded safe expression AST instead of arbitrary JavaScript evaluation. Function/operator allowlists and finite-value checks.
- Improved discontinuity rejection, marching-tetrahedra extraction, shared edge consistency, clipping and gradient normals. Repeated positive powers equal zero are reduced for cases like `(z-1.3)^2=0`.
- Equal-unit aspect ratio defaults for new 2D graphs; chart plot-area measurement rather than the entire outer container.
- Regression handling rejects degenerate/nonfinite data and produces meaningful statistics.

Examples supported now:

```text
x^2+y^2=9
x=2
(3*cos(t),2*sin(t)){0<t<2*pi}
x^2+y^2<=9
x^2/9+y^2/4+z^2=1
z=x+y
x^2+y^2+z^2<=9{z>0}
(cos(t),sin(t),t/3){0<t<4*pi}
((2+0.5*cos(v))*cos(u),(2+0.5*cos(v))*sin(u),0.5*sin(v)){0<u<2*pi}{0<v<2*pi}
```

Limitations: finite numerical resolution can miss tiny features; restricted 2D contour clipping is approximate; parametric restrictions are sampled; general symbolic solving, reusable named parameters/sliders and piecewise syntax remain absent. This is not the complete Desmos language (lists, comprehensions, reusable functions, regressions and other syntax need their own compatibility audit).

### 3D rendering and navigation

- Mathematical coordinates map to Three.js as `(x,z,-y)`, so mathematical z is up. Default view is chosen to show x/y on the ground plane and z vertically.
- Disabled self-shadow artifacts that caused severe surface banding. Regular implicit surfaces use gradient normals with fallback at singularities.
- Worker-based geometry generation, approximately 140 ms input debounce, obsolete-job cancellation.
- Demand-driven drawing instead of continuously rendering an idle scene; orbit damping still redraws while moving.
- Adaptive camera field of view for narrow screens; smaller positive-axis labels and grid.
- One-finger orbit and two-finger dolly/pan are configured. Mobile sidebar/backdrop and dynamic navigation height are implemented.
- Parametric curves/points render using line segments/points. Line appearance is still relatively basic.
- Empty/error surfaces report bounds/restriction problems instead of silently claiming success.

### AI assistant and imports

- Additive layer creation with stable IDs; targeted update/remove/style operations, viewport and 3D camera changes.
- Atomic 3D scene validation before appending layers.
- Bounded function-calling loop: up to 4 rounds/16 actions, returns actual tool results to Gemini so it can repair failures. Mutations are not blindly replayed after network failure.
- Current provider is Google Gemini via `@google/generative-ai`, local model configuration `gemini-2.5-flash`; adapter boundary exists for future providers. The user wants eventual frontier-provider support but simple settings are more important than speculative UI.
- Offline recipes include intersecting planes and several common shapes; these are not a substitute for general model reasoning.
- Chat attachments: PNG/JPEG/WebP images up to 5 MB; CSV/TSV/TXT/MD/JSON text up to 256 KB; up to 3 files in the UI. CSV/TSV local plotting works without a key. PDF is not implemented.
- CSV parser handles quoted fields and rejects invalid numeric rows. Do not assume other older CSV entry points are equally robust.
- Images are sent only on Send. Prior image attachments are not retained as vision context for subsequent turns; text history is bounded.
- API-key modal has direct AI Studio guidance, show/hide, local key-change event, focus handling. Key is used directly from the browser; no platform proxy was added.

### UI and persistence

- Shared import/API-settings styling and focus/Escape handling; responsive chat layout.
- Draft 2D/3D local storage, stable layer IDs, session-specific drafts and URL hydration guard against duplicate StrictMode initialization.
- New homepage and logo exist but were rejected in the latest feedback. Do not mark that design task complete.

### Local MCP integration

- Dependency-free Node 20+ stdio MCP in `mcp-server/`.
- `create_graph`, `upsert_expression`, `get_graph`, `remove_expression`, `set_view`, `export_graph`; legacy one-plot link tools retained.
- Open a live graph once; subsequent mutations update it via loopback SSE with Origin check and capability token.
- Omit expression_id to append; reuse it only to intentionally edit.
- Export gives a token-free snapshot URL that survives process restarts. Live graphs exist only during the MCP process lifetime.
- Manual viewer edits do NOT sync back to MCP; later session updates restore session-owned expressions while preserving unrelated user layers.
- This is a browser bridge, not universal inline MCP Apps support. Remote hosts/browser local-network policy may prevent loopback access.
- Configure `GRAPHLY_BASE_URL=http://127.0.0.1:5173` for this local frontend revision. Published site may lack local features.
- Read `AGENTS.md`, `mcp-server/README.md`, and `integrations/graphly/README.md`; skill source is `integrations/graphly/skills/graphly/SKILL.md`. Inspect actual hidden integration files before assuming package/install completeness.
- After relocating, verify any host MCP configuration uses `/home/tajwar/Downloads/Graphly/mcp-server/index.js`, not an obsolete path. Do not expose unrelated configuration secrets.

## Code map

- `src/App.jsx`: graph state, dataset sidebar, duplicate quick Equation form, 2D sampling, persistence, URL/live hydration, all AI mutation handlers, graph context, navigation.
- `src/components/ThreeDGraph.jsx`: worker lifecycle, mesh materials, orbit controls, axes, sidebar, Active Surfaces legend, viewport.
- `src/lib/expressionSyntax.js`: normalization, safe scalar AST, trailing restrictions, tuple parsing and parametric compilation. Currently treats the first brace as restrictions: piecewise requires deliberate parser changes.
- `src/lib/surfaceEngine.js`, `surface.worker.js`: explicit/implicit/parametric mesh extraction and normals.
- `src/lib/mathEngine.js`, `plot2D.js`, `implicitContours.js`, `inequalityRegions.js`: 2D expression routing, sampling, contouring and regions.
- `src/components/InequalityPlot.jsx`, `PlotMetrics.jsx`: chart overlays and plot geometry.
- `src/lib/graphState.js`, `sceneTools.js`: stable layers, style/scene/view validation.
- `src/components/AIChatbox.jsx` and `.css`: chat UI, provider/tool dispatch, attachments and status.
- `src/lib/aiProvider.js`, `chatTools.js`, `chatToolLoop.js`, `chatIntent.js`, `chatAttachments.js`: provider, tool schema, bounded loop, local intents, file validation.
- `src/components/HomePage.jsx`, `HomePage.css`, `MathBackground.jsx`, `GraphlyLogo.jsx`, `Logo.jsx`: homepage and branding.
- `src/components/ImportPage.jsx`, `ApiKeyModal.jsx`, `SetupUI.css`, `useModalFocus.js`: import and settings UI.
- `src/lib/graphSessionClient.js`, `mcp-server/index.js`, `mcp-server/session.js`: live integration.

## Proposed next implementation sequence (NOT yet done)

1. Restore homepage and conceptual SVG mark using original source (`git show 7cb8ab4:src/App.jsx` and original background files) and the published reference. Apply consistent control styles. Remove duplicate Equation form and unused quickFunction state/handler without removing sidebar functionality.
2. Add a safe mathematical display component. Consider mathjs AST to TeX plus KaTeX, or accessible MathML; no KaTeX dependency is currently installed. Handle tuples, restrictions and later piecewise syntax, with plain-text fallback. Use it in 3D legend and expression previews.
3. Add an expression environment module for named constants/definitions, dependency resolution and cycle/undefined errors. Integrate before 2D sampling AND before 3D worker compilation. Parameter rows should not generate geometry. Add persisted slider bounds/step and accessible keyboard/touch controls; update geometry while dragging with existing debounce. Validate scenes consistently with definitions rather than rejecting symbols prematurely. Keep t/u/v and coordinate variables reserved appropriately.
4. Extend expressionSyntax safely for piecewise `y={x<0:-x,x>=0:x}` and `z={x<0:sin(y),x>=0:cos(y)}`. Distinguish piecewise braces from trailing restrictions. Define first-matching branch and missing-branch behavior, prevent accidental connecting across discontinuities, and maintain AST security.
5. Implement bounded symbolic features with explicit supported scope: simplify, derivative, and exact/analytic polynomial solving up to degree 2 is a reasonable first deliverable. Unsupported expressions should return an honest error rather than a fabricated answer. Integrate in actual UI/chat tools, not just an unused helper.
6. Add atomic 2D scene handler and AI tool. Proposed contract from the previous agent: `onAdd2DScene(layers: [{equation,name,color,opacity}], {bounds:{xMin,xMax,yMin,yMax}})`, tool `add2DScene` with `layers,bounds`. This handler/schema has NOT been added. Mirror 3D atomic validation and stable ID returns.
7. Improve design prompt: plan components, choose curves/surfaces/regions, specify parameter bounds and restrictions, validate tools, repair failures, set useful camera/bounds, preserve other layers. Add original demos such as a parametric butterfly, layered rosette, and trefoil ribbon/centerline. Verify them numerically and visually before claiming support.
8. Complete responsive/browser checks and live image test if a key is configured. Do not confuse emulated touch with physical-device testing. Record exact remaining limitations here and in final response.

Suggested parallel ownership: one agent homepage/logo; one chatbot/tool schemas; one expressionSyntax/symbolic tests; parent App, ThreeDGraph, parameter environment and math display. Agree contracts first. Previous agents exited with usage limits, not completed patches.

## Verification status and commands

Prior completed development turn reported: 117 app tests across 11 files, 4 MCP tests, ESLint and production build passed. These were NOT rerun during this documentation-only turn. Re-run after implementation changes.

```bash
cd /home/tajwar/Downloads/Graphly
npm test
npm run test:mcp
npm run lint
npm run build
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

MCP tests require loopback networking permission. Dev server previously ran at 127.0.0.1:5173; check whether it is still alive before starting a duplicate. The main bundle was approximately 1.94 MB (552 KB gzip), worker about 676 KB; build warned about size and stale Browserslist. Consider meaningful lazy loading later, not blind dependency upgrades.

Existing tests cover math parsing, regressions, render/UI, chatbot, scene tools, advanced equations, inequality regions, implicit contours, mesh topology, and MCP session/security/protocol behavior. Add behavior-focused tests for parameters, dependencies, piecewise seams, symbolic unsupported cases, and representative art.

Previous browser checks: desktop/390x844 phone/820x1180 tablet layouts; import/settings; intersecting planes preserving previous layers; local CSV attachment; parametric ellipse with shaded disk; 3D torus/helix and hemispherical solid. No physical device was tested. No live Gemini image request was verified because a working key was not available then; check current UI status without disclosing key contents. The latest design changes requested by the user have not been visually implemented or tested.

## References and visual evidence

- Published homepage: https://graphly.netlify.app (a web-fetch attempt failed URL safety; use ordinary browser inspection or repository source, not a workaround for any actual browser warning).
- Repo: https://github.com/Tajwarbot/Graphly
- Desmos art: https://www.desmos.com/art
- Desmos 3D: https://www.desmos.com/3d
- Official 3D guide: https://help.desmos.com/hc/en-us/articles/19736835727885-Extending-from-2D-to-3D
- Gemini function calls: https://ai.google.dev/gemini-api/docs/function-calling
- Gemini requests: https://ai.google.dev/gemini-api/docs/generate-content

User screenshots may only exist on the original machine; preserve their descriptions above if these files disappear:
- `/tmp/codex-clipboard-23f7b68f-a0fa-4599-88d6-32c35ac91743.png`: desired published homepage character/layout.
- `/tmp/codex-clipboard-b08ff6c9-c80d-43fc-99c5-72aaa93c6080.png`: inconsistent hamburger control.
- `/tmp/codex-clipboard-921021f9-703e-4196-b9ce-5a091aa17d75.png`: redundant Equation bar.
- `/tmp/codex-clipboard-f67d9479-0893-467d-aa1f-97f72cc13053.png`: raw equation legend.
- `/tmp/codex-clipboard-454077d6-0c20-4d3a-a5e7-792031ed38ec.png`: earlier 3D banding/shadow artifacts.
- `/tmp/codex-clipboard-b1c761a4-c691-42e3-9889-13988841593d.png` and `/tmp/codex-clipboard-a2c35929-10e5-4974-9008-82436f9d2dfd.png`: earlier import/API-key modal consistency concerns.

## Suggested prompt for the next agent

“Read context.md and AGENTS.md in /home/tajwar/Downloads/Graphly. Continue the outstanding Graphly UI, equation-engine and chatbot work listed there. Inspect current git status before editing; distinguish proposed work from implemented code. Keep everything local and do not push/deploy. Use parallel agents if useful, coordinate file ownership, verify representative mathematical artwork in the browser, and update context.md with actual changes and test results.”

## Completed follow-up: expression controls and math display (2026-09-14)

Removed the duplicate top 2D Equation form and its unused state/handler. Sidebar editing remains; mobile access is a labeled Expressions button. The mobile 3D toggle now says Surfaces/Close. Added `src/components/MathExpression.jsx`, a React MathML renderer for powers, fractions, roots and relations with accessible expression labels and plain-text fallback; applied it to the Active 3D Surfaces legend without raw HTML injection. Lint and all 12 app-render tests passed. Browser visual review remains pending. This supersedes open items 4 and the basic legend portion of item 5 above; richer tuple/piecewise display remains possible follow-up.

### Homepage correction completed
- Restored centered Precision Coordinate Plotter hero, compact outlined 2D/3D/import actions, black primary action and restrained mathematical typography based on the original published design.
- Replaced the generic G mark with a custom coordinate-plane, curve and projection SVG logo. Existing Logo exports remain compatible.
- Added visible light harmonic traces, coordinate samples and projection lines. CSS animation respects reduced motion; mobile controls and saved graph accessibility preserved.
- Changed only HomePage.jsx, HomePage.css and GraphlyLogo.jsx for this correction. Local work in Downloads/Graphly; no push.
- Validation: component ESLint passed and all 12 app-render smoke tests passed. Live website fetch was unavailable; original commit 7cb8ab4 supplied design reference.

## Chat art tools and status (agent completed, staged)
- Research: official Desmos Math Art and Parametric Equations help pages describe restrictions/colors/transforms, ordered-pair t curves and u/v surfaces. Recipes use standard mathematical techniques and original compositions, not copied contest artwork.
- Added atomic `add2DScene` schema and AIChatbox dispatch to `onAdd2DScene(layers, {bounds})`. Existing 3D scene contract unchanged.
- Added offline butterfly curve, layered flower, and trefoil ribbon/centreline recipes. Prompt now decomposes artwork into coherent parts, palette and framed scene.
- Chat launch/header show Offline / AI ready (key configured, not claimed network connection) / Working. Corrected parametric 2D tool validation to use actual plot sampler.
- Four new geometry/intent tests pass; changed files lint clean in temporary overlay. Live Gemini response not verified: key is browser-local; use Connect AI and submit a design prompt. Never print keys.
- Files staged in /tmp/graphly-chat-art because Downloads/Graphly is read-only for agent. Parent must apply only artExamples.js, artExamples.test.js, chatIntent.js, chatTools.js, aiProvider.js, AIChatbox.jsx and run integrated tests.

### Piecewise and bounded symbolic engine update (local, verified)

- Added safe scalar piecewise notation, e.g. `y={x<0:-x,x>=0:x}` and `z={x<0:sin(y),x>=0:cos(y)}`. Supports nested branches, arithmetic around branches, function argument commas, strict/inclusive/equality and chained comparisons, first-match semantics, optional final default, and NaN for uncovered domains. Unchosen numeric branches are not evaluated. Existing AST allowlist still validates every branch.
- Trailing restriction braces remain distinct from colon-containing piecewise braces; parametric tuples now ignore commas inside branch braces. Added `splitRelation` and `splitEquation` for top-level routing; integrated mathEngine/surfaceEngine/isInequality so comparisons inside branches do not turn scalar graphs into inequality regions.
- New `src/lib/symbolicMath.js` exports `symbolicMath({operation,expression,variable})`: simplify, differentiate (also accepts derivative), or solve linear/quadratic numeric-coefficient equations over real numbers. Derivative/simplify return expression strings; solve returns symbolic formula strings plus numeric approximations, or an explicit solution-set description. Unsupported transcendental/higher-degree/rational-variable/complex problems throw explanatory errors. This is a bounded helper, not universal CAS. Parent still owns UI/tool integration.
- Added seven regression tests in `src/lib/piecewise.test.js`; full `npm test -- --reporter=dot` passed **124/124**, 12 files. Tests include full 2D sampling and 3D mesh generation for piecewise equations, unsafe-input rejection, nested/strict branch behavior, and symbolic results. No push/deploy.
- Limits: arbitrary piecewise RHS within inequality-region parsing is not yet supported; discontinuous piecewise surface boundaries use the existing finite-resolution mesh sampler (not exact analytic branch-boundary clipping). Symbolic solver coefficients use JavaScript numbers; exact radical formula text is based on those coefficients and is not arbitrary-precision algebra.

## Completed follow-up: shared parameters and symbolic chat integration

Added `expressionEnvironment.js` with safe finite named constants, out-of-order dependency resolution, duplicate/cycle errors and token substitution. 2D sampling, inequality overlays and 3D worker input now resolve definitions such as `a=2` and `b=2*a`. Numeric definition rows expose keyboard/touch native range sliders in both sidebars and persist through existing equation state. Current sliders use a default ±10 extent (expanded for larger values) and .01 step; custom bounds/step and playback are not yet implemented. Parameter rows do not generate geometry. Three environment tests pass. AI scene creation has not yet been made parameter-definition aware.

Connected the bounded symbolic helper to the chatbot tool schema and local commands: `simplify x+x`, `differentiate sin(x)`, `solve x^2-2=0`. Results do not mutate graph layers. Solver supports real linear/quadratic equations with numeric coefficients, not general symbolic systems. Latest combined validation before this integration: 131 app tests, lint and build passed; rerun final checks after integration.

## Completed follow-up: browser artwork and mobile fixes

Verified offline `solve x^2-2=0` returns roots through the actual chat UI. Verified “Build a 2D butterfly” appends an artwork layer while preserving a preexisting disk/ellipse. Browser inspection caught angular parametric sampling and a viewport fit bug: increased 2D parametric samples to 2,400 and made equal-unit bounds contain both requested axes, instead of clipping requested Y extent. Screenshot confirmed smooth full butterfly. At 390x844, inspected expression sidebar; corrected its close control overlapping Graph Settings by making it a separate labeled 44px row. Preview status is Offline: live Gemini/image verification remains blocked by absent configured browser key, not claimed as passed.

## Latest checkpoint (2026-09-14 23:21 local)

131 tests across 14 files, lint, and production build pass after artwork viewport corrections and symbolic integration. Updated AGENTS.md capability guidance. Tested numeric parameter row in the real phone-size sidebar: entering a=2 creates the slider, keyboard interaction changes its equation; corrected dataset name synchronization too. No GitHub push/commit/deployment performed this turn; changes are local and uncommitted. Remaining items include custom slider bounds/step/playback, AI parameter-scene validation, general symbolic solving, piecewise inequality operands, full compatibility beyond these bounded features, physical hardware and a live Gemini image request. Do not present these as completed Desmos parity. Earlier “not implemented” descriptions in this file are historical and superseded by the completed sections below them.

## Project organization request and additional visual check

User requested moving this task to the ChatGPT project Tajwar. list_projects verified Tajwar exists (g-p-6aa82f22cec48191bf83557180f3a696), but exposed app tools cannot reassign an existing Codex task to a ChatGPT project. No move or duplicate was performed; continue in Downloads/Graphly. Verified the trefoil scene adds two layers in the browser; an existing hemisphere occluded the result, so its visibility was toggled off for inspection (not deleted). Tablet 820x1180 shows the ribbon and centreline. Further camera/framing quality review is warranted. Extended MathML display to parametric tuples, which previously fell back to raw text. Do not assume full math typesetting for piecewise expressions yet.

## Completed follow-up: configurable sliders

Parameter sliders now expose min, max and step inputs in both 2D and 3D sidebars. Valid finite settings (min < max, step > 0) persist on their dataset/surface through the existing graph state; commit inputs on blur or Enter. This supersedes the previous default-only slider limitation. Playback remains unimplemented. Lint and 15 app-render/environment tests passed after this change.

## Completed chatbot parameter integration and attachment followups

Removed equation precompilation from AIChatbox tool dispatch so App validates with the current parameter environment. Added setParameter tool routed to onSetParameter(layerId, value, {min,max,step}); optional settings/value and required existing layer ID. Provider guidance now describes parameter scene definitions, piecewise formulas, slider configuration and bounded symbolic capabilities without claiming animation or general CAS. Up to two most recent attached turns are retained in memory and passed back as multimodal context for image/file followups; older payloads are removed, names retained. No browser/API key values inspected or printed; live Gemini remains unverified. Focused chatbot/art/tools tests: 17 passed; changed-file lint passes. No push.

## Completed follow-up: AI parameter scenes and slider playback

Scene validators now resolve existing and new parameter definitions together before applying 2D/3D additions. App single-expression/update handlers use the same environment; setParameter edits a definition and validates finite ordered bounds/positive step. Added tests for parameterized sphere, 2D scenes referring to existing definitions, cycles and invalid tool updates. ParameterSlider has explicit Play/Pause, 250ms updates so 3D worker debounce can complete, stops on unmount and hidden document, and uses stable persisted bounds. Playback is user initiated. Full check results follow below. Agent inequality task did not apply due account-limit approval rejection; piecewise inequality operands remain open.

## Completed follow-up: piecewise inequality operands

Parent implemented top-level comparison and comma splitting for restrictions/inequalities. Comparisons inside piecewise branches no longer split outer inequalities. Both y<={x<0:-x,x>=0:x} regions and z<={x<0:-x,x>=0:x} solids use safe scalar branch evaluation. New regression test checks containment, nonempty shaded polygons and nonempty 3D geometry. This supersedes the preceding unimplemented operand limitation. Finite-resolution boundary sampling remains approximate.

## Completed follow-up: mathematical piecewise display

MathExpression now recursively formats full equations, tuples, trailing restrictions and piecewise cases using native MathML rows. Browser verified the z={x<0:sin(y),cos(y)} legend shows two aligned branches and an otherwise label; accessible labels retain source text. Loaded a parameterized sphere alongside the piecewise surface in a fresh test snapshot. Latest full checks before formatting: 134 tests and lint passed.

## Playback and final verification checkpoint

Browser verified Play changes a=2 over time, Pause stops it, and manual entry restores a=2. The native legend visibly formats a sphere and piecewise cases with fractions/powers/cases. Renamed legend Active expressions because parameter definitions are not surfaces. MCP suite passes all 4 tests; last full application suite is 134 passing tests with clean lint. Browser is still Offline, so no live Gemini/image request has been claimed. Physical devices are unavailable to this session. Changes remain local and uncommitted, no push/deployment.

## Final browser correction

A defensive null-prototype change to the mathjs evaluation scope caused parameter evaluation failures; tests and the browser caught it immediately. Restored mathjs-compatible value scopes while retaining safe error lookup and reserved prototype-related names. Five environment tests pass again. Final screenshot shows the parameterized blue sphere intersecting the magenta piecewise surface, a=2 and no parameter error. Final full test/lint/build run started after this correction; record its actual results before handoff.

Final check results after the correction: **134/134 application tests pass (14 files), lint clean, production build passes**. MCP **4/4 passed** earlier in this turn and MCP files were unchanged afterward. Remaining build advisories: large main/worker bundle and stale Browserslist data. No code pushed or deployed. Live Gemini and physical device tests remain unverified for the reasons above.

## Decluttered graph controls (2026-09-14)

Updated only the render JSX in `src/App.jsx` and `src/components/ThreeDGraph.jsx`. Removed the duplicate 2D plot-color presentation, grouped equation presets and dataset/surface appearance controls into native details sections, and kept all equation inputs, sliders, settings, presets, color actions and delete actions wired to the existing handlers. Added accessible labels and 44px touch targets for compact color and mode controls. Lint and production build pass. The full application suite currently has 140/142 passing tests; two pre-existing symbolic formatting assertions disagree on multiplication-term ordering (`b * y` versus `y * b`). No push/commit/deploy.

## Live Gemini verified and next language work

User supplied a Gemini key; saved only through the local browser's API settings (not in source, docs, or command logs). Actual Gemini request added z=x and z=-x while preserving all three existing layers. First tool attempt used an invalid color; model received validation error and repaired it, proving live function-response repair. Image request still needs a live attachment test. NEVER copy the key into this handoff.

Implementing reusable named functions and bounded numeric lists in shared environment; tests and rendering integration are underway, not yet declare complete. User additionally requests decluttered 2D/3D pages preserving important controls, safe dead-code removal, and continuous context updates. Delegated bounded JSX cleanup and dead-file audit to gpt-5.6-luna agents as explicitly authorized. No push/commit/deploy.

## Safe dead-code audit (2026-09-14)

Removed four conclusively orphaned scaffold files: `src/components/ui/flickering-grid.jsx`, `src/components/ui/liquid-glass-card.jsx`, `src/assets/react.svg`, and `public/vite.svg`. Repository-wide reference search found no imports, dynamic references, HTML references, tests, or deployment configuration references for any of them. Runtime components and public API assets were left intact. Lint passed; an isolated writable copy passed 138/138 application tests and production build. Direct test/build execution in Downloads/Graphly was blocked by the sandbox's read-only `node_modules/.vite-temp`, not by the changes. No push/commit/deploy.

## Completed follow-up: bounded multivariable symbolic algebra (2026-09-14)

`src/lib/symbolicMath.js` now accepts single-letter symbolic constants alongside the selected single-letter variable, while retaining the plotting engine's parsed-AST allowlist. Simplify therefore supports multivariable expressions such as `a*x+a*x+b*y`; differentiate computes partial derivatives, treating every other permitted symbol as constant (for example, d/dx of `a*x^2+b*x*y` is `2*a*x+b*y`). Existing numeric-coefficient real linear/quadratic solve behavior is unchanged.

Added `integrate` (alias `integral`) for validated polynomial expressions in the selected variable through degree 8. Coefficients may contain permitted symbolic constants and safe scalar arithmetic; the response explicitly says to add arbitrary `C` and retain the original expression's domain. This is not a general integrator: transcendental, piecewise, negative/fractional powers, variable denominators, and polynomial degree above 8 are rejected. Symbolic constants remain single letters; multi-letter names are rejected before CAS evaluation. Focused symbolic and preexisting symbolic-operation tests pass: **12/12** across `src/lib/symbolicMath.test.js` and `src/lib/piecewise.test.js`. No push/commit/deploy.

## Completed follow-up: reusable functions and numeric lists

Shared environment now supports reusable one-to-three-argument definitions, nested calls, parameter coefficients, recursion/arity/undefined errors and bounded expansion. Matching-dimensional definitions graph directly: f(t)=t^2 in 2D; g(u,v)=u+v in 3D. Other functions remain reusable without generating inappropriate geometry. Added numeric lists a=[1,2,3], unit integer ranges [1...5], one-based constant indexing a[2], inline lists and equal-length lockstep broadcasting in 2D/3D and scene validation (max100 elements). Multiple 3D results retain their owning expression's styles; 2D curves are separated with null points and inequality overlays expand each variant. No list comprehensions/nested lists/general list-valued functions yet. Five language regression tests pass; latest full run143 tests and lint passed before final integration below.

Connected extended symbolic integrate operation to model tool schema and offline command parser; provider instructions now describe functions, lists, partial derivatives and polynomial integrals truthfully. No code push.

## Completed follow-up: adaptive curves and jump rejection

Added bounded adaptive midpoint refinement for explicit 2D curves and parametric 2D/3D curves. Unresolved jumps/poles produce breaks rather than connecting lines; sample budget is12,000 points with depth8. Explicit 3D meshes probe edge midpoints and reject triangles bridging jumps or poles. Tests verify a finite step curve has no diagonal segment, a parametric 1/t curve breaks, and a step surface has no slanted connecting triangles. Latest full application validation: **146 tests pass across17 files; lint clean**. These are finite numerical safeguards, not proof every arbitrarily small/high-frequency feature is resolved.

## Live image verification completed and cleanup regression fixed

The first live image attempt exposed missing ChevronDown import in new 2D details JSX; parent fixed it and browser verified the2D editor renders. A subsequent submission from the existing workspace was rejected by automatic approval review because it included existing graph context. Used a new isolated synthetic graph (only y=x, fresh chat history), so no existing user graph contents were sent. Live Gemini extracted image rows (0,0),(1,1),(2,4),(3,9), appended Vision test, preserved synthetic line, and returned1 update/0 errors. Browser verified every numeric table cell. The supplied key stays browser-local and is absent from this file/source. Screenshot also confirms f(t)=t^2, a=[-2,0,2], z=f(x)/4+a renders three separate surfaces with decluttered appearance/preset sections. Follow-up image-context check underway.

## Final cleanup and live follow-up verification

Confirmed Gemini retained the prior synthetic image without reattachment: follow-up answered all four exact rows and did not change graph data. At390x844 the cleaned2D sidebar keeps dataset statistics, graph type/trendline/axis selectors, expandable Style, and editable data rows available with no horizontal overflow in the screenshot. Verified missing ChevronDown import correction in live2D path. Browser screenshots confirm reusable functions/list-expanded3D surfaces. Actual physical touch hardware still not tested; do not label viewport emulation as physical-device verification.

Current completed cleanup removed only four reference-audited scaffold assets/components, retained core runtime/data features, and grouped secondary controls. Latest tests146/146, lint clean, production build and4/4MCP tests pass (final import fix verified by subsequent full tests/lint). Browser key is configured and live Gemini text+tool repair+image extraction+image followup are now VERIFIED. Historical Offline notes are superseded.

## Final language-scope correction and status

Fixed nested function argument capture by renaming bound arguments before expanding callees. Regression: a=2, f(t)=a*t, g(a)=f(a)+a must evaluate g(3)=9 (not12). **147/147 app tests pass across17 files; lint clean.** Latest build passes with existing bundle-size advisory; final rebuild follows this scope fix. Four MCP tests passed and MCP code unchanged. Browser viewport reset; kept3D function/list preview and closed isolated image-test tab. No commit/push/deploy. The remaining compatibility limits are explicit, not a claim of full Desmos parity: no nested-list/comprehension language, no universal symbolic solving/integration, and finite numerical feature resolution. Physical-device touch verification is unavailable; live Gemini image/text workflows are now verified.

Final production rebuild after scope correction passed; git diff --check clean. Final verified counts:147 app tests,4 MCP tests, lint, build. Source remains local in Downloads/Graphly.

## User-supplied logo and MathBackground replacement (2026-09-15)

User supplied `/tmp/codex-clipboard-408a3628-777d-4c2b-a79f-eb29ae0de21a.png` as the final logo/favicon and exact replacement canvas code. Copied the image unchanged to `public/graphly-logo.png`; GraphlyLogo now renders this asset, and index.html uses it as the PNG favicon. Removed the prior unreferenced favicon.svg after reference search. Replaced MathBackground.jsx with the supplied code (formatting only), connected it to HomePage in place of the previous SVG waves, removed stale SVG animation helpers/CSS, and kept interactive homepage content above the canvas. The exact supplied implementation continuously animates and does not include the prior reduced-motion behavior. Updated the existing logo smoke test to expect the supplied image. Lint,12 app-render tests and production build pass. Local preview had stopped; restarting for visual verification. No push/deploy.

## Marked 3D build failure and status layout fixed (2026-09-15)

The screenshot showed a global worker failure incorrectly attributed to an ellipsoid and Rendering remaining active. Found a lifecycle race: worker created before the 140ms debounce could fail, then the pending timer could turn loading back on and post to a failed worker. Extracted cancellable surfaceJob: creates workers only when the debounce fires, retries startup/message failure once with a fresh worker, handles synchronous failures, terminates on success/cancel/error, ignores late results, and ends hung builds after30s. Runtime errors now appear once at panel level with Retry rendering, preserving existing meshes; per-equation compile errors remain on their cards. Rendering status occupies its own row and Add Surface cannot shrink/wrap into it. Five regression tests cover cancellation, startup retry, repeated failure, timeout, late results; all17 targeted job/render tests and lint pass. Browser verified existing four-expression function/list/ellipsoid scene renders and loading clears. Screenshot alone cannot prove the original worker loading failure's external cause; fresh local load succeeds. Homepage visual verification also completed: supplied logo and canvas waves appear correctly. No graph data removed, no push/deploy.

Production build and git diff --check also pass after the marked-issue repair; existing large-bundle advisory remains.

## XZ grid and pre-push documentation completed (2026-09-15)

Added a default-visible mathematical XZ grid (y=0), rotating Three.js GridHelper into its XY plane to match Graphly's (x,z,-y) coordinate mapping. Independent Settings switches now read Show XY Plane Grid and Show XZ Plane Grid. Browser verified the vertical grid and both checked controls with existing graph data preserved.

Rewrote README.md with setup prerequisites, locked npm installation, equation examples, Gemini/attachment setup and privacy, Codex/Claude/generic stdio MCP commands, optional skill/plugin installation, live-session lifetime/ownership limits, production build/preview/Netlify instructions, validation commands, repository map, troubleshooting, and accurate capability limits. Checked Codex MCP/skill and Claude MCP commands against official documentation. README local links resolve. Replaced misleading environment example with empty optional key and explicit build exposure note; expanded .gitignore to exclude all .env variants except .env.example.

Pre-push checks:152 app tests and4 MCP tests pass, lint clean, production build passes (existing large-chunk/Browserslist advisories), git diff --check clean. Credential-pattern scan of tracked and unignored files found no Google-key/GitHub-token/private-key matches; this is a pattern scan, not a universal secret audit. MCP test first waited under restricted networking; stopped and reran successfully with loopback permission. All changes remain uncommitted on main in Downloads/Graphly; no staging, commit, fetch, push, or deploy performed. Existing main is ahead of last-fetched origin/main by2 commits; remote freshness has not been checked. User should review/stage/commit locally and check remote state before their later push. Physical-device verification and the documented broader compatibility limits remain.
