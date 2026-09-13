# Graphly Project Roadmap

## Overview
Graphly is a precision graphing calculator and mathematical visualization platform designed for clean, responsive, and functional computation.

---

## Phase Status

- [x] **Phase 1: Triage + brutalist UI pass**
  - Fix mobile touch-action and pinch/pan conflicts with native viewport scrolling.
  - Hardened tick generation, domain range padding, and number formatting for extreme scientific magnitudes and edge cases.
  - Pure brutalist aesthetic redesign: high-contrast `#000000` / `#FFFFFF` base, single electric blue `#0044FF` accent for active states, zero border-radius, exposed 1–2px grid borders, stark Inter (UI) & JetBrains Mono (data/equations) typography.
  - Complete removal of decorative animations, glassmorphism, floating shadows, and pill-shaped gradients.

- [x] **Phase 2: Split into data mode vs. function mode**
  - Restructure app into two structurally separate modes sharing the same brutalist shell.
  - **Data Mode**: Existing AI-scanned table + CSV import + Recharts scatter plot + linear/polynomial regression + Mean/StdDev/R² analysis preserved with full functional parity.
  - **Function Mode**: Dedicated component (`FunctionGraph.jsx`) with brutalist equation input bar (`f(x) =`), active function list with visibility toggles, viewport bounds panel, and empty canvas placeholder ready for Phase 3's Canvas 2D engine.
  - Brutalist two-way mode switcher toggle (`[ DATA ] [ FUNCTION ]`) integrated into the primary navigation bar.

- [x] **Phase 3: Canvas-based function renderer (pan/zoom, infinite curves, asymptotes)**
  - Viewport state as `{ xMin, xMax, yMin, yMax }` in application state.
  - Resample function across visible x-range on every frame at ~1 sample per horizontal pixel.
  - Asymptote and discontinuity detection (no false vertical connecting lines for $1/x$, $\tan(x)$).
  - Unified pointer events pan/zoom toward cursor or touch point with pointer capture.
  - Adaptive gridline spacing and tick labels in JetBrains Mono at any zoom level.

- [x] **Phase 4: Real math parser (mathjs) + implicit equations**
  - Integrated `mathjs` for robust implicit multiplication (`2x`, `2(x+1)`), function library, and safe evaluation.
  - Distinct Marching Squares zero-crossing contour algorithm on a $100 \times 100$ grid with linear edge interpolation for implicit equations (e.g. $x^2 + y^2 = 25$, $x^2/16 + y^2/4 = 1$).

- [x] **Phase 5: 3D graphs (Three.js)**
  - Explicit 3D surfaces $z = f(x, y)$ rendered as dynamic meshes using Three.js and `mathjs`.
  - Full OrbitControls integration (rotate, pan, zoom with mouse and touch).
  - Brutalist 3D aesthetic: flat-shaded MeshStandardMaterial, stark monochrome base with `#0044FF` accent, sharp black Cartesian coordinate axes, and exposed facet wireframe overlay.

- [x] **Phase 6: AI chatbox wired to real app actions**
  - Integrated Google Generative AI SDK function calling / tool declarations (`plotFunction`, `plotImplicitEquation`, `loadDataTable`, `switchTo3D`, `setViewportBounds`).
  - Action transcript and tool execution log embedded directly in the brutalist chat interface.
  - Zero-key heuristic intent engine ensuring immediate local testability.

- [x] **Phase 7: Shareable-link plotting for external AI tools (MCP / GPT Actions)**
  - State hydration in client from `?state=<base64>` and query parameters.
  - Public `/api/plot` endpoint returning clickable URLs for 2D, 3D, and tabular data.
  - Standalone Node MCP Server for Claude Desktop with tools `plot_function`, `plot_data_table`, and `plot_3d_surface`.
  - OpenAPI 3.1 schema for ChatGPT custom GPT Actions in `docs/openapi-gpt-action.json`.

---

## Prospective Future Phases

- [ ] **Phase 8: Real-Time WebSocket Session Pairing**
  - Bi-directional socket pairing between external AI agents and live active browser tabs.
  - Interactive whiteboard synchronization with pairing PIN codes.
