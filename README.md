# **Graphly 📊**

Graphly is a powerful, standalone React application for creating, analyzing, and exporting scientific graphs. It runs entirely in the browser using `localStorage` for persistence, meaning no backend or database configuration is required.

It features an **AI-powered scanner** that turns images of data tables into editable graphs instantly using Google Gemini 3.5 Flash. Users can bring their own free API key for secure, private usage.

![Graphly Demo](https://github.com/user-attachments/assets/cabd606e-fb35-4c74-823c-3c3ba2c42cbc)

## **Features**

### **Core Functionality**
* **🤖 AI Data Scanning:** Upload a photo of a datasheet or handwritten table, and Graphly extracts the data automatically.
* **📈 Mathematical Functions:** Plot complex functions like `sin(x) * x`, `log(x)`, or polynomials just by typing the equation.
* **📊 Rich Analysis:** Automatically calculates Mean, Standard Deviation, and R² (Coefficient of Determination) for your datasets.
* **📥 Import & Export:**
  * Import data via **CSV** or paste directly from Excel.
  * Export graphs as high-quality **PDFs** (via native print).
  * Export datasets as **CSV** files.

---

## Connect AI assistants

The local Node.js MCP server supports live graph sessions: create a graph once, then append or edit individual equations in the same browser tab. See [MCP setup and tools](mcp-server/README.md) and the optional [Codex / Claude Code skill plugin](integrations/graphly/README.md).

Run the updated frontend locally before using the new tools; the hosted website needs this build deployed before it can render live sessions. Client configuration and inline viewer support vary. The existing OpenAPI endpoint and legacy plot tools generate standalone links.

## Equation support and current limits

Enter full equations such as `x^2/9+y^2/4=1` in 2D or `x^2/9+y^2/4+z^2=1` in 3D. Explicit surfaces such as `z=x^2+y^2` also work. New assistant plots append layers; targeted edits use layer IDs. Draft layers are restored from this browser's local storage.

Implicit surfaces are sampled in a finite domain on a background worker. This is a numerical renderer, not a symbolic solver: very small features and general singular zero sets can be missed. Repeated zero factors such as `(z-2)^2=0` are simplified before sampling. Parametric tuples, domain restrictions and inequalities are supported as described below. Live MCP state lasts for the server process; export a snapshot to retain all equations and view settings. Browser edits are currently one-way and do not update MCP state.

Run `npm test`, `npm run test:mcp`, `npm run lint`, and `npm run build` to validate changes. MCP tests require loopback networking.

---

## **Getting Started**

Follow these instructions to run the project locally.

### **Prerequisites**
* [Node.js](https://nodejs.org/) (Version 18 or higher recommended)
* A free [Google Gemini API Key](https://aistudio.google.com/)

### **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tajwarbot/Graphly.git
   cd Graphly
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure Environment (Optional for Dev):**
   * Create a `.env` file in the root directory (copy from `.env.example`):
     ```bash
     cp .env.example .env
     ```
   * Open `.env` and paste your Gemini API key:
     ```env
     VITE_GEMINI_API_KEY=your_actual_api_key_here
     ```
   * **Note:** This is only for local convenience. The app now supports entering your key directly in the UI, which is saved in your browser's Local Storage.

4. **Run the development server:**
   ```bash
   npm run dev
   ```
   Open the link shown in your terminal (usually `http://localhost:5173`) to view the app.

---

## **🛠️ Building for Production**

To create a standalone build (perfect for Netlify, Vercel, or GitHub Pages):

```bash
npm run build
```

This will create a `dist` folder containing your optimized website.

### ⚠️ IMPORTANT: Deployment Security

**DO NOT** set `VITE_GEMINI_API_KEY` in your hosting provider's environment variables (e.g., Netlify/Vercel dashboard).

- If you set this variable in production, your API key will be embedded in the public code and visible to anyone.
- **Leave it empty.** The app is designed to prompt users to enter their own key (BYOK - Bring Your Own Key), which is stored safely in their browser.
- This ensures you incur no costs and leak no secrets.

---

## **📚 Tech Stack**

* **Framework:** [React 19](https://reactjs.org/) + [Vite](https://vite.dev/)
* **Charting:** [Recharts](https://recharts.org/)
* **Styling:** [Tailwind CSS](https://tailwindcss.com/) + `clsx` + `tailwind-merge`
* **Animations:** [Framer Motion](https://www.framer.com/motion/)
* **AI:** [Google Generative AI SDK](https://www.npmjs.com/package/@google/generative-ai)
* **Icons:** [Lucide React](https://lucide.dev/)

---

## **📄 License**

This project is licensed under the [MIT License](LICENSE).

## Assistant and file input

Open **Ask Graphly**, then **Connect AI** to follow the Google AI Studio key link and paste your key. With Gemini connected, the assistant can compose up to 12 surfaces per scene, edit or style named layers, remove requested layers, set bounds/camera, and use tool feedback to repair failed equations. Prompts such as “plot two intersecting planes” and “build a planet with rings” also have local recipes without a key. New plots preserve existing work.

Attach up to three PNG/JPEG/WebP images (5 MB each) or CSV, TSV, TXT, Markdown, and JSON files (256 KB each). CSV/TSV plotting works locally; interpretation of images and other text files requires Gemini. Attachments are sent to Google only when you send the message with a connected key. Unsupported types are rejected before reading. Do not treat a generated scene as an exact reconstruction of a reference image.

`src/lib/aiProvider.js` isolates the current Gemini provider from graph tools and the chat interface. Future provider adapters can translate the same tool schemas and function results without changing graph operations. Additional providers and frontier-model selection are not yet exposed. Provider requests have a timeout, the tool loop is bounded, and applied mutations are not replayed after connection errors.

The default 3D coordinate system has z up, with x and y on the horizontal plane. Full equations are accepted without solving for a dependent variable. Bare expressions retain the usual y=f(x) / z=f(x,y) interpretation. Surface gradients drive smooth normals, self-shadow artifacts are disabled, and the renderer sleeps when idle. Touch controls use one finger to rotate and two fingers to pan/zoom; physical-device testing is still recommended before release.


## More equation forms

- 2D parametric curve: `(3*cos(t),2*sin(t)){0<t<2*pi}`.
- 3D space curve: `(cos(t),sin(t),t/3){0<t<4*pi}`.
- 3D parametric surface: `((3+cos(v))*cos(u),(3+cos(v))*sin(u),sin(v)){0<u<2*pi}{0<v<2*pi}`.
- Restricted surface: `z=x^2+y^2{x^2+y^2<4}`. Trailing restriction groups intersect; comparisons may be chained.
- Shaded 2D region: `x^2+y^2<=9{x>0}`. Strict boundaries are dashed.
- 3D solid boundary: `x^2+y^2+z^2<=9{z>0}`. Solids are clipped to the finite sampling domain; strict/inclusive inequalities share the same visual boundary in 3D.
- Points: `(1,2)` or `(1,2,3)` in the corresponding mode.

Parameter domains default to 0–1; specify bounds for full trigonometric shapes. New 2D graphs use equal coordinate units. Parametric curves and surfaces are finite-resolution approximations; arbitrary functions, symbolic solving, parameter sliders, Boolean unions, piecewise notation and animations are not Desmos-equivalent yet.
