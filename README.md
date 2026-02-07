# **Graphly 📊**

Graphly is a powerful, standalone React application for creating, analyzing, and exporting scientific graphs. It runs entirely in the browser using `localStorage` for persistence, meaning no backend or database configuration is required.

It features an **AI-powered scanner** that turns images of data tables into editable graphs instantly using Google Gemini 2.5 Flash. Users can bring their own free API key for secure, private usage.

![Graphly Demo](https://github.com/user-attachments/assets/cabd606e-fb35-4c74-823c-3c3ba2c42cbc)

## **✨ Features**

### **Core Functionality**
* **🤖 AI Data Scanning:** Upload a photo of a datasheet or handwritten table, and Graphly extracts the data automatically.
* **📈 Mathematical Functions:** Plot complex functions like `sin(x) * x`, `log(x)`, or polynomials just by typing the equation.
* **📊 Rich Analysis:** Automatically calculates Mean, Standard Deviation, and R² (Coefficient of Determination) for your datasets.
* **📥 Import & Export:**
  * Import data via **CSV** or paste directly from Excel.
  * Export graphs as high-quality **PDFs** (via native print).
  * Export datasets as **CSV** files.

---

## **🚀 Getting Started**

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
   * **Note:** This is only for local convenience. The app now supports entering your key directly in the UI, which is saved securely in your browser's Local Storage.

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
