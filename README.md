# **Graphly 📊**

Graphly is a powerful, standalone React application for creating, analyzing, and exporting scientific graphs. It runs entirely in the browser using localStorage for persistence, meaning no backend or database configuration is required.

It features an **AI-powered scanner** that turns images of data tables into editable graphs instantly using Google Gemini.


![Demo](https://github.com/user-attachments/assets/f0c78623-b568-430f-abab-fabf12141be7)


## **✨ Features**

* **🤖 AI Data Scanning:** Upload a photo of a datasheet or handwritten table, and Graphly extracts the data automatically.  
* **📈 Mathematical Functions:** Plot complex functions like sin(x) \* x, log(x), or polynomials just by typing the equation.  
* **📊 Rich Analysis:** Automatically calculates Mean, Standard Deviation, and R² (Coefficient of Determination) for your datasets.  
* **📥 Import & Export:**  
  * Import data via **CSV** or paste directly from Excel.  
  * Export graphs as high-quality **PDFs** (via native print).  
  * Export datasets as **CSV** files.  
* **📱 Mobile Friendly:** Fully responsive design with a slide-out sidebar and touch gestures for zooming/panning on phones and tablets.  
* **🔒 Privacy Focused:** All data stays on your device.

## **🚀 Getting Started**

Follow these instructions to run the project locally.

### **Prerequisites**

* [Node.js](https://nodejs.org/) (Version 16 or higher recommended)  
* A free [Google Gemini API Key](https://aistudio.google.com/)
* (To get a Gemini API key, go to [Google AI Studio](https://aistudio.google.com/), sign in with your Google account, and click "Create API key" (or find it in the Dashboard) to generate a new key, optionally linking it to a Google Cloud project for organization.)

### **Installation**

1. **Clone the repository:**  
   git clone https://github.com/Tajwarbot/Graphly.git  
   cd Graphly

2. **Install dependencies:**  
   npm install

3. **Configure API Key:**  
   * Open src/App.jsx in your code editor.  
   * Locate the configuration section at the top:  
     // \--- CONFIGURATION \---  
     const GEMINI\_API\_KEY \= "YOUR\_GEMINI\_API\_KEY\_HERE"; 

   * Replace "YOUR\_GEMINI\_API\_KEY\_HERE" with your actual API key string.
4. **Run the development server:**  
   npm run dev

   Open the link shown in your terminal (usually http://localhost:5173) to view the app.

## **🛠️ Building for Production**

To create a standalone build (perfect for dragging and dropping onto Netlify or Vercel):

npm run build

This will create a dist folder containing your optimized website.

## **📚 Tech Stack**

* **Framework:** [React](https://reactjs.org/) \+ [Vite](https://vitejs.dev/)  
* **Charting:** [Recharts](https://recharts.org/)  
* **Styling:** [Tailwind CSS](https://tailwindcss.com/)  
* **AI:** [Google Generative AI SDK](https://www.npmjs.com/package/@google/generative-ai)  
* **Icons:** [Lucide React](https://lucide.dev/)

## **📄 License**

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
