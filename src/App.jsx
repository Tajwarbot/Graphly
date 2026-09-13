import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
    Camera, BarChart2, FileText, Check, AlertCircle, RefreshCw,
    ChevronRight, Zap, Settings, Save, Download, Edit2, Plus, Trash2,
    LogOut, Layout, TrendingUp, Grid, Type, Palette, ZoomIn, Home,
    MoreVertical, Share2, ChevronLeft, Calculator, Move, MousePointer2,
    ArrowRightLeft, Eye, EyeOff, Table, Activity, X, FilePlus, FileSpreadsheet, StickyNote, Menu, Sigma, Info, RotateCcw, Minus, Box, Sliders
} from 'lucide-react';
import { GoogleGenerativeAI } from "@google/generative-ai";
import {
    Line, Area, XAxis, YAxis,
    CartesianGrid, ResponsiveContainer,
    Scatter, ComposedChart, ReferenceLine, ReferenceDot, Label, LabelList
} from 'recharts';
import { Key } from 'lucide-react';
import { ApiKeyModal } from './components/ApiKeyModal';
import { FunctionGraph } from './components/FunctionGraph';
import { ThreeDGraph } from './components/ThreeDGraph';
import { AIChatbox } from './components/AIChatbox';
import { Logo, LogoIcon } from './components/Logo';
import { MathBackground } from './components/MathBackground';

// SECURITY: Import security utilities for rate limiting, validation, and API key handling
import {
    rateLimiter,
    InputValidator,
    getGeminiApiKey,
    isApiKeyConfigured
} from './lib/security';

// --- CONFIGURATION ---
const LOCAL_STORAGE_KEY = "graphly_local_data";

// --- MATH & UTILS ---

const THEMES = [
    { name: "Cobalt Blue", color: "#2563EB", bg: "#EFF6FF", border: "#BFDBFE" },
    { name: "Crimson Red", color: "#DC2626", bg: "#FEF2F2", border: "#FECACA" },
    { name: "Emerald Green", color: "#059669", bg: "#ECFDF5", border: "#A7F3D0" },
    { name: "Vivid Amber", color: "#D97706", bg: "#FFFBEB", border: "#FDE68A" },
    { name: "Deep Violet", color: "#7C3AED", bg: "#F5F3FF", border: "#DDD6FE" },
    { name: "Teal Cyan", color: "#0891B2", bg: "#ECFEFF", border: "#A5F3FC" },
    { name: "Hot Rose", color: "#E11D48", bg: "#FFF1F2", border: "#FECDD3" },
    { name: "Midnight Black", color: "#18181B", bg: "#F4F4F5", border: "#E4E4E7" }
];

const TRENDLINE_THEMES = [
    { name: "Crimson", color: "#DC2626" },
    { name: "Cobalt", color: "#2563EB" },
    { name: "Emerald", color: "#059669" },
    { name: "Amber", color: "#D97706" },
    { name: "Violet", color: "#7C3AED" },
    { name: "Dark Slate", color: "#18181B" }
];
import { calculateNiceTicks, formatNumber, getRegressionParams, generateTrendlineData, calculateStats, generateFunctionPoints, compileMathFunction } from './lib/mathEngine.js';

const FUNCTION_PRESETS = [
    { label: 'sin(x)', expr: 'sin(x)' },
    { label: 'x² - 4', expr: 'x^2 - 4' },
    { label: '1 / x', expr: '1/x' },
    { label: 'cos(x) · x', expr: 'cos(x) * x' },
    { label: 'tan(x)', expr: 'tan(x)' },
    { label: 'e^(-x²)', expr: 'exp(-x^2)' },
    { label: '√x', expr: 'sqrt(x)' },
    { label: 'x³ - 3x', expr: 'x^3 - 3*x' }
];

// CSV Parser
const parseCSV = (text) => {
    const lines = text.trim().split(/\r\n|\n/);
    if (lines.length === 0) return [];

    const headers = lines[0].split(/,|;|\t/).map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(/,|;|\t/);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((h, idx) => {
                row[h] = isNaN(values[idx]) ? values[idx].trim() : parseFloat(values[idx]);
            });
            data.push(row);
        }
    }
    return { headers, data };
};

// --- REFINED MINIMALIST UI COMPONENTS ---

const Card = ({ children, className = "" }) => (
    <div className={`bg-white border border-neutral-200 rounded-lg shadow-xs ${className}`}>{children}</div>
);

const Button = ({ onClick, children, variant = "primary", disabled = false, icon: Icon, size = "md", className = "" }) => {
    const sizes = { 
        sm: "py-1.5 px-3 text-xs rounded-md", 
        md: "py-2 px-3.5 text-xs font-medium rounded-md", 
        lg: "py-2.5 px-5 text-sm font-medium rounded-lg", 
        icon: "p-2 rounded-md" 
    };
    const variants = {
        primary: "bg-neutral-900 text-white hover:bg-neutral-800 shadow-xs border border-neutral-900 transition-colors",
        secondary: "bg-white text-neutral-800 border border-neutral-300 hover:bg-neutral-50 shadow-xs transition-colors",
        accent: "bg-[#2563EB] text-white hover:bg-[#1D4ED8] shadow-xs border border-[#2563EB] transition-colors",
        danger: "bg-white text-red-600 border border-red-200 hover:bg-red-50 transition-colors",
        ghost: "bg-transparent text-neutral-700 hover:bg-neutral-100 transition-colors",
        glow: "bg-neutral-900 text-white border border-neutral-900 hover:bg-neutral-800 shadow-sm",
        glowSecondary: "bg-white text-neutral-900 border border-neutral-300 hover:bg-neutral-50 shadow-xs"
    };
    return (
        <button 
            onClick={onClick} 
            disabled={disabled} 
            className={`font-medium flex items-center justify-center gap-2 transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${sizes[size]} ${variants[variant]} ${className}`}
        >
            {Icon && <Icon size={size === 'sm' ? 14 : size === 'lg' ? 18 : 16} />}
            {children}
        </button>
    );
};

// Real-time Function Evaluator (restored from separated 2D toolset)
function FunctionEvaluator({ equation }) {
    const [testX, setTestX] = useState('0');

    const isValid = useMemo(() => {
        if (!equation || typeof equation !== 'string' || !equation.trim()) return false;
        try {
            const fn = compileMathFunction(equation);
            return !!fn;
        } catch {
            return false;
        }
    }, [equation]);

    const yIntercept = useMemo(() => {
        if (!isValid) return '—';
        try {
            const pts = generateFunctionPoints(equation, 0, 0, 1);
            if (pts && pts.length > 0 && pts[0].y !== null && !isNaN(pts[0].y)) {
                return formatNumber(pts[0].y);
            }
            return 'Undefined';
        } catch {
            return 'Undefined';
        }
    }, [equation, isValid]);

    const result = useMemo(() => {
        if (!isValid) return '—';
        try {
            const val = parseFloat(testX);
            if (isNaN(val)) return null;
            const pts = generateFunctionPoints(equation, val, val, 1);
            if (pts && pts.length > 0 && pts[0].y !== null && !isNaN(pts[0].y)) {
                return formatNumber(pts[0].y);
            }
            return 'Undefined';
        } catch {
            return 'Undefined';
        }
    }, [equation, testX, isValid]);

    return (
        <div className="p-2.5 bg-neutral-50/80 border border-neutral-200 rounded-lg space-y-2" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between text-xs gap-2">
                <span className="text-neutral-500 font-medium whitespace-nowrap">Y-Intercept (x = 0):</span>
                <span className="font-mono font-bold text-neutral-900 bg-white px-2 py-0.5 rounded border border-neutral-200 text-xs shadow-2xs truncate max-w-[55%] text-right">
                    {yIntercept ?? '—'}
                </span>
            </div>
            <div className="flex items-center gap-1.5 text-xs pt-1.5 border-t border-neutral-200">
                <span className="text-neutral-500 font-mono font-medium shrink-0">f(</span>
                <input
                    type="number"
                    step="any"
                    value={testX}
                    onChange={(e) => setTestX(e.target.value)}
                    className="w-14 px-1 py-0.5 text-xs font-mono bg-white border border-neutral-200 rounded outline-none focus:border-neutral-900 shadow-2xs text-center"
                    placeholder="0"
                />
                <span className="text-neutral-500 font-mono font-medium shrink-0">) =</span>
                <span className="font-mono font-bold text-blue-600 bg-white px-2 py-0.5 rounded border border-neutral-200 flex-1 text-right truncate shadow-2xs">
                    {result ?? '—'}
                </span>
            </div>
        </div>
    );
}

// --- MAIN APP ---

export default function App() {
    const [view, setView] = useState('dashboard');
    const [appMode, setAppMode] = useState('data'); // 'data' | 'function'
    const [functionList, setFunctionList] = useState([
        { id: 'fn-1', expression: 'sin(x)', color: '#0044FF', visible: true },
        { id: 'fn-2', expression: 'x^2 / 4', color: '#000000', visible: true }
    ]);
    const [viewportBounds, setViewportBounds] = useState({ xMin: -10, xMax: 10, yMin: -10, yMax: 10 });
    const [currentGraph, setCurrentGraph] = useState(null);
    const [isImporting, setIsImporting] = useState(false);
    const [showCSVModal, setShowCSVModal] = useState(false);
    const [csvText, setCsvText] = useState("");
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    // New State for graph dimensions to handle square aspect ratio
    const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });
    const graphContainerRef = useRef(null);
    const activePointers = useRef(new Map());
    const pinchStartDist = useRef(null);
    const pinchStartDomain = useRef(null);

    useEffect(() => {
        document.title = "Graphly";
    }, []);

    const [savedGraphs, setSavedGraphs] = useState([]);
    const [image, setImage] = useState(null);
    const [scanStatus, setScanStatus] = useState("idle");
    const [showApiKeyModal, setShowApiKeyModal] = useState(false);
    const fileInputRef = useRef(null);
    const csvFileInputRef = useRef(null);

    const [zoomDomain, setZoomDomain] = useState({ x: ['auto', 'auto'], y: ['auto', 'auto'] });
    const [isDragging, setIsDragging] = useState(false);
    const lastMousePos = useRef({ x: 0, y: 0 });
    const [expandedDatasetId, setExpandedDatasetId] = useState(null);
    const [showSettings, setShowSettings] = useState(false);

    const [selectedData, setSelectedData] = useState(null);

    // Cursor position for interactive background
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [cursorCoords, setCursorCoords] = useState(null);
    const [quickFunctionExpr, setQuickFunctionExpr] = useState('');

    useEffect(() => {
        const loadGraphs = () => {
            try {
                const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
                if (stored) {
                    setSavedGraphs(JSON.parse(stored));
                }
            } catch (e) {
                console.error("Failed to load local graphs:", e);
            }
        };
        loadGraphs();
    }, []);

    // Hydrate state from shareable URL (?state=... or ?mode=...&fn=...)
    useEffect(() => {
        try {
            const search = window.location.search;
            if (!search) return;
            const params = new URLSearchParams(search);
            const stateParam = params.get('state');
            const modeParam = params.get('mode');
            const fnParam = params.get('fn') || params.get('expression');

            if (stateParam) {
                let jsonStr = '';
                try {
                    jsonStr = decodeURIComponent(escape(atob(stateParam)));
                } catch {
                    jsonStr = atob(stateParam);
                }
                const parsed = JSON.parse(jsonStr);

                if (parsed.mode === '3d') {
                    if (parsed.expression) {
                        setThreeDEquation(parsed.expression);
                    }
                    setAppMode('3d');
                    setView('editor');
                } else if (parsed.mode === 'function' || parsed.mode === '2d') {
                    setAppMode('2d');
                    if (parsed.expression) {
                        handleAIPlotFunction(parsed.expression);
                    }
                    if (parsed.viewportBounds) {
                        setViewportBounds(parsed.viewportBounds);
                    }
                    setView('editor');
                } else if (parsed.mode === 'data') {
                    setAppMode('2d');
                    if (parsed.rows && Array.isArray(parsed.rows)) {
                        handleAILoadDataTable(parsed.name || "Shared Dataset", parsed.rows);
                    } else {
                        createBlankGraph();
                    }
                }
            } else if ((modeParam === 'function' || modeParam === '2d') && fnParam) {
                setAppMode('2d');
                handleAIPlotFunction(fnParam);
                setView('editor');
            } else if (modeParam === '3d') {
                if (fnParam) {
                    setThreeDEquation(fnParam);
                }
                setAppMode('3d');
                setView('editor');
            } else if (modeParam === 'data') {
                setAppMode('2d');
                createBlankGraph();
            }
        } catch (err) {
            console.warn("Could not hydrate graph state from URL:", err);
        }
    }, []);

    // Toggle landing-page class to prevent scrolling
    useEffect(() => {
        const isLandingPage = view === 'dashboard' && savedGraphs.length === 0;
        if (isLandingPage) {
            document.documentElement.classList.add('landing-page');
        } else {
            document.documentElement.classList.remove('landing-page');
        }
        return () => document.documentElement.classList.remove('landing-page');
    }, [view, savedGraphs.length]);

    const saveToLocalStorage = (graphs) => {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(graphs));
        setSavedGraphs(graphs);
    };

    const deleteGraph = (e, id) => {
        e.stopPropagation();
        if (confirm("Are you sure you want to delete this graph? This cannot be undone.")) {
            const newGraphs = savedGraphs.filter(g => g.id !== id);
            saveToLocalStorage(newGraphs);
            if (currentGraph?.id === id) {
                setView('dashboard');
                setCurrentGraph(null);
            }
        }
    };

    // Add Wheel Listener to Graph Container to prevent Browser Zoom
    useEffect(() => {
        const container = graphContainerRef.current;
        if (!container || !currentGraph) return;

        const onWheel = (e) => {
            // If User is holding Ctrl, prevent default browser zoom
            if (e.ctrlKey) {
                e.preventDefault();
            }
        };

        // Passive: false is crucial to be able to preventDefault
        container.addEventListener('wheel', onWheel, { passive: false });

        // Also Resize Observer for Aspect Ratio
        const resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                setContainerSize({ width: entry.contentRect.width, height: entry.contentRect.height });
            }
        });
        resizeObserver.observe(container);

        return () => {
            container.removeEventListener('wheel', onWheel);
            resizeObserver.disconnect();
        };
    }, [view, currentGraph]);

    const printGraph = () => {
        const svgElement = document.querySelector('.recharts-surface');
        if (!svgElement) {
            alert("Could not find graph to download.");
            return;
        }

        // Serialize SVG
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgElement);
        const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(svgBlob);

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const img = new Image();

        // High res for print
        img.onload = () => {
            const padding = 40;
            const titleHeight = 80;
            const width = img.width + (padding * 2);
            const height = img.height + titleHeight + padding;

            canvas.width = width;
            canvas.height = height;

            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, width, height);

            // Title
            ctx.font = "bold 36px sans-serif";
            ctx.fillStyle = "#334155";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(currentGraph.title || "Untitled Graph", width / 2, (titleHeight / 2) + 20);

            // Graph
            ctx.drawImage(img, padding, titleHeight);

            const pngUrl = canvas.toDataURL("image/png");

            // --- PRINTING VIA IFRAME ---
            // This isolates the print content from the main app UI
            const printFrame = document.createElement('iframe');
            printFrame.style.position = 'fixed';
            printFrame.style.right = '0';
            printFrame.style.bottom = '0';
            printFrame.style.width = '0';
            printFrame.style.height = '0';
            printFrame.style.border = '0';
            document.body.appendChild(printFrame);

            // Detect orientation based on screen aspect ratio
            const isLandscape = window.innerWidth > window.innerHeight;
            const pageSize = isLandscape ? 'landscape' : 'portrait';
            // A4 dimensions in mm
            const a4Width = isLandscape ? 297 : 210;
            const a4Height = isLandscape ? 210 : 297;

            const frameDoc = printFrame.contentWindow.document;
            frameDoc.open();
            frameDoc.write(`
            <html>
            <head>
                <style>
                    @page { 
                        size: A4 ${pageSize}; 
                        margin: 10mm; 
                    }
                    body { 
                        margin: 0; 
                        padding: 0; 
                        display: flex; 
                        justify-content: center; 
                        align-items: center; 
                        height: 100vh;
                        width: 100vw;
                        overflow: hidden;
                        background: white;
                    }
                    img {
                        max-width: calc(${a4Width}mm - 20mm);
                        max-height: calc(${a4Height}mm - 20mm);
                        object-fit: contain;
                    }
                </style>
            </head>
            <body>
                <img src="${pngUrl}" />
            </body>
            </html>
        `);
            frameDoc.close();

            // Wait for image to load in iframe then print
            printFrame.onload = () => {
                printFrame.contentWindow.focus();
                printFrame.contentWindow.print();

                // Cleanup
                setTimeout(() => {
                    document.body.removeChild(printFrame);
                    URL.revokeObjectURL(url);
                }, 1000);
            };
        };

        img.src = url;
    };

    const exportCSV = () => {
        if (!currentGraph || !currentGraph.datasets.length) return;
        let csvContent = "data:text/csv;charset=utf-8,";

        let headers = [];
        let maxRows = 0;

        currentGraph.datasets.forEach(ds => {
            headers.push(`${ds.name} (${ds.config.xKey})`);
            headers.push(`${ds.name} (${ds.config.yKey})`);
            if (ds.data.length > maxRows) maxRows = ds.data.length;
        });
        csvContent += headers.join(",") + "\r\n";

        for (let i = 0; i < maxRows; i++) {
            let row = [];
            currentGraph.datasets.forEach(ds => {
                const point = ds.data[i] || {};
                row.push(point[ds.config.xKey] !== undefined ? point[ds.config.xKey] : "");
                row.push(point[ds.config.yKey] !== undefined ? point[ds.config.yKey] : "");
            });
            csvContent += row.join(",") + "\r\n";
        }

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.href = encodedUri;
        link.download = `${currentGraph.title || "export"}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const createBlankGraph = () => {
        const newGraph = {
            title: "Untitled Graph",
            datasets: [{
                id: `ds-${Date.now()}`,
                name: "Dataset 1",
                data: [{ x: 0, y: 0 }],
                visible: true,
                color: THEMES[0].color,
                config: { type: 'scatter', xKey: 'x', yKey: 'y', showTrendline: false, trendlineType: 'linear', trendlineColor: '#ef4444' }
            }],
            // Default Labels to X and Y
            globalConfig: { showGrid: true, enableZoom: true, xAxisLabel: "X", yAxisLabel: "Y", aspectRatio: "auto", showLabels: false },
            annotations: [],
            createdAt: new Date().toISOString()
        };
        setAppMode('2d');
        setCurrentGraph(newGraph);
        setExpandedDatasetId(newGraph.datasets[0].id);
        setIsImporting(false);
        setView('editor');
    };

    const createFunctionGraph = (expression = 'sin(x) * x') => {
        const cleanExpr = expression.replace(/^(y\s*=\s*|f\(x\)\s*=\s*)/i, '').trim();
        const newDs = {
            id: `ds-${Date.now()}`,
            name: `f(x) = ${cleanExpr}`,
            data: [],
            equation: cleanExpr,
            visible: true,
            color: THEMES[0].color,
            config: {
                type: 'function',
                xKey: 'x',
                yKey: 'y',
                showTrendline: false,
                trendlineType: 'linear',
                trendlineColor: '#ef4444'
            }
        };
        const newGraph = {
            title: `Function: ${cleanExpr}`,
            datasets: [newDs],
            globalConfig: { showGrid: true, enableZoom: true, xAxisLabel: "X", yAxisLabel: "Y", aspectRatio: "auto", showLabels: false },
            annotations: [],
            createdAt: new Date().toISOString()
        };
        setAppMode('2d');
        setCurrentGraph(newGraph);
        setExpandedDatasetId(newDs.id);
        setIsImporting(false);
        setView('editor');
    };

    const startImport = () => {
        setIsImporting(true);
        setImage(null);
        setScanStatus("idle");
        setView('scan');
    };

    const handleCSVImport = (parsedData, name = "Imported Data") => {
        if (!parsedData || parsedData.data.length === 0) return;

        const keys = parsedData.headers;
        const xKey = keys[0] || 'x';
        const yKey = keys[1] || keys[0] || 'y';

        const newDataset = {
            id: `ds-${Date.now()}`,
            name: name,
            data: parsedData.data,
            visible: true,
            color: THEMES[(currentGraph?.datasets.length || 0) % THEMES.length].color,
            config: { type: 'scatter', xKey, yKey, showTrendline: false, trendlineType: 'linear', trendlineColor: '#ef4444' }
        };

        if (isImporting && currentGraph) {
            setCurrentGraph(prev => ({ ...prev, datasets: [...prev.datasets, newDataset] }));
            setView('editor');
        } else {
            const newGraph = {
                title: "Imported CSV Graph",
                datasets: [newDataset],
                // Default Labels to X and Y
                globalConfig: { showGrid: true, enableZoom: true, xAxisLabel: "X", yAxisLabel: "Y", aspectRatio: "auto", showLabels: false },
                annotations: [],
                createdAt: new Date().toISOString()
            };
            setCurrentGraph(newGraph);
            setExpandedDatasetId(newDataset.id);
            setView('editor');
        }
        setShowCSVModal(false);
        setIsImporting(false);
    };

    const handleScan = async () => {
        if (!image) return;

        // SECURITY: Check if API key is configured
        if (!isApiKeyConfigured()) {
            setShowApiKeyModal(true);
            return;
        }

        const apiKey = getGeminiApiKey();

        // SECURITY: Rate limiting - prevent API abuse
        const rateLimitResult = rateLimiter.tryConsume();
        if (!rateLimitResult.allowed) {
            setScanStatus("error");
            alert(rateLimitResult.message);
            return;
        }

        setScanStatus("scanning");
        try {
            // SECURITY: Use API key from environment variable
            const genAI = new GoogleGenerativeAI(apiKey);
            // Using gemini-3.5-flash — current GA flash model (1.5 retired Apr 2025)
            const model = genAI.getGenerativeModel({ model: "gemini-3.5-flash" });
            const prompt = `
        Analyze this datasheet/image. Extract tabular data accurately.
        If there are multiple tables or distinct sections, create separate datasets.
        
        CRITICAL: Return ONLY a valid JSON object. No markdown formatting, no backticks, no explanatory text.
        Structure:
        {
          "title": "Document Title",
          "datasets": [
            {
              "name": "Dataset Name",
              "data": [{"Column1": value, "Column2": value}, ...]
            }
          ]
        }
        Strip all units (e.g. "5V" -> 5). Return ONLY pure JSON.
      `;

            const result = await model.generateContent([prompt, { inlineData: { data: image.base64, mimeType: image.file.type } }]);

            if (!result.response) {
                throw new Error("No response from AI model. This might be due to safety filters or connection issues.");
            }

            let text = result.response.text();

            // Robust JSON Extraction: Find the search for { and } to extract the JSON part
            const firstBrace = text.indexOf('{');
            const lastBrace = text.lastIndexOf('}');

            if (firstBrace === -1 || lastBrace === -1) {
                console.error("No JSON object found in response:", text);
                throw new Error("The AI response did not contain a valid data structure. Please try again with a clearer image.");
            }

            text = text.substring(firstBrace, lastBrace + 1);

            // Further sanitization to remove any tricky bits (e.g. trailing commas if present)
            let parsed;
            try {
                parsed = JSON.parse(text);
            } catch (e) {
                console.error("JSON Parse Error:", e, "Original text:", text);
                // Attempt a second pass if simple quotes or trailing commas are an issue
                try {
                    // Very basic cleanup for common AI JSON mistakes
                    const cleanedText = text.replace(/,\s*([}\]])/g, '$1');
                    parsed = JSON.parse(cleanedText);
                } catch {
                    throw new Error("Failed to parse AI response. The data structure was malformed.");
                }
            }

            // SECURITY: Validate and sanitize title from API response
            const titleValidation = InputValidator.validateGraphTitle(parsed.title || "Untitled Graph");
            const sanitizedTitle = titleValidation.valid ? titleValidation.value : "Untitled Graph";

            const newDatasets = (parsed.datasets || [parsed]).map((ds, index) => {
                // SECURITY: Validate and sanitize dataset name
                const nameValidation = InputValidator.validateDatasetName(ds.name || `Dataset ${index + 1}`);
                const sanitizedName = nameValidation.valid ? nameValidation.value : `Dataset ${index + 1}`;

                // SECURITY: Validate and sanitize data array
                const dataValidation = InputValidator.validateDataArray(ds.data || []);
                const sanitizedData = dataValidation.valid ? dataValidation.value : [];

                if (sanitizedData.length === 0) {
                    console.warn(`[Security] Dataset ${index + 1} has no valid data after validation`);
                }

                const sample = sanitizedData[0] || {};
                const keys = Object.keys(sample);
                let xKey = keys[0] || 'x';
                let yKey = keys[1] || keys[0] || 'y';

                return {
                    id: `ds-${Date.now()}-${index}`,
                    name: sanitizedName,
                    data: sanitizedData,
                    visible: true,
                    color: THEMES[(index + (currentGraph?.datasets.length || 0)) % THEMES.length].color,
                    config: {
                        type: 'scatter',
                        xKey: xKey,
                        yKey: yKey,
                        showTrendline: false,
                        trendlineType: 'linear',
                        trendlineColor: '#ef4444'
                    }
                };
            });

            if (isImporting && currentGraph) {
                setCurrentGraph(prev => ({
                    ...prev,
                    datasets: [...prev.datasets, ...newDatasets]
                }));
                // SECURITY: Use sanitized title from validated response
                if (currentGraph.title === "Untitled Graph" && sanitizedTitle !== "Untitled Graph") {
                    setCurrentGraph(prev => ({ ...prev, title: sanitizedTitle }));
                }
            } else {
                const newGraph = {
                    // SECURITY: Use sanitized title
                    title: sanitizedTitle,
                    datasets: newDatasets,
                    globalConfig: {
                        showGrid: true,
                        enableZoom: true,
                        xAxisLabel: "X",
                        yAxisLabel: "Y",
                        aspectRatio: "auto",
                        showLabels: false
                    },
                    annotations: [],
                    createdAt: new Date().toISOString()
                };
                setCurrentGraph(newGraph);
                setExpandedDatasetId(newDatasets[0]?.id);
            }

            setScanStatus("idle");
            setView("editor");
            setIsImporting(false);
        } catch (e) {
            console.error(e);
            setScanStatus("error");
            alert(`Generation Error: ${e.message}`);
        }
    };

    const updateDataset = (id, updates) => {
        setCurrentGraph(prev => ({
            ...prev,
            datasets: prev.datasets.map(ds => ds.id === id ? { ...ds, ...updates } : ds)
        }));
    };

    const deleteDataset = (id, e) => {
        if (e) e.stopPropagation();

        if (currentGraph.datasets.length <= 1) {
            alert("You must have at least one dataset.");
            return;
        }

        setCurrentGraph(prev => ({
            ...prev,
            datasets: prev.datasets.filter(ds => ds.id !== id)
        }));
    };

    const addDataset = (type = 'data') => {
        const newId = `ds-${Date.now()}`;
        const newDs = {
            id: newId,
            name: type === 'function' ? 'New Function' : `New Dataset`,
            // Function datasets use equation string, normal use data array
            data: type === 'function' ? [] : [{ x: 0, y: 0 }],
            equation: type === 'function' ? 'x^2' : undefined,
            visible: true,
            color: THEMES[currentGraph.datasets.length % THEMES.length].color,
            config: {
                type: type === 'function' ? 'function' : 'line',
                xKey: 'x',
                yKey: 'y',
                showTrendline: false,
                trendlineType: 'linear',
                trendlineColor: '#ef4444'
            }
        };
        setCurrentGraph(prev => ({ ...prev, datasets: [...prev.datasets, newDs] }));
        setExpandedDatasetId(newId);
    };

    const addFunctionDataset = (expr) => {
        const cleanExpr = (expr || 'sin(x)').replace(/^(y\s*=\s*|f\(x\)\s*=\s*)/i, '').trim();
        const nextIdx = currentGraph?.datasets?.length || 0;
        const newId = `ds-${Date.now()}`;
        const newDs = {
            id: newId,
            name: `f(x) = ${cleanExpr}`,
            data: [],
            equation: cleanExpr,
            visible: true,
            color: THEMES[nextIdx % THEMES.length].color,
            config: {
                type: 'function',
                xKey: 'x',
                yKey: 'y',
                showTrendline: false,
                trendlineType: 'linear',
                trendlineColor: '#ef4444'
            }
        };
        if (currentGraph) {
            setCurrentGraph(prev => ({ ...prev, datasets: [...prev.datasets, newDs] }));
            setExpandedDatasetId(newId);
        } else {
            createFunctionGraph(cleanExpr);
        }
    };

    const handleQuickFunctionSubmit = (e) => {
        e?.preventDefault();
        if (!quickFunctionExpr.trim()) return;
        addFunctionDataset(quickFunctionExpr.trim());
        setQuickFunctionExpr('');
    };

    const saveGraph = async () => {
        if (!currentGraph) return;

        let updatedGraphs;
        let updatedCurrentGraph = { ...currentGraph };

        if (currentGraph.id) {
            updatedCurrentGraph.updatedAt = new Date().toISOString();
            updatedGraphs = savedGraphs.map(g => g.id === currentGraph.id ? updatedCurrentGraph : g);
        } else {
            updatedCurrentGraph.id = `graph-${Date.now()}`;
            updatedCurrentGraph.createdAt = new Date().toISOString();
            updatedGraphs = [updatedCurrentGraph, ...savedGraphs];
        }

        saveToLocalStorage(updatedGraphs);
        setCurrentGraph(updatedCurrentGraph);
        setView("dashboard");
    };

    // --- RENDER PREPARATION ---

    const globalBounds = useMemo(() => {
        if (!currentGraph) return { xMin: 0, xMax: 10, yMin: 0, yMax: 10 };
        let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
        let hasData = false;

        currentGraph.datasets.forEach(ds => {
            if (!ds.visible) return;

            // If dataset is a function, we don't use it to auto-scale bounds initially
            // unless we want it to fit a default range. 
            // For now, let's only scale to explicit data points.
            if (ds.config.type === 'function') {
                // Optional: Set a default range if ONLY functions exist
                return;
            }

            ds.data.forEach(d => {
                const x = parseFloat(d[ds.config.xKey]);
                const y = parseFloat(d[ds.config.yKey]);
                if (!isNaN(x) && !isNaN(y)) {
                    if (x < xMin) xMin = x;
                    if (x > xMax) xMax = x;
                    if (y < yMin) yMin = y;
                    if (y > yMax) yMax = y;
                    hasData = true;
                }
            });
        });

        if (!hasData) return { xMin: -10, xMax: 10, yMin: -10, yMax: 10 };
        return { xMin, xMax, yMin, yMax };
    }, [currentGraph]);

    const currentDomain = useMemo(() => {
        if (zoomDomain.x[0] !== 'auto') {
            // If Square Aspect Ratio is active, adjust Y domain
            if (currentGraph?.globalConfig?.aspectRatio === 'square' && containerSize.width > 0 && containerSize.height > 0) {
                const xRange = zoomDomain.x[1] - zoomDomain.x[0];
                const ratio = containerSize.height / containerSize.width;
                const yCenter = (zoomDomain.y[1] + zoomDomain.y[0]) / 2;
                const newYRange = xRange * ratio;
                return {
                    x: zoomDomain.x,
                    y: [yCenter - newYRange / 2, yCenter + newYRange / 2]
                };
            }
            return zoomDomain;
        }

        // Initial Domain Calc
        const xDiff = globalBounds.xMax - globalBounds.xMin;
        const yDiff = globalBounds.yMax - globalBounds.yMin;
        const xPad = xDiff > 0 ? xDiff * 0.1 : (Math.abs(globalBounds.xMin) > 0 ? Math.abs(globalBounds.xMin) * 0.1 : 1);
        const yPad = yDiff > 0 ? yDiff * 0.1 : (Math.abs(globalBounds.yMin) > 0 ? Math.abs(globalBounds.yMin) * 0.1 : 1);

        let d = {
            x: [globalBounds.xMin - xPad, globalBounds.xMax + xPad],
            y: [globalBounds.yMin - yPad, globalBounds.yMax + yPad]
        };

        // Square Default?
        if (currentGraph?.globalConfig?.aspectRatio === 'square' && containerSize.width > 0) {
            const xRange = d.x[1] - d.x[0];
            const ratio = containerSize.height / containerSize.width;
            const yCenter = (d.y[1] + d.y[0]) / 2;
            const newYRange = xRange * ratio;
            d.y = [yCenter - newYRange / 2, yCenter + newYRange / 2];
        }

        return d;

    }, [zoomDomain, globalBounds, currentGraph?.globalConfig?.aspectRatio, containerSize]);

    // Calculate ticks with optional manual interval
    const xTicks = useMemo(() => {
        const interval = currentGraph?.globalConfig?.xGridInterval;
        if (interval && interval > 0) {
            const range = currentDomain.x[1] - currentDomain.x[0];
            if (range / interval > 100) {
                return calculateNiceTicks(currentDomain.x[0], currentDomain.x[1]);
            }
            const ticks = [];
            const start = Math.ceil(currentDomain.x[0] / interval) * interval;
            for (let t = start; t <= currentDomain.x[1] && ticks.length < 50; t += interval) {
                ticks.push(t);
            }
            return ticks;
        }
        return calculateNiceTicks(currentDomain.x[0], currentDomain.x[1]);
    }, [currentDomain, currentGraph?.globalConfig?.xGridInterval]);

    const yTicks = useMemo(() => {
        const interval = currentGraph?.globalConfig?.yGridInterval;
        if (interval && interval > 0) {
            const range = currentDomain.y[1] - currentDomain.y[0];
            if (range / interval > 100) {
                return calculateNiceTicks(currentDomain.y[0], currentDomain.y[1]);
            }
            const ticks = [];
            const start = Math.ceil(currentDomain.y[0] / interval) * interval;
            for (let t = start; t <= currentDomain.y[1] && ticks.length < 50; t += interval) {
                ticks.push(t);
            }
            return ticks;
        }
        return calculateNiceTicks(currentDomain.y[0], currentDomain.y[1]);
    }, [currentDomain, currentGraph?.globalConfig?.yGridInterval]);

    // Derived state for Sidebar (contains stats but NO filtering by visibility)
    const allDatasets = useMemo(() => {
        if (!currentGraph) return [];
        return currentGraph.datasets.map(ds => {

            let points = [];
            let stats = { meanX: 0, meanY: 0, stdDevX: 0, stdDevY: 0, n: 0 };
            let trendData = [];
            let trendlineEquation = null;
            let r2 = null;

            if (ds.config.type === 'function' && ds.equation) {
                // Generate function points based on CURRENT view domain
                // This gives the "Desmos" feel of infinite scrolling
                const range = currentDomain.x[1] - currentDomain.x[0];
                const buffer = range * 0.5; // Render a bit outside view
                points = generateFunctionPoints(ds.equation, currentDomain.x[0] - buffer, currentDomain.x[1] + buffer, 400);
            } else {
                // Standard Data Dataset
                points = ds.data.map(d => ({
                    x: parseFloat(d[ds.config.xKey]),
                    y: parseFloat(d[ds.config.yKey]),
                    raw: d
                })).filter(p => !isNaN(p.x) && !isNaN(p.y)).sort((a, b) => a.x - b.x);

                stats = calculateStats(ds.data, ds.config.xKey, ds.config.yKey);

                if (ds.config.showTrendline) {
                    const params = getRegressionParams(points, ds.config.trendlineType);
                    if (params) {
                        trendlineEquation = params.equation;
                        r2 = params.r2;
                        const xRange = currentDomain.x[1] - currentDomain.x[0];
                        const buffer = xRange * 0.5;
                        trendData = generateTrendlineData(params, currentDomain.x[0] - buffer, currentDomain.x[1] + buffer, globalBounds.yMin, globalBounds.yMax);
                    }
                }
            }

            // Return the dataset with enriched display properties, 
            // BUT KEEP ORIGINAL 'equation' (user input) separate from 'trendlineEquation'
            return { ...ds, points, trendData, trendlineEquation, stats, r2 };
        });
    }, [currentGraph, currentDomain, globalBounds]);

    // Derived state for Chart (ONLY visible datasets)
    const visibleDatasets = useMemo(() => {
        return allDatasets.filter(ds => ds.visible);
    }, [allDatasets]);

    const handleWheel = (e) => {
        if (!currentGraph?.globalConfig?.enableZoom) return;
        // Note: preventDefault might not work here if event is passive. 
        // We handle strict prevention in the native listener, but here we calculate scale.

        // Throttling/Smoothing for Trackpads:
        // If deltaMode is 0 (pixel), it's likely a trackpad or high-precision wheel.
        // If delta is small, dampen it.
        let dampening = 0.1;
        if (e.deltaMode === 0) { // Pixel scrolling (Trackpad usually)
            dampening = Math.abs(e.deltaY) < 50 ? 0.02 : 0.05;
        }

        const scale = e.deltaY > 0 ? (1 + dampening) : (1 - dampening);

        const xR = currentDomain.x[1] - currentDomain.x[0];
        const yR = currentDomain.y[1] - currentDomain.y[0];
        const xM = (currentDomain.x[1] + currentDomain.x[0]) / 2;
        const yM = (currentDomain.y[1] + currentDomain.y[0]) / 2;

        const zX = !e.ctrlKey && !e.metaKey;
        const zY = !e.shiftKey;

        setZoomDomain({
            x: zX ? [xM - (xR * scale) / 2, xM + (xR * scale) / 2] : currentDomain.x,
            y: zY ? [yM - (yR * scale) / 2, yM + (yR * scale) / 2] : currentDomain.y
        });
    };

    const zoomIn = () => {
        const scale = 0.8; // zoom in
        const xR = currentDomain.x[1] - currentDomain.x[0];
        const yR = currentDomain.y[1] - currentDomain.y[0];
        const xM = (currentDomain.x[1] + currentDomain.x[0]) / 2;
        const yM = (currentDomain.y[1] + currentDomain.y[0]) / 2;
        setZoomDomain({
            x: [xM - (xR * scale) / 2, xM + (xR * scale) / 2],
            y: [yM - (yR * scale) / 2, yM + (yR * scale) / 2]
        });
    };

    const zoomOut = () => {
        const scale = 1.25; // zoom out
        const xR = currentDomain.x[1] - currentDomain.x[0];
        const yR = currentDomain.y[1] - currentDomain.y[0];
        const xM = (currentDomain.x[1] + currentDomain.x[0]) / 2;
        const yM = (currentDomain.y[1] + currentDomain.y[0]) / 2;
        setZoomDomain({
            x: [xM - (xR * scale) / 2, xM + (xR * scale) / 2],
            y: [yM - (yR * scale) / 2, yM + (yR * scale) / 2]
        });
    };

    const resetView = () => {
        setZoomDomain({ x: ['auto', 'auto'], y: ['auto', 'auto'] });
    };

    // UNIFIED POINTER SUPPORT (Pan & Pinch-to-Zoom)
    const handlePointerDown = (e) => {
        if (selectedData) return;
        if (!currentGraph?.globalConfig?.enableZoom) return;

        activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        try {
            e.currentTarget.setPointerCapture(e.pointerId);
        } catch (err) {
            // ignore if capture fails
        }

        if (activePointers.current.size === 1) {
            setIsDragging(true);
            lastMousePos.current = { x: e.clientX, y: e.clientY };
        } else if (activePointers.current.size === 2) {
            const points = Array.from(activePointers.current.values());
            pinchStartDist.current = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
            pinchStartDomain.current = {
                x: [...currentDomain.x],
                y: [...currentDomain.y]
            };
        }
    };

    const handlePointerMove = (e) => {
        // Track live mathematical coordinates for HUD readout
        if (graphContainerRef.current) {
            const rect = graphContainerRef.current.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            if (px >= 0 && px <= rect.width && py >= 0 && py <= rect.height) {
                const xRange = currentDomain.x[1] - currentDomain.x[0];
                const yRange = currentDomain.y[1] - currentDomain.y[0];
                const plotW = Math.max(1, rect.width - 70);
                const plotH = Math.max(1, rect.height - 60);
                const xFrac = Math.max(0, Math.min(1, (px - 50) / plotW));
                const yFrac = Math.max(0, Math.min(1, (py - 20) / plotH));
                const mx = currentDomain.x[0] + xFrac * xRange;
                const my = currentDomain.y[1] - yFrac * yRange;
                setCursorCoords({ x: mx, y: my });
            }
        }

        if (!activePointers.current.has(e.pointerId)) return;
        activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        // Two-finger pinch zoom
        if (activePointers.current.size === 2 && pinchStartDist.current && pinchStartDomain.current) {
            const points = Array.from(activePointers.current.values());
            const currentDist = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
            if (currentDist > 5 && pinchStartDist.current > 5) {
                const scale = pinchStartDist.current / currentDist;
                const baseDomain = pinchStartDomain.current;
                const xR = baseDomain.x[1] - baseDomain.x[0];
                const yR = baseDomain.y[1] - baseDomain.y[0];
                const xM = (baseDomain.x[1] + baseDomain.x[0]) / 2;
                const yM = (baseDomain.y[1] + baseDomain.y[0]) / 2;

                setZoomDomain({
                    x: [xM - (xR * scale) / 2, xM + (xR * scale) / 2],
                    y: [yM - (yR * scale) / 2, yM + (yR * scale) / 2]
                });
            }
            return;
        }

        // Single-pointer drag pan
        if (isDragging && activePointers.current.size === 1) {
            const dx = e.clientX - lastMousePos.current.x;
            const dy = e.clientY - lastMousePos.current.y;
            const xR = currentDomain.x[1] - currentDomain.x[0];
            const yR = currentDomain.y[1] - currentDomain.y[0];
            const xS = -1 * (dx / (containerSize.width || 500)) * xR;
            const yS = (dy / (containerSize.height || 300)) * yR;
            setZoomDomain({
                x: [currentDomain.x[0] + xS, currentDomain.x[1] + xS],
                y: [currentDomain.y[0] + yS, currentDomain.y[1] + yS]
            });
            lastMousePos.current = { x: e.clientX, y: e.clientY };
        }
    };

    const handlePointerUp = (e) => {
        activePointers.current.delete(e.pointerId);
        try {
            e.currentTarget.releasePointerCapture(e.pointerId);
        } catch (err) {
            // ignore
        }

        if (activePointers.current.size === 0) {
            setIsDragging(false);
            pinchStartDist.current = null;
            pinchStartDomain.current = null;
        } else if (activePointers.current.size === 1) {
            const remaining = Array.from(activePointers.current.values())[0];
            lastMousePos.current = { x: remaining.x, y: remaining.y };
            pinchStartDist.current = null;
            pinchStartDomain.current = null;
        }
    };

    const handlePointClick = (arg1, arg2, ds, type = 'point') => {
        let event = null;
        let payload = null;

        if (arg1 && arg1.stopPropagation) event = arg1;
        else if (arg2 && arg2.stopPropagation) event = arg2;

        if (event) {
            event.stopPropagation();
            event.preventDefault();
        }

        if (arg1 && (arg1.payload || arg1.x !== undefined)) payload = arg1;
        else if (arg2 && (arg2.payload || arg2.x !== undefined)) payload = arg2;

        let data = {};
        if (type === 'trendline') {
            data = {
                isTrendline: true,
                equation: ds.trendlineEquation,
                color: ds.config.trendlineColor || '#ef4444'
            };
        } else {
            const p = (payload && payload.payload) ? payload.payload : payload;
            if (p) {
                data = {
                    x: p.x,
                    y: p.y,
                    xKey: ds.config.xKey,
                    yKey: ds.config.yKey,
                    color: ds.color
                };
            }
        }

        if (data.x !== undefined || data.isTrendline) {
            setSelectedData({
                ...data,
                datasetName: ds.name,
            });
        }
    };

    const handleChartClick = (e) => {
        if (e && e.activePayload && e.activePayload.length > 0) {
            const activeItem = e.activePayload[0];
            const p = activeItem.payload;
            if (p) {
                setSelectedData({
                    x: p.x,
                    y: p.y,
                    datasetName: "Active Point",
                    color: activeItem.color || activeItem.fill
                });
            }
        }
    };

    const handleChartDoubleClick = (e) => {
        if (!showSettings) setShowSettings(true);
        const text = prompt("Enter annotation text:");
        if (text) {
            const xVal = e && e.activeLabel ? e.activeLabel : prompt("Enter X coordinate for annotation:", 0);
            const yVal = prompt("Enter Y coordinate for annotation:", 0);

            if (xVal !== null && yVal !== null) {
                const newAnnotation = {
                    id: Date.now(),
                    x: String(xVal),
                    y: String(yVal),
                    text: text
                };
                setCurrentGraph(prev => ({
                    ...prev,
                    annotations: [...(prev.annotations || []), newAnnotation]
                }));
            }
        }
    };

    // --- AI CHAT TOOL HANDLERS ---
    const [threeDEquation, setThreeDEquation] = useState('sin(x) * cos(y)');

    const handleAIPlotFunction = (expression) => {
        setAppMode('2d');
        setView('editor');
        const cleanExpr = (expression || 'sin(x)').replace(/^(y\s*=\s*|f\(x\)\s*=\s*)/i, '').trim();
        const newDs = {
            id: `ds-${Date.now()}`,
            name: cleanExpr.includes('=') ? cleanExpr : `f(x) = ${cleanExpr}`,
            data: [],
            equation: cleanExpr,
            visible: true,
            color: THEMES[(currentGraph?.datasets?.length || 0) % THEMES.length].color,
            config: {
                type: 'function',
                xKey: 'x',
                yKey: 'y',
                showTrendline: false,
                trendlineType: 'linear',
                trendlineColor: '#ef4444'
            }
        };
        if (currentGraph) {
            setCurrentGraph(prev => ({ ...prev, datasets: [...prev.datasets, newDs] }));
            setExpandedDatasetId(newDs.id);
        } else {
            createFunctionGraph(cleanExpr);
        }
    };

    const handleAIPlotImplicit = (expression) => {
        handleAIPlotFunction(expression);
    };

    const handleAILoadDataTable = (name, rows) => {
        setAppMode('2d');
        setView('editor');
        const newDataset = {
            id: `ds-${Date.now()}`,
            name: name || "AI Data",
            data: rows,
            visible: true,
            color: THEMES[0].color,
            config: { type: 'scatter', xKey: 'x', yKey: 'y', showTrendline: true, trendlineType: 'linear', trendlineColor: '#ef4444' }
        };
        if (currentGraph) {
            setCurrentGraph(prev => ({ ...prev, datasets: [...prev.datasets, newDataset] }));
        } else {
            setCurrentGraph({
                title: name || "AI Data Plot",
                datasets: [newDataset],
                globalConfig: { showGrid: true, enableZoom: true, xAxisLabel: "X", yAxisLabel: "Y", aspectRatio: "auto", showLabels: false },
                annotations: [],
                createdAt: new Date().toISOString()
            });
        }
    };

    const handleAISwitchTo3D = (expression) => {
        if (expression && typeof expression === 'string') {
            setThreeDEquation(expression);
        }
        setAppMode('3d');
        setView('editor');
    };

    const handleAISetViewportBounds = (bounds) => {
        setViewportBounds(bounds);
    };

    return (
        <div className="min-h-screen bg-[#F8FAFC] font-sans text-slate-800 selection:bg-indigo-100 selection:text-indigo-900 overflow-hidden">

            {/* PRINT STYLES - ROBUST */}
            <style>{`
        @media print {
            @page {
                size: landscape;
                margin: 0;
            }
            body {
                margin: 0;
                padding: 0;
                visibility: hidden;
                overflow: hidden;
            }
            
            /* Only show the generated iframe content */
            iframe[style*="fixed"] {
                visibility: visible !important;
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                z-index: 999999 !important;
            }
        }
      `}</style>

            <nav className="sticky top-0 z-50 h-14 bg-white border-b border-neutral-200 flex items-center justify-between px-4 md:px-6">
                <div className="flex items-center gap-3 sm:gap-4">
                    <div className="cursor-pointer" onClick={() => setView('dashboard')}>
                        <Logo size={28} />
                    </div>

                    {/* Mode Switcher: 2D Plotter vs 3D Surface */}
                    <div className="flex bg-neutral-100 p-0.5 rounded-lg border border-neutral-200">
                        <button
                            type="button"
                            onClick={() => {
                                setAppMode('2d');
                                if (view !== 'editor') setView('editor');
                                if (!currentGraph) createBlankGraph();
                            }}
                            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                                appMode !== '3d' ? 'bg-white text-neutral-900 shadow-xs' : 'text-neutral-600 hover:text-neutral-900'
                            }`}
                        >
                            2D Plotter
                        </button>
                        <button
                            type="button"
                            onClick={() => {
                                setAppMode('3d');
                                if (view !== 'editor') setView('editor');
                            }}
                            className={`px-3 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer ${
                                appMode === '3d' ? 'bg-white text-neutral-900 shadow-xs' : 'text-neutral-600 hover:text-neutral-900'
                            }`}
                        >
                            3D Surface
                        </button>
                    </div>
                </div>
                <div className="flex gap-2 items-center">
                    {view === 'editor' && appMode !== '3d' && currentGraph && (
                        <>
                            <Button size="sm" variant="secondary" icon={Download} onClick={() => {
                                const svgElement = document.querySelector('.recharts-surface');
                                if (!svgElement) {
                                    alert("Could not find graph to print.");
                                    return;
                                }
                                const serializer = new XMLSerializer();
                                const svgString = serializer.serializeToString(svgElement);
                                const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
                                const url = URL.createObjectURL(svgBlob);
                                const canvas = document.createElement("canvas");
                                const ctx = canvas.getContext("2d");
                                const img = new Image();
                                img.onload = () => {
                                    const padding = 40;
                                    const titleHeight = 80;
                                    const width = img.width + (padding * 2);
                                    const height = img.height + titleHeight + padding;
                                    canvas.width = width;
                                    canvas.height = height;
                                    ctx.fillStyle = "white";
                                    ctx.fillRect(0, 0, width, height);
                                    ctx.font = "bold 36px sans-serif";
                                    ctx.fillStyle = "#000000";
                                    ctx.textAlign = "center";
                                    ctx.textBaseline = "middle";
                                    ctx.fillText(currentGraph.title || "Untitled Graph", width / 2, (titleHeight / 2) + 20);
                                    ctx.drawImage(img, padding, titleHeight);
                                    const pngUrl = canvas.toDataURL("image/png");

                                    const printFrame = document.createElement('iframe');
                                    printFrame.style.position = 'fixed';
                                    printFrame.style.top = '0';
                                    printFrame.style.left = '0';
                                    printFrame.style.width = '100vw';
                                    printFrame.style.height = '100vh';
                                    printFrame.style.border = '0';
                                    printFrame.style.zIndex = '9999';
                                    document.body.appendChild(printFrame);

                                    const frameDoc = printFrame.contentWindow.document;
                                    frameDoc.open();
                                    frameDoc.write(`
                                <html>
                                <head>
                                    <style>
                                        @page { size: landscape; margin: 0; }
                                        body { 
                                            margin: 0; padding: 0; 
                                            display: flex; 
                                            justify-content: center; 
                                            align-items: center; 
                                            height: 100vh; 
                                            width: 100vw;
                                            overflow: hidden;
                                            background: white;
                                        }
                                        img {
                                            max-width: 95vw;
                                            max-height: 90vh;
                                            object-fit: contain;
                                            box-shadow: none;
                                        }
                                    </style>
                                </head>
                                <body>
                                    <img src="${pngUrl}" />
                                </body>
                                </html>
                            `);
                                    frameDoc.close();
                                    printFrame.onload = () => {
                                        printFrame.contentWindow.focus();
                                        printFrame.contentWindow.print();
                                        setTimeout(() => {
                                            document.body.removeChild(printFrame);
                                            URL.revokeObjectURL(url);
                                        }, 1000);
                                    };
                                };
                                img.src = url;
                            }}><span className="hidden sm:inline">PDF</span></Button>
                            <Button size="sm" variant="secondary" icon={FileSpreadsheet} onClick={exportCSV}><span className="hidden sm:inline">CSV</span></Button>
                            <Button size="sm" variant="primary" icon={Save} onClick={saveGraph}><span className="hidden sm:inline">Save</span></Button>
                            <Button size="sm" variant="secondary" icon={FilePlus} onClick={startImport}><span className="hidden sm:inline">Import</span></Button>
                        </>
                    )}
                    {view === 'dashboard' && (
                        <div className="flex gap-2">
                            <Button onClick={createBlankGraph} icon={Plus} size="sm">
                                <span className="hidden sm:inline">New Graph</span>
                            </Button>
                            <Button onClick={() => { setIsImporting(false); setView('scan'); }} variant="secondary" icon={Camera} size="sm">
                                <span className="hidden sm:inline">Scan</span>
                            </Button>
                            <Button onClick={() => setShowApiKeyModal(true)} variant="secondary" icon={Key} size="sm">
                                <span className="hidden sm:inline">API Key</span>
                            </Button>
                        </div>
                    )}
                </div>
            </nav>

            {view === 'dashboard' && (
                <main className="relative min-h-[calc(100vh-56px)] flex flex-col items-center justify-center px-4 py-12 overflow-hidden bg-white">
                    {/* Ultra-lightweight animated harmonic math wave background */}
                    <MathBackground />

                    {/* Centered, straightforward, elegant Hero */}
                    <div className="relative z-10 max-w-xl w-full text-center space-y-6">
                        <div className="flex justify-center">
                            <div className="p-3 bg-white/90 backdrop-blur-xs border border-neutral-300 rounded-2xl shadow-sm inline-flex items-center gap-3">
                                <LogoIcon size={34} />
                                <span className="font-bold text-2xl text-neutral-900 tracking-tight font-sans">
                                    Graphly
                                </span>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <h1 className="text-2xl sm:text-4xl font-bold text-neutral-900 tracking-tight">
                                Precision Coordinate Plotter
                            </h1>
                            <p className="text-sm sm:text-base text-neutral-600 max-w-md mx-auto leading-relaxed">
                                2D mathematical curves, experimental data regression, and interactive 3D multi-surface visualization.
                            </p>
                        </div>

                        {/* Three Primary Actions: 2D Plotter, 3D Surface, Scan Data */}
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                            <button
                                onClick={() => { setAppMode('2d'); createBlankGraph(); }}
                                className="p-4 bg-neutral-900 text-white rounded-xl border border-neutral-900 hover:bg-neutral-800 transition-all shadow-sm flex flex-col items-center gap-2 group cursor-pointer"
                            >
                                <Plus size={22} className="text-blue-400 group-hover:scale-110 transition-transform" />
                                <span className="text-xs font-bold uppercase tracking-wider">2D Plotter</span>
                                <span className="text-[11px] text-neutral-400 font-mono">Data & Equations</span>
                            </button>

                            <button
                                onClick={() => { setAppMode('3d'); setView('editor'); }}
                                className="p-4 bg-white/95 backdrop-blur-xs text-neutral-900 rounded-xl border border-neutral-300 hover:border-neutral-900 hover:bg-neutral-50 transition-all shadow-xs flex flex-col items-center gap-2 group cursor-pointer"
                            >
                                <Box size={22} className="text-purple-600 group-hover:scale-110 transition-transform" />
                                <span className="text-xs font-bold uppercase tracking-wider">3D Surface</span>
                                <span className="text-[11px] text-neutral-500 font-mono">Multi-Mesh z = f(x,y)</span>
                            </button>

                            <button
                                onClick={() => { setAppMode('2d'); setIsImporting(false); setView('scan'); }}
                                className="p-4 bg-white/95 backdrop-blur-xs text-neutral-900 rounded-xl border border-neutral-300 hover:border-neutral-900 hover:bg-neutral-50 transition-all shadow-xs flex flex-col items-center gap-2 group cursor-pointer"
                            >
                                <Camera size={22} className="text-emerald-600 group-hover:scale-110 transition-transform" />
                                <span className="text-xs font-bold uppercase tracking-wider">Scan & CSV</span>
                                <span className="text-[11px] text-neutral-500 font-mono">AI OCR Import</span>
                            </button>
                        </div>

                        {/* Saved Projects Section */}
                        {savedGraphs.length > 0 && (
                            <div className="pt-8 text-left border-t border-neutral-300 w-full">
                                <div className="flex items-center justify-between mb-3">
                                    <span className="text-xs font-bold font-mono uppercase text-neutral-600">Saved Projects ({savedGraphs.length})</span>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-56 overflow-y-auto pr-1">
                                    {savedGraphs.map(g => (
                                        <div 
                                            key={g.id}
                                            onClick={() => { setAppMode('2d'); setCurrentGraph(g); setView('editor'); }}
                                            className="p-3 bg-white/95 backdrop-blur-xs border border-neutral-300 rounded-lg hover:border-neutral-900 transition-all cursor-pointer shadow-2xs flex items-center justify-between group"
                                        >
                                            <div className="truncate mr-2">
                                                <div className="font-semibold text-xs text-neutral-900 truncate">{g.title || 'Untitled Graph'}</div>
                                                <div className="text-[10px] font-mono text-neutral-500">{g.datasets?.length || 1} dataset(s)</div>
                                            </div>
                                            <button
                                                onClick={(e) => deleteGraph(e, g.id)}
                                                className="opacity-0 group-hover:opacity-100 p-1 text-neutral-400 hover:text-red-600 transition-opacity cursor-pointer"
                                                title="Delete graph"
                                            >
                                                <Trash2 size={13} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </main>
            )}

            {view === 'scan' && (
                <main className="max-w-3xl mx-auto px-4 py-10">
                    <div className="mb-6 border-b-2 border-black pb-4">
                        <div className="flex items-center justify-between">
                            <h1 className="text-2xl font-bold text-black tracking-tight">Import Data</h1>
                            <Button onClick={() => setShowApiKeyModal(true)} variant="secondary" size="sm" icon={Key}>
                                API Key
                            </Button>
                        </div>
                        <p className="text-xs font-mono text-black mt-1">
                            {isImporting ? "Add external data to active graph" : "Create new graph from datasheet or CSV"}
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            className="border-2 border-black bg-white p-6 h-48 flex flex-col justify-between cursor-pointer hover:bg-black hover:text-white transition-none group"
                        >
                            <div className="flex items-center justify-between">
                                <span className="text-xs font-mono font-bold uppercase">[IMAGE SCAN]</span>
                                <Camera size={20} />
                            </div>
                            {image ? (
                                <img src={image.preview} className="h-24 object-contain mx-auto" />
                            ) : (
                                <div>
                                    <div className="font-bold text-base mb-1">Datasheet Image</div>
                                    <div className="text-xs font-mono text-neutral-500 group-hover:text-neutral-300">Upload screenshot or photo of table</div>
                                </div>
                            )}
                        </div>

                        <div
                            onClick={() => setShowCSVModal(true)}
                            className="border-2 border-black bg-white p-6 h-48 flex flex-col justify-between cursor-pointer hover:bg-black hover:text-white transition-none group"
                        >
                            <div className="flex items-center justify-between">
                                <span className="text-xs font-mono font-bold uppercase">[CSV FILE]</span>
                                <FileSpreadsheet size={20} />
                            </div>
                            <div>
                                <div className="font-bold text-base mb-1">Paste / Upload CSV</div>
                                <div className="text-xs font-mono text-neutral-500 group-hover:text-neutral-300">Raw tabular comma-separated values</div>
                            </div>
                        </div>
                    </div>

                    <input type="file" ref={fileInputRef} className="hidden" onChange={e => {
                        if (e.target.files[0]) {
                            const r = new FileReader();
                            r.onloadend = () => setImage({ file: e.target.files[0], base64: r.result.split(',')[1], preview: r.result });
                            r.readAsDataURL(e.target.files[0]);
                        }
                    }} />

                    {image && (
                        <div className="mb-6">
                            <Button className="w-full" variant="primary" size="lg" onClick={handleScan} disabled={scanStatus === 'scanning'} icon={Zap}>
                                {scanStatus === 'scanning' ? 'Analyzing with Gemini AI...' : 'Process Image with AI'}
                            </Button>
                        </div>
                    )}

                    <div className="flex justify-start">
                        <button onClick={() => setView(isImporting ? 'editor' : 'dashboard')} className="text-xs font-mono text-black border border-black px-3 py-1.5 hover:bg-black hover:text-white transition-none">
                            ← Back to {isImporting ? 'Editor' : 'Dashboard'}
                        </button>
                    </div>

                    {/* Brutalist CSV Modal */}
                    {showCSVModal && (
                        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
                            <div className="bg-white border-2 border-black p-6 w-full max-w-lg">
                                <div className="flex items-center justify-between pb-3 mb-4 border-b-2 border-black">
                                    <div className="flex items-center gap-2">
                                        <FileSpreadsheet size={18} />
                                        <h3 className="text-base font-bold text-black">Import CSV Data</h3>
                                    </div>
                                    <button onClick={() => setShowCSVModal(false)} className="p-1 hover:bg-black hover:text-white border border-transparent hover:border-black">
                                        <X size={16} />
                                    </button>
                                </div>
                                <textarea
                                    className="w-full h-40 p-3 bg-white border-2 border-black text-xs font-mono mb-4 outline-none resize-none"
                                    placeholder="x,y&#10;1,2&#10;3,4"
                                    value={csvText}
                                    onChange={e => setCsvText(e.target.value)}
                                />
                                <div className="flex items-center justify-between mb-5 p-2.5 border border-black bg-neutral-50">
                                    <span className="text-xs font-mono text-black">Upload .csv file:</span>
                                    <input
                                        type="file"
                                        accept=".csv"
                                        ref={csvFileInputRef}
                                        className="text-xs font-mono text-black file:mr-2 file:py-1 file:px-2 file:border file:border-black file:text-xs file:bg-white file:text-black hover:file:bg-black hover:file:text-white file:cursor-pointer"
                                        onChange={(e) => {
                                            const file = e.target.files[0];
                                            if (file) {
                                                const reader = new FileReader();
                                                reader.onload = (evt) => setCsvText(evt.target.result);
                                                reader.readAsText(file);
                                            }
                                        }}
                                    />
                                </div>
                                <div className="flex gap-2 justify-end">
                                    <Button variant="secondary" size="md" onClick={() => setShowCSVModal(false)}>Cancel</Button>
                                    <Button variant="primary" size="md" onClick={() => handleCSVImport(parseCSV(csvText))} icon={Check}>Import CSV</Button>
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            )}

            {view === 'editor' && (
                appMode === '3d' ? (
                    <ThreeDGraph initialEquation={threeDEquation} />
                ) : currentGraph ? (
                <div className="relative flex h-[calc(100vh-56px)] overflow-hidden bg-neutral-50">
                    <div className="flex-1 relative bg-neutral-50 flex flex-col overflow-hidden border-r border-neutral-200">
                        {/* Top 2D Function Quick-Add & Preset Bar */}
                        <div className="p-2.5 bg-white border-b border-neutral-200 flex flex-wrap items-center gap-2.5 shrink-0 z-20 shadow-2xs">
                            <div className="flex items-center gap-1.5 font-mono font-semibold text-xs bg-neutral-100 text-neutral-800 border border-neutral-200 px-2.5 py-1.5 rounded-md shadow-2xs">
                                <Calculator size={14} className="text-blue-600" />
                                <span>f(x) =</span>
                            </div>
                            <form onSubmit={handleQuickFunctionSubmit} className="flex-1 min-w-[200px] flex items-center gap-2">
                                <input
                                    type="text"
                                    value={quickFunctionExpr}
                                    onChange={(e) => setQuickFunctionExpr(e.target.value)}
                                    placeholder="Plot equation, e.g. sin(x), x^2 - 4, 1/x, cos(x)*x, exp(-x^2)"
                                    className="flex-1 bg-white border border-neutral-200 px-3 py-1.5 font-mono text-xs text-neutral-900 placeholder:text-neutral-400 focus:outline-none focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900 rounded-md"
                                />
                                <button
                                    type="submit"
                                    className="bg-neutral-900 text-white border border-neutral-900 px-3.5 py-1.5 font-sans text-xs font-semibold hover:bg-neutral-800 transition-colors flex items-center gap-1.5 rounded-md cursor-pointer shadow-2xs shrink-0"
                                >
                                    <Plus size={14} />
                                    <span>Plot Function</span>
                                </button>
                            </form>
                            
                            {/* Function Preset Chips */}
                            <div className="hidden lg:flex items-center gap-1 text-[11px] font-mono text-neutral-500 border-l border-neutral-200 pl-2.5">
                                <span className="text-[10px] uppercase text-neutral-400 font-sans font-semibold mr-1">Presets:</span>
                                {FUNCTION_PRESETS.slice(0, 5).map(preset => (
                                    <button
                                        key={preset.label}
                                        type="button"
                                        onClick={() => addFunctionDataset(preset.expr)}
                                        className="px-2 py-0.5 rounded bg-neutral-100 hover:bg-neutral-200 text-neutral-700 hover:text-neutral-900 transition-colors cursor-pointer"
                                    >
                                        {preset.label}
                                    </button>
                                ))}
                            </div>

                            <button
                                onClick={() => setIsSidebarOpen(true)}
                                className="lg:hidden ml-auto bg-white p-1.5 border border-neutral-200 rounded-md text-neutral-800 hover:bg-neutral-100 transition-colors shadow-xs"
                                aria-label="Open Sidebar"
                            >
                                <Menu size={16} />
                            </button>
                        </div>

                        <div 
                            className="flex-1 p-2 md:p-5 flex flex-col items-center justify-center overflow-hidden relative graph-print-container graph-touch-surface bg-neutral-50"
                            ref={graphContainerRef}
                            style={{ touchAction: 'none' }}
                            onWheel={handleWheel}
                            onPointerDown={handlePointerDown}
                            onPointerMove={handlePointerMove}
                            onPointerUp={handlePointerUp}
                            onPointerCancel={handlePointerUp}
                            onPointerLeave={(e) => {
                                handlePointerUp(e);
                                setCursorCoords(null);
                            }}
                        >
                            {/* COMPACT INFO PILL (Top-Left) */}
                            {selectedData && (
                                <div className="graph-ui-overlay absolute top-3 left-3 z-50 bg-white/95 backdrop-blur-xs border border-neutral-300 rounded-lg shadow-sm px-3 py-1.5 flex items-center gap-3">
                                    <div className="flex items-center gap-2">
                                        <div className="w-2.5 h-2.5 rounded-full border border-white shadow-xs" style={{ backgroundColor: selectedData.color || '#2563EB' }}></div>
                                        <span className="text-xs font-sans font-semibold text-neutral-900 max-w-[130px] truncate">{selectedData.datasetName}</span>
                                    </div>
                                    {selectedData.isTrendline ? (
                                        <span className="font-mono font-bold text-xs text-neutral-900">{selectedData.equation}</span>
                                    ) : (
                                        <div className="flex items-center gap-2 text-xs font-mono border-l border-neutral-200 pl-2 text-neutral-800">
                                            <span>x: {formatNumber(selectedData.x)}</span>
                                            <span>y: {formatNumber(selectedData.y)}</span>
                                        </div>
                                    )}
                                    <button onClick={(e) => { e.stopPropagation(); setSelectedData(null); }} className="text-neutral-500 hover:text-neutral-900 p-0.5 rounded-md hover:bg-neutral-100 ml-1 cursor-pointer">
                                        <X size={12} />
                                    </button>
                                </div>
                            )}

                            {!selectedData && (
                                <div className="graph-ui-overlay absolute top-3 left-3 z-40 bg-white/90 backdrop-blur-xs border border-neutral-200 rounded-md shadow-xs px-2.5 py-1 text-xs text-neutral-500 pointer-events-none flex items-center gap-1.5">
                                    <Info size={12} /> Click point for coordinates
                                </div>
                            )}

                            {/* LIVE CURSOR COORDINATES HUD */}
                            {cursorCoords && (
                                <div className="graph-ui-overlay absolute top-3 right-3 z-40 bg-white/95 backdrop-blur-xs border border-neutral-200 rounded-md shadow-xs px-2.5 py-1 font-mono text-[11px] text-neutral-700 pointer-events-none flex items-center gap-2">
                                    <span className="text-neutral-400 font-sans text-[10px] uppercase font-bold tracking-wider">Coords:</span>
                                    <span>x: {formatNumber(cursorCoords.x)}</span>
                                    <span>y: {formatNumber(cursorCoords.y)}</span>
                                </div>
                            )}

                            {/* Zoom Controls Overlay */}
                            <div className="graph-ui-overlay absolute bottom-4 right-4 z-40 flex flex-col gap-1 bg-white/95 backdrop-blur-xs border border-neutral-200 rounded-lg shadow-sm p-1">
                                <button onClick={zoomIn} className="p-1.5 text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100 rounded-md transition-colors cursor-pointer" title="Zoom In">
                                    <Plus size={16} />
                                </button>
                                <button onClick={zoomOut} className="p-1.5 text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100 rounded-md transition-colors cursor-pointer" title="Zoom Out">
                                    <Minus size={16} />
                                </button>
                                <button onClick={resetView} className="p-1.5 text-neutral-700 hover:text-neutral-900 hover:bg-neutral-100 rounded-md transition-colors cursor-pointer" title="Reset View">
                                    <RotateCcw size={16} />
                                </button>
                            </div>

                            <div 
                                className={`w-full flex-1 bg-white border border-neutral-200 rounded-xl shadow-xs p-2 md:p-4 relative overflow-hidden ${isDragging ? 'cursor-grabbing' : 'cursor-default'}`}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <div className="absolute top-2 left-1/2 transform -translate-x-1/2 font-semibold text-neutral-800 pointer-events-none z-10 text-xs md:text-sm text-center w-3/4 truncate font-sans">
                                    {currentGraph.title}
                                </div>

                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart
                                        margin={{ top: 40, right: 15, bottom: 20, left: 15 }}
                                        onDoubleClick={handleChartDoubleClick}
                                        onClick={handleChartClick}
                                        style={{ outline: 'none' }}
                                    >
                                        {currentGraph.globalConfig?.showGrid !== false && (
                                            <CartesianGrid stroke="#E4E4E7" strokeDasharray="3 3" />
                                        )}

                                        <ReferenceLine x={0} stroke="#27272A" strokeWidth={1.5} />
                                        <ReferenceLine y={0} stroke="#27272A" strokeWidth={1.5} />

                                        {currentGraph.annotations && currentGraph.annotations.map(note => {
                                            const xVal = parseFloat(note.x);
                                            const yVal = parseFloat(note.y);
                                            if (isNaN(xVal) || isNaN(yVal)) return null;
                                            return (
                                                <ReferenceDot key={note.id} x={xVal} y={yVal} r={0}>
                                                    <Label value={note.text} position="top" fill="#18181B" fontSize={11} fontWeight="bold" />
                                                </ReferenceDot>
                                            );
                                        })}

                                        {selectedData && !selectedData.isTrendline && (
                                            <ReferenceDot
                                                x={selectedData.x}
                                                y={selectedData.y}
                                                r={8}
                                                fill={selectedData.color || '#2563EB'}
                                                stroke="#FFFFFF"
                                                strokeWidth={2.5}
                                                isFront={true}
                                            />
                                        )}

                                        <XAxis 
                                            type="number" 
                                            dataKey="x" 
                                            domain={currentDomain.x} 
                                            ticks={xTicks} 
                                            tickFormatter={formatNumber} 
                                            allowDataOverflow 
                                            tick={{ fontSize: 11, fill: '#52525B', fontFamily: 'JetBrains Mono, monospace' }}
                                            axisLine={{ stroke: '#27272A', strokeWidth: 1.5 }}
                                            tickLine={{ stroke: '#A1A1AA' }}
                                        >
                                            <Label value={currentGraph.globalConfig?.xAxisLabel || "X Axis"} offset={-10} position="insideBottom" style={{ fontSize: '11px', fill: '#27272A', fontWeight: 600, fontFamily: 'Inter, sans-serif' }} />
                                        </XAxis>
                                        <YAxis 
                                            type="number" 
                                            dataKey="y"
                                            domain={currentDomain.y} 
                                            ticks={yTicks} 
                                            tickFormatter={formatNumber} 
                                            allowDataOverflow 
                                            tick={{ fontSize: 11, fill: '#52525B', fontFamily: 'JetBrains Mono, monospace' }}
                                            axisLine={{ stroke: '#27272A', strokeWidth: 1.5 }}
                                            tickLine={{ stroke: '#A1A1AA' }}
                                        >
                                            <Label value={currentGraph.globalConfig?.yAxisLabel || "Y Axis"} angle={-90} offset={10} position="insideLeft" style={{ fontSize: '11px', fill: '#27272A', fontWeight: 600, fontFamily: 'Inter, sans-serif' }} />
                                        </YAxis>

                                        {visibleDatasets.map(ds => (
                                            <React.Fragment key={ds.id}>
                                                {ds.config.type === 'function' && (
                                                    <Line
                                                        key={`func-${ds.id}`}
                                                        data={ds.points}
                                                        dataKey="y"
                                                        stroke={ds.color || '#2563EB'}
                                                        strokeWidth={2.5}
                                                        dot={false}
                                                        connectNulls={false}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                        activeDot={{ r: 5, stroke: '#FFFFFF', strokeWidth: 2, onClick: (e, p) => handlePointClick(p, e, ds) }}
                                                        onClick={(e, p) => handlePointClick(p, e, ds, 'function')}
                                                    />
                                                )}

                                                {ds.trendData.length > 0 && (
                                                    <Line
                                                        key={`trend-${ds.id}`}
                                                        data={ds.trendData}
                                                        dataKey="y"
                                                        stroke={ds.config.trendlineColor || ds.color || '#DC2626'}
                                                        strokeWidth={2}
                                                        strokeDasharray="4 4"
                                                        dot={false}
                                                        activeDot={false}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                        cursor="pointer"
                                                        onClick={(p, e) => handlePointClick(p, e, ds, 'trendline')}
                                                    />
                                                )}

                                                {ds.config.type === 'area' && (
                                                    <Area
                                                        key={`area-${ds.id}`}
                                                        data={ds.points}
                                                        dataKey="y"
                                                        stroke={ds.color || '#2563EB'}
                                                        fill={ds.color || '#2563EB'}
                                                        fillOpacity={0.15}
                                                        strokeWidth={2}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                    />
                                                )}

                                                {ds.config.type === 'line' && (
                                                    <Line
                                                        key={`line-${ds.id}`}
                                                        data={ds.points}
                                                        dataKey="y"
                                                        stroke={ds.color || '#2563EB'}
                                                        strokeWidth={2}
                                                        dot={{ r: 4, fill: ds.color || '#2563EB', stroke: '#FFFFFF', strokeWidth: 1.5 }}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                        activeDot={{ r: 6, stroke: '#FFFFFF', strokeWidth: 2, onClick: (e, p) => handlePointClick(p, e, ds) }}
                                                        onClick={(e, p) => handlePointClick(p, e, ds)}
                                                    />
                                                )}

                                                {ds.config.type === 'scatter' && (
                                                    <Scatter
                                                        key={`scatter-${ds.id}`}
                                                        data={ds.points}
                                                        name={ds.name}
                                                        dataKey="y"
                                                        fill={ds.color || '#2563EB'}
                                                        isAnimationActive={false}
                                                        onClick={(p, e) => handlePointClick(p, e, ds)}
                                                        cursor="pointer"
                                                        shape={(props) => {
                                                            const { cx, cy } = props;
                                                            if (typeof cx !== 'number' || typeof cy !== 'number' || isNaN(cx) || isNaN(cy)) return null;
                                                            return (
                                                                <circle
                                                                    cx={cx}
                                                                    cy={cy}
                                                                    r={6}
                                                                    fill={ds.color || '#2563EB'}
                                                                    stroke="#FFFFFF"
                                                                    strokeWidth={1.5}
                                                                />
                                                            );
                                                        }}
                                                    >
                                                        {currentGraph.globalConfig?.showLabels && (
                                                            <LabelList
                                                                dataKey="y"
                                                                position="top"
                                                                offset={10}
                                                                content={(props) => {
                                                                    const { x, y, value, index } = props;
                                                                    const point = ds.points && ds.points[index];
                                                                    if (!point) return null;
                                                                    const displayX = point.x !== undefined ? point.x : index;
                                                                    return (
                                                                        <text x={x} y={y - 10} fill="#000000" fontSize={10} textAnchor="middle" fontWeight="bold" fontFamily="JetBrains Mono, monospace">
                                                                            ({formatNumber(displayX)}, {formatNumber(value)})
                                                                        </text>
                                                                    );
                                                                }}
                                                            />
                                                        )}
                                                    </Scatter>
                                                )}
                                            </React.Fragment>
                                        ))}
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    {isSidebarOpen && (
                        <div
                            className="fixed inset-0 bg-black/40 z-40 lg:hidden"
                            onClick={() => setIsSidebarOpen(false)}
                        />
                    )}

                    {/* Refined Minimalist Sidebar */}
                    <div className={`
                        fixed inset-y-0 right-0 z-50 w-76 sm:w-80 bg-white border-l border-neutral-300 flex flex-col shadow-xl
                        lg:relative lg:translate-x-0 lg:w-80 lg:z-auto lg:h-full lg:shadow-none
                        xl:relative xl:translate-x-0 xl:w-80 xl:z-auto xl:h-full
                        ${isSidebarOpen ? 'translate-x-0' : 'translate-x-full'}
                    `}>
                        <button
                            onClick={() => setIsSidebarOpen(false)}
                            className="lg:hidden xl:hidden absolute top-2.5 right-2.5 p-1 rounded-md border border-neutral-300 text-neutral-600 hover:bg-neutral-100 z-50 cursor-pointer"
                        >
                            <X size={18} />
                        </button>

                        <div className="flex border-b border-neutral-300 shrink-0 bg-neutral-100/60 p-1.5 gap-1.5">
                            <button
                                onClick={() => setShowSettings(false)}
                                className={`flex-1 py-1.5 rounded-md text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${!showSettings ? 'bg-white text-neutral-900 shadow-2xs border border-neutral-300 font-bold' : 'text-neutral-600 hover:text-neutral-900 border border-transparent'}`}
                            >
                                <Layout size={13} /> Datasets
                            </button>
                            <button
                                onClick={() => setShowSettings(true)}
                                className={`flex-1 py-1.5 rounded-md text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${showSettings ? 'bg-white text-neutral-900 shadow-2xs border border-neutral-300 font-bold' : 'text-neutral-600 hover:text-neutral-900 border border-transparent'}`}
                            >
                                <Settings size={13} /> Graph Settings
                            </button>
                        </div>

                        {showSettings ? (
                            <div className="flex-1 p-4 space-y-5 overflow-y-auto">
                                <div className="border border-neutral-200 rounded-lg p-3 bg-white shadow-2xs">
                                    <label className="flex items-center justify-between cursor-pointer">
                                        <span className="text-xs font-semibold text-neutral-800 flex items-center gap-1.5">
                                            <Eye size={14} className="text-neutral-500" /> Annotate All Points
                                        </span>
                                        <input
                                            type="checkbox"
                                            checked={currentGraph.globalConfig?.showLabels === true}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, showLabels: e.target.checked } })}
                                            className="w-4 h-4 rounded border border-neutral-300 text-neutral-900 focus:ring-0 cursor-pointer"
                                        />
                                    </label>
                                </div>

                                <div>
                                    <label className="text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-1.5 block">Graph Title</label>
                                    <input
                                        className="w-full p-2 bg-white border border-neutral-200 rounded-lg text-xs font-sans text-neutral-900 outline-none focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900"
                                        value={currentGraph.title}
                                        onChange={e => setCurrentGraph({ ...currentGraph, title: e.target.value })}
                                        placeholder="Graph Title"
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-1.5 block">X Axis Label</label>
                                        <input
                                            className="w-full p-2 bg-white border border-neutral-200 rounded-lg text-xs font-sans text-neutral-900 outline-none focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900"
                                            value={currentGraph.globalConfig?.xAxisLabel || ''}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, xAxisLabel: e.target.value } })}
                                            placeholder="X"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-1.5 block">Y Axis Label</label>
                                        <input
                                            className="w-full p-2 bg-white border border-neutral-200 rounded-lg text-xs font-sans text-neutral-900 outline-none focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900"
                                            value={currentGraph.globalConfig?.yAxisLabel || ''}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, yAxisLabel: e.target.value } })}
                                            placeholder="Y"
                                        />
                                    </div>
                                </div>

                                <div className="pt-3 border-t border-neutral-200">
                                    <label className="text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-1.5 block">Aspect Ratio</label>
                                    <select
                                        className="w-full p-2 bg-white border border-neutral-200 rounded-lg text-xs font-sans text-neutral-900 outline-none focus:border-neutral-900"
                                        value={currentGraph.globalConfig?.aspectRatio || 'auto'}
                                        onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, aspectRatio: e.target.value } })}
                                    >
                                        <option value="auto">Auto (Fill)</option>
                                        <option value="square">Square (1:1)</option>
                                    </select>
                                </div>

                                <div className="pt-3 border-t border-neutral-200">
                                    <div className="flex items-center justify-between mb-2">
                                        <label className="text-xs font-semibold text-neutral-700 uppercase tracking-wider flex items-center gap-1.5">
                                            <Sliders size={13} className="text-blue-600" /> Viewport Bounds
                                        </label>
                                        <span className="text-[10px] font-mono text-neutral-400">Cartesian</span>
                                    </div>
                                    <div className="p-3 bg-neutral-50/80 border border-neutral-200 rounded-lg space-y-3">
                                        <div>
                                            <label className="text-[11px] font-medium text-neutral-600 block mb-1">X Domain [Min, Max]</label>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="flex items-center bg-white border border-neutral-200 rounded-md px-2 py-1 shadow-2xs">
                                                    <span className="text-[10px] text-neutral-400 font-mono mr-1">min</span>
                                                    <input
                                                        type="number"
                                                        step="any"
                                                        value={Number(currentDomain.x[0]).toFixed(2)}
                                                        onChange={(e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val)) {
                                                                setZoomDomain(prev => ({
                                                                    x: [val, prev.x[1] !== 'auto' ? prev.x[1] : currentDomain.x[1]],
                                                                    y: prev.y[0] !== 'auto' ? prev.y : currentDomain.y
                                                                }));
                                                            }
                                                        }}
                                                        className="w-full text-xs font-mono outline-none text-neutral-900 bg-transparent"
                                                    />
                                                </div>
                                                <div className="flex items-center bg-white border border-neutral-200 rounded-md px-2 py-1 shadow-2xs">
                                                    <span className="text-[10px] text-neutral-400 font-mono mr-1">max</span>
                                                    <input
                                                        type="number"
                                                        step="any"
                                                        value={Number(currentDomain.x[1]).toFixed(2)}
                                                        onChange={(e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val)) {
                                                                setZoomDomain(prev => ({
                                                                    x: [prev.x[0] !== 'auto' ? prev.x[0] : currentDomain.x[0], val],
                                                                    y: prev.y[0] !== 'auto' ? prev.y : currentDomain.y
                                                                }));
                                                            }
                                                        }}
                                                        className="w-full text-xs font-mono outline-none text-neutral-900 bg-transparent"
                                                    />
                                                </div>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="text-[11px] font-medium text-neutral-600 block mb-1">Y Range [Min, Max]</label>
                                            <div className="grid grid-cols-2 gap-2">
                                                <div className="flex items-center bg-white border border-neutral-200 rounded-md px-2 py-1 shadow-2xs">
                                                    <span className="text-[10px] text-neutral-400 font-mono mr-1">min</span>
                                                    <input
                                                        type="number"
                                                        step="any"
                                                        value={Number(currentDomain.y[0]).toFixed(2)}
                                                        onChange={(e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val)) {
                                                                setZoomDomain(prev => ({
                                                                    x: prev.x[0] !== 'auto' ? prev.x : currentDomain.x,
                                                                    y: [val, prev.y[1] !== 'auto' ? prev.y[1] : currentDomain.y[1]]
                                                                }));
                                                            }
                                                        }}
                                                        className="w-full text-xs font-mono outline-none text-neutral-900 bg-transparent"
                                                    />
                                                </div>
                                                <div className="flex items-center bg-white border border-neutral-200 rounded-md px-2 py-1 shadow-2xs">
                                                    <span className="text-[10px] text-neutral-400 font-mono mr-1">max</span>
                                                    <input
                                                        type="number"
                                                        step="any"
                                                        value={Number(currentDomain.y[1]).toFixed(2)}
                                                        onChange={(e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val)) {
                                                                setZoomDomain(prev => ({
                                                                    x: prev.x[0] !== 'auto' ? prev.x : currentDomain.x,
                                                                    y: [prev.y[0] !== 'auto' ? prev.y[0] : currentDomain.y[0], val]
                                                                }));
                                                            }
                                                        }}
                                                        className="w-full text-xs font-mono outline-none text-neutral-900 bg-transparent"
                                                    />
                                                </div>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-3 gap-1.5 pt-1">
                                            <button
                                                type="button"
                                                onClick={() => setZoomDomain({ x: [-10, 10], y: [-10, 10] })}
                                                className="py-1 px-2 text-[10px] font-mono font-medium bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-200 rounded transition-colors cursor-pointer shadow-2xs text-center"
                                            >
                                                [-10, 10]
                                            </button>
                                            <button
                                                type="button"
                                                onClick={() => setZoomDomain({ x: [-6.28, 6.28], y: [-3.5, 3.5] })}
                                                className="py-1 px-2 text-[10px] font-mono font-medium bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-200 rounded transition-colors cursor-pointer shadow-2xs text-center"
                                            >
                                                [-2π, 2π]
                                            </button>
                                            <button
                                                type="button"
                                                onClick={resetView}
                                                className="py-1 px-2 text-[10px] font-mono font-medium bg-white hover:bg-neutral-100 text-neutral-700 border border-neutral-200 rounded transition-colors cursor-pointer shadow-2xs text-center"
                                            >
                                                Fit Data
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <div className="pt-3 border-t border-neutral-200">
                                    <label className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={currentGraph.globalConfig?.showGrid !== false}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, showGrid: e.target.checked } })}
                                            className="rounded border border-neutral-300 text-neutral-900 cursor-pointer"
                                        />
                                        <span className="text-xs font-semibold text-neutral-800">Show Grid Lines</span>
                                    </label>
                                </div>

                                {currentGraph.globalConfig?.showGrid !== false && (
                                    <div className="space-y-3 pl-3 border-l-2 border-neutral-200">
                                        <div>
                                            <label className="text-xs font-medium text-neutral-600 block mb-1">X Grid Interval</label>
                                            <input
                                                type="number"
                                                step="any"
                                                placeholder="Auto"
                                                className="w-full p-2 text-xs bg-white border border-neutral-200 rounded-md font-mono outline-none focus:border-neutral-900"
                                                value={currentGraph.globalConfig?.xGridInterval || ''}
                                                onChange={e => setCurrentGraph({
                                                    ...currentGraph,
                                                    globalConfig: {
                                                        ...currentGraph.globalConfig,
                                                        xGridInterval: e.target.value ? parseFloat(e.target.value) : null
                                                    }
                                                })}
                                            />
                                        </div>
                                        <div>
                                            <label className="text-xs font-medium text-neutral-600 block mb-1">Y Grid Interval</label>
                                            <input
                                                type="number"
                                                step="any"
                                                placeholder="Auto"
                                                className="w-full p-2 text-xs bg-white border border-neutral-200 rounded-md font-mono outline-none focus:border-neutral-900"
                                                value={currentGraph.globalConfig?.yGridInterval || ''}
                                                onChange={e => setCurrentGraph({
                                                    ...currentGraph,
                                                    globalConfig: {
                                                        ...currentGraph.globalConfig,
                                                        yGridInterval: e.target.value ? parseFloat(e.target.value) : null
                                                    }
                                                })}
                                            />
                                        </div>
                                    </div>
                                )}

                                <div className="pt-3 border-t border-neutral-200 pb-12">
                                    <label className="text-xs font-semibold text-neutral-700 uppercase tracking-wider mb-2 block flex items-center gap-1.5">
                                        <StickyNote size={13} className="text-neutral-500" /> Annotations
                                    </label>
                                    <div className="space-y-2">
                                        {currentGraph.annotations?.map((note, idx) => (
                                            <div key={note.id} className="flex gap-1.5 items-center border border-neutral-200 rounded-md p-1.5 bg-white shadow-2xs">
                                                <input
                                                    className="w-12 p-1 text-xs bg-neutral-50 border border-neutral-200 rounded font-mono text-center outline-none"
                                                    value={note.x}
                                                    onChange={e => {
                                                        const newNotes = [...currentGraph.annotations];
                                                        newNotes[idx].x = e.target.value;
                                                        setCurrentGraph({ ...currentGraph, annotations: newNotes });
                                                    }}
                                                />
                                                <input
                                                    className="w-12 p-1 text-xs bg-neutral-50 border border-neutral-200 rounded font-mono text-center outline-none"
                                                    value={note.y}
                                                    onChange={e => {
                                                        const newNotes = [...currentGraph.annotations];
                                                        newNotes[idx].y = e.target.value;
                                                        setCurrentGraph({ ...currentGraph, annotations: newNotes });
                                                    }}
                                                />
                                                <input
                                                    className="flex-1 p-1 text-xs bg-neutral-50 border border-neutral-200 rounded font-sans outline-none"
                                                    value={note.text}
                                                    onChange={e => {
                                                        const newNotes = [...currentGraph.annotations];
                                                        newNotes[idx].text = e.target.value;
                                                        setCurrentGraph({ ...currentGraph, annotations: newNotes });
                                                    }}
                                                />
                                                <button onClick={() => setCurrentGraph(prev => ({ ...prev, annotations: prev.annotations.filter((_, i) => i !== idx) }))} className="text-neutral-400 hover:text-red-600 p-1 transition-colors cursor-pointer">
                                                    <X size={12} />
                                                </button>
                                            </div>
                                        ))}
                                        <Button variant="secondary" size="sm" className="w-full text-xs" onClick={() => setCurrentGraph(prev => ({ ...prev, annotations: [...(prev.annotations || []), { id: Date.now(), x: "0", y: "0", text: "Note" }] }))}>
                                            + Add Annotation
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex-1 flex flex-col overflow-hidden">
                                <div className="p-3 border-b border-neutral-200 flex justify-between items-center bg-white">
                                    <span className="text-xs font-semibold text-neutral-700">Datasets ({allDatasets.length})</span>
                                    <div className="flex gap-1.5">
                                        <button onClick={() => addDataset('function')} className="text-xs font-semibold rounded-md border border-neutral-200 bg-white hover:bg-neutral-50 hover:border-neutral-300 text-neutral-800 px-2.5 py-1.5 transition-all flex items-center gap-1.5 cursor-pointer shadow-2xs">
                                            <Sigma size={13} className="text-blue-600" /> + Function
                                        </button>
                                        <button onClick={() => addDataset('data')} className="text-xs font-semibold rounded-md border border-neutral-200 bg-white hover:bg-neutral-50 hover:border-neutral-300 text-neutral-800 px-2.5 py-1.5 transition-all flex items-center gap-1.5 cursor-pointer shadow-2xs">
                                            <Plus size={13} className="text-emerald-600" /> + Table
                                        </button>
                                    </div>
                                </div>
                                <div className="flex-1 overflow-y-auto p-3 space-y-3">
                                    {allDatasets.map((ds) => (
                                        <div key={ds.id} className="border border-neutral-200 rounded-xl bg-white shadow-xs overflow-hidden transition-all">
                                            <div className="flex items-center gap-2.5 p-3 bg-white border-b border-neutral-100 cursor-pointer hover:bg-neutral-50/70" onClick={() => setExpandedDatasetId(expandedDatasetId === ds.id ? null : ds.id)}>
                                                <button onClick={(e) => { e.stopPropagation(); updateDataset(ds.id, { visible: !ds.visible }); }} className={`p-1.5 rounded-md border transition-colors cursor-pointer ${ds.visible ? 'border-neutral-200 text-neutral-700 hover:bg-neutral-100' : 'border-neutral-200 bg-neutral-100 text-neutral-400'}`} title={ds.visible ? "Hide dataset" : "Show dataset"}>
                                                    {ds.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                                                </button>

                                                <div className="w-3.5 h-3.5 rounded-full ring-2 ring-white shadow-2xs shrink-0" style={{ backgroundColor: ds.color || '#2563EB' }}></div>

                                                <input
                                                    className="flex-1 bg-transparent text-xs font-semibold text-neutral-900 outline-none border-b border-transparent focus:border-neutral-400 py-0.5"
                                                    value={ds.name}
                                                    onClick={(e) => e.stopPropagation()}
                                                    onChange={(e) => updateDataset(ds.id, { name: e.target.value })}
                                                />

                                                <ChevronLeft size={16} className={`text-neutral-400 transition-transform ${expandedDatasetId === ds.id ? '-rotate-90' : ''}`} />
                                            </div>

                                            {expandedDatasetId === ds.id && (
                                                <div className="p-3.5 space-y-4">
                                                    {ds.config.type !== 'function' && (
                                                        <div className="border border-neutral-200 rounded-lg p-3 bg-neutral-50/60">
                                                            <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-neutral-600 border-b border-neutral-200 pb-1.5 mb-2.5">
                                                                <Activity size={13} className="text-blue-600" /> Dataset Statistics
                                                            </div>
                                                            <div className="grid grid-cols-2 gap-2">
                                                                <div className="bg-white border border-neutral-200 rounded-md p-2 shadow-2xs">
                                                                    <div className="text-[10px] font-semibold uppercase text-neutral-400">Mean X</div>
                                                                    <div className="text-xs font-mono font-bold text-neutral-900 mt-0.5">{formatNumber(ds.stats.meanX)}</div>
                                                                </div>
                                                                <div className="bg-white border border-neutral-200 rounded-md p-2 shadow-2xs">
                                                                    <div className="text-[10px] font-semibold uppercase text-neutral-400">Mean Y</div>
                                                                    <div className="text-xs font-mono font-bold text-neutral-900 mt-0.5">{formatNumber(ds.stats.meanY)}</div>
                                                                </div>
                                                                <div className="bg-white border border-neutral-200 rounded-md p-2 shadow-2xs">
                                                                    <div className="text-[10px] font-semibold uppercase text-neutral-400">Std Dev Y</div>
                                                                    <div className="text-xs font-mono font-bold text-neutral-900 mt-0.5">{formatNumber(ds.stats.stdDevY)}</div>
                                                                </div>
                                                                <div className="bg-white border border-neutral-200 rounded-md p-2 shadow-2xs">
                                                                    <div className="text-[10px] font-semibold uppercase text-neutral-400">Fit (R²)</div>
                                                                    <div className="text-xs font-mono font-bold text-blue-600 mt-0.5">{ds.r2 !== null ? formatNumber(ds.r2) : 'N/A'}</div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="grid grid-cols-2 gap-3">
                                                        <div>
                                                            <label className="text-xs font-medium text-neutral-600 mb-1 block">Graph Type</label>
                                                            {ds.config.type === 'function' ? (
                                                                <div className="w-full text-xs p-2 bg-neutral-100 border border-neutral-200 rounded-md font-mono text-neutral-700">Function</div>
                                                            ) : (
                                                                <select value={ds.config.type} onChange={e => updateDataset(ds.id, { config: { ...ds.config, type: e.target.value } })} className="w-full text-xs p-2 bg-white border border-neutral-200 rounded-md outline-none cursor-pointer text-neutral-800 focus:border-neutral-400">
                                                                    <option value="scatter">Scatter Plot</option>
                                                                    <option value="line">Connected Line</option>
                                                                    <option value="area">Filled Area</option>
                                                                </select>
                                                            )}
                                                        </div>
                                                        {ds.config.type !== 'function' && (
                                                            <div>
                                                                <label className="text-xs font-medium text-neutral-600 mb-1 block">Trendline Fit</label>
                                                                <select
                                                                    value={ds.config.showTrendline ? ds.config.trendlineType : 'none'}
                                                                    onChange={e => updateDataset(ds.id, { config: { ...ds.config, showTrendline: e.target.value !== 'none', trendlineType: e.target.value === 'none' ? 'linear' : e.target.value } })}
                                                                    className="w-full text-xs p-2 bg-white border border-neutral-200 rounded-md outline-none cursor-pointer text-neutral-800 focus:border-neutral-400"
                                                                >
                                                                    <option value="none">None</option>
                                                                    <option value="linear">Linear (y = mx + b)</option>
                                                                    <option value="quadratic">Quadratic (Polynomial)</option>
                                                                    <option value="exponential">Exponential</option>
                                                                    <option value="power">Power</option>
                                                                    <option value="logarithmic">Logarithmic</option>
                                                                </select>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {ds.config.type === 'function' ? (
                                                        <div className="space-y-3.5">
                                                            <div>
                                                                <div className="flex items-center justify-between mb-1">
                                                                    <label className="text-xs font-medium text-neutral-700">Equation: f(x) =</label>
                                                                    {ds.points?.error ? (
                                                                        <span className="text-[10px] font-mono text-red-600 bg-red-50 border border-red-200 px-1.5 py-0.5 rounded shadow-2xs">Syntax Error</span>
                                                                    ) : (
                                                                        <span className="text-[10px] font-mono text-emerald-600 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded shadow-2xs">Valid Equation</span>
                                                                    )}
                                                                </div>
                                                                <input
                                                                    className="w-full p-2 bg-white border border-neutral-200 rounded-md font-mono text-xs text-neutral-900 outline-none focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900 shadow-2xs"
                                                                    value={ds.equation || ''}
                                                                    onChange={e => {
                                                                        const eq = e.target.value;
                                                                        updateDataset(ds.id, {
                                                                            equation: eq,
                                                                            name: eq ? `f(x) = ${eq}` : 'f(x)'
                                                                        });
                                                                    }}
                                                                    onClick={e => e.stopPropagation()}
                                                                    placeholder="sin(x) * x"
                                                                />
                                                                {ds.points?.error ? (
                                                                    <p className="text-[11px] font-mono text-red-600 mt-1">
                                                                        {ds.points.error.message || 'Check equation syntax'}
                                                                    </p>
                                                                ) : (
                                                                    <p className="text-[11px] text-neutral-400 mt-1">Supported: sin, cos, tan, log, sqrt, abs, pi, e, ^</p>
                                                                )}
                                                            </div>

                                                            {/* Quick Expression Preset Chips */}
                                                            <div>
                                                                <div className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider mb-1.5">Equation Presets</div>
                                                                <div className="flex flex-wrap gap-1">
                                                                    {FUNCTION_PRESETS.map(preset => (
                                                                        <button
                                                                            key={preset.label}
                                                                            type="button"
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                updateDataset(ds.id, {
                                                                                    equation: preset.expr,
                                                                                    name: `f(x) = ${preset.expr}`
                                                                                });
                                                                            }}
                                                                            className="px-2 py-0.5 text-[11px] font-mono rounded-md bg-neutral-100 hover:bg-neutral-200 text-neutral-700 hover:text-neutral-900 border border-neutral-200 transition-colors cursor-pointer shadow-2xs"
                                                                        >
                                                                            {preset.label}
                                                                        </button>
                                                                    ))}
                                                                </div>
                                                            </div>

                                                            {/* Live Function Evaluator */}
                                                            <div>
                                                                <div className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider mb-1.5">Function Evaluation</div>
                                                                <FunctionEvaluator equation={ds.equation} />
                                                            </div>

                                                            {/* High-contrast Color Swatches */}
                                                            <div>
                                                                <div className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider mb-1.5">Curve Color</div>
                                                                <div className="flex items-center gap-2">
                                                                    {THEMES.map(theme => (
                                                                        <button
                                                                            key={theme.name}
                                                                            type="button"
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                updateDataset(ds.id, { color: theme.color });
                                                                            }}
                                                                            className={`w-5 h-5 rounded-full border transition-transform cursor-pointer shadow-2xs ${
                                                                                ds.color === theme.color ? 'ring-2 ring-neutral-900 scale-110' : 'border-neutral-200 hover:scale-105'
                                                                            }`}
                                                                            style={{ backgroundColor: theme.color }}
                                                                            title={theme.name}
                                                                        />
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ) : (
                                                        <div className="grid grid-cols-2 gap-3">
                                                            <div>
                                                                <label className="text-xs font-medium text-neutral-600 mb-1 block">X Variable</label>
                                                                <select value={ds.config.xKey} onChange={e => updateDataset(ds.id, { config: { ...ds.config, xKey: e.target.value } })} className="w-full text-xs p-2 bg-white border border-neutral-200 rounded-md outline-none cursor-pointer text-neutral-800 focus:border-neutral-400">
                                                                    {Object.keys(ds.data[0] || {}).map(k => <option key={k} value={k}>{k}</option>)}
                                                                </select>
                                                            </div>
                                                            <div>
                                                                <label className="text-xs font-medium text-neutral-600 mb-1 block">Y Variable</label>
                                                                <select value={ds.config.yKey} onChange={e => updateDataset(ds.id, { config: { ...ds.config, yKey: e.target.value } })} className="w-full text-xs p-2 bg-white border border-neutral-200 rounded-md outline-none cursor-pointer text-neutral-800 focus:border-neutral-400">
                                                                    {Object.keys(ds.data[0] || {}).map(k => <option key={k} value={k}>{k}</option>)}
                                                                </select>
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="flex flex-col gap-3 border-t border-neutral-100 pt-3">
                                                        <div className="flex flex-col gap-1.5">
                                                            <span className="text-xs font-medium text-neutral-700">Plot Color</span>
                                                            <div className="flex flex-wrap gap-2">
                                                                {THEMES.map(t => (
                                                                    <button 
                                                                        key={t.color} 
                                                                        onClick={() => updateDataset(ds.id, { color: t.color })} 
                                                                        className={`w-6 h-6 rounded-md border border-black/10 transition-transform hover:scale-110 cursor-pointer ${ds.color === t.color ? 'ring-2 ring-neutral-900 ring-offset-2 scale-110' : ''}`} 
                                                                        style={{ backgroundColor: t.color }} 
                                                                        title={t.name}
                                                                    />
                                                                ))}
                                                            </div>
                                                        </div>
                                                        {ds.config.type !== 'function' && (
                                                            <div className="flex flex-col gap-1.5">
                                                                <span className="text-xs font-medium text-neutral-700">Trendline Color</span>
                                                                <div className="flex flex-wrap gap-2">
                                                                    {TRENDLINE_THEMES.map(t => (
                                                                        <button 
                                                                            key={t.color} 
                                                                            onClick={() => updateDataset(ds.id, { config: { ...ds.config, trendlineColor: t.color } })} 
                                                                            className={`w-6 h-6 rounded-md border border-black/10 transition-transform hover:scale-110 cursor-pointer ${ds.config.trendlineColor === t.color ? 'ring-2 ring-neutral-900 ring-offset-2 scale-110' : ''}`} 
                                                                            style={{ backgroundColor: t.color }} 
                                                                            title={t.name}
                                                                        />
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {ds.config.type !== 'function' && (
                                                        <div className="border border-neutral-200 rounded-lg overflow-hidden">
                                                            <div className="bg-neutral-50 px-3 py-2 border-b border-neutral-200 flex justify-between items-center">
                                                                <span className="text-xs font-semibold text-neutral-700 flex items-center gap-1.5">
                                                                    <Table size={13} className="text-neutral-500" /> Data Points ({ds.data.length})
                                                                </span>
                                                                <button onClick={() => {
                                                                    const newRow = Object.keys(ds.data[0] || { [ds.config.xKey]: 0, [ds.config.yKey]: 0 }).reduce((acc, k) => ({ ...acc, [k]: 0 }), {});
                                                                    updateDataset(ds.id, { data: [...ds.data, newRow] });
                                                                }} className="rounded-md bg-neutral-900 text-white hover:bg-neutral-800 px-2.5 py-1 text-xs font-medium flex items-center gap-1 transition-colors cursor-pointer shadow-2xs">
                                                                    + Add Row
                                                                </button>
                                                            </div>
                                                            <div className="max-h-52 overflow-y-auto">
                                                                <table className="w-full text-xs">
                                                                    <thead className="bg-neutral-50/80 border-b border-neutral-200 sticky top-0">
                                                                        <tr>
                                                                            <th className="p-2 text-left font-semibold text-neutral-600 border-r border-neutral-200 w-1/2">{ds.config.xKey}</th>
                                                                            <th className="p-2 text-left font-semibold text-neutral-600 w-1/2">{ds.config.yKey}</th>
                                                                        </tr>
                                                                    </thead>
                                                                    <tbody>
                                                                        {ds.data.map((row, rIdx) => (
                                                                            <tr key={rIdx} className="border-b border-neutral-100 last:border-b-0 hover:bg-neutral-50/80">
                                                                                <td className="p-0 border-r border-neutral-100 relative">
                                                                                    <input className="w-full px-2 py-1.5 bg-transparent outline-none font-mono text-xs text-neutral-800" value={row[ds.config.xKey]} onChange={e => {
                                                                                        const nd = [...ds.data]; nd[rIdx] = { ...nd[rIdx], [ds.config.xKey]: e.target.value };
                                                                                        updateDataset(ds.id, { data: nd });
                                                                                    }} />
                                                                                </td>
                                                                                <td className="p-0 relative flex items-center">
                                                                                    <input className="w-full px-2 py-1.5 bg-transparent outline-none font-mono text-xs text-neutral-800" value={row[ds.config.yKey]} onChange={e => {
                                                                                        const nd = [...ds.data]; nd[rIdx] = { ...nd[rIdx], [ds.config.yKey]: e.target.value };
                                                                                        updateDataset(ds.id, { data: nd });
                                                                                    }} />
                                                                                    <button onClick={() => updateDataset(ds.id, { data: ds.data.filter((_, i) => i !== rIdx) })} className="p-1.5 text-neutral-300 hover:text-red-600 mr-1 cursor-pointer transition-colors" title="Delete row">
                                                                                        <Trash2 size={13} />
                                                                                    </button>
                                                                                </td>
                                                                            </tr>
                                                                        ))}
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="pt-2 border-t border-neutral-100 flex justify-end">
                                                        <button onClick={(e) => deleteDataset(ds.id, e)} className="rounded-md border border-red-200 text-red-600 hover:bg-red-50 px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer">
                                                            <Trash2 size={13} /> Delete Dataset
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center p-12 text-center h-[calc(100vh-56px)] bg-neutral-50/50">
                        <div className="max-w-md border border-neutral-200 rounded-2xl p-8 bg-white shadow-xs text-left">
                            <div className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-neutral-100 text-neutral-600 mb-3">
                                2D PLOTTER // NO ACTIVE GRAPH
                            </div>
                            <h2 className="text-xl font-bold text-neutral-900 mb-2">No Plot Loaded</h2>
                            <p className="text-xs text-neutral-600 mb-6 leading-relaxed">
                                Plot experimental data tables, calculate regression trendlines, or plot mathematical functions on the 2D coordinate plane.
                            </p>
                            <div className="flex flex-wrap gap-2">
                                <Button variant="primary" size="md" onClick={createBlankGraph} icon={Plus}>Create Data Plot</Button>
                                <Button variant="secondary" size="md" onClick={() => createFunctionGraph('sin(x) * x')} icon={Calculator}>Plot Function</Button>
                                <Button variant="secondary" size="md" onClick={() => { setIsImporting(false); setView('scan'); }} icon={Camera}>Scan Image</Button>
                            </div>
                        </div>
                    </div>
                )
            )}
            {/* API Key Modal Component */}
            <ApiKeyModal
                isOpen={showApiKeyModal}
                onClose={() => setShowApiKeyModal(false)}
                onSave={() => {}}
            />
            {/* AI Chatbox with real application tools */}
            <AIChatbox
                onPlotFunction={handleAIPlotFunction}
                onPlotImplicit={handleAIPlotImplicit}
                onLoadDataTable={handleAILoadDataTable}
                onSwitchTo3D={handleAISwitchTo3D}
                onSetViewportBounds={handleAISetViewportBounds}
            />
        </div>
    );
}
