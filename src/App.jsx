import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
    Camera, BarChart2, FileText, Check, AlertCircle, RefreshCw,
    ChevronRight, Zap, Settings, Save, Download, Edit2, Plus, Trash2,
    LogOut, Layout, TrendingUp, Grid, Type, Palette, ZoomIn, Home,
    MoreVertical, Share2, ChevronLeft, Calculator, Move, MousePointer2,
    ArrowRightLeft, Eye, EyeOff, Table, Activity, X, FilePlus, FileSpreadsheet, StickyNote, Menu, Sigma, Info, RotateCcw, Minus
} from 'lucide-react';
import { GoogleGenerativeAI } from "@google/generative-ai";
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis,
    CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart,
    Scatter, ComposedChart, ReferenceLine, ReferenceDot, Label, ErrorBar, LabelList
} from 'recharts';
import { FlickeringGrid } from './components/ui/flickering-grid';
import { LiquidGlassCard } from './components/ui/liquid-glass-card';

// SECURITY: Import security utilities for rate limiting, validation, and API key handling
import {
    rateLimiter,
    InputValidator,
    getGeminiApiKey,
    isApiKeyConfigured,
    createRateLimitError,
    createValidationError
} from './lib/security';

// --- CONFIGURATION ---
// SECURITY: API key now loaded from environment variables (see .env file)
// Never hardcode API keys in source code
const LOCAL_STORAGE_KEY = "graphly_local_data";

// --- MATH & UTILS ---

const THEMES = [
    { name: "Ocean", color: "#0ea5e9" },
    { name: "Sunset", color: "#f97316" },
    { name: "Forest", color: "#10b981" },
    { name: "Berry", color: "#d946ef" },
    { name: "Midnight", color: "#6366f1" },
    { name: "Cherry", color: "#ef4444" },
    { name: "Teal", color: "#14b8a6" },
    { name: "Noir", color: "#1e293b" }
];

const calculateNiceTicks = (min, max, maxTicks = 8) => {
    if (min === max || min === 'auto' || max === 'auto') return [];
    const range = max - min;
    if (range <= 0) return [min];
    const roughStep = range / (maxTicks - 1);
    const exponent = Math.floor(Math.log10(roughStep));
    const fraction = roughStep / Math.pow(10, exponent);
    let niceFraction;
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
    const step = niceFraction * Math.pow(10, exponent);
    const start = Math.ceil(min / step) * step;
    const end = Math.floor(max / step) * step;
    const ticks = [];
    const epsilon = step / 1000;
    for (let t = start; t <= end + epsilon; t += step) {
        ticks.push(t);
    }
    return ticks;
};

const formatNumber = (num) => {
    if (typeof num !== 'number') return num;
    if (num === 0) return 0;
    const abs = Math.abs(num);
    if (abs >= 10000 || abs < 0.001) {
        return num.toExponential(2);
    }
    return parseFloat(num.toFixed(3));
};

const formatEquationNumber = (num) => {
    if (Math.abs(num) < 0.001 && num !== 0) return num.toExponential(2);
    return num.toFixed(3);
};

// Full Regression Suite
const getRegressionParams = (points, type = 'linear') => {
    const n = points.length;
    const validPoints = points.filter(p => !isNaN(p.x) && !isNaN(p.y)).sort((a, b) => a.x - b.x);
    if (validPoints.length < 2) return null;

    let r2 = null;

    const calculateR2 = (predictedY, actualY) => {
        const yMean = actualY.reduce((a, b) => a + b, 0) / actualY.length;
        const ssRes = actualY.reduce((sum, y, i) => sum + Math.pow(y - predictedY[i], 2), 0);
        const ssTot = actualY.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
        return ssTot === 0 ? 0 : 1 - (ssRes / ssTot);
    };

    // Determinant helper
    const det3x3 = (m) => {
        return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };

    if (type === 'linear') {
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        validPoints.forEach(p => { sumX += p.x; sumY += p.y; sumXY += p.x * p.y; sumX2 += p.x * p.x; });
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        const preds = validPoints.map(p => slope * p.x + intercept);
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, slope, intercept, r2, equation: `y = ${formatEquationNumber(slope)}x + ${formatEquationNumber(intercept)}` };
    }
    else if (type === 'quadratic') {
        let s00 = n, s10 = 0, s20 = 0, s30 = 0, s40 = 0, s01 = 0, s11 = 0, s21 = 0;
        validPoints.forEach(p => {
            const x = p.x; const y = p.y;
            s10 += x; s20 += x * x; s30 += x * x * x; s40 += x * x * x * x;
            s01 += y; s11 += x * y; s21 += x * x * y;
        });
        const M = [[s00, s10, s20], [s10, s20, s30], [s20, s30, s40]];
        const Det = det3x3(M);
        if (Det === 0) return null;
        const c = det3x3([[s01, s10, s20], [s11, s20, s30], [s21, s30, s40]]) / Det;
        const b = det3x3([[s00, s01, s20], [s10, s11, s30], [s20, s21, s40]]) / Det;
        const a = det3x3([[s00, s10, s01], [s10, s20, s11], [s20, s30, s21]]) / Det;

        const preds = validPoints.map(p => a * p.x * p.x + b * p.x + c);
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, a, b, c, r2, equation: `y = ${formatEquationNumber(a)}x² + ${formatEquationNumber(b)}x + ${formatEquationNumber(c)}` };
    }
    else if (type === 'exponential') {
        const v = validPoints.filter(p => p.y > 0);
        if (v.length < 2) return null;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        const N = v.length;
        v.forEach(p => {
            const lny = Math.log(p.y);
            sumX += p.x; sumY += lny; sumXY += p.x * lny; sumX2 += p.x * p.x;
        });
        const b = (N * sumXY - sumX * sumY) / (N * sumX2 - sumX * sumX);
        const a = Math.exp((sumY - b * sumX) / N);

        const preds = validPoints.map(p => a * Math.exp(b * p.x));
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, a, b, r2, equation: `y = ${formatEquationNumber(a)}e^(${formatEquationNumber(b)}x)` };
    }
    else if (type === 'power') {
        const v = validPoints.filter(p => p.x > 0 && p.y > 0);
        if (v.length < 2) return null;
        let sumlnX = 0, sumlnY = 0, sumlnXlnY = 0, sumlnX2 = 0;
        const N = v.length;
        v.forEach(p => {
            const lnx = Math.log(p.x);
            const lny = Math.log(p.y);
            sumlnX += lnx; sumlnY += lny; sumlnXlnY += lnx * lny; sumlnX2 += lnx * lnx;
        });
        const b = (N * sumlnXlnY - sumlnX * sumlnY) / (N * sumlnX2 - sumlnX * sumlnX);
        const a = Math.exp((sumlnY - b * sumlnX) / N);

        const preds = validPoints.map(p => a * Math.pow(p.x, b));
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, a, b, r2, equation: `y = ${formatEquationNumber(a)}x^${formatEquationNumber(b)}` };
    }
    else if (type === 'logarithmic') {
        const v = validPoints.filter(p => p.x > 0);
        if (v.length < 2) return null;
        let sumLnX = 0, sumY = 0, sumLnXY = 0, sumLnX2 = 0;
        const N = v.length;
        v.forEach(p => {
            const lnx = Math.log(p.x);
            sumLnX += lnx; sumY += p.y; sumLnXY += lnx * p.y; sumLnX2 += lnx * lnx;
        });
        const b = (N * sumLnXY - sumLnX * sumY) / (N * sumLnX2 - sumLnX * sumLnX);
        const a = (sumY - b * sumLnX) / N;

        const preds = validPoints.map(p => a + b * Math.log(p.x));
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, a, b, r2, equation: `y = ${formatEquationNumber(a)} + ${formatEquationNumber(b)}ln(x)` };
    }
    return null;
};

const generateTrendlineData = (params, xMin, xMax, yMinData, yMaxData) => {
    if (!params || xMin === 'auto' || xMax === 'auto') return [];
    const points = [];
    const resolution = 150;
    const step = (xMax - xMin) / (resolution - 1);
    const yRange = Math.abs(yMaxData - yMinData) || 10;
    const SAFE_MAX = yMaxData + (yRange * 5);
    const SAFE_MIN = yMinData - (yRange * 5);

    for (let i = 0; i < resolution; i++) {
        const x = xMin + (i * step);
        let y = null;
        if (params.type === 'linear') y = params.slope * x + params.intercept;
        else if (params.type === 'quadratic') y = params.a * x * x + params.b * x + params.c;
        else if (params.type === 'exponential') y = params.a * Math.exp(params.b * x);
        else if (params.type === 'power') y = params.a * Math.pow(x, params.b);
        else if (params.type === 'logarithmic') {
            if (x > 0) y = params.a + params.b * Math.log(x);
        }

        if (y !== null && !isNaN(y)) {
            if (y > SAFE_MAX) y = SAFE_MAX;
            if (y < SAFE_MIN) y = SAFE_MIN;
            points.push({ x, y });
        }
    }
    return points;
};

// Calculate Basic Statistics
const calculateStats = (data, xKey, yKey) => {
    if (!data) return { meanX: 0, meanY: 0, stdDevX: 0, stdDevY: 0, n: 0 };
    const validData = data.filter(d => !isNaN(parseFloat(d[xKey])) && !isNaN(parseFloat(d[yKey])));
    const n = validData.length;
    if (n === 0) return { meanX: 0, meanY: 0, stdDevX: 0, stdDevY: 0, n: 0 };

    const sumX = validData.reduce((acc, val) => acc + parseFloat(val[xKey]), 0);
    const sumY = validData.reduce((acc, val) => acc + parseFloat(val[yKey]), 0);
    const meanX = sumX / n;
    const meanY = sumY / n;

    const sumSqDiffX = validData.reduce((acc, val) => acc + Math.pow(parseFloat(val[xKey]) - meanX, 2), 0);
    const sumSqDiffY = validData.reduce((acc, val) => acc + Math.pow(parseFloat(val[yKey]) - meanY, 2), 0);

    const stdDevX = Math.sqrt(sumSqDiffX / n);
    const stdDevY = Math.sqrt(sumSqDiffY / n);

    return { meanX, meanY, stdDevX, stdDevY, n };
};

// Function Evaluator for Graphing
const generateFunctionPoints = (equation, xMin = -10, xMax = 10, resolution = 200) => {
    try {
        const points = [];
        const step = (xMax - xMin) / resolution;

        let jsEq = equation.toLowerCase()
            .replace(/\s+/g, '') // remove spaces
            .replace(/\^/g, '**')
            .replace(/(\d)([a-z(])/g, '$1*$2')
            .replace(/(\))([a-z0-9])/g, '$1*$2')
            .replace(/sin/g, 'Math.sin')
            .replace(/cos/g, 'Math.cos')
            .replace(/tan/g, 'Math.tan')
            .replace(/log/g, 'Math.log10')
            .replace(/ln/g, 'Math.log')
            .replace(/sqrt/g, 'Math.sqrt')
            .replace(/abs/g, 'Math.abs')
            .replace(/pi/g, 'Math.PI')
            .replace(/e/g, 'Math.E');

        const f = new Function('x', `try { return ${jsEq}; } catch(e) { return NaN; }`);

        for (let x = xMin; x <= xMax; x += step) {
            const y = f(x);
            if (!isNaN(y) && isFinite(y)) {
                points.push({ x, y });
            }
        }
        return points;
    } catch (e) {
        return [];
    }
};

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

// --- UI COMPONENTS ---

const Card = ({ children, className = "" }) => (
    <div className={`bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden ${className}`}>{children}</div>
);

const Button = ({ onClick, children, variant = "primary", disabled = false, icon: Icon, size = "md", className = "" }) => {
    const sizes = { sm: "py-1.5 px-3 text-xs", md: "py-2.5 px-4 text-sm", lg: "py-3.5 px-6 text-base", icon: "p-2" };
    const variants = {
        primary: "bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm hover:shadow-lg hover:-translate-y-0.5",
        secondary: "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50 hover:border-slate-300 hover:shadow-md",
        danger: "bg-red-50 text-red-600 hover:bg-red-100",
        ghost: "text-slate-500 hover:bg-slate-100",
        glow: "bg-indigo-600 text-white btn-glow shadow-lg hover:shadow-xl",
        glowSecondary: "bg-white text-slate-700 border-2 border-slate-200 btn-glow-secondary hover:border-indigo-400"
    };
    return (
        <button onClick={onClick} disabled={disabled} className={`font-medium flex items-center justify-center gap-2 rounded-xl transition-all duration-300 active:scale-95 disabled:opacity-50 ${sizes[size]} ${variants[variant]} ${className}`}>
            {Icon && <Icon size={size === 'sm' ? 14 : size === 'lg' ? 20 : 18} />}
            {children}
        </button>
    );
};

// --- MAIN APP ---

export default function App() {
    const [view, setView] = useState('dashboard');
    const [currentGraph, setCurrentGraph] = useState(null);
    const [isImporting, setIsImporting] = useState(false);
    const [showCSVModal, setShowCSVModal] = useState(false);
    const [csvText, setCsvText] = useState("");
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    // New State for graph dimensions to handle square aspect ratio
    const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });
    const graphContainerRef = useRef(null);
    const lastTouchDistance = useRef(null);

    useEffect(() => {
        document.title = "Graphly";
    }, []);

    // Force mobile version and lock browser zoom on non-desktop devices
    useEffect(() => {
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
            (navigator.maxTouchPoints > 0 && navigator.platform === 'MacIntel');

        if (isMobile) {
            // Prevent browser zoom on mobile
            document.body.style.touchAction = 'pan-x pan-y';

            // Prevent pinch zoom outside graph area
            const preventZoom = (e) => {
                if (e.touches.length > 1) {
                    e.preventDefault();
                }
            };

            document.addEventListener('touchmove', preventZoom, { passive: false });

            return () => {
                document.removeEventListener('touchmove', preventZoom);
                document.body.style.touchAction = '';
            };
        }
    }, []);

    const [savedGraphs, setSavedGraphs] = useState([]);
    const [image, setImage] = useState(null);
    const [scanStatus, setScanStatus] = useState("idle");
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
        setCurrentGraph(newGraph);
        setExpandedDatasetId(newGraph.datasets[0].id);
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
        const apiKey = getGeminiApiKey();
        if (!apiKey) {
            setScanStatus("error");
            alert("Gemini API key is not configured. Please set VITE_GEMINI_API_KEY in your .env file.");
            return;
        }

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
            const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-09-2025" });
            const prompt = `
        Analyze this datasheet/image. Extract tabular data.
        If there are multiple tables or distinct sections, create separate datasets.
        
        Return JSON:
        {
          "title": "Document Title",
          "datasets": [
            {
              "name": "Table 1 (e.g. Forward Bias)",
              "data": [{"Voltage": 0.1, "Current": 0.01}, ...]
            }
          ]
        }
        Strip units. Return ONLY pure JSON. No comments.
      `;

            const result = await model.generateContent([prompt, { inlineData: { data: image.base64, mimeType: image.file.type } }]);
            let text = result.response.text();
            text = text.replace(/```json/g, '').replace(/```/g, '').replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '').trim();
            const parsed = JSON.parse(text);

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
        const xPad = (globalBounds.xMax - globalBounds.xMin) * 0.1 || 1;
        const yPad = (globalBounds.yMax - globalBounds.yMin) * 0.1 || 1;

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
            const ticks = [];
            const start = Math.ceil(currentDomain.x[0] / interval) * interval;
            for (let t = start; t <= currentDomain.x[1]; t += interval) {
                ticks.push(t);
            }
            return ticks;
        }
        return calculateNiceTicks(currentDomain.x[0], currentDomain.x[1]);
    }, [currentDomain, currentGraph?.globalConfig?.xGridInterval]);

    const yTicks = useMemo(() => {
        const interval = currentGraph?.globalConfig?.yGridInterval;
        if (interval && interval > 0) {
            const ticks = [];
            const start = Math.ceil(currentDomain.y[0] / interval) * interval;
            for (let t = start; t <= currentDomain.y[1]; t += interval) {
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
                points = generateFunctionPoints(ds.equation, currentDomain.x[0] - buffer, currentDomain.x[1] + buffer);
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

    const handleMouseDown = (e) => {
        if (selectedData) return;
        if (!currentGraph?.globalConfig?.enableZoom) return;
        setIsDragging(true);
        lastMousePos.current = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;
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
    };

    // TOUCH SUPPORT
    const handleTouchStart = (e) => {
        if (selectedData) return;
        if (!currentGraph?.globalConfig?.enableZoom) return;

        if (e.touches.length === 2) {
            // Pinch start
            const d = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
            lastTouchDistance.current = d;
        } else {
            setIsDragging(true);
            lastMousePos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        }
    };

    const handleTouchMove = (e) => {
        if (e.touches.length === 2 && lastTouchDistance.current) {
            // Pinch Zoom
            const d = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );

            const delta = d - lastTouchDistance.current;
            const scale = delta > 0 ? 0.95 : 1.05; // Pinch out (zoom in) vs Pinch in (zoom out)
            lastTouchDistance.current = d;

            const xR = currentDomain.x[1] - currentDomain.x[0];
            const yR = currentDomain.y[1] - currentDomain.y[0];
            const xM = (currentDomain.x[1] + currentDomain.x[0]) / 2;
            const yM = (currentDomain.y[1] + currentDomain.y[0]) / 2;

            setZoomDomain({
                x: [xM - (xR * scale) / 2, xM + (xR * scale) / 2],
                y: [yM - (yR * scale) / 2, yM + (yR * scale) / 2]
            });
            return;
        }

        if (!isDragging) return;
        const dx = e.touches[0].clientX - lastMousePos.current.x;
        const dy = e.touches[0].clientY - lastMousePos.current.y;
        const xR = currentDomain.x[1] - currentDomain.x[0];
        const yR = currentDomain.y[1] - currentDomain.y[0];
        const xS = -1 * (dx / (containerSize.width || 500)) * xR;
        const yS = (dy / (containerSize.height || 300)) * yR;
        setZoomDomain({
            x: [currentDomain.x[0] + xS, currentDomain.x[1] + xS],
            y: [currentDomain.y[0] + yS, currentDomain.y[1] + yS]
        });
        lastMousePos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
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

            <nav className={`sticky top-0 z-50 h-16 flex items-center justify-between px-4 md:px-6 transition-all duration-500 ${view === 'dashboard' && savedGraphs.length === 0 ? 'bg-transparent border-transparent' : 'bg-white/80 backdrop-blur-md border-b border-slate-200/60'}`}>
                <div className="flex items-center gap-3 cursor-pointer group" onClick={() => setView('dashboard')}>
                    <div className="w-9 h-9 bg-gradient-to-tr from-indigo-600 to-violet-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-200/50 border border-white/20 transition-transform duration-300 group-hover:scale-110">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                            <circle cx="5" cy="19" r="2" />
                            <circle cx="19" cy="5" r="2" />
                            <circle cx="19" cy="19" r="2" />
                            <path d="M5 19L19 5" />
                            <path d="M5 19L19 19" />
                        </svg>
                    </div>
                    <span className="font-bold text-xl tracking-tight text-slate-900 hidden sm:inline transition-colors duration-300 group-hover:text-indigo-600">Graphly</span>
                </div>
                <div className="flex gap-2">
                    {view === 'editor' && (
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
                                    ctx.fillStyle = "#334155";
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
                            <Button size="sm" icon={Save} onClick={saveGraph}><span className="hidden sm:inline">Save</span></Button>
                            <Button size="sm" variant="secondary" icon={FilePlus} onClick={startImport}><span className="hidden sm:inline">Import</span></Button>
                        </>
                    )}
                    {view === 'dashboard' && savedGraphs.length > 0 && (
                        <div className="flex gap-2 animate-fade-in">
                            <Button onClick={createBlankGraph} icon={Plus} size="sm" className="transition-all duration-300 hover:scale-105"><span className="hidden sm:inline">Create</span></Button>
                            <Button onClick={() => { setIsImporting(false); setView('scan'); }} variant="secondary" icon={Camera} size="sm" className="transition-all duration-300 hover:scale-105"><span className="hidden sm:inline">Scan</span></Button>
                        </div>
                    )}
                </div>
            </nav>

            {/* Full-page background for landing - extends behind header */}
            {view === 'dashboard' && savedGraphs.length === 0 && (
                <div className="fixed inset-0 z-0 pointer-events-none">
                    {/* Animated Flickering Grid Background - Full Viewport */}
                    <FlickeringGrid
                        className="absolute inset-0"
                        squareSize={4}
                        gridGap={6}
                        color="#6366f1"
                        maxOpacity={0.15}
                        flickerChance={0.1}
                    />

                    {/* Gradient Orbs */}
                    <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-gradient-to-br from-indigo-200/50 to-violet-200/50 rounded-full blur-3xl -translate-y-1/3 translate-x-1/3 pointer-events-none"></div>
                    <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-gradient-to-tr from-blue-200/40 to-cyan-200/40 rounded-full blur-3xl translate-y-1/3 -translate-x-1/3 pointer-events-none"></div>

                    {/* Colorful gradient band at bottom for glassmorphism visibility */}
                    <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-indigo-500/30 via-violet-400/20 to-transparent pointer-events-none"></div>
                </div>
            )}

            {view === 'dashboard' && (
                <main className={`max-w-6xl mx-auto px-4 py-6 md:px-6 md:py-10 ${savedGraphs.length === 0 ? 'overflow-hidden max-w-none px-0 py-0 h-screen' : 'h-[calc(100vh-64px)] overflow-y-auto'}`}>
                    {savedGraphs.length === 0 ? (
                        /* ==========================================
                           LANDING PAGE - Full Screen Hero Only
                           ========================================== */
                        <div
                            className="h-full relative flex items-start px-4 pt-8"
                            onMouseMove={(e) => setMousePos({ x: e.clientX, y: e.clientY })}
                        >
                            {/* Hero Content Section */}
                            <section className="relative w-full max-w-6xl mx-auto mt-8">
                                {/* Content is now relative to the fixed background */}

                                {/* Interactive Floating Elements - Math symbols + Graph elements */}
                                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                                    {/* Sigma Symbol */}
                                    <div
                                        className="absolute text-indigo-200/60 text-6xl font-serif select-none transition-transform duration-700 ease-out"
                                        style={{
                                            top: '5%',
                                            left: '5%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.02}px, ${(mousePos.y - window.innerHeight / 2) * 0.02}px)`
                                        }}
                                    >∑</div>

                                    {/* Pi Symbol */}
                                    <div
                                        className="absolute text-violet-200/50 text-5xl font-serif select-none transition-transform duration-500 ease-out"
                                        style={{
                                            top: '15%',
                                            right: '12%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.03}px, ${(mousePos.y - window.innerHeight / 2) * 0.025}px)`
                                        }}
                                    >π</div>

                                    {/* Integral Symbol */}
                                    <div
                                        className="absolute text-blue-200/50 text-7xl font-serif select-none transition-transform duration-600 ease-out"
                                        style={{
                                            bottom: '25%',
                                            left: '3%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.015}px, ${(mousePos.y - window.innerHeight / 2) * -0.02}px)`
                                        }}
                                    >∫</div>

                                    {/* Delta Symbol */}
                                    <div
                                        className="absolute text-indigo-300/40 text-4xl font-serif select-none transition-transform duration-800 ease-out"
                                        style={{
                                            top: '55%',
                                            right: '8%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.025}px, ${(mousePos.y - window.innerHeight / 2) * -0.015}px)`
                                        }}
                                    >Δ</div>

                                    {/* Mini Bar Chart - Interactive SVG */}
                                    <div
                                        className="absolute select-none transition-transform duration-500 ease-out"
                                        style={{
                                            top: '8%',
                                            left: '25%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.025}px, ${(mousePos.y - window.innerHeight / 2) * 0.02}px)`
                                        }}
                                    >
                                        <svg width="60" height="50" viewBox="0 0 60 50" className="opacity-30">
                                            <rect x="5" y="30" width="10" height="20" fill="#6366f1" rx="2" />
                                            <rect x="20" y="15" width="10" height="35" fill="#8b5cf6" rx="2" />
                                            <rect x="35" y="25" width="10" height="25" fill="#a855f7" rx="2" />
                                            <rect x="50" y="10" width="10" height="40" fill="#6366f1" rx="2" />
                                        </svg>
                                    </div>

                                    {/* Mini Line Chart */}
                                    <div
                                        className="absolute select-none transition-transform duration-600 ease-out"
                                        style={{
                                            bottom: '35%',
                                            right: '20%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.03}px, ${(mousePos.y - window.innerHeight / 2) * 0.025}px)`
                                        }}
                                    >
                                        <svg width="80" height="40" viewBox="0 0 80 40" className="opacity-25">
                                            <polyline
                                                points="5,35 20,25 35,30 50,15 65,20 75,5"
                                                fill="none"
                                                stroke="#6366f1"
                                                strokeWidth="3"
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                            />
                                            <circle cx="5" cy="35" r="3" fill="#6366f1" />
                                            <circle cx="35" cy="30" r="3" fill="#8b5cf6" />
                                            <circle cx="75" cy="5" r="3" fill="#6366f1" />
                                        </svg>
                                    </div>

                                    {/* Pie Chart Slice */}
                                    <div
                                        className="absolute select-none transition-transform duration-700 ease-out"
                                        style={{
                                            top: '40%',
                                            left: '12%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.02}px, ${(mousePos.y - window.innerHeight / 2) * -0.025}px)`
                                        }}
                                    >
                                        <svg width="50" height="50" viewBox="0 0 50 50" className="opacity-25">
                                            <circle cx="25" cy="25" r="20" fill="none" stroke="#e0e7ff" strokeWidth="8" />
                                            <circle
                                                cx="25" cy="25" r="20"
                                                fill="none"
                                                stroke="#6366f1"
                                                strokeWidth="8"
                                                strokeDasharray="75 125"
                                                strokeLinecap="round"
                                            />
                                        </svg>
                                    </div>

                                    {/* Scatter Plot Points */}
                                    <div
                                        className="absolute select-none transition-transform duration-500 ease-out"
                                        style={{
                                            top: '20%',
                                            left: '40%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.018}px, ${(mousePos.y - window.innerHeight / 2) * 0.015}px)`
                                        }}
                                    >
                                        <svg width="60" height="50" viewBox="0 0 60 50" className="opacity-20">
                                            <circle cx="10" cy="40" r="4" fill="#6366f1" />
                                            <circle cx="20" cy="30" r="5" fill="#8b5cf6" />
                                            <circle cx="35" cy="35" r="3" fill="#a855f7" />
                                            <circle cx="45" cy="20" r="4" fill="#6366f1" />
                                            <circle cx="55" cy="15" r="5" fill="#8b5cf6" />
                                        </svg>
                                    </div>

                                    {/* Trend Arrow Up */}
                                    <div
                                        className="absolute select-none transition-transform duration-600 ease-out"
                                        style={{
                                            bottom: '20%',
                                            left: '35%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.022}px, ${(mousePos.y - window.innerHeight / 2) * -0.02}px) rotate(-45deg)`
                                        }}
                                    >
                                        <svg width="40" height="40" viewBox="0 0 40 40" className="opacity-20">
                                            <path d="M5 35 L35 5" stroke="#10b981" strokeWidth="4" strokeLinecap="round" />
                                            <path d="M20 5 L35 5 L35 20" stroke="#10b981" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                                        </svg>
                                    </div>

                                    {/* Data Grid */}
                                    <div
                                        className="absolute select-none transition-transform duration-700 ease-out"
                                        style={{
                                            top: '65%',
                                            right: '30%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.015}px, ${(mousePos.y - window.innerHeight / 2) * 0.018}px)`
                                        }}
                                    >
                                        <svg width="50" height="40" viewBox="0 0 50 40" className="opacity-15">
                                            <rect x="0" y="0" width="15" height="12" fill="#6366f1" rx="2" />
                                            <rect x="18" y="0" width="15" height="12" fill="#8b5cf6" rx="2" />
                                            <rect x="36" y="0" width="15" height="12" fill="#a855f7" rx="2" />
                                            <rect x="0" y="14" width="15" height="12" fill="#a855f7" rx="2" />
                                            <rect x="18" y="14" width="15" height="12" fill="#6366f1" rx="2" />
                                            <rect x="36" y="14" width="15" height="12" fill="#8b5cf6" rx="2" />
                                            <rect x="0" y="28" width="15" height="12" fill="#8b5cf6" rx="2" />
                                            <rect x="18" y="28" width="15" height="12" fill="#a855f7" rx="2" />
                                            <rect x="36" y="28" width="15" height="12" fill="#6366f1" rx="2" />
                                        </svg>
                                    </div>

                                    {/* Theta */}
                                    <div
                                        className="absolute text-indigo-200/35 text-4xl font-serif select-none transition-transform duration-700 ease-out"
                                        style={{
                                            top: '3%',
                                            left: '60%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.015}px, ${(mousePos.y - window.innerHeight / 2) * 0.02}px)`
                                        }}
                                    >θ</div>

                                    {/* Coordinate axes icon */}
                                    <div
                                        className="absolute select-none transition-transform duration-500 ease-out"
                                        style={{
                                            bottom: '15%',
                                            left: '55%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.028}px, ${(mousePos.y - window.innerHeight / 2) * -0.018}px)`
                                        }}
                                    >
                                        <svg width="45" height="45" viewBox="0 0 45 45" className="opacity-20">
                                            <path d="M5 40 L5 5" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" />
                                            <path d="M5 40 L40 40" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" />
                                            <path d="M5 5 L2 10 M5 5 L8 10" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" />
                                            <path d="M40 40 L35 37 M40 40 L35 43" stroke="#6366f1" strokeWidth="2" strokeLinecap="round" />
                                        </svg>
                                    </div>

                                    {/* Function curve */}
                                    <div
                                        className="absolute select-none transition-transform duration-600 ease-out"
                                        style={{
                                            top: '30%',
                                            right: '5%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.022}px, ${(mousePos.y - window.innerHeight / 2) * 0.015}px)`
                                        }}
                                    >
                                        <svg width="60" height="40" viewBox="0 0 60 40" className="opacity-20">
                                            <path d="M5 35 Q 15 5, 30 20 T 55 10" fill="none" stroke="#8b5cf6" strokeWidth="3" strokeLinecap="round" />
                                        </svg>
                                    </div>

                                    {/* Plus-minus */}
                                    <div
                                        className="absolute text-violet-300/35 text-3xl select-none transition-transform duration-600 ease-out"
                                        style={{
                                            top: '50%',
                                            left: '22%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * -0.022}px, ${(mousePos.y - window.innerHeight / 2) * 0.015}px)`
                                        }}
                                    >±</div>

                                    {/* Infinity Symbol */}
                                    <div
                                        className="absolute text-cyan-200/40 text-5xl select-none transition-transform duration-700 ease-out"
                                        style={{
                                            bottom: '40%',
                                            right: '3%',
                                            transform: `translate(${(mousePos.x - window.innerWidth / 2) * 0.035}px, ${(mousePos.y - window.innerHeight / 2) * 0.03}px)`
                                        }}
                                    >∞</div>
                                </div>

                                <div className="relative max-w-6xl mx-auto">
                                    <div className="grid lg:grid-cols-2 gap-12 items-center">
                                        {/* Left Content */}
                                        <div className="text-center lg:text-left">
                                            <div className="inline-flex items-center gap-2 bg-indigo-50 text-indigo-700 px-4 py-2 rounded-full text-sm font-medium mb-6 animate-fade-in-up">
                                                <Zap size={16} />
                                                <span>AI-Powered Graph Generation</span>
                                            </div>

                                            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-slate-900 leading-tight mb-6 animate-fade-in-up delay-100" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                                                Visualize Data.
                                                <br />
                                                <span className="bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
                                                    Discover Insights.
                                                </span>
                                            </h1>

                                            <p className="text-lg md:text-xl text-slate-600 mb-8 max-w-lg mx-auto lg:mx-0 animate-fade-in-up delay-200" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                                                Transform raw data into beautiful, accurate graphs. Scan datasheets, plot functions, and analyze trends with intelligent precision.
                                            </p>

                                            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start animate-fade-in-up delay-300" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                                                <Button onClick={createBlankGraph} variant="glow" size="lg" icon={Plus}>
                                                    Create Graph
                                                </Button>
                                                <Button onClick={() => { setIsImporting(false); setView('scan'); }} variant="glowSecondary" size="lg" icon={Camera}>
                                                    Scan Image
                                                </Button>
                                            </div>

                                            {/* Trust Indicators */}
                                            <div className="mt-10 flex items-center gap-6 justify-center lg:justify-start text-sm text-slate-500 animate-fade-in-up delay-400" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                                                <div className="flex items-center gap-2">
                                                    <Check size={16} className="text-green-500" />
                                                    <span>100% Free</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <Check size={16} className="text-green-500" />
                                                    <span>No Account Needed</span>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Right - Animated Graph Visual */}
                                        <div className="relative flex items-center justify-center animate-fade-in delay-200" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                                            <div className="hero-graphic-inner relative w-full max-w-md aspect-square">
                                                {/* Animated Geometric Graph */}
                                                <svg viewBox="0 0 400 400" className="w-full h-full" fill="none">
                                                    {/* Outer rotating ring */}
                                                    <g className="animate-slow-rotate" style={{ transformOrigin: 'center' }}>
                                                        <polygon
                                                            points="200,40 340,120 340,280 200,360 60,280 60,120"
                                                            stroke="url(#gradient1)"
                                                            strokeWidth="2"
                                                            fill="none"
                                                            opacity="0.6"
                                                        />
                                                    </g>

                                                    {/* Inner hexagons */}
                                                    <g style={{ transformOrigin: 'center' }}>
                                                        <polygon
                                                            points="200,80 300,140 300,260 200,320 100,260 100,140"
                                                            stroke="url(#gradient2)"
                                                            strokeWidth="2.5"
                                                            fill="none"
                                                        />
                                                        <polygon
                                                            points="200,120 260,160 260,240 200,280 140,240 140,160"
                                                            stroke="#1e293b"
                                                            strokeWidth="3"
                                                            fill="none"
                                                        />
                                                    </g>

                                                    {/* Data points */}
                                                    <circle cx="200" cy="80" r="6" fill="#6366f1" />
                                                    <circle cx="300" cy="140" r="5" fill="#8b5cf6" />
                                                    <circle cx="300" cy="260" r="5" fill="#8b5cf6" />
                                                    <circle cx="200" cy="320" r="6" fill="#6366f1" />
                                                    <circle cx="100" cy="260" r="5" fill="#8b5cf6" />
                                                    <circle cx="100" cy="140" r="5" fill="#8b5cf6" />

                                                    {/* Center point with glow */}
                                                    <circle cx="200" cy="200" r="12" fill="#4f46e5" filter="url(#glow)" />
                                                    <circle cx="200" cy="200" r="6" fill="white" />

                                                    {/* Connecting lines representing data flow */}
                                                    <path d="M200,80 L200,200 M300,140 L200,200 M100,260 L200,200" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="4,4" opacity="0.5" />

                                                    {/* Gradients and filters */}
                                                    <defs>
                                                        <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                                                            <stop offset="0%" stopColor="#6366f1" />
                                                            <stop offset="100%" stopColor="#8b5cf6" />
                                                        </linearGradient>
                                                        <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                                                            <stop offset="0%" stopColor="#1e293b" />
                                                            <stop offset="100%" stopColor="#475569" />
                                                        </linearGradient>
                                                        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                                                            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                                                            <feMerge>
                                                                <feMergeNode in="coloredBlur" />
                                                                <feMergeNode in="SourceGraphic" />
                                                            </feMerge>
                                                        </filter>
                                                    </defs>
                                                </svg>

                                                {/* Floating labels */}
                                                <div className="absolute top-8 right-8 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-lg text-xs font-mono text-slate-700 animate-float" style={{ animationDelay: '0.5s' }}>
                                                    f(x) = x²
                                                </div>
                                                <div className="absolute bottom-12 left-4 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-lg text-xs font-medium text-indigo-600 animate-float" style={{ animationDelay: '1s' }}>
                                                    R² = 0.998
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </section>

                            {/* Footer - Fixed at bottom with LiquidGlass effect */}
                            <footer className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50">
                                <LiquidGlassCard
                                    shadowIntensity="md"
                                    blurIntensity="lg"
                                    borderRadius="9999px"
                                    glowIntensity="sm"
                                    draggable={false}
                                    className="px-6 py-2.5"
                                >
                                    <p className="text-sm text-slate-700 font-medium whitespace-nowrap">
                                        © 2025 Graphly. Developed by <a href="https://github.com/Tajwarbot" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:text-indigo-700 transition-colors font-semibold hover:underline">Ahmad Taki Tajwar</a>
                                    </p>
                                </LiquidGlassCard>
                            </footer>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {savedGraphs.map(g => (
                                <Card key={g.id} className="group hover:shadow-md transition-shadow cursor-pointer relative">
                                    <div onClick={() => { setCurrentGraph(g); setView('editor'); }} className="p-6">
                                        <h3 className="font-bold text-lg mb-2">{g.title}</h3>
                                        <div className="flex items-center justify-between text-slate-500 text-xs mt-4">
                                            <span className="bg-indigo-50 text-indigo-700 px-2 py-1 rounded-full">{g.datasets?.length || 1} Datasets</span>
                                            <span>{new Date(g.updatedAt || g.createdAt).toLocaleDateString()}</span>
                                        </div>
                                    </div>
                                    <button
                                        onClick={(e) => deleteGraph(e, g.id)}
                                        className="absolute top-3 right-3 p-1.5 text-slate-300 hover:text-red-500 hover:bg-red-50 rounded-full opacity-0 group-hover:opacity-100 transition-all"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </Card>
                            ))}
                        </div>
                    )}
                </main>
            )}

            {view === 'scan' && (
                <main className="max-w-4xl mx-auto px-6 h-[calc(100vh-64px)] overflow-hidden flex flex-col justify-center">
                    {/* Header with animation */}
                    <div className="text-center mb-10 animate-fade-in-up">
                        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-indigo-100 to-violet-100 rounded-2xl mb-4">
                            <Camera className="text-indigo-600" size={28} />
                        </div>
                        <h1 className="text-3xl font-bold text-slate-900 mb-2">Import Data</h1>
                        <p className="text-slate-500">
                            {isImporting ? "Add data to your existing graph." : "Create a new graph from external data."}
                        </p>
                    </div>

                    {/* Import Cards with animations */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            className="group card-hover bg-white border-2 border-dashed border-slate-200 rounded-2xl h-52 flex flex-col items-center justify-center cursor-pointer hover:border-indigo-400 hover:bg-gradient-to-br hover:from-indigo-50/50 hover:to-violet-50/50 transition-all duration-300 animate-fade-in-up delay-100"
                            style={{ opacity: 0, animationFillMode: 'forwards' }}
                        >
                            {image ? <img src={image.preview} className="h-full object-contain p-4 rounded-xl" /> : (
                                <>
                                    <div className="w-14 h-14 bg-slate-100 group-hover:bg-indigo-100 rounded-xl flex items-center justify-center mb-3 transition-colors duration-300">
                                        <Camera size={28} className="text-slate-400 group-hover:text-indigo-600 transition-colors duration-300" />
                                    </div>
                                    <span className="text-sm font-semibold text-slate-700">Scan Image</span>
                                    <span className="text-xs text-slate-400 mt-1">Upload datasheet or table image</span>
                                </>
                            )}
                        </div>

                        <div
                            onClick={() => setShowCSVModal(true)}
                            className="group card-hover bg-white border-2 border-dashed border-slate-200 rounded-2xl h-52 flex flex-col items-center justify-center cursor-pointer hover:border-indigo-400 hover:bg-gradient-to-br hover:from-indigo-50/50 hover:to-violet-50/50 transition-all duration-300 animate-fade-in-up delay-200"
                            style={{ opacity: 0, animationFillMode: 'forwards' }}
                        >
                            <div className="w-14 h-14 bg-slate-100 group-hover:bg-indigo-100 rounded-xl flex items-center justify-center mb-3 transition-colors duration-300">
                                <FileSpreadsheet size={28} className="text-slate-400 group-hover:text-indigo-600 transition-colors duration-300" />
                            </div>
                            <span className="text-sm font-semibold text-slate-700">Paste / Upload CSV</span>
                            <span className="text-xs text-slate-400 mt-1">Import from spreadsheet data</span>
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
                        <div className="animate-fade-in-up">
                            <Button className="w-full" variant="glow" size="lg" onClick={handleScan} disabled={scanStatus === 'scanning'} icon={Zap}>
                                {scanStatus === 'scanning' ? 'Analyzing with AI...' : 'Generate from Image'}
                            </Button>
                        </div>
                    )}

                    <div className="mt-8 flex justify-center animate-fade-in-up delay-300" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                        <button onClick={() => setView(isImporting ? 'editor' : 'dashboard')} className="text-slate-400 hover:text-slate-600 text-sm font-medium transition-colors">
                            ← Back to {isImporting ? 'Editor' : 'Dashboard'}
                        </button>
                    </div>

                    {/* CSV Modal with glassmorphism */}
                    {showCSVModal && (
                        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-md animate-fade-in">
                            <div className="glass-dark bg-white/95 rounded-2xl shadow-2xl p-6 w-full max-w-lg mx-4 animate-scale-in border border-white/50">
                                <div className="flex items-center gap-3 mb-5">
                                    <div className="w-10 h-10 bg-indigo-100 rounded-xl flex items-center justify-center">
                                        <FileSpreadsheet className="text-indigo-600" size={20} />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-slate-900">Import CSV Data</h3>
                                        <p className="text-xs text-slate-500">Paste your data or upload a file</p>
                                    </div>
                                </div>
                                <textarea
                                    className="w-full h-40 p-4 bg-slate-50 border border-slate-200 rounded-xl text-sm font-mono mb-4 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-400 outline-none resize-none transition-all"
                                    placeholder="Paste CSV data here (e.g. x,y&#10;1,2&#10;3,4)"
                                    value={csvText}
                                    onChange={e => setCsvText(e.target.value)}
                                />
                                <div className="flex justify-between items-center mb-5 p-3 bg-slate-50 rounded-lg">
                                    <span className="text-xs text-slate-500 font-medium">Or upload a file</span>
                                    <input
                                        type="file"
                                        accept=".csv"
                                        ref={csvFileInputRef}
                                        className="text-xs text-slate-500 file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200 file:transition-colors file:cursor-pointer"
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
                                <div className="flex gap-3 justify-end">
                                    <Button variant="secondary" size="md" onClick={() => setShowCSVModal(false)}>Cancel</Button>
                                    <Button variant="primary" size="md" onClick={() => handleCSVImport(parseCSV(csvText))} icon={Check}>Import Data</Button>
                                </div>
                            </div>
                        </div>
                    )}
                </main>
            )}

            {view === 'editor' && currentGraph && (
                <div className="relative flex h-[calc(100vh-64px)] overflow-hidden">

                    <div className="flex-1 relative bg-slate-50 flex flex-col overflow-hidden">
                        <div className="absolute inset-0 bg-[radial-gradient(#e2e8f0_1px,transparent_1px)] [background-size:20px_20px] opacity-50"></div>

                        <button
                            onClick={() => setIsSidebarOpen(true)}
                            className="lg:hidden absolute top-4 right-4 z-30 bg-white p-2 rounded-full shadow-md border border-slate-200 text-slate-600 hover:text-indigo-600 transition-colors"
                        >
                            <Menu size={20} />
                        </button>

                        <div className="flex-1 p-2 md:p-8 flex flex-col items-center justify-center overflow-hidden relative graph-print-container"
                            ref={graphContainerRef}
                            onWheel={handleWheel}
                            onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={() => setIsDragging(false)} onMouseLeave={() => setIsDragging(false)}
                            onTouchStart={handleTouchStart} onTouchMove={handleTouchMove} onTouchEnd={() => setIsDragging(false)}>

                            {/* COMPACT INFO PILL (Top-Left) */}
                            {selectedData && (
                                <div className="graph-ui-overlay absolute top-2 left-2 z-50 bg-white/95 backdrop-blur border border-slate-300 rounded-full shadow-md px-3 py-1.5 flex items-center gap-3">
                                    <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: selectedData.color || '#6366f1' }}></div>
                                        <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider max-w-[80px] truncate">{selectedData.datasetName}</span>
                                    </div>
                                    {selectedData.isTrendline ? (
                                        <span className="font-mono font-bold text-xs text-slate-800">{selectedData.equation}</span>
                                    ) : (
                                        <div className="flex items-center gap-2 text-xs font-mono border-l border-slate-200 pl-2">
                                            <span className="text-slate-600">x: {formatNumber(selectedData.x)}</span>
                                            <span className="text-slate-600">y: {formatNumber(selectedData.y)}</span>
                                        </div>
                                    )}
                                    <button onClick={(e) => { e.stopPropagation(); setSelectedData(null); }} className="text-slate-400 hover:text-slate-600 ml-1">
                                        <X size={14} />
                                    </button>
                                </div>
                            )}

                            {!selectedData && (
                                <div className="graph-ui-overlay absolute top-2 left-2 z-40 bg-white/50 backdrop-blur rounded-full px-3 py-1 text-[10px] text-slate-400 pointer-events-none flex items-center gap-1">
                                    <Info size={10} /> Click on points for details
                                </div>
                            )}

                            {/* Zoom Controls Overlay - Enhanced */}
                            <div className="graph-ui-overlay absolute bottom-6 right-6 z-40 flex flex-col gap-2">
                                <button onClick={zoomIn} className="p-2.5 bg-white/95 backdrop-blur shadow-lg rounded-xl text-slate-600 hover:text-indigo-600 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 transition-all duration-200 hover:scale-105 hover:shadow-xl" title="Zoom In">
                                    <Plus size={18} />
                                </button>
                                <button onClick={zoomOut} className="p-2.5 bg-white/95 backdrop-blur shadow-lg rounded-xl text-slate-600 hover:text-indigo-600 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 transition-all duration-200 hover:scale-105 hover:shadow-xl" title="Zoom Out">
                                    <Minus size={18} />
                                </button>
                                <button onClick={resetView} className="p-2.5 bg-white/95 backdrop-blur shadow-lg rounded-xl text-slate-600 hover:text-indigo-600 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 transition-all duration-200 hover:scale-105 hover:shadow-xl" title="Reset View">
                                    <RotateCcw size={18} />
                                </button>
                            </div>

                            <div className={`w-full flex-1 bg-white rounded-xl md:rounded-2xl shadow-xl border border-slate-200 p-2 md:p-4 relative ${isDragging ? 'cursor-grabbing' : 'cursor-default'}`}
                                onClick={(e) => e.stopPropagation()}
                            >
                                <div className="absolute top-2 left-1/2 transform -translate-x-1/2 font-bold text-slate-700 pointer-events-none z-10 text-sm md:text-base text-center w-3/4 truncate">
                                    {currentGraph.title}
                                </div>

                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart
                                        margin={{ top: 50, right: 10, bottom: 20, left: 10 }}
                                        onDoubleClick={handleChartDoubleClick}
                                        onClick={handleChartClick}
                                        style={{ outline: 'none' }}
                                    >
                                        {currentGraph.globalConfig?.showGrid !== false && (
                                            <CartesianGrid strokeDasharray="" stroke="#e2e8f0" />
                                        )}

                                        <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={2} />
                                        <ReferenceLine y={0} stroke="#94a3b8" strokeWidth={2} />

                                        {currentGraph.annotations && currentGraph.annotations.map(note => {
                                            const xVal = parseFloat(note.x);
                                            const yVal = parseFloat(note.y);
                                            if (isNaN(xVal) || isNaN(yVal)) return null;
                                            return (
                                                <ReferenceDot key={note.id} x={xVal} y={yVal} r={0}>
                                                    <Label value={note.text} position="top" fill="#374151" fontSize={12} fontWeight="bold" />
                                                </ReferenceDot>
                                            );
                                        })}

                                        {selectedData && !selectedData.isTrendline && (
                                            <ReferenceDot
                                                x={selectedData.x}
                                                y={selectedData.y}
                                                r={8}
                                                fill={selectedData.color}
                                                stroke="white"
                                                strokeWidth={3}
                                                isFront={true}
                                            />
                                        )}

                                        <XAxis type="number" dataKey="x" domain={currentDomain.x} ticks={xTicks} tickFormatter={formatNumber} allowDataOverflow tick={{ fontSize: 10, fill: '#64748b' }}>
                                            <Label value={currentGraph.globalConfig?.xAxisLabel || "X Axis"} offset={-10} position="insideBottom" style={{ fontSize: '10px', fill: '#94a3b8', fontWeight: 600 }} />
                                        </XAxis>
                                        <YAxis type="number" domain={currentDomain.y} ticks={yTicks} tickFormatter={formatNumber} allowDataOverflow tick={{ fontSize: 10, fill: '#64748b' }}>
                                            <Label value={currentGraph.globalConfig?.yAxisLabel || "Y Axis"} angle={-90} offset={10} position="insideLeft" style={{ fontSize: '10px', fill: '#94a3b8', fontWeight: 600 }} />
                                        </YAxis>

                                        {visibleDatasets.map(ds => (
                                            <React.Fragment key={ds.id}>
                                                {ds.config.type === 'function' && (
                                                    <Line
                                                        key={`func-${ds.id}`}
                                                        data={ds.points}
                                                        dataKey="y"
                                                        stroke={ds.color}
                                                        strokeWidth={3}
                                                        dot={false}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                        activeDot={{ r: 6, onClick: (e, p) => handlePointClick(p, e, ds) }}
                                                        onClick={(e, p) => handlePointClick(p, e, ds, 'function')}
                                                    />
                                                )}

                                                {ds.trendData.length > 0 && (
                                                    <Line
                                                        key={`trend-${ds.id}`}
                                                        data={ds.trendData}
                                                        dataKey="y"
                                                        stroke={ds.config.trendlineColor || '#ef4444'}
                                                        strokeWidth={3}
                                                        strokeDasharray="0"
                                                        dot={false}
                                                        activeDot={false}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                        cursor="pointer"
                                                        onClick={(p, e) => handlePointClick(p, e, ds, 'trendline')}
                                                    />
                                                )}

                                                {ds.config.type === 'area' && (
                                                    <>
                                                        <defs>
                                                            <linearGradient id={`grad-${ds.id}`} x1="0" y1="0" x2="0" y2="1">
                                                                <stop offset="5%" stopColor={ds.color} stopOpacity={0.3} />
                                                                <stop offset="95%" stopColor={ds.color} stopOpacity={0} />
                                                            </linearGradient>
                                                        </defs>
                                                        <Area
                                                            key={`area-${ds.id}`}
                                                            data={ds.points}
                                                            dataKey="y"
                                                            stroke={ds.color}
                                                            fill={`url(#grad-${ds.id})`}
                                                            strokeWidth={2}
                                                            activeDot={{ r: 6, onClick: (e, p) => handlePointClick(p, e, ds) }}
                                                            isAnimationActive={false}
                                                            type="monotone"
                                                            dot={{ r: 4, strokeWidth: 0, fill: ds.color, onClick: (p, e) => handlePointClick(p, e, ds), cursor: 'pointer' }}
                                                        >
                                                            {currentGraph.globalConfig?.showLabels && (
                                                                <LabelList dataKey="y" position="top" content={(props) => {
                                                                    const { x, y, value, payload, index } = props;
                                                                    const point = payload || (ds.points && ds.points[index]);
                                                                    if (!point) return null;
                                                                    const displayX = point.x !== undefined ? point.x : (point.payload?.x ?? index);
                                                                    return (
                                                                        <text x={x} y={y - 12} fill={ds.color} fontSize={10} textAnchor="middle" fontWeight="bold">
                                                                            ({formatNumber(displayX)}, {formatNumber(value)})
                                                                        </text>
                                                                    );
                                                                }} />
                                                            )}
                                                        </Area>
                                                    </>
                                                )}
                                                {ds.config.type === 'line' && (
                                                    <Line
                                                        key={`line-${ds.id}`}
                                                        data={ds.points}
                                                        dataKey="y"
                                                        stroke={ds.color}
                                                        strokeWidth={2}
                                                        isAnimationActive={false}
                                                        type="monotone"
                                                        activeDot={{ r: 6, onClick: (e, p) => handlePointClick(p, e, ds) }}
                                                        dot={{ r: 4, stroke: 'white', strokeWidth: 2, fill: ds.color, onClick: (p, e) => handlePointClick(p, e, ds), cursor: 'pointer' }}
                                                    >
                                                        {currentGraph.globalConfig?.showLabels && (
                                                            <LabelList dataKey="y" position="top" content={(props) => {
                                                                const { x, y, value, payload, index } = props;
                                                                const point = payload || (ds.points && ds.points[index]);
                                                                if (!point) return null;
                                                                const displayX = point.x !== undefined ? point.x : (point.payload?.x ?? index);
                                                                return (
                                                                    <text x={x} y={y - 12} fill={ds.color} fontSize={10} textAnchor="middle" fontWeight="bold">
                                                                        ({formatNumber(displayX)}, {formatNumber(value)})
                                                                    </text>
                                                                );
                                                            }} />
                                                        )}
                                                    </Line>
                                                )}
                                                {ds.config.type === 'scatter' && (
                                                    <Scatter
                                                        key={`scatter-${ds.id}`}
                                                        data={ds.points}
                                                        name={ds.name}
                                                        dataKey="y"
                                                        fill={ds.color}
                                                        isAnimationActive={false}
                                                        onClick={(p, i, e) => handlePointClick(p, e, ds)}
                                                        cursor="pointer"
                                                        activeDot={false}
                                                    >
                                                        {currentGraph.globalConfig?.showLabels && (
                                                            <LabelList dataKey="y" position="top" content={(props) => {
                                                                const { x, y, value, payload, index } = props;
                                                                const point = payload || (ds.points && ds.points[index]);
                                                                if (!point) return null;
                                                                const displayX = point.x !== undefined ? point.x : (point.payload?.x ?? index);
                                                                return (
                                                                    <text x={x} y={y - 12} fill={ds.color} fontSize={10} textAnchor="middle" fontWeight="bold">
                                                                        ({formatNumber(displayX)}, {formatNumber(value)})
                                                                    </text>
                                                                );
                                                            }} />
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
                            className="fixed inset-0 bg-black/20 z-40 lg:hidden"
                            onClick={() => setIsSidebarOpen(false)}
                        />
                    )}

                    <div className={`
               fixed inset-y-0 right-0 z-50 w-80 bg-white shadow-2xl transform transition-transform duration-300 ease-in-out flex flex-col
               lg:relative lg:translate-x-0 lg:w-96 lg:shadow-xl lg:z-auto lg:h-full lg:border-l lg:border-slate-200
               xl:relative xl:translate-x-0 xl:w-96 xl:shadow-xl xl:z-auto xl:h-full xl:border-l xl:border-slate-200
               ${isSidebarOpen ? 'translate-x-0' : 'translate-x-full'}
            `}>

                        <button
                            onClick={() => setIsSidebarOpen(false)}
                            className="lg:hidden xl:hidden absolute top-3 right-3 p-1 text-slate-400 hover:text-slate-600 z-50"
                        >
                            <X size={20} />
                        </button>

                        <div className="flex border-b border-slate-100 shrink-0 pt-2 lg:pt-0 bg-slate-50/50">
                            <button
                                onClick={() => setShowSettings(false)}
                                className={`flex-1 py-3.5 text-sm font-semibold transition-all duration-300 relative ${!showSettings ? 'text-indigo-600 bg-white' : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'}`}
                            >
                                <Layout size={14} className="inline mr-2" /> Datasets
                                {!showSettings && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-indigo-500 to-violet-500"></div>}
                            </button>
                            <button
                                onClick={() => setShowSettings(true)}
                                className={`flex-1 py-3.5 text-sm font-semibold transition-all duration-300 relative ${showSettings ? 'text-indigo-600 bg-white' : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'}`}
                            >
                                <Settings size={14} className="inline mr-2" /> Graph Settings
                                {showSettings && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-indigo-500 to-violet-500"></div>}
                            </button>
                        </div>

                        {showSettings ? (
                            <div className="flex-1 p-4 space-y-6 overflow-y-auto">
                                <div className="bg-indigo-50/50 p-4 rounded-xl border border-indigo-100">
                                    <label className="flex items-center justify-between cursor-pointer">
                                        <div className="flex items-center gap-2">
                                            <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center text-indigo-600">
                                                <Eye size={16} />
                                            </div>
                                            <span className="text-sm font-bold text-slate-700">Annotate All Points</span>
                                        </div>
                                        <input
                                            type="checkbox"
                                            checked={currentGraph.globalConfig?.showLabels === true}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, showLabels: e.target.checked } })}
                                            className="w-5 h-5 rounded text-indigo-600 focus:ring-indigo-500 border-slate-300 transition-all cursor-pointer"
                                        />
                                    </label>
                                </div>

                                <div>
                                    <label className="text-xs font-bold text-slate-400 uppercase mb-2 block">Graph Title</label>
                                    <input
                                        className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400 transition-all"
                                        value={currentGraph.title}
                                        onChange={e => setCurrentGraph({ ...currentGraph, title: e.target.value })}
                                        placeholder="Enter graph title..."
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="text-xs font-bold text-slate-400 uppercase mb-2 block">X Axis Label</label>
                                        <input
                                            className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400 transition-all"
                                            value={currentGraph.globalConfig?.xAxisLabel || ''}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, xAxisLabel: e.target.value } })}
                                            placeholder="e.g. Time (s)"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-xs font-bold text-slate-400 uppercase mb-2 block">Y Axis Label</label>
                                        <input
                                            className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400 transition-all"
                                            value={currentGraph.globalConfig?.yAxisLabel || ''}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, yAxisLabel: e.target.value } })}
                                            placeholder="e.g. Velocity (m/s)"
                                        />
                                    </div>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <label className="text-xs font-bold text-slate-400 uppercase mb-2 block">Aspect Ratio</label>
                                    <select
                                        className="w-full p-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-400"
                                        value={currentGraph.globalConfig?.aspectRatio || 'auto'}
                                        onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, aspectRatio: e.target.value } })}
                                    >
                                        <option value="auto">Auto (Fill)</option>
                                        <option value="square">Square (1:1)</option>
                                    </select>
                                </div>

                                <div className="pt-4 border-t border-slate-100">
                                    <label className="flex items-center gap-2 cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={currentGraph.globalConfig?.showGrid !== false}
                                            onChange={e => setCurrentGraph({ ...currentGraph, globalConfig: { ...currentGraph.globalConfig, showGrid: e.target.checked } })}
                                            className="rounded text-indigo-600 focus:ring-indigo-500 border-gray-300"
                                        />
                                        <span className="text-sm text-slate-700">Show Grid Lines</span>
                                    </label>
                                </div>

                                {currentGraph.globalConfig?.showGrid !== false && (
                                    <div className="space-y-3 mt-3 pl-4 border-l-2 border-indigo-100">
                                        <div>
                                            <label className="text-xs text-slate-500 block mb-1">X Grid Interval</label>
                                            <input
                                                type="number"
                                                step="any"
                                                placeholder="Auto"
                                                className="w-full p-2 text-sm bg-slate-50 border border-slate-200 rounded-lg outline-none focus:ring-2 focus:ring-indigo-100"
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
                                            <label className="text-xs text-slate-500 block mb-1">Y Grid Interval</label>
                                            <input
                                                type="number"
                                                step="any"
                                                placeholder="Auto"
                                                className="w-full p-2 text-sm bg-slate-50 border border-slate-200 rounded-lg outline-none focus:ring-2 focus:ring-indigo-100"
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

                                <div className="pt-4 border-t border-slate-100 pb-20 lg:pb-0">
                                    <label className="text-xs font-bold text-slate-400 uppercase mb-3 block flex items-center gap-1"><StickyNote size={12} /> Annotations</label>
                                    <div className="space-y-2">
                                        {currentGraph.annotations?.map((note, idx) => (
                                            <div key={note.id} className="flex gap-2 items-center bg-slate-50 p-2 rounded-md border border-slate-100">
                                                <input
                                                    className="w-12 p-1 text-xs bg-white border rounded text-center outline-none"
                                                    value={note.x}
                                                    onChange={e => {
                                                        const newNotes = [...currentGraph.annotations];
                                                        newNotes[idx].x = e.target.value;
                                                        setCurrentGraph({ ...currentGraph, annotations: newNotes });
                                                    }}
                                                />
                                                <input
                                                    className="w-12 p-1 text-xs bg-white border rounded text-center outline-none"
                                                    value={note.y}
                                                    onChange={e => {
                                                        const newNotes = [...currentGraph.annotations];
                                                        newNotes[idx].y = e.target.value;
                                                        setCurrentGraph({ ...currentGraph, annotations: newNotes });
                                                    }}
                                                />
                                                <input
                                                    className="flex-1 p-1 text-xs bg-transparent border-b border-transparent focus:border-indigo-300 outline-none"
                                                    value={note.text}
                                                    onChange={e => {
                                                        const newNotes = [...currentGraph.annotations];
                                                        newNotes[idx].text = e.target.value;
                                                        setCurrentGraph({ ...currentGraph, annotations: newNotes });
                                                    }}
                                                />
                                                <button onClick={() => setCurrentGraph(prev => ({ ...prev, annotations: prev.annotations.filter((_, i) => i !== idx) }))} className="text-slate-300 hover:text-red-500"><X size={12} /></button>
                                            </div>
                                        ))}
                                        <Button variant="secondary" size="sm" className="w-full text-xs" onClick={() => setCurrentGraph(prev => ({ ...prev, annotations: [...(prev.annotations || []), { id: Date.now(), x: "0", y: "0", text: "Note" }] }))}>
                                            + Add Annotation
                                        </Button>
                                        <p className="text-[10px] text-slate-400 text-center mt-1">Double-click on chart to add quickly</p>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex-1 flex flex-col overflow-hidden">
                                <div className="p-3 border-b border-slate-100 flex justify-end gap-2 bg-slate-50/30">
                                    <button onClick={() => addDataset('function')} className="text-xs font-medium text-indigo-600 hover:bg-indigo-50 px-3 py-1.5 rounded-md transition-colors flex items-center gap-1"><Sigma size={14} /> Function</button>
                                    <button onClick={() => addDataset('data')} className="text-xs font-medium text-indigo-600 hover:bg-indigo-50 px-3 py-1.5 rounded-md transition-colors flex items-center gap-1"><Plus size={14} /> Table</button>
                                </div>
                                <div className="flex-1 overflow-y-auto p-2 space-y-2 pb-20 lg:pb-2">
                                    {allDatasets.map((ds, idx) => (
                                        <div key={ds.id} className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm transition-all hover:shadow-md">
                                            <div className="flex items-center gap-3 p-3 bg-slate-50/80 border-b border-slate-100 cursor-pointer" onClick={() => setExpandedDatasetId(expandedDatasetId === ds.id ? null : ds.id)}>
                                                <button onClick={(e) => { e.stopPropagation(); updateDataset(ds.id, { visible: !ds.visible }); }} className={`p-1.5 rounded-md transition-colors ${ds.visible ? 'text-slate-600 hover:bg-slate-200' : 'text-slate-300 hover:bg-slate-100'}`}>
                                                    {ds.visible ? <Eye size={16} /> : <EyeOff size={16} />}
                                                </button>

                                                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: ds.color }}></div>

                                                <input
                                                    className="flex-1 bg-transparent text-sm font-semibold text-slate-700 outline-none"
                                                    value={ds.name}
                                                    onClick={(e) => e.stopPropagation()}
                                                    onChange={(e) => updateDataset(ds.id, { name: e.target.value })}
                                                />

                                                <ChevronLeft size={16} className={`text-slate-400 transition-transform ${expandedDatasetId === ds.id ? '-rotate-90' : ''}`} />
                                            </div>

                                            {expandedDatasetId === ds.id && (
                                                <div className="p-4 space-y-5 animate-in slide-in-from-top-2 duration-200">

                                                    {ds.config.type !== 'function' && (
                                                        <div className="bg-indigo-50/50 p-3 rounded-lg border border-indigo-100 mb-3">
                                                            <h4 className="text-[10px] font-bold text-indigo-400 uppercase mb-2 flex items-center gap-1"><Activity size={10} /> Analysis</h4>
                                                            <div className="grid grid-cols-2 gap-y-1 gap-x-4 text-xs">
                                                                <div className="flex justify-between">
                                                                    <span className="text-slate-500">Mean X:</span>
                                                                    <span className="font-mono">{formatNumber(ds.stats.meanX)}</span>
                                                                </div>
                                                                <div className="flex justify-between">
                                                                    <span className="text-slate-500">Mean Y:</span>
                                                                    <span className="font-mono">{formatNumber(ds.stats.meanY)}</span>
                                                                </div>
                                                                <div className="flex justify-between">
                                                                    <span className="text-slate-500">StdDev Y:</span>
                                                                    <span className="font-mono">{formatNumber(ds.stats.stdDevY)}</span>
                                                                </div>
                                                                {ds.r2 !== null && (
                                                                    <div className="flex justify-between">
                                                                        <span className="text-slate-500">R²:</span>
                                                                        <span className="font-mono font-bold text-indigo-600">{formatNumber(ds.r2)}</span>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="grid grid-cols-2 gap-3">
                                                        <div>
                                                            <label className="text-[10px] uppercase font-bold text-slate-400 mb-1 block">Graph Type</label>
                                                            {ds.config.type === 'function' ? (
                                                                <div className="w-full text-xs p-1.5 bg-slate-100 border border-slate-200 rounded text-slate-500 italic">Function</div>
                                                            ) : (
                                                                <select value={ds.config.type} onChange={e => updateDataset(ds.id, { config: { ...ds.config, type: e.target.value } })} className="w-full text-xs p-1.5 bg-slate-50 border border-slate-200 rounded outline-none focus:ring-1 focus:ring-indigo-500">
                                                                    <option value="scatter">Scatter</option>
                                                                    <option value="line">Line</option>
                                                                    <option value="area">Area</option>
                                                                </select>
                                                            )}
                                                        </div>
                                                        {ds.config.type !== 'function' && (
                                                            <div>
                                                                <label className="text-[10px] uppercase font-bold text-slate-400 mb-1 block">Trendline</label>
                                                                <select
                                                                    value={ds.config.showTrendline ? ds.config.trendlineType : 'none'}
                                                                    onChange={e => updateDataset(ds.id, { config: { ...ds.config, showTrendline: e.target.value !== 'none', trendlineType: e.target.value === 'none' ? 'linear' : e.target.value } })}
                                                                    className="w-full text-xs p-1.5 bg-slate-50 border border-slate-200 rounded outline-none focus:ring-1 focus:ring-indigo-500"
                                                                >
                                                                    <option value="none">None</option>
                                                                    <option value="linear">Linear</option>
                                                                    <option value="quadratic">Quadratic</option>
                                                                    <option value="exponential">Exponential</option>
                                                                    <option value="power">Power</option>
                                                                    <option value="logarithmic">Logarithmic</option>
                                                                </select>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {ds.config.type === 'function' ? (
                                                        <div>
                                                            <label className="text-[10px] uppercase font-bold text-slate-400 mb-1 block">f(x) =</label>
                                                            <input
                                                                className="w-full p-2 bg-white border border-slate-300 rounded font-mono text-sm focus:ring-2 focus:ring-indigo-500 outline-none"
                                                                value={ds.equation || ''}
                                                                onChange={e => updateDataset(ds.id, { equation: e.target.value })}
                                                                onClick={e => e.stopPropagation()}
                                                                placeholder="e.g. sin(x) * x"
                                                            />
                                                            <p className="text-[10px] text-slate-400 mt-1">Supported: sin, cos, tan, log, sqrt, abs, pi, e, ^</p>
                                                        </div>
                                                    ) : (
                                                        <div className="space-y-2">
                                                            <div className="flex items-center gap-2">
                                                                <span className="text-xs font-mono text-slate-500 w-4">X</span>
                                                                <select value={ds.config.xKey} onChange={e => updateDataset(ds.id, { config: { ...ds.config, xKey: e.target.value } })} className="flex-1 text-xs p-1.5 bg-slate-50 border border-slate-200 rounded outline-none">
                                                                    {Object.keys(ds.data[0] || {}).map(k => <option key={k} value={k}>{k}</option>)}
                                                                </select>
                                                            </div>
                                                            <div className="flex items-center gap-2">
                                                                <span className="text-xs font-mono text-slate-500 w-4">Y</span>
                                                                <select value={ds.config.yKey} onChange={e => updateDataset(ds.id, { config: { ...ds.config, yKey: e.target.value } })} className="flex-1 text-xs p-1.5 bg-slate-50 border border-slate-200 rounded outline-none">
                                                                    {Object.keys(ds.data[0] || {}).map(k => <option key={k} value={k}>{k}</option>)}
                                                                </select>
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="flex items-center justify-between border-t border-slate-100 pt-3">
                                                        <div className="flex flex-col gap-1">
                                                            <span className="text-[10px] font-bold text-slate-400">DATA COLOR</span>
                                                            <div className="flex gap-1">
                                                                {THEMES.map(t => (
                                                                    <button key={t.color} onClick={() => updateDataset(ds.id, { color: t.color })} className={`w-3 h-3 rounded-full ${ds.color === t.color ? 'ring-1 ring-offset-1 ring-slate-400 scale-110' : ''}`} style={{ backgroundColor: t.color }} />
                                                                ))}
                                                            </div>
                                                        </div>
                                                        {ds.config.type !== 'function' && (
                                                            <div className="flex flex-col gap-1 items-end">
                                                                <span className="text-[10px] font-bold text-slate-400">TREND COLOR</span>
                                                                <div className="flex gap-1">
                                                                    {['#000', '#ef4444', '#22c55e', '#3b82f6'].map(c => (
                                                                        <button key={c} onClick={() => updateDataset(ds.id, { config: { ...ds.config, trendlineColor: c } })} className={`w-3 h-3 rounded-full ${ds.config.trendlineColor === c ? 'ring-1 ring-offset-1 ring-slate-400 scale-110' : ''}`} style={{ backgroundColor: c }} />
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>

                                                    {ds.config.type !== 'function' && (
                                                        <div className="border border-slate-200 rounded-lg overflow-hidden">
                                                            <div className="bg-slate-50 px-3 py-2 border-b border-slate-200 flex justify-between items-center">
                                                                <span className="text-[10px] font-bold text-slate-500 uppercase flex items-center gap-1"><Table size={12} /> Data Points</span>
                                                                <button onClick={() => {
                                                                    const newRow = Object.keys(ds.data[0] || { [ds.config.xKey]: 0, [ds.config.yKey]: 0 }).reduce((acc, k) => ({ ...acc, [k]: 0 }), {});
                                                                    updateDataset(ds.id, { data: [...ds.data, newRow] });
                                                                }} className="text-indigo-600 hover:text-indigo-700 text-xs font-medium flex items-center gap-1"><Plus size={12} /> Add</button>
                                                            </div>
                                                            <div className="max-h-32 overflow-y-auto">
                                                                <table className="w-full text-xs">
                                                                    <thead className="bg-slate-50 text-slate-400 font-medium">
                                                                        <tr>
                                                                            <th className="p-2 text-left font-normal border-r border-slate-100 w-1/2">{ds.config.xKey}</th>
                                                                            <th className="p-2 text-left font-normal w-1/2">{ds.config.yKey}</th>
                                                                        </tr>
                                                                    </thead>
                                                                    <tbody className="divide-y divide-slate-50">
                                                                        {ds.data.map((row, rIdx) => (
                                                                            <tr key={rIdx} className="group hover:bg-slate-50">
                                                                                <td className="p-0 border-r border-slate-100 relative">
                                                                                    <input className="w-full p-2 bg-transparent outline-none focus:bg-white text-slate-600" value={row[ds.config.xKey]} onChange={e => {
                                                                                        const nd = [...ds.data]; nd[rIdx] = { ...nd[rIdx], [ds.config.xKey]: e.target.value };
                                                                                        updateDataset(ds.id, { data: nd });
                                                                                    }} />
                                                                                </td>
                                                                                <td className="p-0 relative">
                                                                                    <input className="w-full p-2 bg-transparent outline-none focus:bg-white text-slate-600" value={row[ds.config.yKey]} onChange={e => {
                                                                                        const nd = [...ds.data]; nd[rIdx] = { ...nd[rIdx], [ds.config.yKey]: e.target.value };
                                                                                        updateDataset(ds.id, { data: nd });
                                                                                    }} />
                                                                                    <button onClick={() => updateDataset(ds.id, { data: ds.data.filter((_, i) => i !== rIdx) })} className="absolute right-1 top-1.5 opacity-0 group-hover:opacity-100 text-slate-300 hover:text-red-500"><Trash2 size={12} /></button>
                                                                                </td>
                                                                            </tr>
                                                                        ))}
                                                                    </tbody>
                                                                </table>
                                                            </div>
                                                        </div>
                                                    )}

                                                    <div className="pt-2 border-t border-slate-100 flex justify-end">
                                                        <button onClick={(e) => deleteDataset(ds.id, e)} className="text-red-500 hover:text-red-600 text-xs flex items-center gap-1"><Trash2 size={12} /> Delete Dataset</button>
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
            )}
        </div>
    );
}