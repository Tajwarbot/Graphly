import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { getGeminiApiKey, isApiKeyConfigured } from '../lib/security';
import { compileMathFunction } from '../lib/mathEngine';
import { Terminal, X, CheckCircle2, Minus, Maximize2, Sparkles, Send } from 'lucide-react';

export const GEOMETRIC_3D_SURFACES = [
    {
        names: ['hyperboloid', 'hyperbolic paraboloid', 'saddle', 'monkey saddle', 'pringle'],
        equation: '(x^2 - y^2) / 4',
        label: 'Hyperbolic Paraboloid (Saddle)'
    },
    {
        names: ['two sheeted hyperboloid', 'hyperboloid of two sheets', 'hyperboloid 2 sheets'],
        equation: 'sqrt(x^2 + y^2 + 1)',
        label: 'Two-Sheeted Hyperboloid'
    },
    {
        names: ['paraboloid', 'elliptic paraboloid', 'bowl', 'cup'],
        equation: '(x^2 + y^2) / 6',
        label: 'Elliptic Paraboloid'
    },
    {
        names: ['sombrero', 'mexican hat', 'hat surface'],
        equation: '2 * sin(sqrt(x^2 + y^2)) / (sqrt(x^2 + y^2) + 0.1)',
        label: 'Sombrero Surface'
    },
    {
        names: ['ripple', 'ripples', 'water ripple', 'waves', 'cross wave', 'wave surface'],
        equation: 'sin(x) * cos(y)',
        label: 'Ripple Wave'
    },
    {
        names: ['gaussian hill', 'gaussian', 'bell surface', 'hill'],
        equation: '3 * exp(-(x^2 + y^2) / 4)',
        label: 'Gaussian Hill'
    },
    {
        names: ['sphere', 'hemisphere', 'dome'],
        equation: 'sqrt(max(0, 25 - x^2 - y^2))',
        label: 'Hemisphere Surface'
    },
    {
        names: ['cone', 'conic surface'],
        equation: 'sqrt(x^2 + y^2) / 2',
        label: 'Cone Surface'
    },
    {
        names: ['cutting plane', 'plane', 'flat plane', 'ramp'],
        equation: '0.4 * x',
        label: 'Cutting Plane'
    }
];

export const GEOMETRIC_2D_CURVES = [
    {
        names: ['circle', 'unit circle'],
        implicit: true,
        equation: 'x^2 + y^2 = 25',
        label: 'Circle'
    },
    {
        names: ['ellipse', 'oval'],
        implicit: true,
        equation: 'x^2 / 16 + y^2 / 9 = 1',
        label: 'Ellipse'
    },
    {
        names: ['hyperbola'],
        implicit: true,
        equation: 'x^2 - y^2 = 9',
        label: 'Hyperbola'
    },
    {
        names: ['parabola'],
        implicit: false,
        equation: 'x^2',
        label: 'Parabola'
    },
    {
        names: ['cubic'],
        implicit: false,
        equation: 'x^3 - 3*x',
        label: 'Cubic Curve'
    },
    {
        names: ['sine wave', 'sinusoid', 'sine curve'],
        implicit: false,
        equation: 'sin(x)',
        label: 'Sine Wave'
    },
    {
        names: ['cosine wave', 'cosine curve'],
        implicit: false,
        equation: 'cos(x)',
        label: 'Cosine Wave'
    },
    {
        names: ['tangent wave', 'tangent curve'],
        implicit: false,
        equation: 'tan(x)',
        label: 'Tangent Wave'
    },
    {
        names: ['sigmoid', 'logistic curve'],
        implicit: false,
        equation: '1 / (1 + exp(-x))',
        label: 'Sigmoid Curve'
    },
    {
        names: ['gaussian curve', 'normal distribution', 'bell curve'],
        implicit: false,
        equation: 'exp(-x^2)',
        label: 'Gaussian Curve'
    },
    {
        names: ['exponential growth', 'exponential'],
        implicit: false,
        equation: 'exp(x)',
        label: 'Exponential Function'
    },
    {
        names: ['reciprocal', 'asymptote curve'],
        implicit: false,
        equation: '1 / x',
        label: 'Reciprocal (1/x)'
    }
];

const AI_TOOLS_DECLARATION = [
    {
        functionDeclarations: [
            {
                name: "plotFunction",
                description: "Plot an explicit 2D mathematical curve y = f(x), e.g. sin(x), x^2, 1/x, 2x + 1. Must be pure math expression without natural language text.",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        expression: {
                            type: "STRING",
                            description: "The mathematical expression in terms of x, e.g. 'x^2', 'sin(x)', '1/x'"
                        }
                    },
                    required: ["expression"]
                }
            },
            {
                name: "plotImplicitEquation",
                description: "Plot an implicit 2D equation involving x and y, e.g. x^2 + y^2 = 25 or sin(x) = cos(y).",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        expression: {
                            type: "STRING",
                            description: "The implicit equation string, e.g. 'x^2 + y^2 = 25'"
                        }
                    },
                    required: ["expression"]
                }
            },
            {
                name: "loadDataTable",
                description: "Load a table of numerical data points into Data Mode for scatter plotting and regression analysis.",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        name: {
                            type: "STRING",
                            description: "Descriptive name for this dataset"
                        },
                        rows: {
                            type: "ARRAY",
                            description: "Array of point objects containing x and y coordinates",
                            items: {
                                type: "OBJECT",
                                properties: {
                                    x: { type: "NUMBER", description: "X coordinate" },
                                    y: { type: "NUMBER", description: "Y coordinate" }
                                },
                                required: ["x", "y"]
                            }
                        }
                    },
                    required: ["rows"]
                }
            },
            {
                name: "switchTo3D",
                description: "Switch application to 3D mode and plot a 3D explicit surface z = f(x, y).",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        expression: {
                            type: "STRING",
                            description: "The 3D surface expression in terms of x and y, e.g. 'sin(x) * cos(y)' or '(x^2 - y^2) / 4'"
                        }
                    },
                    required: ["expression"]
                }
            },
            {
                name: "setViewportBounds",
                description: "Zoom or pan the 2D viewport by setting coordinate domain [xMin, xMax] and range [yMin, yMax].",
                parameters: {
                    type: "OBJECT",
                    properties: {
                        xMin: { type: "NUMBER" },
                        xMax: { type: "NUMBER" },
                        yMin: { type: "NUMBER" },
                        yMax: { type: "NUMBER" }
                    },
                    required: ["xMin", "xMax", "yMin", "yMax"]
                }
            }
        ]
    }
];

// Verify if a token string represents an authentic mathematical formula
export function isMathExpression(expr) {
    if (!expr || typeof expr !== 'string') return false;
    const clean = expr.trim();
    if (!clean) return false;

    // Tokens: words, numbers, or symbols
    const words = clean.toLowerCase().match(/[a-z]+/g) || [];
    const allowedMathKeywords = new Set([
        'x', 'y', 'z', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'ln', 'sqrt', 'cbrt',
        'abs', 'round', 'floor', 'ceil', 'pi', 'e', 'min', 'max', 'pow'
    ]);

    for (const w of words) {
        if (!allowedMathKeywords.has(w)) {
            return false;
        }
    }

    try {
        const fn = compileMathFunction(clean);
        if (!fn) return false;
        const v0 = fn(0, 0);
        const v1 = fn(1, 1);
        const v2 = fn(2, 2);
        return (
            (typeof v0 === 'number' && Number.isFinite(v0)) ||
            (typeof v1 === 'number' && Number.isFinite(v1)) ||
            (typeof v2 === 'number' && Number.isFinite(v2))
        );
    } catch {
        return false;
    }
}

// Fallback rule-based NLP intent parser for instant, offline, and reliable execution
export function parseIntentLocally(prompt, executeTool) {
    if (!prompt || typeof prompt !== 'string') {
        return { text: "Please provide a command or equation.", actions: [] };
    }

    const trimmed = prompt.trim();
    const lower = trimmed.toLowerCase();
    const actions = [];

    // 1. Check for named 3D quadric surfaces / shapes
    for (const surf of GEOMETRIC_3D_SURFACES) {
        if (surf.names.some(n => lower.includes(n))) {
            const res = executeTool('switchTo3D', { expression: surf.equation });
            actions.push({ name: 'switchTo3D', args: { expression: surf.equation }, result: res.message });
            return {
                text: `Identified 3D geometric surface: ${surf.label}. Switched to 3D view and plotted z = ${surf.equation}.`,
                actions
            };
        }
    }

    // 2. Check for explicit 3D mode request
    if (lower.includes('3d') || lower.includes('surface') || lower.includes('z =') || lower.includes('z=')) {
        let expr = 'sin(x) * cos(y)';
        const zMatch = prompt.match(/z\s*=\s*([^,\n]+)/i);
        const plotMatch = prompt.match(/plot\s+(.+)/i);
        if (zMatch && zMatch[1]) {
            expr = zMatch[1].trim();
        } else if (plotMatch && plotMatch[1]) {
            expr = plotMatch[1].replace(/in 3d|3d surface|3d/gi, '').trim();
        }
        // Strip common conversational artifacts
        expr = expr.replace(/^(a|an|the)\s+/i, '').replace(/^(surface|plot)\s+/i, '').trim();

        const res = executeTool('switchTo3D', { expression: expr });
        actions.push({ name: 'switchTo3D', args: { expression: expr }, result: res.message });
        return { text: `Switched to 3D view and plotted surface z = ${expr}.`, actions };
    }

    // 3. Check for named 2D curves (circle, ellipse, hyperbola, parabola, etc.)
    for (const curve of GEOMETRIC_2D_CURVES) {
        if (curve.names.some(n => lower.includes(n))) {
            let eq = curve.equation;
            if (curve.label === 'Circle') {
                const rMatch = prompt.match(/radius\s*(?:of|=)?\s*([0-9.]+)/i) || prompt.match(/r\s*=\s*([0-9.]+)/i);
                if (rMatch) {
                    const r = parseFloat(rMatch[1]);
                    eq = `x^2 + y^2 = ${r * r}`;
                }
            }

            if (curve.implicit) {
                const res = executeTool('plotImplicitEquation', { expression: eq });
                actions.push({ name: 'plotImplicitEquation', args: { expression: eq }, result: res.message });
                return { text: `Plotted 2D ${curve.label}: ${eq}.`, actions };
            } else {
                const res = executeTool('plotFunction', { expression: eq });
                actions.push({ name: 'plotFunction', args: { expression: eq }, result: res.message });
                return { text: `Plotted 2D ${curve.label}: y = ${eq}.`, actions };
            }
        }
    }

    // 4. Check for implicit 2D equations (e.g. x^2 + y^2 = 25 or sin(x) = cos(y))
    // Note: Disambiguate explicit forms like "y = x^2" or "plot y = sin(x)"
    const isExplicitYEq = /^(?:plot\s+)?(?:y|f\(x\))\s*=/i.test(trimmed);
    if (!isExplicitYEq && lower.includes('=') && lower.includes('x') && lower.includes('y')) {
        let expr = 'x^2 + y^2 = 25';
        const eqMatch = prompt.match(/([xXyY0-9\^+\-*/\s.()]+=[^,\n]+)/);
        if (eqMatch) {
            expr = eqMatch[1].trim();
        }
        const res = executeTool('plotImplicitEquation', { expression: expr });
        actions.push({ name: 'plotImplicitEquation', args: { expression: expr }, result: res.message });
        return { text: `Plotted implicit 2D equation: ${expr}.`, actions };
    }

    // 5. Check for data table intent
    if (lower.includes('table') || lower.includes('data points') || lower.includes('scatter') || lower.includes('dataset')) {
        const rows = [
            { x: 1, y: 2.5 },
            { x: 2, y: 4.8 },
            { x: 3, y: 6.9 },
            { x: 4, y: 9.1 },
            { x: 5, y: 11.2 }
        ];
        const res = executeTool('loadDataTable', { name: "AI Sample Data", rows });
        actions.push({ name: 'loadDataTable', args: { name: "AI Sample Data", rows }, result: res.message });
        return { text: `Loaded 5 structured experimental data points into Data Mode with regression analysis.`, actions };
    }

    // 6. Check for Zoom / Viewport commands
    const shouldZoomOut = lower.includes('zoom out') || lower.includes('expand');
    const shouldZoomIn = lower.includes('zoom in');

    // 7. Explicit Function / Math Expression Extraction
    let cleanCandidate = trimmed;
    // Strip conversational lead-ins
    cleanCandidate = cleanCandidate.replace(/^(can you please |please |can you |could you )/i, '');
    cleanCandidate = cleanCandidate.replace(/^(plot|graph|draw|show me|visualize)\s+/i, '');
    cleanCandidate = cleanCandidate.replace(/^(a|an|the)\s+/i, '');
    cleanCandidate = cleanCandidate.replace(/^(function|curve|equation)\s+/i, '');
    cleanCandidate = cleanCandidate.replace(/^(y\s*=\s*|f\(x\)\s*=\s*)/i, '');
    // Strip zoom instructions from equation candidate
    cleanCandidate = cleanCandidate.replace(/(and\s+)?(zoom out|zoom in|expand)/gi, '').trim();

    // Check if the candidate contains both x and y without '=' (indicates a 3D explicit surface z = f(x, y))
    if (/\bx\b/i.test(cleanCandidate) && /\by\b/i.test(cleanCandidate) && !cleanCandidate.includes('=')) {
        if (isMathExpression(cleanCandidate)) {
            const res3D = executeTool('switchTo3D', { expression: cleanCandidate });
            actions.push({ name: 'switchTo3D', args: { expression: cleanCandidate }, result: res3D.message });
            return {
                text: `Detected multi-variable expression: plotted 3D surface z = ${cleanCandidate}.`,
                actions
            };
        }
    }

    // Validate if the candidate is mathematically valid
    const isMathValid = isMathExpression(cleanCandidate);

    if (isMathValid) {
        const resFunc = executeTool('plotFunction', { expression: cleanCandidate });
        actions.push({ name: 'plotFunction', args: { expression: cleanCandidate }, result: resFunc.message });

        if (shouldZoomOut) {
            const resZoom = executeTool('setViewportBounds', { xMin: -50, xMax: 50, yMin: -20, yMax: 2500 });
            actions.push({ name: 'setViewportBounds', args: { xMin: -50, xMax: 50, yMin: -20, yMax: 2500 }, result: resZoom.message });
        } else if (shouldZoomIn) {
            const resZoom = executeTool('setViewportBounds', { xMin: -2, xMax: 2, yMin: -2, yMax: 2 });
            actions.push({ name: 'setViewportBounds', args: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 }, result: resZoom.message });
        }

        return {
            text: `Plotted function y = ${cleanCandidate}${shouldZoomOut ? ' and expanded viewport.' : '.'}`,
            actions
        };
    }

    // If pure zoom request without equation
    if (shouldZoomOut || shouldZoomIn) {
        const bounds = shouldZoomOut
            ? { xMin: -50, xMax: 50, yMin: -20, yMax: 2500 }
            : { xMin: -2, xMax: 2, yMin: -2, yMax: 2 };
        const resZoom = executeTool('setViewportBounds', bounds);
        actions.push({ name: 'setViewportBounds', args: bounds, result: resZoom.message });
        return { text: `Adjusted viewport domain and range.`, actions };
    }

    // Friendly fallback: do not pollute canvas with invalid text
    return {
        text: `I couldn't identify a valid mathematical expression in "${prompt}". Try formulas like "sin(x)", "x^2 - 4", or named surfaces like "hyperboloid", "paraboloid", "saddle", or "ripple".`,
        actions: []
    };
}

export function AIChatbox({
    onPlotFunction,
    onPlotImplicit,
    onLoadDataTable,
    onSwitchTo3D,
    onSetViewportBounds
}) {
    const [isOpen, setIsOpen] = useState(false);
    const [isMinimized, setIsMinimized] = useState(false);
    const [messages, setMessages] = useState([
        {
            id: 'init-1',
            role: 'assistant',
            text: "Graphly AI Assistant ready. Ask me to plot 2D curves, implicit shapes (circle x^2 + y^2 = 25), 3D surfaces (hyperboloids, paraboloids, ripples), or load data tables.",
            actions: []
        }
    ]);
    const [input, setInput] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const chatEndRef = useRef(null);

    // Only auto-scroll when user or assistant adds subsequent messages, NOT on initial modal opening
    useEffect(() => {
        if (isOpen && !isMinimized && messages.length > 1) {
            chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, isOpen, isMinimized]);

    // Direct Tool Executor
    const executeTool = (name, args) => {
        try {
            if (name === 'plotFunction') {
                const expr = args.expression || 'x';
                onPlotFunction(expr);
                return { success: true, message: `Plotted function y = ${expr}` };
            }
            if (name === 'plotImplicitEquation') {
                const expr = args.expression || 'x^2 + y^2 = 25';
                onPlotImplicit(expr);
                return { success: true, message: `Plotted implicit equation ${expr}` };
            }
            if (name === 'loadDataTable') {
                const nameStr = args.name || 'AI Generated Data';
                const rows = Array.isArray(args.rows) ? args.rows : [{ x: 1, y: 1 }];
                onLoadDataTable(nameStr, rows);
                return { success: true, message: `Loaded ${rows.length} rows into Data Mode: ${nameStr}` };
            }
            if (name === 'switchTo3D') {
                const expr = args.expression || 'sin(x) * cos(y)';
                onSwitchTo3D(expr);
                return { success: true, message: `Switched to 3D surface: z = ${expr}` };
            }
            if (name === 'setViewportBounds') {
                const { xMin, xMax, yMin, yMax } = args;
                onSetViewportBounds({ xMin, xMax, yMin, yMax });
                return { success: true, message: `Updated viewport to X[${xMin}, ${xMax}], Y[${yMin}, ${yMax}]` };
            }
            return { success: false, message: `Unknown tool ${name}` };
        } catch (err) {
            return { success: false, message: `Error executing ${name}: ${err.message}` };
        }
    };

    const handleSend = async (e) => {
        e?.preventDefault();
        const promptText = input.trim();
        if (!promptText || isProcessing) return;

        setInput('');
        const userMsg = {
            id: `msg-${Date.now()}`,
            role: 'user',
            text: promptText,
            actions: []
        };
        setMessages(prev => [...prev, userMsg]);
        setIsProcessing(true);

        const apiKey = getGeminiApiKey();

        if (apiKey && isApiKeyConfigured()) {
            try {
                const genAI = new GoogleGenerativeAI(apiKey);
                const model = genAI.getGenerativeModel({
                    model: "gemini-1.5-flash",
                    tools: AI_TOOLS_DECLARATION,
                    systemInstruction: `You are Graphly AI Assistant. When users ask for 3D surfaces (hyperboloid, paraboloid, saddle, sombrero, ripple), call switchTo3D with pure equation like (x^2-y^2)/4. When asked for implicit curves (circle x^2+y^2=25), call plotImplicitEquation. For 2D explicit curves, call plotFunction with mathematical expression only (no English words like 'a hyperboloid').`
                });

                const chat = model.startChat();
                const result = await chat.sendMessage(promptText);
                const response = await result.response;
                const functionCalls = response.functionCalls();

                const executedActions = [];
                if (functionCalls && functionCalls.length > 0) {
                    for (const call of functionCalls) {
                        const execRes = executeTool(call.name, call.args);
                        executedActions.push({
                            name: call.name,
                            args: call.args,
                            result: execRes.message
                        });
                    }
                }

                // If model made no tool call directly, run local intent parser as safety check
                if (executedActions.length === 0) {
                    const fallback = parseIntentLocally(promptText, executeTool);
                    executedActions.push(...fallback.actions);
                }

                const replyText = response.text() || "Action completed on active screen.";
                setMessages(prev => [
                    ...prev,
                    {
                        id: `msg-${Date.now() + 1}`,
                        role: 'assistant',
                        text: replyText,
                        actions: executedActions
                    }
                ]);
            } catch (error) {
                console.warn("Gemini API call failed, falling back to local intent parser:", error);
                const fallback = parseIntentLocally(promptText, executeTool);
                setMessages(prev => [
                    ...prev,
                    {
                        id: `msg-${Date.now() + 1}`,
                        role: 'assistant',
                        text: `${fallback.text} (Executed via local tool engine)`,
                        actions: fallback.actions
                    }
                ]);
            } finally {
                setIsProcessing(false);
            }
        } else {
            // Offline / heuristic tool engine
            setTimeout(() => {
                const fallback = parseIntentLocally(promptText, executeTool);
                setMessages(prev => [
                    ...prev,
                    {
                        id: `msg-${Date.now() + 1}`,
                        role: 'assistant',
                        text: fallback.text,
                        actions: fallback.actions
                    }
                ]);
                setIsProcessing(false);
            }, 250);
        }
    };

    const handleQuickPrompt = (txt) => {
        setInput(txt);
    };

    return (
        <div className="fixed bottom-4 left-4 z-40 font-sans">
            {!isOpen ? (
                <button
                    onClick={() => { setIsOpen(true); setIsMinimized(false); }}
                    className="bg-neutral-900 text-white border border-neutral-900 px-3.5 py-2 font-sans text-xs font-semibold hover:bg-neutral-800 transition-colors flex items-center gap-2 rounded-lg shadow-md cursor-pointer"
                >
                    <Terminal size={14} className="text-blue-400" />
                    <span>AI Copilot</span>
                </button>
            ) : isMinimized ? (
                <div className="w-64 bg-neutral-900 text-white border border-neutral-800 rounded-lg shadow-xl p-2.5 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Terminal size={14} className="text-blue-400" />
                        <span className="font-sans text-xs font-bold uppercase tracking-wider">AI Copilot</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => setIsMinimized(false)}
                            className="p-1 text-neutral-400 hover:text-white rounded transition-colors cursor-pointer"
                            title="Expand"
                        >
                            <Maximize2 size={13} />
                        </button>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="p-1 text-neutral-400 hover:text-white rounded transition-colors cursor-pointer"
                            title="Close"
                        >
                            <X size={13} />
                        </button>
                    </div>
                </div>
            ) : (
                <div className="w-80 sm:w-96 h-[460px] bg-white border border-neutral-300 rounded-xl flex flex-col shadow-2xl overflow-hidden">
                    {/* Header */}
                    <div className="px-3 py-2.5 bg-neutral-900 text-white flex items-center justify-between border-b border-neutral-800 shrink-0">
                        <div className="flex items-center gap-2">
                            <Terminal size={14} className="text-blue-400" />
                            <span className="font-sans text-xs font-bold uppercase tracking-wider">AI Copilot</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <button
                                onClick={() => setIsMinimized(true)}
                                className="p-1 text-neutral-400 hover:text-white rounded transition-colors cursor-pointer"
                                title="Minimize"
                            >
                                <Minus size={13} />
                            </button>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="p-1 text-neutral-400 hover:text-white rounded transition-colors cursor-pointer"
                                title="Close"
                            >
                                <X size={13} />
                            </button>
                        </div>
                    </div>

                    {/* Messages Transcript */}
                    <div className="flex-1 overflow-y-auto p-3 space-y-2.5 bg-white">
                        {messages.map((m) => (
                            <div
                                key={m.id}
                                className={`text-xs ${
                                    m.role === 'user' 
                                        ? 'bg-neutral-100 border border-neutral-300 rounded-lg p-2.5 ml-6' 
                                        : 'bg-white border border-neutral-200 rounded-lg p-2.5 mr-2 shadow-2xs'
                                }`}
                            >
                                <div className="font-mono text-[10px] font-bold text-neutral-500 uppercase mb-1">
                                    {m.role === 'user' ? 'USER' : 'GRAPHLY AI'}
                                </div>
                                <div className="text-neutral-900 leading-relaxed font-sans">{m.text}</div>

                                {/* Executed Tool Action Badges */}
                                {m.actions && m.actions.length > 0 && (
                                    <div className="mt-2 space-y-1.5 pt-2 border-t border-neutral-100">
                                        <div className="text-[9px] font-mono font-bold uppercase text-neutral-400">
                                            Executed Tool Call
                                        </div>
                                        {m.actions.map((act, idx) => (
                                            <div key={idx} className="p-2 bg-neutral-50 border border-neutral-200 rounded-md font-mono text-[10px]">
                                                <div className="flex items-center gap-1.5 font-bold text-neutral-900">
                                                    <CheckCircle2 size={12} className="text-blue-600 shrink-0" />
                                                    <span className="truncate">{act.name}({JSON.stringify(act.args)})</span>
                                                </div>
                                                <div className="text-neutral-600 mt-0.5 text-[9px] pl-4">
                                                    &rarr; {act.result}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                        {isProcessing && (
                            <div className="p-2 border border-neutral-200 border-dashed rounded-lg font-mono text-[10px] text-neutral-500 flex items-center gap-2 bg-neutral-50">
                                <span className="animate-spin text-blue-600 font-bold">&bull;</span>
                                <span>Evaluating tool calls & executing action...</span>
                            </div>
                        )}
                        <div ref={chatEndRef} />
                    </div>

                    {/* Quick Suggestion Chips */}
                    <div className="px-2.5 py-2 border-t border-neutral-200 bg-neutral-50 flex gap-1.5 overflow-x-auto shrink-0">
                        {[
                            { label: 'Hyperboloid (3D)', prompt: 'plot a hyperboloid' },
                            { label: 'Circle', prompt: 'plot circle x^2 + y^2 = 25' },
                            { label: 'Parabola & Zoom', prompt: 'plot y = x^2 and zoom out' },
                            { label: 'Sombrero (3D)', prompt: 'plot sombrero in 3D' },
                            { label: 'Ripple Wave', prompt: 'plot a ripple surface' },
                            { label: 'Sample Data', prompt: 'load sample data table' }
                        ].map((q, idx) => (
                            <button
                                key={idx}
                                type="button"
                                onClick={() => handleQuickPrompt(q.prompt)}
                                className="whitespace-nowrap px-2 py-1 bg-white border border-neutral-200 hover:border-neutral-900 text-[10px] font-mono font-medium text-neutral-700 hover:text-neutral-900 rounded transition-colors cursor-pointer shadow-2xs shrink-0"
                            >
                                {q.label}
                            </button>
                        ))}
                    </div>

                    {/* Input Bar */}
                    <form onSubmit={handleSend} className="p-2.5 border-t border-neutral-200 bg-white flex items-center gap-2 shrink-0">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type prompt (e.g. plot a hyperboloid)..."
                            className="flex-1 px-3 py-1.5 font-mono text-xs text-neutral-900 bg-neutral-50 border border-neutral-200 rounded-lg focus:outline-none focus:bg-white focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900 placeholder:text-neutral-400"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isProcessing}
                            className="bg-neutral-900 text-white px-3.5 py-1.5 font-sans text-xs font-semibold rounded-lg hover:bg-neutral-800 transition-colors disabled:opacity-40 cursor-pointer shrink-0"
                        >
                            Send
                        </button>
                    </form>
                </div>
            )}
        </div>
    );
}
