import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Plus, Trash2, Eye, EyeOff, ZoomIn, Minus, RotateCcw, Sliders, Calculator, MousePointer } from 'lucide-react';
import { calculateNiceTicks, formatNumber, compileMathFunction } from '../lib/mathEngine.js';

export function FunctionGraph({
    functionList = [],
    onAddFunction,
    onRemoveFunction,
    onToggleFunctionVisibility,
    onUpdateFunctionExpression,
    viewportBounds = { xMin: -10, xMax: 10, yMin: -10, yMax: 10 },
    onUpdateViewportBounds,
    onResetViewport
}) {
    const [newExpr, setNewExpr] = useState('');
    const [showRangePanel, setShowRangePanel] = useState(false);
    const [cursorCoords, setCursorCoords] = useState(null);

    const canvasRef = useRef(null);
    const containerRef = useRef(null);

    // Pointer state for pan and pinch
    const activePointers = useRef(new Map());
    const panStartBounds = useRef(null);
    const panStartPos = useRef({ x: 0, y: 0 });
    const pinchStartDist = useRef(null);
    const pinchStartMidpoint = useRef(null);
    const pinchStartBounds = useRef(null);

    const handleAddSubmit = (e) => {
        e?.preventDefault();
        if (!newExpr.trim()) return;
        onAddFunction(newExpr.trim());
        setNewExpr('');
    };

    // Redraw Canvas
    const drawCanvas = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        if (width <= 0 || height <= 0) return;

        // Sync internal canvas resolution with device pixel ratio
        if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
        }

        ctx.save();
        ctx.scale(dpr, dpr);

        const { xMin, xMax, yMin, yMax } = viewportBounds;
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;

        // Coordinate conversion functions
        const toScreenX = (x) => ((x - xMin) / xRange) * width;
        const toScreenY = (y) => (1 - (y - yMin) / yRange) * height;
        const toMathX = (px) => xMin + (px / width) * xRange;

        // 1. Clear & Background
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, width, height);

        // 2. Adaptive Gridlines
        const targetTicksX = Math.max(4, Math.floor(width / 95));
        const targetTicksY = Math.max(4, Math.floor(height / 75));
        const xTicks = calculateNiceTicks(xMin, xMax, targetTicksX);
        const yTicks = calculateNiceTicks(yMin, yMax, targetTicksY);

        ctx.lineWidth = 1;
        ctx.strokeStyle = '#EEEEEE';

        // Vertical grid lines
        xTicks.forEach(tx => {
            const sx = Math.round(toScreenX(tx)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(sx, 0);
            ctx.lineTo(sx, height);
            ctx.stroke();
        });

        // Horizontal grid lines
        yTicks.forEach(ty => {
            const sy = Math.round(toScreenY(ty)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, sy);
            ctx.lineTo(width, sy);
            ctx.stroke();
        });

        // 3. Primary Cartesian Axes (x=0, y=0)
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#000000';

        if (xMin <= 0 && xMax >= 0) {
            const sx0 = Math.round(toScreenX(0)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(sx0, 0);
            ctx.lineTo(sx0, height);
            ctx.stroke();
        }

        if (yMin <= 0 && yMax >= 0) {
            const sy0 = Math.round(toScreenY(0)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, sy0);
            ctx.lineTo(width, sy0);
            ctx.stroke();
        }

        // 4. Tick Labels in JetBrains Mono
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = '#555555';

        // X-axis numeric labels
        const labelY = (yMin <= 0 && yMax >= 0) 
            ? Math.min(height - 6, Math.max(14, toScreenY(0) + 14)) 
            : height - 6;
        ctx.textAlign = 'center';
        xTicks.forEach(tx => {
            if (tx === 0 && (xMin <= 0 && xMax >= 0 && yMin <= 0 && yMax >= 0)) return;
            const sx = toScreenX(tx);
            ctx.fillText(formatNumber(tx), sx, labelY);
        });

        // Y-axis numeric labels
        const labelX = (xMin <= 0 && xMax >= 0) 
            ? Math.min(width - 4, Math.max(34, toScreenX(0) - 6)) 
            : 32;
        ctx.textAlign = 'right';
        yTicks.forEach(ty => {
            const sy = toScreenY(ty);
            ctx.fillText(formatNumber(ty), labelX, sy + 3);
        });

        // 5. Dynamic Resampling: Explicit Functions & Implicit Contours (Marching Squares)
        functionList.forEach(fn => {
            if (!fn.visible || !fn.expression) return;
            const compiled = compileMathFunction(fn.expression);
            if (!compiled) return;

            ctx.strokeStyle = fn.color || '#0044FF';
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';

            if (compiled.type === 'implicit') {
                // ==========================================
                // IMPLICIT CODE PATH: Marching Squares 2D Grid
                // ==========================================
                const cols = 100;
                const rows = 100;
                const dx = xRange / cols;
                const dy = yRange / rows;
                const grid = new Float64Array((cols + 1) * (rows + 1));

                let idx = 0;
                for (let j = 0; j <= rows; j++) {
                    const y = yMin + j * dy;
                    for (let i = 0; i <= cols; i++) {
                        const x = xMin + i * dx;
                        const val = compiled(x, y);
                        grid[idx++] = Number.isFinite(val) ? val : NaN;
                    }
                }

                ctx.beginPath();
                const getVal = (i, j) => grid[j * (cols + 1) + i];

                for (let j = 0; j < rows; j++) {
                    const y0 = yMin + j * dy;
                    const y1 = y0 + dy;

                    for (let i = 0; i < cols; i++) {
                        const x0 = xMin + i * dx;
                        const x1 = x0 + dx;

                        const v0 = getVal(i, j);         // Bottom-Left
                        const v1 = getVal(i + 1, j);     // Bottom-Right
                        const v2 = getVal(i + 1, j + 1); // Top-Right
                        const v3 = getVal(i, j + 1);     // Top-Left

                        if (isNaN(v0) || isNaN(v1) || isNaN(v2) || isNaN(v3)) continue;

                        const b0 = v0 > 0 ? 1 : 0;
                        const b1 = v1 > 0 ? 2 : 0;
                        const b2 = v2 > 0 ? 4 : 0;
                        const b3 = v3 > 0 ? 8 : 0;
                        const caseId = b0 | b1 | b2 | b3;

                        if (caseId === 0 || caseId === 15) continue;

                        const interp = (va, vb) => {
                            const diff = vb - va;
                            if (Math.abs(diff) < 1e-12) return 0.5;
                            const t = -va / diff;
                            return Math.max(0, Math.min(1, t));
                        };

                        const pBottom = { x: x0 + interp(v0, v1) * dx, y: y0 };
                        const pRight  = { x: x1, y: y0 + interp(v1, v2) * dy };
                        const pTop    = { x: x0 + interp(v3, v2) * dx, y: y1 };
                        const pLeft   = { x: x0, y: y0 + interp(v0, v3) * dy };

                        const drawSegment = (pA, pB) => {
                            ctx.moveTo(toScreenX(pA.x), toScreenY(pA.y));
                            ctx.lineTo(toScreenX(pB.x), toScreenY(pB.y));
                        };

                        switch (caseId) {
                            case 1:
                            case 14:
                                drawSegment(pBottom, pLeft);
                                break;
                            case 2:
                            case 13:
                                drawSegment(pBottom, pRight);
                                break;
                            case 3:
                            case 12:
                                drawSegment(pLeft, pRight);
                                break;
                            case 4:
                            case 11:
                                drawSegment(pTop, pRight);
                                break;
                            case 5: {
                                const vCenter = (v0 + v1 + v2 + v3) / 4;
                                if (vCenter > 0) {
                                    drawSegment(pLeft, pTop);
                                    drawSegment(pBottom, pRight);
                                } else {
                                    drawSegment(pBottom, pLeft);
                                    drawSegment(pTop, pRight);
                                }
                                break;
                            }
                            case 6:
                            case 9:
                                drawSegment(pBottom, pTop);
                                break;
                            case 7:
                            case 8:
                                drawSegment(pTop, pLeft);
                                break;
                            case 10: {
                                const vCenter = (v0 + v1 + v2 + v3) / 4;
                                if (vCenter > 0) {
                                    drawSegment(pBottom, pLeft);
                                    drawSegment(pTop, pRight);
                                } else {
                                    drawSegment(pLeft, pTop);
                                    drawSegment(pBottom, pRight);
                                }
                                break;
                            }
                        }
                    }
                }
                ctx.stroke();
            } else {
                // ==========================================
                // EXPLICIT CODE PATH: 1D Horizontal Resampler
                // ==========================================
                let inPath = false;
                let prevY = null;
                const stepPx = 1;

                ctx.beginPath();

                for (let px = 0; px <= width; px += stepPx) {
                    const mx = toMathX(px);
                    const my = compiled(mx);

                    if (typeof my !== 'number' || !Number.isFinite(my) || isNaN(my)) {
                        if (inPath) {
                            ctx.stroke();
                            ctx.beginPath();
                            inPath = false;
                        }
                        prevY = null;
                        continue;
                    }

                    const py = toScreenY(my);

                    if (inPath && prevY !== null) {
                        const dy = Math.abs(my - prevY);
                        const signFlip = (my > 0 && prevY < 0) || (my < 0 && prevY > 0);

                        if ((signFlip && dy > yRange * 0.4) || dy > yRange * 2.5) {
                            ctx.stroke();
                            ctx.beginPath();
                            inPath = false;
                            prevY = my;
                            continue;
                        }
                    }

                    const clampedPy = Math.max(-height * 2, Math.min(height * 3, py));

                    if (!inPath) {
                        ctx.moveTo(px, clampedPy);
                        inPath = true;
                    } else {
                        ctx.lineTo(px, clampedPy);
                    }

                    prevY = my;
                }

                if (inPath) {
                    ctx.stroke();
                }
            }
        });

        ctx.restore();
    }, [viewportBounds, functionList]);

    // Redraw on dependencies or resize
    useEffect(() => {
        let animId = requestAnimationFrame(drawCanvas);
        const handleResize = () => {
            cancelAnimationFrame(animId);
            animId = requestAnimationFrame(drawCanvas);
        };
        window.addEventListener('resize', handleResize);
        return () => {
            cancelAnimationFrame(animId);
            window.removeEventListener('resize', handleResize);
        };
    }, [drawCanvas]);

    // Zoom centered at a point
    const zoomAtPoint = useCallback((factor, mousePx, mousePy) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        const { xMin, xMax, yMin, yMax } = viewportBounds;
        const xRange = xMax - xMin;
        const yRange = yMax - yMin;

        // Math coords under cursor
        const cx = xMin + (mousePx / width) * xRange;
        const cy = yMin + (1 - mousePy / height) * yRange;

        const newXMin = cx - (cx - xMin) * factor;
        const newXMax = cx + (xMax - cx) * factor;
        const newYMin = cy - (cy - yMin) * factor;
        const newYMax = cy + (yMax - cy) * factor;

        // Boundary guard against infinite degeneration
        if (newXMax - newXMin > 1e-12 && newXMax - newXMin < 1e20) {
            onUpdateViewportBounds({
                xMin: newXMin,
                xMax: newXMax,
                yMin: newYMin,
                yMax: newYMax
            });
        }
    }, [viewportBounds, onUpdateViewportBounds]);

    // Wheel Zoom handler (zooms toward cursor)
    const handleWheel = (e) => {
        e.preventDefault();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const mousePx = e.clientX - rect.left;
        const mousePy = e.clientY - rect.top;

        // Smooth zoom factor
        const factor = e.deltaY < 0 ? 0.88 : 1.14;
        zoomAtPoint(factor, mousePx, mousePy);
    };

    // Pointer Events for Pan & Pinch
    const handlePointerDown = (e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.setPointerCapture(e.pointerId);

        activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (activePointers.current.size === 1) {
            panStartBounds.current = { ...viewportBounds };
            panStartPos.current = { x: e.clientX, y: e.clientY };
        } else if (activePointers.current.size === 2) {
            const pts = Array.from(activePointers.current.values());
            const dx = pts[1].x - pts[0].x;
            const dy = pts[1].y - pts[0].y;
            pinchStartDist.current = Math.hypot(dx, dy);
            const rect = canvas.getBoundingClientRect();
            pinchStartMidpoint.current = {
                px: (pts[0].x + pts[1].x) / 2 - rect.left,
                py: (pts[0].y + pts[1].y) / 2 - rect.top
            };
            pinchStartBounds.current = { ...viewportBounds };
        }
    };

    const handlePointerMove = (e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();

        // Update live cursor coordinates
        const mousePx = e.clientX - rect.left;
        const mousePy = e.clientY - rect.top;
        if (mousePx >= 0 && mousePx <= rect.width && mousePy >= 0 && mousePy <= rect.height) {
            const { xMin, xMax, yMin, yMax } = viewportBounds;
            const mx = xMin + (mousePx / rect.width) * (xMax - xMin);
            const my = yMin + (1 - mousePy / rect.height) * (yMax - yMin);
            setCursorCoords({ x: mx, y: my });
        } else {
            setCursorCoords(null);
        }

        if (!activePointers.current.has(e.pointerId)) return;
        activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (activePointers.current.size === 1 && panStartBounds.current) {
            // Drag-to-Pan
            const dx = e.clientX - panStartPos.current.x;
            const dy = e.clientY - panStartPos.current.y;
            const xSpan = panStartBounds.current.xMax - panStartBounds.current.xMin;
            const ySpan = panStartBounds.current.yMax - panStartBounds.current.yMin;

            const dMathX = -(dx / rect.width) * xSpan;
            const dMathY = (dy / rect.height) * ySpan;

            onUpdateViewportBounds({
                xMin: panStartBounds.current.xMin + dMathX,
                xMax: panStartBounds.current.xMax + dMathX,
                yMin: panStartBounds.current.yMin + dMathY,
                yMax: panStartBounds.current.yMax + dMathY
            });
        } else if (activePointers.current.size === 2 && pinchStartDist.current && pinchStartBounds.current) {
            // Pinch-to-Zoom
            const pts = Array.from(activePointers.current.values());
            const currentDist = Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y);
            if (currentDist > 5 && pinchStartDist.current > 5) {
                const factor = pinchStartDist.current / currentDist;
                const { px, py } = pinchStartMidpoint.current;
                const bounds = pinchStartBounds.current;
                const xSpan = bounds.xMax - bounds.xMin;
                const ySpan = bounds.yMax - bounds.yMin;
                const cx = bounds.xMin + (px / rect.width) * xSpan;
                const cy = bounds.yMin + (1 - py / rect.height) * ySpan;

                onUpdateViewportBounds({
                    xMin: cx - (cx - bounds.xMin) * factor,
                    xMax: cx + (bounds.xMax - cx) * factor,
                    yMin: cy - (cy - bounds.yMin) * factor,
                    yMax: cy + (bounds.yMax - cy) * factor
                });
            }
        }
    };

    const handlePointerUp = (e) => {
        if (canvasRef.current?.hasPointerCapture(e.pointerId)) {
            canvasRef.current.releasePointerCapture(e.pointerId);
        }
        activePointers.current.delete(e.pointerId);

        if (activePointers.current.size === 1) {
            const remaining = Array.from(activePointers.current.values())[0];
            panStartPos.current = { x: remaining.x, y: remaining.y };
            panStartBounds.current = { ...viewportBounds };
        } else if (activePointers.current.size === 0) {
            panStartBounds.current = null;
            pinchStartDist.current = null;
            pinchStartMidpoint.current = null;
            pinchStartBounds.current = null;
        }
    };

    return (
        <div className="relative flex h-[calc(100vh-56px)] overflow-hidden bg-white">
            {/* Main Canvas Workspace */}
            <div className="flex-1 relative bg-white flex flex-col overflow-hidden border-r-2 border-black">
                {/* Top Brutalist Equation Bar */}
                <div className="p-3 bg-white border-b-2 border-black flex items-center gap-3">
                    <div className="flex items-center gap-2 font-mono font-bold text-sm bg-neutral-100 border border-black px-2.5 py-1.5">
                        <Calculator size={16} className="text-black" />
                        <span>f(x) =</span>
                    </div>
                    <form onSubmit={handleAddSubmit} className="flex-1 flex items-center gap-2">
                        <input
                            type="text"
                            value={newExpr}
                            onChange={(e) => setNewExpr(e.target.value)}
                            placeholder="e.g. x^2 + y^2 = 25, 2x, sin(2x), 1/x, tan(x)"
                            className="flex-1 bg-white border border-black px-3 py-1.5 font-mono text-sm text-black placeholder:text-neutral-400 focus:outline-none focus:ring-1 focus:ring-black rounded-none"
                        />
                        <button
                            type="submit"
                            className="bg-black text-white border border-black px-4 py-1.5 font-mono text-xs font-bold hover:bg-white hover:text-black transition-none flex items-center gap-1.5 rounded-none"
                        >
                            <Plus size={14} />
                            <span>PLOT FUNCTION</span>
                        </button>
                    </form>
                </div>

                {/* Canvas Viewport Area */}
                <div 
                    ref={containerRef}
                    className="flex-1 relative flex items-center justify-center bg-white p-2 md:p-4 overflow-hidden graph-touch-surface select-none" 
                    style={{ touchAction: 'none' }}
                >
                    <canvas
                        ref={canvasRef}
                        id="function-canvas-renderer"
                        className="w-full h-full border-2 border-black bg-white cursor-crosshair"
                        style={{ display: 'block', touchAction: 'none' }}
                        onWheel={handleWheel}
                        onPointerDown={handlePointerDown}
                        onPointerMove={handlePointerMove}
                        onPointerUp={handlePointerUp}
                        onPointerCancel={handlePointerUp}
                        onPointerLeave={(e) => {
                            handlePointerUp(e);
                            setCursorCoords(null);
                        }}
                    />

                    {/* Cursor Position Readout */}
                    {cursorCoords && (
                        <div className="absolute top-6 left-6 font-mono text-[11px] font-bold bg-white border border-black px-2.5 py-1 z-20 shadow-none pointer-events-none">
                            x: {formatNumber(cursorCoords.x)} &nbsp;|&nbsp; y: {formatNumber(cursorCoords.y)}
                        </div>
                    )}

                    {/* Viewport Range Badges */}
                    <div className="absolute top-6 right-6 font-mono text-[10px] bg-white border border-black px-2 py-1 z-20 pointer-events-none">
                        [{formatNumber(viewportBounds.xMin)}, {formatNumber(viewportBounds.xMax)}]
                    </div>

                    {/* Floating Zoom & Reset Controls */}
                    <div className="absolute bottom-6 right-6 flex flex-col gap-1 z-20">
                        <button
                            onClick={() => {
                                const canvas = canvasRef.current;
                                if (!canvas) return;
                                const rect = canvas.getBoundingClientRect();
                                zoomAtPoint(0.8, rect.width / 2, rect.height / 2);
                            }}
                            title="Zoom In"
                            className="w-8 h-8 bg-white border-2 border-black flex items-center justify-center text-black hover:bg-black hover:text-white transition-none rounded-none font-bold"
                        >
                            <ZoomIn size={14} />
                        </button>
                        <button
                            onClick={() => {
                                const canvas = canvasRef.current;
                                if (!canvas) return;
                                const rect = canvas.getBoundingClientRect();
                                zoomAtPoint(1.25, rect.width / 2, rect.height / 2);
                            }}
                            title="Zoom Out"
                            className="w-8 h-8 bg-white border-2 border-black flex items-center justify-center text-black hover:bg-black hover:text-white transition-none rounded-none font-bold"
                        >
                            <Minus size={14} />
                        </button>
                        <button
                            onClick={onResetViewport}
                            title="Reset Origin [-10, 10]"
                            className="w-8 h-8 bg-white border-2 border-black flex items-center justify-center text-black hover:bg-black hover:text-white transition-none rounded-none font-bold"
                        >
                            <RotateCcw size={14} />
                        </button>
                    </div>
                </div>
            </div>

            {/* Right Sidebar: Functions & Range Settings */}
            <div className="w-80 bg-white flex flex-col h-full overflow-hidden border-l-0">
                {/* Sidebar Header Tabs */}
                <div className="flex border-b-2 border-black">
                    <button
                        onClick={() => setShowRangePanel(false)}
                        className={`flex-1 py-3 text-xs font-bold border-r border-black flex items-center justify-center gap-1.5 transition-none rounded-none ${
                            !showRangePanel ? 'bg-black text-white' : 'bg-white text-black hover:bg-neutral-100'
                        }`}
                    >
                        <Calculator size={14} />
                        <span>FUNCTIONS ({functionList.length})</span>
                    </button>
                    <button
                        onClick={() => setShowRangePanel(true)}
                        className={`flex-1 py-3 text-xs font-bold flex items-center justify-center gap-1.5 transition-none rounded-none ${
                            showRangePanel ? 'bg-black text-white' : 'bg-white text-black hover:bg-neutral-100'
                        }`}
                    >
                        <Sliders size={14} />
                        <span>BOUNDS</span>
                    </button>
                </div>

                {/* Sidebar Content */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {!showRangePanel ? (
                        <div className="space-y-3">
                            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-neutral-500">
                                Active Expressions
                            </div>
                            {functionList.length === 0 ? (
                                <div className="p-4 border border-black border-dashed text-center">
                                    <p className="text-xs text-neutral-500 font-mono">No functions entered.</p>
                                    <p className="text-[11px] text-black mt-1 font-sans">Type an equation like 1/x or sin(x) above to plot.</p>
                                </div>
                            ) : (
                                functionList.map((fn, index) => (
                                    <div key={fn.id || index} className="p-3 border border-black bg-white space-y-2">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <div
                                                    className="w-3 h-3 border border-black"
                                                    style={{ backgroundColor: fn.color || '#0044FF' }}
                                                />
                                                <span className="font-mono text-xs font-bold text-black flex items-center gap-1.5">
                                                    {compileMathFunction(fn.expression)?.type === 'implicit' ? (
                                                        <>
                                                            <span>F_{index + 1}(x,y)</span>
                                                            <span className="text-[9px] bg-neutral-200 border border-black px-1 py-0.5 font-mono">IMPLICIT</span>
                                                        </>
                                                    ) : (
                                                        <span>f_{index + 1}(x)</span>
                                                    )}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <button
                                                    onClick={() => onToggleFunctionVisibility(fn.id)}
                                                    className="p-1 text-black border border-transparent hover:border-black hover:bg-black hover:text-white transition-none"
                                                    title={fn.visible ? "Hide Function" : "Show Function"}
                                                >
                                                    {fn.visible ? <Eye size={13} /> : <EyeOff size={13} />}
                                                </button>
                                                <button
                                                    onClick={() => onRemoveFunction(fn.id)}
                                                    className="p-1 text-black border border-transparent hover:border-black hover:bg-black hover:text-white transition-none"
                                                    title="Delete Function"
                                                >
                                                    <Trash2 size={13} />
                                                </button>
                                            </div>
                                        </div>
                                        <input
                                            type="text"
                                            value={fn.expression}
                                            onChange={(e) => onUpdateFunctionExpression(fn.id, e.target.value)}
                                            className="w-full bg-neutral-50 border border-black px-2 py-1 font-mono text-xs text-black focus:outline-none focus:bg-white rounded-none"
                                        />
                                    </div>
                                ))
                            )}
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-neutral-500">
                                Viewport Coordinates
                            </div>
                            <div className="p-3 border border-black space-y-3">
                                <div>
                                    <label className="text-[10px] font-mono font-bold block mb-1">X DOMAIN [MIN, MAX]</label>
                                    <div className="grid grid-cols-2 gap-2">
                                        <input
                                            type="number"
                                            value={viewportBounds.xMin}
                                            onChange={(e) => onUpdateViewportBounds({ ...viewportBounds, xMin: parseFloat(e.target.value) || 0 })}
                                            className="border border-black p-1.5 font-mono text-xs rounded-none bg-white"
                                        />
                                        <input
                                            type="number"
                                            value={viewportBounds.xMax}
                                            onChange={(e) => onUpdateViewportBounds({ ...viewportBounds, xMax: parseFloat(e.target.value) || 0 })}
                                            className="border border-black p-1.5 font-mono text-xs rounded-none bg-white"
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="text-[10px] font-mono font-bold block mb-1">Y RANGE [MIN, MAX]</label>
                                    <div className="grid grid-cols-2 gap-2">
                                        <input
                                            type="number"
                                            value={viewportBounds.yMin}
                                            onChange={(e) => onUpdateViewportBounds({ ...viewportBounds, yMin: parseFloat(e.target.value) || 0 })}
                                            className="border border-black p-1.5 font-mono text-xs rounded-none bg-white"
                                        />
                                        <input
                                            type="number"
                                            value={viewportBounds.yMax}
                                            onChange={(e) => onUpdateViewportBounds({ ...viewportBounds, yMax: parseFloat(e.target.value) || 0 })}
                                            className="border border-black p-1.5 font-mono text-xs rounded-none bg-white"
                                        />
                                    </div>
                                </div>
                                <button
                                    onClick={onResetViewport}
                                    className="w-full bg-white text-black border border-black py-1.5 font-mono text-xs font-bold hover:bg-black hover:text-white transition-none rounded-none"
                                >
                                    DEFAULT BOUNDS [-10, 10]
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
