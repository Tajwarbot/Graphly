import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { compileMathFunction } from '../lib/mathEngine.js';
import { RotateCcw, Box, Eye, EyeOff, Sliders, Play, Plus, Trash2, Layers, ChevronDown, ChevronRight, Menu, X } from 'lucide-react';

const PRESETS = [
    { name: 'Ripple', expr: 'sin(x) * cos(y)' },
    { name: 'Saddle', expr: '(x^2 - y^2) / 4' },
    { name: 'Sombrero', expr: '2 * sin(sqrt(x^2 + y^2)) / (sqrt(x^2 + y^2) + 0.1)' },
    { name: 'Gaussian Hill', expr: '3 * exp(-(x^2 + y^2) / 4)' },
    { name: 'Paraboloid', expr: '(x^2 + y^2) / 6' },
    { name: 'Cross Wave', expr: 'sin(x) + cos(y)' },
    { name: 'Cutting Plane', expr: '0.4 * x' }
];

const SURFACE_COLORS = [
    { name: "Cobalt Blue", hex: "#2563EB" },
    { name: "Crimson Red", hex: "#DC2626" },
    { name: "Emerald Green", hex: "#059669" },
    { name: "Vivid Amber", hex: "#D97706" },
    { name: "Deep Violet", hex: "#7C3AED" },
    { name: "Teal Cyan", hex: "#0891B2" },
    { name: "Hot Rose", hex: "#E11D48" },
    { name: "Midnight Black", hex: "#18181B" }
];

export function ThreeDGraph({ initialEquation }) {
    const containerRef = useRef(null);
    const canvasRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const cameraRef = useRef(null);
    const controlsRef = useRef(null);
    const surfacesGroupRef = useRef(null);

    // Multiple surfaces state
    const [surfaces, setSurfaces] = useState([
        {
            id: 'surf-1',
            name: 'Surface 1',
            equation: initialEquation || 'sin(x) * cos(y)',
            color: '#2563EB',
            visible: true,
            opacity: 0.85,
            wireframeMode: 'both' // 'both' | 'solid' | 'wireframe'
        }
    ]);

    useEffect(() => {
        if (initialEquation) {
            setSurfaces(prev => {
                const next = [...prev];
                if (next.length > 0) {
                    next[0] = { ...next[0], equation: initialEquation };
                } else {
                    next.push({
                        id: 'surf-1',
                        name: 'Surface 1',
                        equation: initialEquation,
                        color: '#2563EB',
                        visible: true,
                        opacity: 0.85,
                        wireframeMode: 'both'
                    });
                }
                return next;
            });
            setExpandedSurfaceId('surf-1');
        }
    }, [initialEquation]);

    const [activeTab, setActiveTab] = useState('surfaces'); // 'surfaces' | 'settings'
    const [expandedSurfaceId, setExpandedSurfaceId] = useState('surf-1');
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    // Global 3D scene parameters
    const [gridSegments, setGridSegments] = useState(60);
    const [range, setRange] = useState(5);
    const [heightScale, setHeightScale] = useState(1);
    const [showAxes, setShowAxes] = useState(true);
    const [showGrid, setShowGrid] = useState(true);

    // Build or update all 3D surface meshes in scene
    const updateSurfaces = useCallback(() => {
        if (!sceneRef.current || !surfacesGroupRef.current) return;
        const group = surfacesGroupRef.current;

        // Dispose previous children
        while (group.children.length > 0) {
            const child = group.children[0];
            group.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        }

        const size = range * 2;
        const segs = gridSegments;

        surfaces.forEach((surf) => {
            if (!surf.visible) return;

            const compiled = compileMathFunction(surf.equation);
            if (!compiled) return;

            const geometry = new THREE.PlaneGeometry(size, size, segs, segs);
            geometry.rotateX(-Math.PI / 2);

            const posAttr = geometry.attributes.position;
            for (let i = 0; i < posAttr.count; i++) {
                const x = posAttr.getX(i);
                const y = -posAttr.getZ(i);
                let zVal = compiled(x, y);

                if (!Number.isFinite(zVal) || isNaN(zVal)) {
                    zVal = 0;
                }
                const clampedZ = Math.max(-size * 2, Math.min(size * 2, zVal * heightScale));
                posAttr.setY(i, clampedZ);
            }

            posAttr.needsUpdate = true;
            geometry.computeVertexNormals();

            // Surface Material
            const isTransparent = surf.opacity < 1;
            const material = new THREE.MeshStandardMaterial({
                color: new THREE.Color(surf.color),
                flatShading: true,
                roughness: 0.5,
                metalness: 0.1,
                side: THREE.DoubleSide,
                transparent: isTransparent,
                opacity: surf.opacity,
                wireframe: surf.wireframeMode === 'wireframe',
                depthWrite: !isTransparent // Proper alpha blending
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            group.add(mesh);

            // Faceted Wireframe Overlay
            if (surf.wireframeMode === 'both') {
                const wireGeo = new THREE.WireframeGeometry(geometry);
                const wireMat = new THREE.LineBasicMaterial({
                    color: 0x18181B,
                    linewidth: 1,
                    transparent: true,
                    opacity: 0.2
                });
                const wireMesh = new THREE.LineSegments(wireGeo, wireMat);
                group.add(wireMesh);
            }
        });
    }, [surfaces, gridSegments, range, heightScale]);

    // Initialize Three.js Scene
    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        const width = container.clientWidth || 800;
        const height = container.clientHeight || 600;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xFFFFFF);
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(16, 14, 18);
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            antialias: true,
            powerPreference: 'high-performance'
        });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        rendererRef.current = renderer;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 0, 0);
        controlsRef.current = controls;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.7);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xFFFFFF, 0.9);
        dirLight.position.set(20, 35, 20);
        dirLight.castShadow = true;
        scene.add(dirLight);

        const fillLight = new THREE.DirectionalLight(0xEEEEEE, 0.4);
        fillLight.position.set(-20, -10, -20);
        scene.add(fillLight);

        // Ground Reference Grid
        const gridHelper = new THREE.GridHelper(24, 24, 0x18181B, 0xD4D4D8);
        gridHelper.position.y = -0.01;
        gridHelper.name = 'referenceGrid';
        scene.add(gridHelper);

        // Cartesian Axes
        const axesGroup = new THREE.Group();
        axesGroup.name = 'cartesianAxes';

        const createAxis = (start, end, color = 0x71717A) => {
            const geom = new THREE.BufferGeometry().setFromPoints([start, end]);
            const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
            return new THREE.Line(geom, mat);
        };

        const createAxisLabel = (text, colorHex, position) => {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');

            // Circular badge with high-contrast border
            ctx.fillStyle = '#FFFFFF';
            ctx.strokeStyle = colorHex;
            ctx.lineWidth = 8;
            ctx.beginPath();
            ctx.arc(64, 64, 50, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Coordinate Axis Text
            ctx.font = 'bold 48px "JetBrains Mono", monospace, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = colorHex;
            ctx.fillText(text, 64, 64);

            const texture = new THREE.CanvasTexture(canvas);
            texture.minFilter = THREE.LinearFilter;
            const spriteMat = new THREE.SpriteMaterial({ 
                map: texture, 
                depthTest: false, 
                transparent: true 
            });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.position.copy(position);
            sprite.scale.set(1.4, 1.4, 1.4);
            return sprite;
        };

        // Red for X, Blue for Z (Height), Green for Y (Depth)
        axesGroup.add(createAxis(new THREE.Vector3(-12.5, 0, 0), new THREE.Vector3(12.5, 0, 0), 0xDC2626)); // X (Red)
        axesGroup.add(createAxis(new THREE.Vector3(0, -10.5, 0), new THREE.Vector3(0, 10.5, 0), 0x2563EB)); // Z Height (Blue)
        axesGroup.add(createAxis(new THREE.Vector3(0, 0, -12.5), new THREE.Vector3(0, 0, 12.5), 0x059669)); // Y Depth (Green)

        // Axis Tip Coordinate Text Labels
        axesGroup.add(createAxisLabel('+X', '#DC2626', new THREE.Vector3(13.2, 0, 0)));
        axesGroup.add(createAxisLabel('-X', '#DC2626', new THREE.Vector3(-13.2, 0, 0)));
        axesGroup.add(createAxisLabel('+Z', '#2563EB', new THREE.Vector3(0, 11.2, 0)));
        axesGroup.add(createAxisLabel('-Z', '#2563EB', new THREE.Vector3(0, -11.2, 0)));
        axesGroup.add(createAxisLabel('+Y', '#059669', new THREE.Vector3(0, 0, -13.2)));
        axesGroup.add(createAxisLabel('-Y', '#059669', new THREE.Vector3(0, 0, 13.2)));

        scene.add(axesGroup);

        // Group dedicated to surface meshes
        const surfacesGroup = new THREE.Group();
        scene.add(surfacesGroup);
        surfacesGroupRef.current = surfacesGroup;

        // Render loop
        let animId;
        const animate = () => {
            animId = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Resize Observer
        const resizeObserver = new ResizeObserver((entries) => {
            for (let entry of entries) {
                const { width: w, height: h } = entry.contentRect;
                if (w > 0 && h > 0) {
                    camera.aspect = w / h;
                    camera.updateProjectionMatrix();
                    renderer.setSize(w, h);
                }
            }
        });
        resizeObserver.observe(container);

        return () => {
            cancelAnimationFrame(animId);
            resizeObserver.disconnect();
            controls.dispose();
            renderer.dispose();
        };
    }, []);

    // Toggle Grid and Axes visibility
    useEffect(() => {
        if (!sceneRef.current) return;
        const grid = sceneRef.current.getObjectByName('referenceGrid');
        if (grid) grid.visible = showGrid;
        const axes = sceneRef.current.getObjectByName('cartesianAxes');
        if (axes) axes.visible = showAxes;
    }, [showGrid, showAxes]);

    // Redraw surfaces on change
    useEffect(() => {
        updateSurfaces();
    }, [updateSurfaces]);

    // Surface management handlers
    const addSurface = (presetExpr = null) => {
        const nextIndex = surfaces.length;
        const nextColor = SURFACE_COLORS[nextIndex % SURFACE_COLORS.length].hex;
        const newId = `surf-${Date.now()}`;
        const newSurf = {
            id: newId,
            name: `Surface ${nextIndex + 1}`,
            equation: presetExpr || (nextIndex === 1 ? '(x^2 - y^2) / 4' : 'cos(x) * sin(y)'),
            color: nextColor,
            visible: true,
            opacity: 0.85,
            wireframeMode: 'both'
        };
        setSurfaces(prev => [...prev, newSurf]);
        setExpandedSurfaceId(newId);
    };

    const updateSurface = (id, updates) => {
        setSurfaces(prev => prev.map(s => s.id === id ? { ...s, ...updates } : s));
    };

    const deleteSurface = (id, e) => {
        if (e) e.stopPropagation();
        if (surfaces.length <= 1) {
            alert("At least one 3D surface must remain.");
            return;
        }
        setSurfaces(prev => prev.filter(s => s.id !== id));
    };

    const resetCamera = () => {
        if (!cameraRef.current || !controlsRef.current) return;
        cameraRef.current.position.set(16, 14, 18);
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
    };

    return (
        <div className="relative flex h-[calc(100vh-56px)] overflow-hidden bg-white">
            {/* Main 3D Canvas Area */}
            <div className="flex-1 relative bg-white flex flex-col overflow-hidden">
                {/* 3D Viewport */}
                <div 
                    ref={containerRef}
                    className="flex-1 relative w-full h-full bg-white overflow-hidden select-none cursor-grab active:cursor-grabbing"
                    style={{ touchAction: 'none' }}
                >
                    <canvas ref={canvasRef} className="w-full h-full block" />

                    {/* Active Surfaces Legend HUD */}
                    <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-xs border border-neutral-300 rounded-xl p-3 pointer-events-auto z-10 max-w-xs shadow-sm">
                        <div className="text-[10px] font-mono font-semibold uppercase tracking-wider text-neutral-600 mb-1.5 flex items-center justify-between">
                            <span>Active 3D Surfaces ({surfaces.filter(s => s.visible).length})</span>
                        </div>
                        <div className="space-y-1.5 max-h-32 overflow-y-auto">
                            {surfaces.map((s) => (
                                <div key={s.id} className="flex items-center gap-2 text-xs">
                                    <div className="w-2.5 h-2.5 rounded-full ring-1 ring-white shrink-0 shadow-2xs" style={{ backgroundColor: s.color }} />
                                    <span className={`font-mono text-[11px] truncate ${s.visible ? 'text-neutral-900 font-semibold' : 'text-neutral-400 line-through'}`}>
                                        z = {s.equation}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="text-[10px] font-mono text-neutral-500 mt-2 pt-2 border-t border-neutral-200 flex gap-2">
                            <span>x,y ∈ [{-range}, {range}]</span>
                            <span>•</span>
                            <span>{gridSegments}x{gridSegments} mesh</span>
                        </div>
                    </div>

                    {/* Mobile Sidebar Toggle Button */}
                    <div className="absolute top-4 right-4 z-20 md:hidden">
                        <button
                            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                            className="bg-white/95 backdrop-blur-xs border border-neutral-300 rounded-lg p-2 text-neutral-800 hover:bg-neutral-100 shadow-sm transition-colors cursor-pointer"
                            aria-label="Toggle Surfaces Panel"
                        >
                            {isSidebarOpen ? <X size={16} /> : <Menu size={16} />}
                        </button>
                    </div>

                    {/* Navigation HUD Instructions */}
                    <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-xs border border-neutral-300 rounded-md px-3 py-1.5 font-mono text-[11px] text-neutral-700 pointer-events-none z-10 shadow-2xs">
                        [L-CLICK] Rotate &nbsp;•&nbsp; [R-CLICK / SHIFT] Pan &nbsp;•&nbsp; [SCROLL] Zoom
                    </div>

                    {/* Reset Camera Button */}
                    <div className="absolute bottom-4 right-4 z-10">
                        <button
                            onClick={resetCamera}
                            title="Reset 3D Perspective"
                            className="bg-white border border-neutral-300 rounded-lg px-3 py-2 text-neutral-800 hover:bg-neutral-100 transition-colors shadow-xs flex items-center gap-1.5 text-xs font-semibold cursor-pointer"
                        >
                            <RotateCcw size={14} />
                            <span>Reset View</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Overlay Backdrop */}
            {isSidebarOpen && (
                <div 
                    className="fixed inset-0 bg-black/30 backdrop-blur-xs z-30 md:hidden"
                    onClick={() => setIsSidebarOpen(false)}
                />
            )}

            {/* Right Sidebar: Compact Multi-Surface Manager & Parameters */}
            <div className={`
                fixed inset-y-0 right-0 z-40 w-76 sm:w-80 bg-white flex flex-col h-full overflow-hidden border-l border-neutral-300 shadow-xl
                md:relative md:translate-x-0 md:w-72 lg:w-80 md:z-auto md:shadow-none transition-transform duration-200
                ${isSidebarOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-0'}
            `}>
                {/* Segmented Control Header */}
                <div className="flex border-b border-neutral-300 shrink-0 bg-neutral-100/60 p-1.5 gap-1.5">
                    <button
                        onClick={() => setActiveTab('surfaces')}
                        className={`flex-1 py-1.5 rounded-md text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                            activeTab === 'surfaces' ? 'bg-white text-neutral-900 shadow-2xs border border-neutral-300 font-bold' : 'text-neutral-600 hover:text-neutral-900 border border-transparent'
                        }`}
                    >
                        <Layers size={13} /> Surfaces ({surfaces.length})
                    </button>
                    <button
                        onClick={() => setActiveTab('settings')}
                        className={`flex-1 py-1.5 rounded-md text-xs font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer ${
                            activeTab === 'settings' ? 'bg-white text-neutral-900 shadow-2xs border border-neutral-300 font-bold' : 'text-neutral-600 hover:text-neutral-900 border border-transparent'
                        }`}
                    >
                        <Sliders size={13} /> Settings
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-3 space-y-4 pb-20">
                    {activeTab === 'surfaces' ? (
                        <>
                            {/* Action Row */}
                            <div className="flex items-center justify-between">
                                <span className="text-xs font-bold uppercase tracking-wider text-neutral-700">Equations & Layers</span>
                                <button
                                    onClick={() => addSurface()}
                                    className="rounded-md bg-neutral-900 text-white hover:bg-neutral-800 px-2.5 py-1 text-xs font-semibold flex items-center gap-1 transition-colors cursor-pointer shadow-2xs"
                                >
                                    <Plus size={12} /> Add Surface
                                </button>
                            </div>

                            {/* Surface Cards List */}
                            <div className="space-y-3">
                                {surfaces.map((s, idx) => (
                                    <div key={s.id} className="border border-neutral-200 rounded-xl bg-white shadow-xs overflow-hidden transition-all">
                                        {/* Card Header */}
                                        <div 
                                            className="flex items-center gap-2.5 p-3 bg-white border-b border-neutral-100 cursor-pointer hover:bg-neutral-50/70"
                                            onClick={() => setExpandedSurfaceId(expandedSurfaceId === s.id ? null : s.id)}
                                        >
                                            <button
                                                onClick={(e) => { e.stopPropagation(); updateSurface(s.id, { visible: !s.visible }); }}
                                                className={`p-1.5 rounded-md border transition-colors cursor-pointer ${s.visible ? 'border-neutral-200 text-neutral-700 hover:bg-neutral-100' : 'border-neutral-200 bg-neutral-100 text-neutral-400'}`}
                                                title={s.visible ? "Hide surface" : "Show surface"}
                                            >
                                                {s.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                                            </button>

                                            <div className="w-3.5 h-3.5 rounded-full ring-2 ring-white shadow-2xs shrink-0" style={{ backgroundColor: s.color }} />

                                            <input
                                                className="flex-1 bg-transparent text-xs font-semibold text-neutral-900 outline-none border-b border-transparent focus:border-neutral-400 py-0.5"
                                                value={s.name}
                                                onClick={(e) => e.stopPropagation()}
                                                onChange={(e) => updateSurface(s.id, { name: e.target.value })}
                                            />

                                            <ChevronDown size={15} className={`text-neutral-400 transition-transform ${expandedSurfaceId === s.id ? 'rotate-180' : ''}`} />
                                        </div>

                                        {/* Expanded Surface Details */}
                                        {expandedSurfaceId === s.id && (
                                            <div className="p-3.5 space-y-3.5 bg-neutral-50/30">
                                                {/* Equation Input */}
                                                <div>
                                                    <label className="text-xs font-medium text-neutral-600 mb-1 block">z = f(x, y)</label>
                                                    <input
                                                        className="w-full p-2 bg-white border border-neutral-200 rounded-md font-mono text-xs text-neutral-900 outline-none focus:border-neutral-900 focus:ring-1 focus:ring-neutral-900"
                                                        value={s.equation}
                                                        onChange={(e) => updateSurface(s.id, { equation: e.target.value })}
                                                        placeholder="sin(x) * cos(y)"
                                                    />
                                                </div>

                                                {/* Color Picker Swatches */}
                                                <div>
                                                    <span className="text-xs font-medium text-neutral-700 mb-1.5 block">Surface Color</span>
                                                    <div className="flex flex-wrap gap-2">
                                                        {SURFACE_COLORS.map(c => (
                                                            <button
                                                                key={c.hex}
                                                                onClick={() => updateSurface(s.id, { color: c.hex })}
                                                                className={`w-6 h-6 rounded-md border border-black/10 transition-transform hover:scale-110 cursor-pointer ${
                                                                    s.color === c.hex ? 'ring-2 ring-neutral-900 ring-offset-2 scale-110' : ''
                                                                }`}
                                                                style={{ backgroundColor: c.hex }}
                                                                title={c.name}
                                                            />
                                                        ))}
                                                    </div>
                                                </div>

                                                {/* Shading / Wireframe Mode */}
                                                <div>
                                                    <span className="text-xs font-medium text-neutral-700 mb-1.5 block">Shading & Facets</span>
                                                    <div className="flex bg-neutral-100 p-0.5 rounded-lg border border-neutral-200">
                                                        <button
                                                            onClick={() => updateSurface(s.id, { wireframeMode: 'both' })}
                                                            className={`flex-1 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
                                                                s.wireframeMode === 'both' ? 'bg-white text-neutral-900 shadow-2xs' : 'text-neutral-600 hover:text-neutral-900'
                                                            }`}
                                                        >
                                                            Faceted
                                                        </button>
                                                        <button
                                                            onClick={() => updateSurface(s.id, { wireframeMode: 'solid' })}
                                                            className={`flex-1 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
                                                                s.wireframeMode === 'solid' ? 'bg-white text-neutral-900 shadow-2xs' : 'text-neutral-600 hover:text-neutral-900'
                                                            }`}
                                                        >
                                                            Solid
                                                        </button>
                                                        <button
                                                            onClick={() => updateSurface(s.id, { wireframeMode: 'wireframe' })}
                                                            className={`flex-1 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
                                                                s.wireframeMode === 'wireframe' ? 'bg-white text-neutral-900 shadow-2xs' : 'text-neutral-600 hover:text-neutral-900'
                                                            }`}
                                                        >
                                                            Wire
                                                        </button>
                                                    </div>
                                                </div>

                                                {/* Opacity Slider */}
                                                <div>
                                                    <div className="flex justify-between text-xs font-medium text-neutral-600 mb-1">
                                                        <span>Surface Opacity</span>
                                                        <span className="font-mono">{Math.round(s.opacity * 100)}%</span>
                                                    </div>
                                                    <input
                                                        type="range"
                                                        min="0.1"
                                                        max="1.0"
                                                        step="0.05"
                                                        value={s.opacity}
                                                        onChange={(e) => updateSurface(s.id, { opacity: parseFloat(e.target.value) })}
                                                        className="w-full accent-neutral-900 cursor-pointer"
                                                    />
                                                </div>

                                                {/* Delete Button */}
                                                {surfaces.length > 1 && (
                                                    <div className="pt-2 border-t border-neutral-200 flex justify-end">
                                                        <button
                                                            onClick={(e) => deleteSurface(s.id, e)}
                                                            className="rounded-md border border-red-200 text-red-600 hover:bg-red-50 px-2.5 py-1 text-xs font-medium flex items-center gap-1 transition-colors cursor-pointer"
                                                        >
                                                            <Trash2 size={12} /> Remove Surface
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>

                            {/* Preset Quick Loader */}
                            <div className="pt-2 border-t border-neutral-200">
                                <div className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                                    Add From Presets
                                </div>
                                <div className="grid grid-cols-2 gap-2">
                                    {PRESETS.map((p) => (
                                        <button
                                            key={p.name}
                                            onClick={() => addSurface(p.expr)}
                                            className="p-2 border border-neutral-200 rounded-lg bg-white text-left hover:border-neutral-400 hover:bg-neutral-50/70 transition-all cursor-pointer shadow-2xs"
                                        >
                                            <div className="font-semibold text-xs text-neutral-900">{p.name}</div>
                                            <div className="text-[10px] font-mono text-neutral-500 truncate mt-0.5">{p.expr}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </>
                    ) : (
                        /* Global 3D Scene Settings */
                        <div className="space-y-5">
                            <div className="space-y-4 p-3.5 border border-neutral-200 rounded-xl bg-neutral-50/50">
                                <div>
                                    <div className="flex justify-between text-xs font-semibold text-neutral-700 mb-1">
                                        <span>Domain Bounds [±X, ±Y]</span>
                                        <span className="font-mono text-neutral-900">±{range}</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="2"
                                        max="15"
                                        step="1"
                                        value={range}
                                        onChange={(e) => setRange(parseFloat(e.target.value))}
                                        className="w-full accent-neutral-900 cursor-pointer"
                                    />
                                </div>

                                <div>
                                    <div className="flex justify-between text-xs font-semibold text-neutral-700 mb-1">
                                        <span>Mesh Grid Density</span>
                                        <span className="font-mono text-neutral-900">{gridSegments} x {gridSegments}</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="20"
                                        max="100"
                                        step="10"
                                        value={gridSegments}
                                        onChange={(e) => setGridSegments(parseInt(e.target.value))}
                                        className="w-full accent-neutral-900 cursor-pointer"
                                    />
                                </div>

                                <div>
                                    <div className="flex justify-between text-xs font-semibold text-neutral-700 mb-1">
                                        <span>Height (Z) Scale</span>
                                        <span className="font-mono text-neutral-900">{heightScale.toFixed(1)}x</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="0.2"
                                        max="3"
                                        step="0.2"
                                        value={heightScale}
                                        onChange={(e) => setHeightScale(parseFloat(e.target.value))}
                                        className="w-full accent-neutral-900 cursor-pointer"
                                    />
                                </div>
                            </div>

                            <div className="space-y-3 pt-2 border-t border-neutral-200">
                                <label className="flex items-center justify-between cursor-pointer text-xs font-semibold text-neutral-800">
                                    <span>Show Cartesian Axes (X, Y, Z)</span>
                                    <input
                                        type="checkbox"
                                        checked={showAxes}
                                        onChange={(e) => setShowAxes(e.target.checked)}
                                        className="rounded border border-neutral-300 text-neutral-900 cursor-pointer"
                                    />
                                </label>
                                <label className="flex items-center justify-between cursor-pointer text-xs font-semibold text-neutral-800">
                                    <span>Show Ground Reference Grid</span>
                                    <input
                                        type="checkbox"
                                        checked={showGrid}
                                        onChange={(e) => setShowGrid(e.target.checked)}
                                        className="rounded border border-neutral-300 text-neutral-900 cursor-pointer"
                                    />
                                </label>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
