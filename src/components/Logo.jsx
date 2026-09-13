import React from 'react';

export function LogoIcon({ size = 26, className = "" }) {
    return (
        <svg 
            width={size} 
            height={size} 
            viewBox="0 0 64 64" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg"
            className={`shrink-0 ${className}`}
        >
            <defs>
                <linearGradient id="logoCurveGrad" x1="12" y1="52" x2="52" y2="12" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#2563EB" />
                    <stop offset="50%" stopColor="#7C3AED" />
                    <stop offset="100%" stopColor="#0891B2" />
                </linearGradient>
                <linearGradient id="logoPlaneGrad" x1="14" y1="14" x2="50" y2="50" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#2563EB" stopOpacity="0.12" />
                    <stop offset="100%" stopColor="#7C3AED" stopOpacity="0.04" />
                </linearGradient>
            </defs>

            {/* Outer Precision Boundary (Isometric Cube / Hexagon outline) */}
            <polygon 
                points="32,6 56,20 56,46 32,58 8,46 8,20" 
                fill="#FFFFFF" 
                stroke="#18181B" 
                strokeWidth="2.5" 
                strokeLinejoin="round" 
            />
            
            {/* Coordinate plane facet fill */}
            <polygon points="32,32 56,20 56,46 32,58" fill="url(#logoPlaneGrad)" />
            
            {/* 3D Coordinate Axis Spokes from Center Origin */}
            <line x1="32" y1="32" x2="32" y2="6" stroke="#18181B" strokeWidth="2" strokeLinecap="round" />
            <line x1="32" y1="32" x2="56" y2="46" stroke="#18181B" strokeWidth="2" strokeLinecap="round" />
            <line x1="32" y1="32" x2="8" y2="46" stroke="#18181B" strokeWidth="2" strokeLinecap="round" />
            
            {/* Grid tick marks on axis */}
            <line x1="30" y1="19" x2="34" y2="19" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="19" y1="39" x2="21" y2="43" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" />
            <line x1="43" y1="43" x2="45" y2="39" stroke="#71717A" strokeWidth="1.5" strokeLinecap="round" />

            {/* Parametric Wave Curve Slicing Space */}
            <path 
                d="M 12,42 C 20,24 24,40 32,32 C 40,24 44,40 52,22" 
                stroke="url(#logoCurveGrad)" 
                strokeWidth="3.5" 
                strokeLinecap="round" 
                fill="none" 
            />

            {/* Precision Origin and Sample Points */}
            <circle cx="32" cy="32" r="3" fill="#FFFFFF" stroke="#18181B" strokeWidth="2" />
            <circle cx="21" cy="30" r="2.5" fill="#FFFFFF" stroke="#2563EB" strokeWidth="2" />
            <circle cx="43" cy="34" r="2.5" fill="#FFFFFF" stroke="#7C3AED" strokeWidth="2" />
            <circle cx="52" cy="22" r="2.5" fill="#2563EB" stroke="#FFFFFF" strokeWidth="1.5" />
        </svg>
    );
}

export function Logo({ size = 26, showText = true, className = "" }) {
    return (
        <div className={`flex items-center gap-2.5 select-none ${className}`}>
            <LogoIcon size={size} />
            {showText && (
                <span className="font-bold text-base text-neutral-900 tracking-tight leading-none">
                    Graphly
                </span>
            )}
        </div>
    );
}
