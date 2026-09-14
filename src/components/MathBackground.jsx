import React, { useEffect, useRef } from 'react';

export function MathBackground() {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animId;
        let t = 0;
        let width = 0;
        let height = 0;

        const handleResize = () => {
            const parent = canvas.parentElement;
            if (!parent) return;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = parent.clientWidth;
            height = parent.clientHeight;
            canvas.width = Math.floor(width * dpr);
            canvas.height = Math.floor(height * dpr);
            ctx.resetTransform();
            ctx.scale(dpr, dpr);
        };

        handleResize();
        window.addEventListener('resize', handleResize);

        const render = () => {
            t += 0.01;
            ctx.clearRect(0, 0, width, height);

            // Subtle Cartesian coordinate background grid
            const gridSize = 44;
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.035)';

            for (let x = 0; x <= width; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            for (let y = 0; y <= height; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }

            const centerY = height * 0.52;

            // Wave 1: Harmonic Blue Sine Wave
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = 'rgba(37, 99, 235, 0.22)';
            ctx.beginPath();
            for (let x = 0; x <= width; x += 6) {
                const y = centerY + Math.sin(x * 0.007 + t) * 65 + Math.cos(x * 0.014 - t * 0.6) * 35;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Wave 2: Violet Harmonic Wave
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = 'rgba(124, 58, 237, 0.18)';
            ctx.beginPath();
            for (let x = 0; x <= width; x += 6) {
                const y = centerY + Math.cos(x * 0.005 - t * 0.7) * 75 + Math.sin(x * 0.011 + t * 0.5) * 45;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Wave 3: Modulated Cyan Wave
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = 'rgba(8, 145, 178, 0.16)';
            ctx.beginPath();
            for (let x = 0; x <= width; x += 8) {
                const envelope = Math.sin(x * 0.003 + t * 0.3);
                const y = centerY + envelope * Math.sin(x * 0.018 + t * 1.1) * 60;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Moving Coordinate Sample Nodes
            for (let i = 0; i < 6; i++) {
                const sampleX = ((t * 36 + i * (width / 5)) % (width + 80)) - 40;
                const sampleY = centerY + Math.sin(sampleX * 0.007 + t) * 65 + Math.cos(sampleX * 0.014 - t * 0.6) * 35;

                // Outer ring
                ctx.fillStyle = '#FFFFFF';
                ctx.strokeStyle = '#2563EB';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(sampleX, sampleY, 3.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // Drop line to axis
                ctx.strokeStyle = 'rgba(37, 99, 235, 0.12)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(sampleX, sampleY);
                ctx.lineTo(sampleX, centerY);
                ctx.stroke();
            }

            animId = requestAnimationFrame(render);
        };

        render();

        return () => {
            cancelAnimationFrame(animId);
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <div className="absolute inset-0 pointer-events-none overflow-hidden select-none z-0">
            <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
    );
}
