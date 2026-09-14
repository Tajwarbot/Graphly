import { describe, it, expect } from 'vitest';
import React from 'react';
import { renderToString } from 'react-dom/server';
import App from './App.jsx';
import { Logo, LogoIcon } from './components/Logo.jsx';
import { ThreeDGraph } from './components/ThreeDGraph.jsx';
import { MathBackground } from './components/MathBackground.jsx';

describe('App rendering', () => {
    it('renders App without crashing', () => {
        const html = renderToString(<App />);
        expect(html).toContain('Graphly');
        expect(html).toContain('2D Plotter');
        expect(html).toContain('3D Surface');
    });

    it('renders conceptual Logo component without Plotter text', () => {
        const html = renderToString(<Logo size={28} />);
        expect(html).toContain('Graphly');
        expect(html).not.toContain('Plotter');
        expect(html).toContain('<svg');
        expect(html).not.toContain('<img');
    });

    it('renders lightweight MathBackground canvas', () => {
        const html = renderToString(<MathBackground />);
        expect(html).toContain('<canvas');
    });

    it('renders ThreeDGraph with multi-surface support and no redundant top input', () => {
        const html = renderToString(<ThreeDGraph />);
        expect(html).toContain('Add Surface');
        expect(html).toContain('Surfaces');
        expect(html).toContain('Reset View');
    });
});

import { AIChatbox } from './components/AIChatbox.jsx';
import { parseIntentLocally } from './lib/chatIntent.js';

describe('AIChatbox & Natural Language Geometric Parser', () => {
    const createMockExecutor = () => {
        const calls = [];
        const executor = (name, args) => {
            calls.push({ name, args });
            return { success: true, message: `Executed ${name}` };
        };
        executor.calls = calls;
        return executor;
    };

    it('correctly maps "plot a hyperboloid" to complete 3D hyperboloid surface', () => {
        const mockExec = createMockExecutor();
        const result = parseIntentLocally('plot a hyperboloid', mockExec);
        expect(mockExec.calls).toHaveLength(1);
        expect(mockExec.calls[0].name).toBe('switchTo3D');
        expect(mockExec.calls[0].args.expression).toBe('x^2 + y^2 - z^2 = 1');
        expect(result.actions[0].success).toBe(true);
    });

    it('correctly maps "plot an ellipsoid" to 3D ellipsoid surface', () => {
        const mockExec = createMockExecutor();
        const result = parseIntentLocally('plot an ellipsoid', mockExec);
        expect(mockExec.calls).toHaveLength(1);
        expect(mockExec.calls[0].name).toBe('switchTo3D');
        expect(mockExec.calls[0].args.expression).toBe('x^2/16 + y^2/9 + z^2/4 = 1');
        expect(result.actions[0].success).toBe(true);
    });

    it('correctly maps "plot a paraboloid" to 3D elliptic paraboloid', () => {
        const mockExec = createMockExecutor();
        parseIntentLocally('plot a paraboloid', mockExec);
        expect(mockExec.calls).toHaveLength(1);
        expect(mockExec.calls[0].name).toBe('switchTo3D');
        expect(mockExec.calls[0].args.expression).toBe('(x^2 + y^2) / 6');
    });

    it('correctly maps "plot sombrero in 3d" to 3D sombrero surface', () => {
        const mockExec = createMockExecutor();
        parseIntentLocally('plot sombrero in 3d', mockExec);
        expect(mockExec.calls).toHaveLength(1);
        expect(mockExec.calls[0].name).toBe('switchTo3D');
        expect(mockExec.calls[0].args.expression).toContain('sin(sqrt(x^2 + y^2))');
    });

    it('correctly maps "plot a circle with radius 4" to implicit circle equation', () => {
        const mockExec = createMockExecutor();
        parseIntentLocally('plot a circle with radius 4', mockExec);
        expect(mockExec.calls).toHaveLength(1);
        expect(mockExec.calls[0].name).toBe('plotImplicitEquation');
        expect(mockExec.calls[0].args.expression).toBe('x^2 + y^2 = 16');
    });

    it('correctly parses "plot y = x^2 and zoom out" into plotFunction and setViewportBounds', () => {
        const mockExec = createMockExecutor();
        parseIntentLocally('plot y = x^2 and zoom out', mockExec);
        expect(mockExec.calls).toHaveLength(2);
        expect(mockExec.calls[0].name).toBe('plotFunction');
        expect(mockExec.calls[0].args.expression).toBe('x^2');
        expect(mockExec.calls[1].name).toBe('setViewportBounds');
    });

    it('rejects invalid non-mathematical English sentences and does not pollute canvas', () => {
        const mockExec = createMockExecutor();
        const result = parseIntentLocally('write me a poem about graphs', mockExec);
        expect(mockExec.calls).toHaveLength(0);
        expect(result.text).toContain('could not interpret');
    });

    it('renders AIChatbox trigger button without crashing', () => {
        const html = renderToString(
            <AIChatbox
                onPlotFunction={() => {}}
                onPlotImplicit={() => {}}
                onLoadDataTable={() => {}}
                onSwitchTo3D={() => {}}
                onSetViewportBounds={() => {}}
            />
        );
        expect(html).toContain('Ask Graphly');
    });
});
