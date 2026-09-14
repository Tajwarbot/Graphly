import { artExampleRequest } from './artExamples.js';
import { compileParametric, splitRestrictions } from './expressionSyntax.js';
import { compileSurface } from './surfaceEngine.js';

export const GEOMETRIC_3D_SURFACES = [
    { names: ['monkey saddle'], equation: 'x^3 - 3*x*y^2', label: 'Monkey Saddle' },
    { names: ['hyperboloid'], equation: 'x^2 + y^2 - z^2 = 1', label: 'One-Sheeted Hyperboloid' },
    { names: ['sphere'], equation: 'x^2 + y^2 + z^2 = 25', label: 'Sphere' },
    {
        names: ['hyperbolic paraboloid', 'saddle', 'pringle'],
        equation: '(x^2 - y^2) / 4',
        label: 'Hyperbolic Paraboloid (Saddle)'
    },
    {
        names: ['two sheeted hyperboloid', 'hyperboloid of two sheets', 'hyperboloid 2 sheets'],
        equation: 'z^2 - x^2 - y^2 = 1',
        label: 'Two-Sheeted Hyperboloid'
    },
    {
        names: ['paraboloid', 'elliptic paraboloid', 'bowl', 'cup'],
        equation: '(x^2 + y^2) / 6',
        label: 'Elliptic Paraboloid'
    },
    {
        names: ['ellipsoid', '3d ellipse', 'egg surface', 'oval 3d', 'football'],
        equation: 'x^2/16 + y^2/9 + z^2/4 = 1',
        label: 'Ellipsoid Surface'
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
        names: ['hemisphere', 'dome'],
        equation: 'sqrt(25 - x^2 - y^2)',
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

// Compile for syntax and symbols; sampling a few points incorrectly rejects valid domains.
export function isMathExpression(expr) {
    try { compileSurface(expr); return true; } catch { try { return !!compileParametric(expr,2); } catch { return false; } }
}

export function parseIntentLocally(prompt, executeTool, context = {}) {
    const symbolic = prompt.trim().match(/^(simplify|differentiate|integrate|solve)\s+(.+?)(?:\s+(?:for|with respect to)\s+([a-z]))?$/i);
    if (symbolic) {
        const result = executeTool('symbolicMath', {operation:symbolic[1].toLowerCase(), expression:symbolic[2], variable:symbolic[3] || 'x'});
        return {text:result.success ? [result.expression || result.solutionSet || `${result.variable} = ${result.solutions.join(', ')}`, result.note].filter(Boolean).join('\n') : result.message, actions:[]};
    }

    if (typeof prompt !== 'string' || !prompt.trim()) return { text: 'Enter an equation or describe a graph.', actions: [] };
    const lower = prompt.toLowerCase();
    const run = (name, args) => {
        const result = executeTool(name, args);
        return { text: result.message, actions: [{ name, args, result: result.message, success: result.success }] };
    };
    if (/^(?:what|why|how|explain|describe|tell me)\b/i.test(prompt.trim())) return {text:'Connect AI for mathematical explanations and custom designs, or enter an equation to plot locally.',actions:[]};
    if (/\b(planet|saturn)\b/.test(lower) && /\b(plot|build|make|create|draw|graph)\b/.test(lower)) return run('addScene', {surfaces:[{equation:'x^2+y^2+z^2=4',name:'Planet',color:'#b88759'},{equation:'(sqrt(x^2+y^2)-3.3)^2+z^2/0.04=0.36',name:'Ring',color:'#8d9dae'}],bounds:{xMin:-5,xMax:5,yMin:-5,yMax:5,zMin:-5,zMax:5},camera:{position:[9,-11,7],target:[0,0,0]}});
    if (/\btorus\b/.test(lower) && /\b(plot|build|make|create|draw|graph)\b/.test(lower)) return run('switchTo3D',{expression:'(sqrt(x^2+y^2)-3)^2+z^2=1'});
    if (/\b(?:two|2)\s+(?:intersecting\s+)?planes\b/.test(lower) || /\bplanes\s+(?:that\s+)?intersect\b/.test(lower)) {
        const results = ['z = x', 'z = -x'].map(expression => run('switchTo3D', { expression }));
        return { text: 'Two planes intersect along the y-axis. ' + results.map(r => r.text).join('. '), actions: results.flatMap(r => r.actions) };
    }
    // A semicolon/newline separates equations; validate the whole request before adding any.
    const scene = prompt.replace(/^(?:please\s+)?(?:plot|graph|add|draw)\s+/i, '').split(/[;\n]+/).map(s => s.trim()).filter(Boolean);
    if (scene.length > 1) {
        if (scene.length > 12 || !scene.every(isMathExpression)) return { text: 'Provide up to 12 valid equations separated by semicolons. No changes were made.', actions: [] };
        const in3D = scene.some(eq => /\bz\b/.test(eq));
        const results = scene.map(expression => run(in3D ? 'switchTo3D' : expression.includes('=') ? 'plotImplicitEquation' : 'plotFunction', { expression }));
        return { text: results.map(r => r.text).join('\n'), actions: results.flatMap(r => r.actions) };
    }
    const art = artExampleRequest(prompt);
    if (art) return run(art.dimension === '2d' ? 'add2DScene' : 'addScene', art.dimension === '2d'
        ? { layers: art.layers, bounds: art.bounds }
        : { surfaces: art.layers, bounds: art.bounds, camera: art.camera });
    // Offline mode must not invent data or replace a graph for an edit request.
    if (/\b(replace|change|remove|delete|edit|make it)\b/.test(lower)) {
        return { text: 'Connect AI to edit existing items by name. Use the expression list to edit an item, or ask me to add a new equation.', actions: [] };
    }
    if (/\b(table|dataset|scatter|data points)\b/.test(lower)) {
        const rows = [...prompt.matchAll(/\(\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*\)/gi)].map(m => ({ x: Number(m[1]), y: Number(m[2]) }));
        if (!rows.length) return { text: 'Provide the points to add, for example: table (1, 2), (2, 5), (3, 9).', actions: [] };
        return run('loadDataTable', { name: 'Data points', rows });
    }
    let candidate = prompt.trim().replace(/^(?:can you please |could you please |please |can you |could you )/i, '')
        .replace(/^(?:plot|graph|draw|show me|visualize|add)\s+/i, '')
        .replace(/^(?:a|an|the)\s+/i, '')
        .replace(/^(?:function|curve|equation|surface)\s+/i, '')
        .replace(/\s+(?:in\s+)?3d(?:\s+mode)?$/i, '')
        .replace(/\s+(?:and\s+)?zoom (?:out|in)$/i, '').trim();
    candidate = candidate.replace(/^(?:circle|ellipse|sphere|ellipsoid)\s+(?=[xyz])/i, '');
    const is3D = /\b3d\b|\bz\b/i.test(prompt);
    const explicitY = /^(?:y|f\(x\))\s*=/i.test(candidate);
    if (explicitY && !/\by\b/.test(candidate.split('=').slice(1).join('='))) candidate = candidate.replace(/^(?:y|f\(x\))\s*=\s*/i, '');
    try {
        const tuple2D=compileParametric(candidate,2);
        if(tuple2D)return run('plotImplicitEquation',{expression:candidate});
    } catch { /* May be a 3D tuple. */ }
    try { if(compileParametric(candidate,3))return run('switchTo3D',{expression:candidate}); } catch { /* Nonparametric equation. */ }
    if (isMathExpression(candidate) && /[<>≤≥]/.test(splitRestrictions(candidate).base)) return run(is3D || context.dimension==='3d' ? 'switchTo3D' : 'plotImplicitEquation',{expression:candidate});
    if (isMathExpression(candidate)) {
        const tool = is3D || (!explicitY && !candidate.includes('=') && /\by\b/.test(candidate)) ? 'switchTo3D'
            : candidate.includes('=') ? 'plotImplicitEquation' : 'plotFunction';
        const result = run(tool, { expression: candidate });
        if (/zoom (out|in)/i.test(prompt) && result.actions[0].success !== false && tool !== 'switchTo3D') {
            const r = /zoom out/i.test(prompt) ? 20 : 2;
            const zoom = run('setViewportBounds', { xMin: -r, xMax: r, yMin: -r, yMax: r });
            result.actions.push(...zoom.actions);
            result.text += '. ' + zoom.text;
        }
        return result;
    }
    // Match longer names first so a two-sheeted hyperboloid is not a generic hyperboloid.
    const named = [...GEOMETRIC_3D_SURFACES.map(item => ({ ...item, tool: 'switchTo3D' })),
        ...GEOMETRIC_2D_CURVES.map(item => ({ ...item, tool: item.implicit ? 'plotImplicitEquation' : 'plotFunction' }))]
        .flatMap(item => item.names.map(name => ({ ...item, name })))
        .sort((a, b) => b.name.length - a.name.length)
        .find(item => new RegExp('\\b' + item.name + '\\b', 'i').test(lower));
    if (named && !/[=]/.test(candidate) && !/^(?:what|why|how|explain|describe)\b/i.test(prompt)) {
        let equation = named.equation;
        const radius = prompt.match(/(?:radius\s*(?:of|=)?|\br\s*=)\s*(\d+(?:\.\d+)?)/i);
        if (named.label === 'Circle') equation = `x^2 + y^2 = ${radius ? Number(radius[1]) ** 2 : lower.includes('unit circle') ? 1 : 25}`;
        if (named.label === 'Sphere' && radius) equation = `x^2 + y^2 + z^2 = ${Number(radius[1]) ** 2}`;
        return run(named.tool, { expression: equation });
    }
    if (/^zoom (in|out)$/i.test(prompt.trim())) {
        const r = /out/i.test(prompt) ? 20 : 2;
        return run('setViewportBounds', { xMin: -r, xMax: r, yMin: -r, yMax: r });
    }
    return { text: 'I could not interpret that request offline. Enter an equation such as sin(x) or x^2 + y^2 + z^2 = 25, or connect an API key for conversational assistance.', actions: [] };
}

