import { parse } from 'mathjs';
export const normalizeMath = (input) =>
    String(input || '')
        .trim()
        .replace(/²/g, '^2')
        .replace(/³/g, '^3')
        .replace(/−/g, '-')
        .replace(/π/g, 'pi')
        .replace(/≤/g, '<=')
        .replace(/≥/g, '>=')
        .replace(/\bln\s*\(/g, 'log(');
const functions = [
    'sin',
    'cos',
    'tan',
    'sec',
    'csc',
    'cot',
    'asin',
    'acos',
    'atan',
    'atan2',
    'sinh',
    'cosh',
    'tanh',
    'sqrt',
    'cbrt',
    'abs',
    'exp',
    'log',
    'log10',
    'log2',
    'min',
    'max',
    'floor',
    'ceil',
    'round',
    'sign'
];
export function scalar(expression, variables = ['x', 'y', 'z']) {
    if (!expression || expression.length > 4000)
        throw new Error('Use an expression of 1–4,000 characters.');
    const ast = parse(expression);
    ast.traverse((node) => {
        if (
            node.isAssignmentNode ||
            node.isFunctionAssignmentNode ||
            node.isBlockNode ||
            node.isAccessorNode ||
            node.isArrayNode ||
            node.isObjectNode ||
            node.isConditionalNode ||
            node.isRangeNode ||
            (node.isOperatorNode &&
                !['+', '-', '*', '/', '^'].includes(node.op))
        )
            throw new Error('Only mathematical expressions are supported.');
        if (
            node.isSymbolNode &&
            ![...variables, 'pi', 'e', 'tau', ...functions].includes(node.name)
        )
            throw new Error(`Unknown symbol: ${node.name}`);
        if (node.isFunctionNode && !functions.includes(node.fn.name))
            throw new Error('Unknown mathematical function.');
    });
    const compiled = ast.compile();
    return (scope) => {
        try {
            const value = compiled.evaluate(scope);
            return typeof value === 'number' && Number.isFinite(value)
                ? value
                : NaN;
        } catch {
            return NaN;
        }
    };
}
export function splitRestrictions(input) {
    const raw = normalizeMath(input),
        index = raw.indexOf('{');
    if (index < 0) return { base: raw, restrictions: [] };
    const suffix = raw.slice(index),
        restrictions = [...suffix.matchAll(/\{([^{}]+)\}/g)].map((m) => m[1]);
    if (suffix.replace(/\{[^{}]+\}/g, '').trim() || !restrictions.length)
        throw new Error('Use trailing restrictions such as {x > 0}{y < 2}.');
    return { base: raw.slice(0, index).trim(), restrictions };
}
export function compileRestrictions(restrictions, variables = ['x', 'y', 'z']) {
    const fields = [];
    for (const restriction of restrictions)
        for (const condition of restriction.split(',')) {
            const parts = condition.split(/(<=|>=|<|>)/).map((s) => s.trim());
            if (parts.length < 3 || parts.length % 2 === 0)
                throw new Error(
                    'Restrictions need comparisons, for example -2 < x < 2.'
                );
            for (let i = 0; i < parts.length - 2; i += 2) {
                const a = scalar(parts[i], variables),
                    b = scalar(parts[i + 2], variables),
                    sign = parts[i + 1].startsWith('<') ? 1 : -1;
                fields.push((scope) => sign * (a(scope) - b(scope)));
            }
        }
    return fields;
}
export const allowedBy = (fields, scope) =>
    fields.every((f) => Number.isFinite(f(scope)) && f(scope) <= 1e-10);
export function tupleComponents(base) {
    if (!base.startsWith('(') || !base.endsWith(')')) return null;
    const text = base.slice(1, -1),
        parts = [];
    let depth = 0,
        start = 0;
    for (let i = 0; i < text.length; i++) {
        if (text[i] === '(') depth++;
        if (text[i] === ')') depth--;
        if (text[i] === ',' && depth === 0) {
            parts.push(text.slice(start, i));
            start = i + 1;
        }
    }
    if (!parts.length) return null;
    parts.push(text.slice(start));
    return parts;
}
export function compileParametric(input, dimension = 3) {
    const { base, restrictions } = splitRestrictions(input),
        tuple = tupleComponents(base);
    if (!tuple) return null;
    if (tuple.length !== dimension)
        throw new Error(`Use ${dimension} coordinates in this graph.`);
    const usesUV = /\b[uv]\b/.test(base),
        usesT = /\bt\b/.test(base);
    if (usesUV && usesT)
        throw new Error('Use t for curves, or u and v for surfaces.');
    if (usesUV && dimension === 2)
        throw new Error('Use t for a 2D parametric curve.');
    const parameters = usesUV ? ['u', 'v'] : ['t'];
    const coordinates = tuple.map((expr) => scalar(expr, parameters));
    const fields = compileRestrictions(restrictions, [
        ...parameters,
        'x',
        'y',
        'z'
    ]);
    const ranges = Object.fromEntries(parameters.map((p) => [p, [0, 1]]));
    // Tighten parameter intervals to their explicit numeric bounds. Other
    // restrictions remain predicates and are clipped during sampling.
    for (const restriction of restrictions)
        for (const condition of restriction.split(',')) {
            const parts = condition.split(/(<=|>=|<|>)/).map((s) => s.trim());
            for (let i = 0; i < parts.length - 2; i += 2)
                for (const p of parameters) {
                    const left = parts[i],
                        right = parts[i + 2],
                        less = parts[i + 1].startsWith('<');
                    if (
                        left === p &&
                        !/[a-z]/i.test(right.replace(/pi/g, ''))
                    ) {
                        const value = scalar(right, [])({});
                        if (Number.isFinite(value))
                            ranges[p][less ? 1 : 0] = value;
                    }
                    if (
                        right === p &&
                        !/[a-z]/i.test(left.replace(/pi/g, ''))
                    ) {
                        const value = scalar(left, [])({});
                        if (Number.isFinite(value))
                            ranges[p][less ? 0 : 1] = value;
                    }
                }
        }
    for (const [a, b] of Object.values(ranges))
        if (!Number.isFinite(a) || !Number.isFinite(b) || a >= b)
            throw new Error('Parameter bounds must be finite and increasing.');
    return {
        kind: usesUV ? 'surface' : usesT ? 'curve' : 'point',
        ranges,
        fields,
        point(scope) {
            return coordinates.map((fn) => fn(scope));
        }
    };
}
export function parametricCurve(input, dimension = 2, resolution = 600) {
    const fn = compileParametric(input, dimension);
    if (!fn || fn.kind === 'surface')
        throw new Error('Enter a parametric curve using t.');
    const [min, max] = fn.ranges.t,
        points = [];
    const n =
        fn.kind === 'point' ? 0 : Math.min(2000, Math.max(20, resolution));
    for (let i = 0; i <= n; i++) {
        const t = n ? min + ((max - min) * i) / n : 0,
            p = fn.point({ t });
        const scope = { t, x: p[0], y: p[1], z: p[2] ?? 0 };
        points.push(
            p.every(Number.isFinite) && allowedBy(fn.fields, scope) ? p : null
        );
    }
    return points;
}
