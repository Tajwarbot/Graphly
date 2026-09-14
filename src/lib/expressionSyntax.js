import { adaptiveCurve } from './adaptiveCurve.js';
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
    if (expression.includes('{')) return compilePiecewise(expression, variables);
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
function splitTop(text, delimiter) {
    let depth = 0, start = 0;
    const parts = [];
    for (let i = 0; i < text.length; i++) {
        if ('({['.includes(text[i])) depth++;
        if (')}]'.includes(text[i])) depth--;
        if (text[i] === delimiter && depth === 0) { parts.push(text.slice(start, i).trim()); start = i + 1; }
    }
    parts.push(text.slice(start).trim()); return parts;
}
function condition(source, variables) {
    const parts = normalizeMath(source).split(/(<=|>=|!=|==|<|>|=)/).map(s => s.trim());
    if (parts.length < 3 || parts.length % 2 === 0) throw new Error('Piecewise conditions need comparisons.');
    const terms = parts.filter((_, i) => i % 2 === 0).map(term => scalar(term, variables));
    return scope => {
        const values = terms.map(fn => fn(scope));
        return values.every(Number.isFinite) && parts.filter((_, i) => i % 2 === 1).every((op, i) => {
            const a = values[i], b = values[i + 1];
            return op === '<' ? a < b : op === '>' ? a > b : op === '<=' ? a <= b : op === '>=' ? a >= b : op === '!=' ? a !== b : a === b;
        });
    };
}
function braceGroups(raw) {
    let depth = 0, start = -1; const groups = [];
    for (let i = 0; i < raw.length; i++) {
        if (raw[i] === '{') { if (depth === 0) start = i; depth++; }
        if (raw[i] === '}') { if (--depth < 0) throw new Error('Unexpected closing brace.'); if (depth === 0) groups.push({ start, end: i + 1, text: raw.slice(start + 1, i) }); }
    }
    if (depth) throw new Error('Close every restriction or piecewise brace.');
    return groups;
}
function compilePiecewise(expression, variables) {
    let rewritten = '', cursor = 0;
    const pieces = braceGroups(expression).map((group, i) => {
        const branches = splitTop(group.text, ',').map((branch, j, all) => {
            const parts = splitTop(branch, ':');
            if (parts.length === 1 && j === all.length - 1) return { value: scalar(parts[0], variables), matches: () => true };
            if (parts.length !== 2) throw new Error('Use {x < 0: -x, x >= 0: x}.');
            return { value: scalar(parts[1], variables), matches: condition(parts[0], variables) };
        });
        const symbol = `piecewiseInternal${i}`;
        if (expression.includes(symbol) || variables.includes(symbol)) throw new Error('Reserved piecewise symbol.');
        rewritten += expression.slice(cursor, group.start) + '(' + symbol + ')'; cursor = group.end;
        return { symbol, branches };
    });
    rewritten += expression.slice(cursor);
    const evaluate = scalar(rewritten, [...variables, ...pieces.map(p => p.symbol)]);
    return scope => {
        const expanded = { ...scope };
        for (const piece of pieces) { const branch = piece.branches.find(b => b.matches(scope)); expanded[piece.symbol] = branch ? branch.value(scope) : NaN; }
        return evaluate(expanded);
    };
}
export function splitRestrictions(input) {
    const raw = normalizeMath(input), groups = braceGroups(raw);
    let end = raw.length; const restrictions = [];
    for (let i = groups.length - 1; i >= 0; i--) {
        const group = groups[i];
        if (raw.slice(group.end, end).trim() || group.text.includes(':') || !/[<>]/.test(group.text)) break;
        restrictions.unshift(group.text); end = group.start;
    }
    return { base: raw.slice(0, end).trim(), restrictions };
}
export function compileRestrictions(restrictions, variables = ['x', 'y', 'z']) {
    const fields = [];
    for (const restriction of restrictions)
        for (const condition of splitConditions(restriction)) {
            const parts = splitComparisons(condition);
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
        if ('({'.includes(text[i])) depth++;
        if (')}'.includes(text[i])) depth--;
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
        for (const condition of splitConditions(restriction)) {
            const parts = splitComparisons(condition);
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
    const [min,max]=fn.ranges.t;
    const evaluate=t=>{
        const p=fn.point({t});
        return p.every(Number.isFinite)&&allowedBy(fn.fields,{t,x:p[0],y:p[1],z:p[2]??0})?p:null;
    };
    if(fn.kind==='point')return [evaluate(0)];
    return adaptiveCurve(evaluate,min,max,Math.min(2000,Math.max(20,resolution)));

}

/** Find only the outer graph relation; branch predicates are scalar syntax. */
export function splitRelation(input) {
    let depth = 0;
    for (let i = 0; i < input.length; i++) {
        if ('({['.includes(input[i])) depth++;
        if (')}]'.includes(input[i])) depth--;
        if (depth === 0 && /[<>=]/.test(input[i])) {
            const operator = input[i] + (input[i + 1] === '=' ? '=' : '');
            return { left: input.slice(0, i).trim(), operator, right: input.slice(i + operator.length).trim() };
        }
    }
    return null;
}
export function splitEquation(input) {
    return splitTop(input, '=');
}

export function splitComparisons(input) {
    const parts = []; let depth=0, start=0;
    for(let i=0;i<input.length;i++) {
        if ('({['.includes(input[i])) depth++;
        if (')}]'.includes(input[i])) depth--;
        if(depth===0 && /[<>]/.test(input[i])) {
            parts.push(input.slice(start,i).trim());
            const op=input[i]+(input[i+1]==='='?'=':'');
            parts.push(op); i+=op.length-1; start=i+1;
        }
    }
    parts.push(input.slice(start).trim()); return parts;
}
export const splitConditions = input => splitTop(input, ',');
