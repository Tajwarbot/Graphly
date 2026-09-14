import { parse } from 'mathjs';
import { splitRestrictions, splitRelation, normalizeMath, splitComparisons, splitConditions, scalar } from './expressionSyntax.js';

const functions = new Set(['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'sqrt', 'cbrt', 'abs', 'exp', 'log', 'log10', 'log2', 'min', 'max', 'floor', 'ceil', 'round', 'sign']);
const symbols = new Set(['x', 'y', 'pi', 'e']);
const cache = new Map();

export function isInequality(expression) {
    return typeof expression === 'string' && /[<>]/.test(splitRelation(normalizeMath(expression))?.operator || '');
}

/** Each condition is a finite scalar residual <= 0; chains represent intersection. */
export function compileInequality(expression) {
    if (typeof expression !== 'string' || expression.length > 2000) throw new Error('Enter an inequality shorter than 2000 characters.');
    const raw = expression.trim().replace(/≤/g, '<=').replace(/≥/g, '>=').replace(/²/g, '^2').replace(/³/g, '^3').replace(/−/g, '-').replace(/\bln\s*\(/g, 'log(');
    if (cache.has(raw)) return cache.get(raw);
    const {base,restrictions}=splitRestrictions(raw);
    if(restrictions.length){
        const constraints=[base,...restrictions.flatMap(splitConditions)].flatMap(r=>compileInequality(r).constraints);
        return {constraints,contains(x,y){return constraints.every(c=>{const v=c.evaluate(x,y);return Number.isFinite(v)&&(c.strict?v<0:v<=0);});}};
    }
    const parts = splitComparisons(raw);
    if (parts.length < 3 || parts.length > 9 || parts.length % 2 !== 1) throw new Error('Use an inequality such as y > x or -1 < y < 1.');
    const terms = parts.filter((_, index) => index % 2 === 0).map(term => {
        if (!term.trim()) throw new Error('Both sides of an inequality need an expression.');
        if (term.includes('{')) { const evaluate=scalar(term,['x','y']); return (x,y)=>evaluate({x,y}); }
        const ast = parse(term);
        let count = 0;
        ast.traverse(node => {
            if (++count > 300) throw new Error('This expression is too complex.');
            if (node.isOperatorNode) {
                if (!['+', '-', '*', '/', '^'].includes(node.op)) throw new Error('Only arithmetic expressions are supported.');
            } else if (node.isFunctionNode) {
                if (!node.fn.isSymbolNode || !functions.has(node.fn.name)) throw new Error('Unsupported mathematical function.');
            } else if (node.isSymbolNode) {
                if (!symbols.has(node.name) && !functions.has(node.name)) throw new Error(`Unknown symbol: ${node.name}`);
            } else if (node.isConstantNode) {
                if (typeof node.value !== 'number' || !Number.isFinite(node.value)) throw new Error('Only finite numeric constants are supported.');
            } else if (!node.isParenthesisNode) throw new Error('Only mathematical expressions are supported.');
        });
        const compiled = ast.compile();
        return (x, y) => {
            try { const value = compiled.evaluate({ x, y }); return typeof value === 'number' && Number.isFinite(value) ? value : NaN; }
            catch { return NaN; }
        };
    });
    const constraints = terms.slice(1).map((right, index) => {
        const left = terms[index];
        const operator = parts[index * 2 + 1];
        const direction = operator.startsWith('<') ? 1 : -1;
        return { strict: operator.length === 1, boundary: `${parts[index * 2].trim()} = ${parts[index * 2 + 2].trim()}`, evaluate: (x, y) => direction * (left(x, y) - right(x, y)) };
    });
    const result = { constraints, contains(x, y) { return constraints.every(({ evaluate, strict }) => { const value = evaluate(x, y); return Number.isFinite(value) && (strict ? value < 0 : value <= 0); }); } };
    if (cache.size >= 100) cache.delete(cache.keys().next().value);
    cache.set(raw, result);
    return result;
}

function crossing(a, b, evaluate, av, bv) {
    if (av === 0) return a;
    if (bv === 0) return b;
    let low = 0, high = 1;
    for (let i = 0; i < 16; i++) {
        const t = (low + high) / 2;
        const value = evaluate(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
        if (!Number.isFinite(value)) return null;
        if ((value < 0) === (av < 0)) low = t; else high = t;
    }
    const t = (low + high) / 2;
    const point = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    // Do not fabricate an equality boundary where a residual changes sign across a pole.
    if (Math.abs(evaluate(...point)) > Math.max(1, Math.abs(av), Math.abs(bv)) * .01) return null;
    return point;
}

function clip(polygon, evaluate, strict) {
    const output = [];
    for (let i = 0; i < polygon.length; i++) {
        const a = polygon[i], b = polygon[(i + 1) % polygon.length];
        const av = evaluate(...a), bv = evaluate(...b);
        if (!Number.isFinite(av) || !Number.isFinite(bv)) return [];
        const insideA = strict ? av < 0 : av <= 0;
        const insideB = strict ? bv < 0 : bv <= 0;
        if (insideA) output.push(a);
        if (insideA !== insideB) {
            const point = crossing(a, b, evaluate, av, bv);
            if (!point) return [];
            output.push(point);
        }
    }
    return output;
}

/** Sample finite cells, then clip triangles to every inequality. Coordinates remain mathematical. */
export function buildInequalityRegion(expression, bounds, resolution = 64) {
    const { xMin, xMax, yMin, yMax } = bounds || {};
    if (![xMin, xMax, yMin, yMax].every(Number.isFinite) || xMin >= xMax || yMin >= yMax || !Number.isFinite(xMax - xMin) || !Number.isFinite(yMax - yMin)) throw new Error('Region bounds must be finite and increasing.');
    if (!Number.isFinite(resolution)) throw new Error('Resolution must be finite.');
    const n = Math.max(8, Math.min(128, Math.round(resolution)));
    const compiled = compileInequality(expression);
    const polygons = [];
    const dx = (xMax - xMin) / n, dy = (yMax - yMin) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const a = [xMin + i * dx, yMin + j * dy];
        const b = [xMin + (i + 1) * dx, a[1]];
        const c = [b[0], yMin + (j + 1) * dy];
        const d = [a[0], c[1]];
        for (let polygon of [[a, b, c], [a, c, d]]) {
            for (const condition of compiled.constraints) {
                polygon = clip(polygon, condition.evaluate, condition.strict);
                if (polygon.length < 3) break;
            }
            if (polygon.length >= 3) polygons.push(polygon);
        }
    }
    return { polygons, boundaries: compiled.constraints.map(({ boundary, strict }) => ({ expression: boundary, strict })) };
}

/** Render all cells in ONE filled path so adjacent cells do not accumulate opacity. */
export function regionToSvgPath(region, project = point => point) {
    return region.polygons.map(polygon => polygon.map((point, index) => {
        const [x, y] = project(point);
        return `${index ? 'L' : 'M'}${x},${y}`;
    }).join('') + 'Z').join('');
}
