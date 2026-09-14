import { allowedBy } from './expressionSyntax.js';
import { compileSurface } from './surfaceEngine.js';

/** Contour F(x,y)=0. Returns [[x,y],[x,y]] line segments in mathematical coordinates.
 * Splitting each cell about its sampled centre resolves saddle ambiguity locally.
 */
export function generateImplicitSegments(equation, bounds, resolution = 100) {
    if (!bounds || ![...bounds.x, ...bounds.y].every(Number.isFinite) || bounds.x[0] >= bounds.x[1] || bounds.y[0] >= bounds.y[1]) throw new Error('Invalid contour bounds.');
    if (/\bz\b/.test(equation)) throw new Error('A 2D curve cannot contain z.');
    const fn = compileSurface(equation);
    const n = Math.max(12, Math.min(200, Math.round(resolution)));
    const dx = (bounds.x[1] - bounds.x[0]) / n, dy = (bounds.y[1] - bounds.y[0]) / n;
    const value = p => fn.evaluate(p[0], p[1]);
    const segments = [];
    const edge = (a, b, va, vb) => {
        if (va === 0) return a;
        if (vb === 0) return b;
        let lo = 0, hi = 1;
        for (let k = 0; k < 12; k++) {
            const t = (lo + hi) / 2, p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            const v = value(p);
            if (!Number.isFinite(v)) return null;
            if ((v < 0) === (va < 0)) lo = t; else hi = t;
        }
        const t = (lo + hi) / 2, p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        return Math.abs(value(p)) <= Math.max(1, Math.abs(va), Math.abs(vb)) * 0.002 ? p : null;
    };
    const grid = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => {
        const p = [bounds.x[0] + i * dx, bounds.y[0] + j * dy];
        return { p, v: value(p) };
    }));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const corners = [grid[i][j], grid[i+1][j], grid[i+1][j+1], grid[i][j+1]];
        const p = [bounds.x[0] + (i + 0.5) * dx, bounds.y[0] + (j + 0.5) * dy];
        const center = { p, v: value(p) };
        for (let k = 0; k < 4; k++) {
            const tri = [corners[k], corners[(k + 1) % 4], center];
            if (!tri.every(q => Number.isFinite(q.v))) continue;
            const crossings = [];
            for (let e = 0; e < 3; e++) {
                const a = tri[e], b = tri[(e + 1) % 3];
                if ((a.v < 0) !== (b.v < 0)) crossings.push(edge(a.p, b.p, a.v, b.v));
            }
            if (crossings.length === 2 && crossings.every(Boolean) && Math.hypot(crossings[0][0] - crossings[1][0], crossings[0][1] - crossings[1][1]) > 1e-12) {
                const valid=p=>allowedBy(fn.fields,{x:p[0],y:p[1],z:0});
                if(valid(crossings[0]) && valid(crossings[1])) segments.push(crossings);
            }
        }
    }
    return segments;
}

/** Recharts-ready contour fragments separated by explicit null points. */
export function generateImplicitPoints(equation, bounds, resolution = 100) {
    return generateImplicitSegments(equation, { x: [bounds.xMin, bounds.xMax], y: [bounds.yMin, bounds.yMax] }, resolution)
        .flatMap(([a, b]) => [{ x: a[0], y: a[1] }, { x: b[0], y: b[1] }, { x: null, y: null }]);
}
