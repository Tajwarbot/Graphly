import { parse } from 'mathjs';

import {
    normalizeMath,
    scalar,
    splitRestrictions,
    compileRestrictions,
    compileParametric,
    allowedBy
} from './expressionSyntax.js';
const cache = new Map();
function zeroBase(expression) {
    let node = parse(expression);
    for (;;) {
        if (node.isParenthesisNode) {
            node = node.content;
            continue;
        }
        if (
            node.isOperatorNode &&
            node.op === '^' &&
            node.args[1].isConstantNode &&
            Number.isInteger(Number(node.args[1].value)) &&
            Number(node.args[1].value) > 0
        ) {
            node = node.args[0];
            continue;
        }
        if (node.isFunctionNode && node.fn.name === 'abs') {
            node = node.args[0];
            continue;
        }
        return node.toString();
    }
}
export function compileSurface(input) {
    const raw = normalizeMath(input);
    if (!raw) throw new Error('Enter an equation or expression.');
    if (cache.has(raw)) return cache.get(raw);
    const parametric = compileParametric(raw, 3);
    if (parametric) return { parametric, implicit: false };
    const { base, restrictions } = splitRestrictions(raw);
    const fields = compileRestrictions(restrictions);
    const relation = base.match(/^(.*?)(<=|>=|<|>)(.*)$/);
    if (relation) {
        const constraints = compileRestrictions([base]);
        return {
            implicit: true,
            fields,
            inequality: true,
            evaluate(x, y, z = 0) {
                return Math.max(...constraints.map((f) => f({ x, y, z })));
            }
        };
    }
    const parts = (relation ? `${relation[1]}=${relation[3]}` : base).split(
        '='
    );
    if (parts.length > 2)
        throw new Error('Use one equality sign per equation.');
    const implicit =
        !!relation ||
        (parts.length === 2 &&
            !(parts[0].trim() === 'z' && !/\bz\b/.test(parts[1])));
    let expression = implicit
        ? `(${parts[0]})-(${parts[1]})`
        : parts.length === 2
          ? parts[1]
          : base;
    // Removing repeated factors preserves their zero set and reveals roots that
    // have no sign change, such as (z-2)^2=0 and abs(x^2+y^2-4)=0.
    if (implicit && !relation && parts[1].trim() === '0')
        expression = zeroBase(parts[0]);
    else if (implicit && !relation && parts[0].trim() === '0')
        expression = zeroBase(parts[1]);
    const evaluate = scalar(expression);
    if (!implicit && /\bz\b/.test(expression))
        throw new Error('Use an equation such as x^2+y^2+z^2=9.');
    const sign = relation?.[2].startsWith('>') ? -1 : 1;
    const fn = {
        implicit,
        fields,
        inequality: !!relation,
        evaluate(x, y, z = 0) {
            return sign * evaluate({ x, y, z });
        }
    };
    if (cache.size >= 200) cache.delete(cache.keys().next().value);
    cache.set(raw, fn);
    return fn;
}

// Output is in mathematical coordinates; Three.js coordinate mapping happens in the view.
export function buildSurface(input, range = 5, resolution = 36) {
    const fn = { ...compileSurface(input) };
    if (fn.parametric) return buildParametricSurface(fn.parametric, resolution);
    const n = Math.max(8, Math.min(64, Math.round(resolution)));
    const bounds =
        typeof range === 'number'
            ? {
                  xMin: -range,
                  xMax: range,
                  yMin: -range,
                  yMax: range,
                  zMin: -range,
                  zMax: range
              }
            : { ...range, zMin: range.zMin ?? -5, zMax: range.zMax ?? 5 };
    for (const axis of ['x', 'y', 'z'])
        if (
            !Number.isFinite(bounds[`${axis}Min`]) ||
            !Number.isFinite(bounds[`${axis}Max`]) ||
            bounds[`${axis}Min`] >= bounds[`${axis}Max`]
        )
            throw new Error('Surface bounds must be finite and increasing.');
    const minimum = [bounds.xMin, bounds.yMin, bounds.zMin];
    const steps = [
        (bounds.xMax - bounds.xMin) / n,
        (bounds.yMax - bounds.yMin) / n,
        (bounds.zMax - bounds.zMin) / n
    ];
    const step = Math.min(...steps);
    const heightLimit =
        Math.max(Math.abs(bounds.zMin), Math.abs(bounds.zMax)) * 4;
    const vertices = [];
    if (fn.inequality) {
        const field = fn.evaluate;
        fn.evaluate = (x, y, z) =>
            Math.max(
                field(x, y, z),
                ...fn.fields.map((f) => f({ x, y, z })),
                bounds.xMin - x,
                x - bounds.xMax,
                bounds.yMin - y,
                y - bounds.yMax,
                bounds.zMin - z,
                z - bounds.zMax
            );
    }
    const push = (a, b, c) => {
        if (!fn.inequality && fn.fields.length) {
            const polygon = clipPolygon([a, b, c], fn.fields);
            for (let i = 1; i < polygon.length - 1; i++)
                vertices.push(...polygon[0], ...polygon[i], ...polygon[i + 1]);
            return;
        }
        if (fn.implicit) {
            const u = b.map((v, i) => v - a[i]),
                v = c.map((w, i) => w - a[i]);
            const normal = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]
            ];
            const length = Math.hypot(...normal);
            if (length < 1e-12) return;
            const mid = a.map((x, i) => (x + b[i] + c[i]) / 3);
            const epsilon = step * 0.01;
            const plus = fn.evaluate(
                ...mid.map((x, i) => x + (normal[i] / length) * epsilon)
            );
            const minus = fn.evaluate(
                ...mid.map((x, i) => x - (normal[i] / length) * epsilon)
            );
            if (plus < minus) [b, c] = [c, b];
        }
        for (const p of [a, b, c]) vertices.push(...p);
    };
    if (!fn.implicit) {
        const points = [];
        for (let i = 0; i <= n; i++)
            for (let j = 0; j <= n; j++) {
                const x = bounds.xMin + i * steps[0],
                    y = bounds.yMin + j * steps[1];
                points.push([x, y, fn.evaluate(x, y)]);
            }
        const triangle = (a, b, c) => {
            if (
                ![a, b, c].every(
                    (p) =>
                        Number.isFinite(p[2]) && Math.abs(p[2]) <= heightLimit
                )
            )
                return;
            // Reject large jumps rather than connecting across poles.
            if (
                Math.max(a[2], b[2], c[2]) - Math.min(a[2], b[2], c[2]) >
                heightLimit / 2
            )
                return;
            push(a, b, c);
        };
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++) {
                const k = i * (n + 1) + j,
                    a = points[k],
                    b = points[k + n + 1],
                    c = points[k + 1],
                    d = points[k + n + 2];
                triangle(a, b, c);
                triangle(b, d, c);
            }
    } else {
        const stride = n + 1,
            values = new Float64Array(stride ** 3);
        const idx = (i, j, k) => (i * stride + j) * stride + k;
        for (let i = 0; i <= n; i++)
            for (let j = 0; j <= n; j++)
                for (let k = 0; k <= n; k++)
                    values[idx(i, j, k)] = fn.evaluate(
                        minimum[0] + i * steps[0],
                        minimum[1] + j * steps[1],
                        minimum[2] + k * steps[2]
                    );
        const corners = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ];
        const tetrahedra = [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6]
        ];
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                for (let k = 0; k < n; k++) {
                    const points = corners.map(([a, b, c]) => [
                        minimum[0] + (i + a) * steps[0],
                        minimum[1] + (j + b) * steps[1],
                        minimum[2] + (k + c) * steps[2]
                    ]);
                    const v = corners.map(
                        ([a, b, c]) => values[idx(i + a, j + b, k + c)]
                    );
                    if (
                        !v.every(Number.isFinite) ||
                        v.every((x) => x >= 0) ||
                        v.every((x) => x < 0)
                    )
                        continue;
                    const edge = (a, b) => {
                        // Use the same endpoint order on shared edges. Exact roots remain
                        // exact rather than becoming tiny sliver triangles around grid nodes.
                        if (v[a] === 0) return points[a];
                        if (v[b] === 0) return points[b];
                        if (
                            points[a].some(
                                (value, axis) =>
                                    value !== points[b][axis] &&
                                    points[a]
                                        .slice(0, axis)
                                        .every(
                                            (previous, d) =>
                                                previous === points[b][d]
                                        ) &&
                                    value > points[b][axis]
                            )
                        )
                            [a, b] = [b, a];
                        let lo = 0,
                            hi = 1;
                        // Refine the root and reject discontinuities masquerading as zero crossings.
                        for (let t = 0; t < 12; t++) {
                            const m = (lo + hi) / 2,
                                p = points[a].map(
                                    (x, d) => x + m * (points[b][d] - x)
                                );
                            const f = fn.evaluate(...p);
                            if (!Number.isFinite(f)) return null;
                            if (f < 0 === v[a] < 0) lo = m;
                            else hi = m;
                        }
                        const m = (lo + hi) / 2,
                            p = points[a].map(
                                (x, d) => x + m * (points[b][d] - x)
                            );
                        if (
                            Math.abs(fn.evaluate(...p)) >
                            Math.max(1, Math.abs(v[a]), Math.abs(v[b])) * 0.01
                        )
                            return null;
                        return p;
                    };
                    for (const tet of tetrahedra) {
                        const inside = tet.filter((a) => v[a] < 0),
                            outside = tet.filter((a) => v[a] >= 0);
                        if (!inside.length || !outside.length) continue;
                        if (inside.length === 1 || outside.length === 1) {
                            const one = inside.length === 1 ? inside : outside,
                                many = inside.length === 1 ? outside : inside;
                            const tri = many.map((b) => edge(one[0], b));
                            if (tri.every(Boolean)) push(...tri);
                        } else {
                            const [a, b] = inside,
                                [c, d] = outside;
                            const p = [
                                edge(a, c),
                                edge(a, d),
                                edge(b, c),
                                edge(b, d)
                            ];
                            if (p.every(Boolean)) {
                                push(p[0], p[1], p[2]);
                                push(p[1], p[3], p[2]);
                            }
                        }
                    }
                }
    }
    return new Float32Array(vertices);
}

// Field gradients give smooth shading independent of irregular tetrahedral triangle
// sizes. Evaluate once per welded position, with geometric normals as a fallback at
// singularities where the gradient has no direction.
export function buildSurfaceNormals(input, vertices) {
    const fn = compileSurface(input);
    const normals = new Float32Array(vertices.length);
    const cache = new Map();
    for (let i = 0; i < vertices.length; i += 3) {
        const point = Array.from(vertices.subarray(i, i + 3));
        const key = point.join(',');
        let normal = cache.get(key);
        if (!normal) {
            const gradient = point.map((value, axis) => {
                const h = 1e-5 * Math.max(1, Math.abs(value));
                const a = [...point],
                    b = [...point];
                a[axis] += h;
                b[axis] -= h;
                if (!fn.implicit && axis === 2) return 1;
                const derivative =
                    (fn.evaluate(...a) - fn.evaluate(...b)) / (2 * h);
                return fn.implicit ? derivative : -derivative;
            });
            const length = Math.hypot(...gradient);
            normal =
                Number.isFinite(length) && length > 1e-10
                    ? gradient.map((v) => v / length)
                    : [0, 0, 0];
            cache.set(key, normal);
        }
        if (normal.every((v) => v === 0)) {
            const start = Math.floor(i / 9) * 9;
            const a = Array.from(vertices.subarray(start, start + 3));
            const b = Array.from(vertices.subarray(start + 3, start + 6)).map(
                (v, j) => v - a[j]
            );
            const c = Array.from(vertices.subarray(start + 6, start + 9)).map(
                (v, j) => v - a[j]
            );
            const face = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0]
            ];
            const length = Math.hypot(...face) || 1;
            normals.set(
                face.map((v) => v / length),
                i
            );
        } else normals.set(normal, i);
    }
    return normals;
}

function clipPolygon(points, fields) {
    let polygon = points;
    for (const field of fields) {
        const out = [];
        for (let i = 0; i < polygon.length; i++) {
            const a = polygon[i],
                b = polygon[(i + 1) % polygon.length];
            const value = (p) => field({ x: p[0], y: p[1], z: p[2] });
            const va = value(a),
                vb = value(b),
                insideA = Number.isFinite(va) && va <= 0,
                insideB = Number.isFinite(vb) && vb <= 0;
            if (insideA) out.push(a);
            if (
                insideA !== insideB &&
                Number.isFinite(va) &&
                Number.isFinite(vb)
            ) {
                let lo = 0,
                    hi = 1;
                for (let j = 0; j < 16; j++) {
                    const t = (lo + hi) / 2,
                        p = a.map((v, k) => v + t * (b[k] - v));
                    if (value(p) <= 0 === insideA) lo = t;
                    else hi = t;
                }
                out.push(a.map((v, k) => v + ((lo + hi) / 2) * (b[k] - v)));
            }
        }
        polygon = out;
        if (!polygon.length) break;
    }
    return polygon;
}
function buildParametricSurface(fn, resolution) {
    if (fn.kind !== 'surface')
        throw new Error('Use the curve renderer for a tuple in t.');
    const n = Math.max(12, Math.min(128, Math.round(resolution * 2))),
        points = [],
        vertices = [];
    for (let i = 0; i <= n; i++)
        for (let j = 0; j <= n; j++) {
            const u =
                    fn.ranges.u[0] +
                    (i / n) * (fn.ranges.u[1] - fn.ranges.u[0]),
                v =
                    fn.ranges.v[0] +
                    (j / n) * (fn.ranges.v[1] - fn.ranges.v[0]);
            const p = fn.point({ u, v });
            points.push(
                p.every(Number.isFinite) &&
                    allowedBy(fn.fields, { u, v, x: p[0], y: p[1], z: p[2] })
                    ? p
                    : null
            );
        }
    const triangle = (...ps) => {
        if (ps.every(Boolean)) vertices.push(...ps.flat());
    };
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            const k = i * (n + 1) + j;
            triangle(points[k], points[k + n + 1], points[k + 1]);
            triangle(points[k + n + 1], points[k + n + 2], points[k + 1]);
        }
    return new Float32Array(vertices);
}
