import { describe, expect, it } from 'vitest';
import { buildSurface } from './surfaceEngine.js';

// Residuals alone do not detect holes or duplicate triangles. Closed quadrics and
// a genus-one torus must also have two faces meeting at every welded mesh edge.
function topology(vertices) {
    const points = new Set(), edges = new Map();
    for (let i = 0; i < vertices.length; i += 9) {
        const triangle = [0, 3, 6].map(offset => Array.from(vertices.slice(i + offset, i + offset + 3)).map(v => Math.round(v / 1e-5)).join(','));
        triangle.forEach(p => points.add(p));
        for (let j = 0; j < 3; j++) {
            const edge = [triangle[j], triangle[(j + 1) % 3]].sort().join('|');
            edges.set(edge, (edges.get(edge) ?? 0) + 1);
        }
    }
    return { incidence: [...edges.values()], euler: points.size - edges.size + vertices.length / 9 };
}
describe('closed implicit mesh topology', () => {
    it.each([
        ['x^2+y^2+z^2=9', 2],
        ['x^2/9+y^2/4+z^2=1', 2],
        ['(sqrt(x^2+y^2)-3)^2+z^2=1', 0],
    ])('%s is closed with the expected Euler characteristic', (equation, euler) => {
        const vertices = buildSurface(equation, 5, 24);
        expect(vertices.length).toBeGreaterThan(0);
        const result = topology(vertices);
        expect(result.incidence.every(count => count === 2)).toBe(true);
        expect(result.euler).toBe(euler);
    });
});
