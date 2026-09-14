import { describe, it, expect } from 'vitest';
import { compileInequality, buildInequalityRegion, regionToSvgPath } from './lib/inequalityRegions';
const bounds = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 };
const area = region => region.polygons.reduce((sum, points) => sum + Math.abs(points.reduce((s, a, i) => { const b = points[(i + 1) % points.length]; return s + a[0] * b[1] - a[1] * b[0]; }, 0)) / 2, 0);

describe('2D inequality regions', () => {
    it('shades a disk, not just its boundary', () => {
        const region = buildInequalityRegion('x^2+y^2<=9', bounds, 64);
        expect(area(region)).toBeCloseTo(9 * Math.PI, 1);
        expect(compileInequality('x²+y²≤9').contains(0, 0)).toBe(true);
        expect(compileInequality('x²+y²≤9').contains(4, 0)).toBe(false);
        expect(region.boundaries).toEqual([{ expression: 'x^2+y^2 = 9', strict: false }]);
    });
    it('clips a diagonal half plane and distinguishes strict boundaries', () => {
        const compiled = compileInequality('y>x');
        expect(compiled.contains(1, 1)).toBe(false);
        expect(compiled.contains(1, 2)).toBe(true);
        expect(area(buildInequalityRegion('y>x', bounds, 16))).toBeCloseTo(32, 3);
        expect(compiled.constraints[0].strict).toBe(true);
    });
    it('intersects chained bounds as a strip', () => {
        const region = buildInequalityRegion('-1<y<1', bounds, 32);
        expect(area(region)).toBeCloseTo(16, 3);
        expect(compileInequality('-1<y<1').contains(0, 2)).toBe(false);
        expect(region.polygons.flat().every(([, y]) => y >= -1 - 1e-5 && y <= 1 + 1e-5)).toBe(true);
    });
    it('supports reversed and mixed chained comparisons', () => {
        expect(compileInequality('1 >= y > -1').contains(0, 1)).toBe(true);
        expect(compileInequality('1 >= y > -1').contains(0, -1)).toBe(false);
    });
    it('does not shade undefined sqrt domain', () => {
        const region = buildInequalityRegion('y < sqrt(x)', bounds, 32);
        expect(region.polygons.flat().every(([x]) => x >= 0)).toBe(true);
    });
    it('returns no region for contradictory bounds', () => {
        expect(area(buildInequalityRegion('1 < y < -1', bounds, 16))).toBe(0);
    });
    it('does not shade a strict equality that is false everywhere', () => {
        expect(area(buildInequalityRegion('0 < 0', bounds, 8))).toBe(0);
        expect(area(buildInequalityRegion('0 <= 0', bounds, 8))).toBeCloseTo(64);
    });
    it('projects filled paths through screen scales', () => {
        expect(regionToSvgPath({ polygons: [[[0, 0], [1, 0], [0, 1]]] }, ([x, y]) => [x * 10, 20 - y * 10])).toBe('M0,20L10,20L0,10Z');
    });
    it.each(['a=1<y', 'import("x")<y', 'x[1]<y', '[1,2]<y', 'z<1', 'x<', 'x==y', 'x < Infinity'])('rejects invalid or unsafe input %s', input => {
        expect(() => compileInequality(input)).toThrow();
    });
    it('rejects invalid bounds and resolution', () => {
        expect(() => buildInequalityRegion('y>x', { ...bounds, xMax: Infinity })).toThrow();
        expect(() => buildInequalityRegion('y>x', { ...bounds, xMin: 4 })).toThrow();
        expect(() => buildInequalityRegion('y>x', bounds, NaN)).toThrow();
    });
});
