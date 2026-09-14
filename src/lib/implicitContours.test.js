import { describe, expect, it } from 'vitest';
import { generateImplicitPoints } from './implicitContours.js';
const bounds = { xMin: -4, xMax: 4, yMin: -4, yMax: 4 };
describe('implicit 2D contours', () => {
    it('plots a complete ellipse with explicit fragment breaks', () => {
        const points = generateImplicitPoints('x^2/9+y^2/4=1', bounds, 48);
        const valid = points.filter(p => p.x !== null);
        expect(valid.length).toBeGreaterThan(100);
        expect(Math.min(...valid.map(p => p.y))).toBeLessThan(-1.99);
        expect(Math.max(...valid.map(p => p.y))).toBeGreaterThan(1.99);
        expect(Math.min(...valid.map(p => p.x))).toBeLessThan(-2.99);
        expect(Math.max(...valid.map(p => p.x))).toBeGreaterThan(2.99);
        expect(valid.every(p => Math.abs(p.x*p.x/9+p.y*p.y/4-1) < 0.001)).toBe(true);
        expect(points.every((p,i) => i % 3 !== 2 || (p.x === null && p.y === null))).toBe(true);
    });
    it('does not mistake a pole for an implicit root', () => {
        expect(generateImplicitPoints('1/x=0', bounds, 48)).toHaveLength(0);
    });
    it('supports vertical lines and rejects z and invalid syntax', () => {
        const points = generateImplicitPoints('x=1', bounds, 48).filter(p => p.x !== null);
        expect(points.length).toBeGreaterThan(0);
        expect(points.every(p => Math.abs(p.x-1) < 0.001)).toBe(true);
        expect(() => generateImplicitPoints('z=1', bounds)).toThrow();
        expect(() => generateImplicitPoints('x+', bounds)).toThrow();
    });
});
