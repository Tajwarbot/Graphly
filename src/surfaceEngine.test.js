import { describe, it, expect } from 'vitest';
import { compileSurface, buildSurface } from './lib/surfaceEngine.js';
import { appendDataset, makeDataset, makeSurface } from './lib/graphState.js';

describe('3D equation engine', () => {
    it('accepts explicit equations and expressions with both horizontal variables', () => {
        expect(compileSurface('z = x^2 + y^2').evaluate(2,3)).toBe(13);
        expect(compileSurface('sin(x)*cos(y)').evaluate(1,2)).toBeCloseTo(Math.sin(1)*Math.cos(2));
    });
    it('compiles full implicit equations in any coordinate orientation', () => {
        expect(compileSurface('x²/9 + y²/4 + z² = 1').evaluate(3,0,0)).toBe(0);
        expect(compileSurface('x = y^2 + z^2').evaluate(2,1,1)).toBe(0);
        expect(compileSurface('x^2 + y^2 = 4').implicit).toBe(true);
        expect(()=>compileSurface('banana(x)')).toThrow();
        expect(()=>compileSurface('z+1')).toThrow();
    });
    it('builds both halves of an ellipsoid with vertices satisfying its equation', () => {
        const positions=buildSurface('x^2/9 + y^2/4 + z^2 = 1',4,16);
        expect(positions.length).toBeGreaterThan(100);
        let minZ=Infinity,maxZ=-Infinity;
        for(let i=0;i<positions.length;i+=3) {
            const [x,y,z]=positions.slice(i,i+3);
            expect(Math.abs(x*x/9+y*y/4+z*z-1)).toBeLessThan(0.001);
            minZ=Math.min(minZ,z);maxZ=Math.max(maxZ,z);
        }
        expect(minZ).toBeLessThan(-0.99);expect(maxZ).toBeGreaterThan(0.99);
    });
    it('never fabricates flat points outside an explicit square root domain', () => {
        const positions=buildSurface('sqrt(1-x^2-y^2)',2,16);
        expect(positions.length).toBeGreaterThan(0);
        for(let i=0;i<positions.length;i+=3) {
            const [x,y,z]=positions.slice(i,i+3);
            expect(x*x+y*y).toBeLessThanOrEqual(1.00001);
            expect(z*z+x*x+y*y).toBeCloseTo(1,5);
        }
    });
    it('does not generate an implicit surface across a pole', () => {
        expect(buildSurface('1/x = 0',2,15).length).toBe(0);
    });
});
describe('Non-destructive workspace operations', () => {
    it('preserves each dataset when adding several items to an initially empty graph', () => {
        const a=makeDataset({equation:'x^2'}), b=makeDataset({rows:[{x:1,y:2}],name:'Observations'});
        const graph=appendDataset(appendDataset(null,a),b);
        expect(graph.datasets.map(d=>d.id)).toEqual([a.id,b.id]);
        expect(graph.datasets[0].equation).toBe('x^2');
        expect(a.id).not.toBe(b.id);
        expect(makeSurface('x').id).not.toBe(makeSurface('x').id);
    });
});
