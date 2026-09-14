import { describe, it, expect } from 'vitest';
import { generateFunctionPoints } from './mathEngine.js';
describe('explicit function equation plotting', () => {
    it.each(['x^2', 'y = x^2', 'f(x) = x^2'])('plots %s through the same compiler', equation => {
        expect(generateFunctionPoints(equation, -2, 2, 4)).toEqual([{x:-2,y:4},{x:-1,y:1},{x:0,y:0},{x:1,y:1},{x:2,y:4}]);
    });
    it('evaluates a full equation at a single requested x', () => {
        expect(generateFunctionPoints('y = sin(x)', 0, 0, 1)).toEqual([{x:0,y:0}]);
    });
    it('rejects implicit equations from explicit sampling', () => {
        expect(generateFunctionPoints('y = y^2+x')).toHaveLength(0);
    });
});
