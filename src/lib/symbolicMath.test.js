import { describe, expect, it } from 'vitest';
import { symbolicMath } from './symbolicMath.js';

describe('bounded symbolic math', () => {
    it('simplifies multivariable expressions with single-letter symbolic constants', () => {
        const result = symbolicMath({ operation: 'simplify', expression: 'a*x + a*x + b*y', variable: 'x' });
        expect(result.expression).toBe('2 * a * x + b * y');
    });

    it('takes partial derivatives while treating other symbols as constants', () => {
        const result = symbolicMath({ operation: 'differentiate', expression: 'a*x^2 + b*x*y + c*y', variable: 'x' });
        expect(result.expression).toBe('2 * a * x + y * b');
    });

    it('integrates bounded polynomials with symbolic coefficients', () => {
        const result = symbolicMath({ operation: 'integrate', expression: 'a*x^2 + b*x + c', variable: 'x' });
        expect(result.operation).toBe('integrate');
        expect(result.expression).toBe('c * x + b * x ^ 2 / 2 + a * x ^ 3 / 3');
        expect(result.note).toContain('arbitrary constant C');
        expect(result.note).toContain('domain');
    });

    it('rejects non-polynomial integrands and out-of-scope symbols', () => {
        expect(() => symbolicMath({ operation: 'integrate', expression: 'sin(x)' })).toThrow('polynomial');
        expect(() => symbolicMath({ operation: 'simplify', expression: 'alpha*x' })).toThrow('single-letter');
    });
});
