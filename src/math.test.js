import { describe, it, expect } from 'vitest';
import {
    calculateNiceTicks,
    formatNumber,
    formatEquationNumber,
    getRegressionParams,
    calculateStats,
    generateFunctionPoints
} from './lib/mathEngine.js';

describe('Math Engine Characterization Tests', () => {

    describe('Function Evaluation (generateFunctionPoints)', () => {
        it('evaluates simple function x', () => {
            const points = generateFunctionPoints('x', 0, 10, 10);
            expect(points.length).toBeGreaterThan(0);
            expect(points[0].y).toBe(0);
            expect(points[points.length - 1].y).toBe(10);
        });

        it('evaluates x^2 (represented as x^2 in string, converted to ** in implementation)', () => {
            const points = generateFunctionPoints('x^2', 0, 5, 5);
            expect(points[0].y).toBe(0);
            expect(points[points.length - 1].y).toBe(25);
        });

        it('evaluates 2*x + 3', () => {
            const points = generateFunctionPoints('2*x + 3', 0, 5, 5);
            expect(points[0].y).toBe(3);
            expect(points[1].y).toBe(5); // x=1
            expect(points[points.length - 1].y).toBe(13); // x=5
        });

        it('evaluates sin(x)', () => {
            const points = generateFunctionPoints('sin(x)', 0, Math.PI / 2, 2);
            expect(points[0].y).toBe(0);
            expect(points[points.length - 1].y).toBeCloseTo(1, 5);
        });

        it('evaluates cos(x)', () => {
            const points = generateFunctionPoints('cos(x)', 0, Math.PI, 2);
            expect(points[0].y).toBe(1);
            expect(points[points.length - 1].y).toBeCloseTo(-1, 5);
        });

        it('evaluates sqrt(x)', () => {
            const points = generateFunctionPoints('sqrt(x)', 0, 4, 4);
            expect(points[0].y).toBe(0);
            expect(points[points.length - 1].y).toBe(2);
        });

        it('handles zero-domain safely without infinite loop (min === max)', () => {
            const points = generateFunctionPoints('x', 5, 5, 200);
            expect(points.length).toBe(1);
            expect(points[0].x).toBe(5);
            expect(points[0].y).toBe(5);
        });

        // Domain behavior tests
        it('handles invalid domain for sqrt(x)', () => {
            const points = generateFunctionPoints('sqrt(x)', -5, -1, 4);
            expect(points.every(p => p.y === null)).toBe(true);
            expect(points.length).toBe(5);
            // TODO: Ensure future math engine explicitly handles domain errors gracefully without relying on JS NaN filtering
        });

        it('handles invalid domain for log(x)', () => {
            const points = generateFunctionPoints('log(x)', -5, 0, 5);
            expect(points.every(p => p.y === null)).toBe(true);
            expect(points.length).toBe(6);
        });

        // Discontinuities tests
        it('handles discontinuities like 1/x', () => {
            const points = generateFunctionPoints('1/x', -1, 1, 2); 
            expect(points.length).toBe(3);
            expect(points[0].y).toBe(-1);
            expect(points[1].y).toBeNull();
            expect(points[2].y).toBe(1);
            // TODO: Future math engine should explicitly signal asymptotes rather than silently dropping points
        });

        it('handles discontinuities like tan(x) - characterizes current behavior', () => {
            const points = generateFunctionPoints('tan(x)', Math.PI / 2 - 0.1, Math.PI / 2 + 0.1, 2);
            // tan(PI/2) in JS is a very large number (not Infinity due to float imprecision), 
            // so it might actually generate a point! This is unsafe/fragile behavior.
            expect(points.length).toBeGreaterThan(0);
            // We just characterize that it returns something finite, not explicitly handling the asymptote
            expect(Number.isFinite(points[0].y)).toBe(true);
        });
    });

    describe('Regression (getRegressionParams)', () => {
        const linearData = [
            { x: 0, y: 0 },
            { x: 1, y: 2 },
            { x: 2, y: 4 },
            { x: 3, y: 6 }
        ];

        it('calculates linear regression', () => {
            const result = getRegressionParams(linearData, 'linear');
            expect(result).not.toBeNull();
            expect(result.type).toBe('linear');
            expect(result.slope).toBeCloseTo(2);
            expect(result.intercept).toBeCloseTo(0);
            expect(result.r2).toBeCloseTo(1);
        });

        const quadData = [
            { x: 0, y: 0 },
            { x: 1, y: 1 },
            { x: 2, y: 4 },
            { x: 3, y: 9 }
        ];

        it('calculates quadratic regression', () => {
            const result = getRegressionParams(quadData, 'quadratic');
            expect(result).not.toBeNull();
            expect(result.type).toBe('quadratic');
            expect(result.a).toBeCloseTo(1);
            expect(result.b).toBeCloseTo(0);
            expect(result.c).toBeCloseTo(0);
            expect(result.r2).toBeCloseTo(1);
        });

        const expData = [
            { x: 0, y: 1 },
            { x: 1, y: Math.E },
            { x: 2, y: Math.E * Math.E }
        ];

        it('calculates exponential regression', () => {
            const result = getRegressionParams(expData, 'exponential');
            expect(result).not.toBeNull();
            expect(result.type).toBe('exponential');
            expect(result.a).toBeCloseTo(1);
            expect(result.b).toBeCloseTo(1);
        });
    });

    describe('Statistics (calculateStats)', () => {
        it('calculates mean and standard deviation correctly', () => {
            const data = [
                { xKey: 2, yKey: 4 },
                { xKey: 4, yKey: 8 },
                { xKey: 6, yKey: 12 },
                { xKey: 8, yKey: 16 }
            ];
            
            const stats = calculateStats(data, 'xKey', 'yKey');
            expect(stats.n).toBe(4);
            expect(stats.meanX).toBe(5); // (2+4+6+8)/4 = 20/4 = 5
            expect(stats.meanY).toBe(10); // (4+8+12+16)/4 = 40/4 = 10
            
            // Pop std dev for X: values are 2,4,6,8. Mean=5. 
            // Var = (9 + 1 + 1 + 9)/4 = 20/4 = 5. StdDev = sqrt(5) ≈ 2.236
            expect(stats.stdDevX).toBeCloseTo(Math.sqrt(5));
            expect(stats.stdDevY).toBeCloseTo(Math.sqrt(20));
        });

        it('handles empty data', () => {
            const stats = calculateStats([], 'x', 'y');
            expect(stats).toEqual({ meanX: 0, meanY: 0, stdDevX: 0, stdDevY: 0, n: 0 });
        });
    });

    describe('Tick Generation (calculateNiceTicks)', () => {
        it('generates ticks for 0 to 10', () => {
            const ticks = calculateNiceTicks(0, 10, 6);
            expect(ticks).toEqual([0, 2, 4, 6, 8, 10]);
        });

        it('generates ticks for -10 to 10', () => {
            const ticks = calculateNiceTicks(-10, 10, 5);
            expect(ticks).toEqual([-10, -5, 0, 5, 10]);
        });

        it('generates ticks for small range 0 to 0.1', () => {
            const ticks = calculateNiceTicks(0, 0.1, 6);
            expect(ticks).toEqual([0, 0.02, 0.04, 0.06, 0.08, 0.1]);
        });

        it('generates ticks for large range 0 to 1000', () => {
            const ticks = calculateNiceTicks(0, 1000, 6);
            expect(ticks).toEqual([0, 200, 400, 600, 800, 1000]);
        });

        it('handles unusual ranges (e.g. min == max) - characterizes current behavior', () => {
            const ticks = calculateNiceTicks(5, 5);
            expect(ticks).toEqual([]);
        });

        it('handles extreme large magnitudes without hanging', () => {
            const ticks = calculateNiceTicks(0, 1e12, 6);
            expect(ticks.length).toBeGreaterThan(1);
            expect(ticks.length).toBeLessThanOrEqual(10);
            expect(ticks[0]).toBe(0);
        });

        it('handles extreme small magnitudes without floating point drift', () => {
            const ticks = calculateNiceTicks(0, 1e-6, 6);
            expect(ticks.length).toBeGreaterThan(1);
            expect(ticks.every(t => Number.isFinite(t))).toBe(true);
        });

        it('handles non-finite ranges gracefully', () => {
            expect(calculateNiceTicks(NaN, 10)).toEqual([]);
            expect(calculateNiceTicks(0, Infinity)).toEqual([]);
        });
    });

    describe('Number Formatting (formatNumber & formatEquationNumber)', () => {
        it('formats ordinary numbers', () => {
            expect(formatNumber(5)).toBe(5);
            expect(formatNumber(-5)).toBe(-5);
        });

        it('formats decimals, rounding to 3 decimal places', () => {
            expect(formatNumber(3.14159)).toBe(3.142);
        });

        it('formats very large values in exponential notation', () => {
            expect(formatNumber(15000)).toBe('1.50e+4');
            expect(formatNumber(1e12)).toBe('1.00e+12');
        });

        it('formats very small values in exponential notation', () => {
            expect(formatNumber(0.0005)).toBe('5.00e-4');
            expect(formatNumber(1e-9)).toBe('1.00e-9');
        });

        it('formats zero correctly', () => {
            expect(formatNumber(0)).toBe(0);
            expect(formatEquationNumber(0)).toBe('0.000');
        });

        it('handles non-finite numbers safely', () => {
            expect(formatNumber(NaN)).toBe('—');
            expect(formatNumber(Infinity)).toBe('Infinity');
        });
    });
    describe('AST Parser & Evaluator', () => {
        it('handles implicit multiplication', () => {
            const expectEq = (eq, val) => {
                const points = generateFunctionPoints(eq, 3, 3, 200);
                expect(points[0].y).toBeCloseTo(val, 5);
            };
            expectEq('2x', 6);
            expectEq('2sin(x)', 2 * Math.sin(3));
            expectEq('3(x+1)', 12);
            expectEq('(x+1)(x-1)', 8);
            expectEq('x(x+1)', 12);
        });

        it('supports all required functions', () => {
            const fns = ['exp', 'asin', 'acos', 'atan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'ceil', 'floor', 'round', 'sign'];
            fns.forEach(fn => {
                const eq = `${fn}(x)`;
                const points = generateFunctionPoints(eq, 0.5, 0.5, 200);
                expect(points).not.toBeNull();
                expect(points.error).toBeUndefined();
            });
        });

        it('supports constants', () => {
            expect(generateFunctionPoints('pi', 1, 1, 200)[0].y).toBe(Math.PI);
            expect(generateFunctionPoints('e', 1, 1, 200)[0].y).toBe(Math.E);
            expect(generateFunctionPoints('tau', 1, 1, 200)[0].y).toBe(Math.PI * 2);
        });

        it('reports useful parse errors with positions', () => {
            const p1 = generateFunctionPoints('2**x');
            expect(p1.error).toBeInstanceOf(Error);
            expect(p1.error.position).toBeDefined();

            const p2 = generateFunctionPoints('sin(');
            expect(p2.error).toBeDefined();

            const p3 = generateFunctionPoints('x+');
            expect(p3.error).toBeDefined();

            const p4 = generateFunctionPoints('x,,2');
            expect(p4.error).toBeDefined();
        });

        it('prevents arbitrary javascript execution', () => {
            const badInputs = [
                'fetch("http://example.com")',
                'while(true){}',
                '(function(){while(true){}})()',
                'eval("1+1")',
                'constructor',
                '__proto__'
            ];
            badInputs.forEach(input => {
                const result = generateFunctionPoints(input);
                expect(result.error).toBeDefined();
            });
        });

        it('handles discontinuities with null gap markers', () => {
            const p1 = generateFunctionPoints('1/x', -1, 1, 200);
            expect(p1.some(p => p.y === null)).toBe(true);

            const p2 = generateFunctionPoints('tan(x)', Math.PI/2 - 0.2, Math.PI/2 + 0.2, 200);
            expect(p2.some(p => p.y === null)).toBe(true);

            const p3 = generateFunctionPoints('sqrt(x)', -1, 1, 100);
            // domain -1 to 0 should be null
            expect(p3[0].y).toBeNull();
            expect(p3[p3.length - 1].y).not.toBeNull();
        });

        it('preserves regressions (no bad regex replacements)', () => {
            expect(generateFunctionPoints('exp(0)', 0, 0, 200)[0].y).toBe(1);
            expect(generateFunctionPoints('sec(0)', 0, 0, 200)[0].y).toBe(1);
        });
    });
});
