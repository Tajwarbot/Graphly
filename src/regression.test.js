import { describe, expect, it } from 'vitest';
import { calculateStats, getRegressionParams } from './lib/mathEngine.js';

describe('Regression data validation and domain handling', () => {
    const invalid = [{ x: NaN, y: 4 }, { x: Infinity, y: 1 }, { x: null, y: 2 }, { x: 2, y: Infinity }];
    it('fits a line using only finite numeric rows', () => {
        const result = getRegressionParams([{ x: 1, y: 5 }, { x: 2, y: 7 }, { x: 3, y: 9 }, ...invalid]);
        expect(result.slope).toBeCloseTo(2);
        expect(result.intercept).toBeCloseTo(3);
        expect(result.r2).toBeCloseTo(1);
    });
    it('fits a quadratic with invalid rows omitted from its sample count', () => {
        const result = getRegressionParams([{ x: -1, y: 2 }, { x: 0, y: 3 }, { x: 1, y: 6 }, ...invalid], 'quadratic');
        expect(result.a).toBeCloseTo(1);
        expect(result.b).toBeCloseTo(2);
        expect(result.c).toBeCloseTo(3);
    });
    it.each(['linear', 'quadratic', 'exponential', 'power', 'logarithmic'])('returns no %s fit for a vertical dataset', type => {
        expect(getRegressionParams([{ x: 2, y: 1 }, { x: 2, y: 3 }, { x: 2, y: 5 }], type)).toBeNull();
    });
    it('computes logarithmic goodness of fit on the fitted domain', () => {
        const result = getRegressionParams([{ x: -1, y: 99 }, { x: 1, y: 2 }, { x: Math.E, y: 5 }, { x: Math.E ** 2, y: 8 }], 'logarithmic');
        expect(result.a).toBeCloseTo(2);
        expect(result.b).toBeCloseTo(3);
        expect(result.r2).toBeCloseTo(1);
    });
    it('computes power goodness of fit on the fitted domain', () => {
        const result = getRegressionParams([{ x: -1, y: -2 }, { x: 1, y: 2 }, { x: 4, y: 4 }, { x: 9, y: 6 }], 'power');
        expect(result.a).toBeCloseTo(2);
        expect(result.b).toBeCloseTo(0.5);
        expect(result.r2).toBeCloseTo(1);
    });
    it('computes exponential goodness of fit on positive observations', () => {
        const result = getRegressionParams([{ x: 0, y: -4 }, { x: 1, y: 2 }, { x: 2, y: 4 }, { x: 3, y: 8 }], 'exponential');
        expect(result.r2).toBeCloseTo(1);
    });
    it('recognizes a perfect horizontal fit', () => {
        const result = getRegressionParams([{ x: 0, y: 4 }, { x: 1, y: 4 }, { x: 2, y: 4 }]);
        expect(result.r2).toBe(1);
    });
    it('excludes infinite rows from descriptive statistics', () => {
        expect(calculateStats([{ x: '1', y: '2' }, { x: Infinity, y: 3 }, { x: 3, y: 4 }], 'x', 'y')).toEqual({ n: 2, meanX: 2, meanY: 3, stdDevX: 1, stdDevY: 1 });
    });
});
