import { describe, it, expect, vi } from 'vitest';
import { parseIntentLocally, isMathExpression } from './lib/chatIntent.js';

describe('safe additive chat requests', () => {
    const execute = () => vi.fn(() => ({ success: true, message: 'Added' }));
    it('uses complete quadrics and distinguishes two sheets from a saddle', () => {
        const tool = execute();
        parseIntentLocally('plot a two sheeted hyperboloid', tool);
        expect(tool).toHaveBeenLastCalledWith('switchTo3D', { expression: 'z^2 - x^2 - y^2 = 1' });
        parseIntentLocally('plot a sphere', tool);
        expect(tool).toHaveBeenLastCalledWith('switchTo3D', { expression: 'x^2 + y^2 + z^2 = 25' });
    });
    it('keeps full implicit equations', () => {
        const tool = execute();
        parseIntentLocally('plot x^2 + y^2 + z^2 = 9', tool);
        expect(tool).toHaveBeenCalledWith('switchTo3D', { expression: 'x^2 + y^2 + z^2 = 9' });
    });
    it('does not manufacture user data', () => {
        const tool = execute();
        parseIntentLocally('load a table', tool);
        expect(tool).not.toHaveBeenCalled();
        parseIntentLocally('table (1, 2), (-3, 4)', tool);
        expect(tool).toHaveBeenCalledWith('loadDataTable', { name: 'Data points', rows: [{ x: 1, y: 2 }, { x: -3, y: 4 }] });
    });
    it('reports failure faithfully', () => {
        const result = parseIntentLocally('plot a sphere', () => ({ success: false, message: 'Surface limit reached' }));
        expect(result.text).toBe('Surface limit reached');
        expect(result.actions[0].success).toBe(false);
    });
    it('does not turn explanations and edits into new plots', () => {
        const tool = execute();
        parseIntentLocally('what is a sphere?', tool);
        parseIntentLocally('change the sphere radius', tool);
        expect(tool).not.toHaveBeenCalled();
    });
    it('validates syntax without rejecting domains away from origin', () => {
        expect(isMathExpression('sqrt(x-10)')).toBe(true);
        expect(isMathExpression('hello world')).toBe(false);
        expect(isMathExpression('x +')).toBe(false);
    });
});
