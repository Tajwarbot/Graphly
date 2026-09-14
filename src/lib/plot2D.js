import {
    compileParametric,
    parametricCurve,
    splitRestrictions,
    compileRestrictions,
    allowedBy
} from './expressionSyntax.js';
import { compileMathFunction, generateFunctionPoints } from './mathEngine.js';
import { generateImplicitPoints } from './implicitContours.js';
import { compileInequality, isInequality } from './inequalityRegions.js';
export function samplePlot2D(equation, bounds) {
    const { base, restrictions } = splitRestrictions(equation);
    const parametric = compileParametric(equation, 2);
    if (parametric)
        return parametricCurve(equation, 2).map((p) =>
            p ? { x: p[0], y: p[1] } : { x: null, y: null }
        );
    if (isInequality(base)) {
        compileInequality(base);
        return [];
    }
    const fn = compileMathFunction(base);
    if (!fn) throw new Error('Enter a valid equation.');
    if (fn.type === 'implicit') return generateImplicitPoints(equation, bounds);
    const fields = compileRestrictions(restrictions, ['x', 'y']);
    const buffer = (bounds.xMax - bounds.xMin) * 0.5;
    const points = generateFunctionPoints(
        base,
        bounds.xMin - buffer,
        bounds.xMax + buffer,
        500
    );
    if (points.error) throw points.error;
    return points.map((p) => (allowedBy(fields, p) ? p : { x: p.x, y: null }));
}
