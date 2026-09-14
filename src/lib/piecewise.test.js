import { describe, it, expect } from 'vitest';
import { scalar, splitRestrictions, compileParametric } from './expressionSyntax.js';
import { compileMathFunction, generateFunctionPoints } from './mathEngine.js';
import { compileSurface, buildSurface } from './surfaceEngine.js';
import { isInequality } from './inequalityRegions.js';
import { symbolicMath } from './symbolicMath.js';

describe('piecewise syntax', () => {
    it('uses strict comparisons, first match, lazy branches and fallback', () => {
        const fn = scalar('{x<0:-x,x>=0:x}', ['x']);
        expect(fn({x:-3})).toBe(3); expect(fn({x:0})).toBe(0);
        expect(scalar('{x<0:sqrt(-x),sqrt(x)}')({x:4})).toBe(2);
        expect(scalar('{x<0:1}')({x:0})).toBeNaN();
        expect(scalar('{x>=0:1,x>=0:2}')({x:0})).toBe(1);
    });
    it('supports nesting, arithmetic, function commas and chained predicates', () => {
        expect(scalar('2+{x<0:{y<0:3,4},max(x,2)}')({x:-1,y:-1})).toBe(5);
        expect(scalar('{-1<x<=1:5,0}')({x:-1})).toBe(0);
        expect(compileParametric('({t<0:-t,t},t)',2).point({t:-1})).toEqual([1,-1]);
    });
    it('preserves trailing restrictions and rejects unsafe branches', () => {
        expect(splitRestrictions('y={x<0:-x,x>=0:x}{x<3}')).toEqual({base:'y={x<0:-x,x>=0:x}',restrictions:['x<3']});
        expect(() => scalar('{x<0:import("bad"),1}')).toThrow();
        expect(() => scalar('{x<0:a=1,2}')).toThrow();
        expect(() => scalar('{x<0:1')).toThrow();
    });
    it('routes 2D and 3D as explicit equations rather than inequalities', () => {
        const equation='y={x<0:-x,x>=0:x}';
        expect(isInequality(equation)).toBe(false);
        const fn=compileMathFunction(equation); expect(fn.type).toBe('explicit'); expect(fn(-2)).toBe(2);
        expect(generateFunctionPoints(equation,-2,2,20).every(p=>p.y===Math.abs(p.x))).toBe(true);
        const surface=compileSurface('z={x<0:sin(y),x>=0:cos(y)}');
        expect(surface.implicit).toBe(false); expect(surface.evaluate(-1,0)).toBe(0); expect(surface.evaluate(1,0)).toBe(1);
        expect(buildSurface('z={x<0:sin(y),x>=0:cos(y)}',2,8).length).toBeGreaterThan(0);
    });
});
describe('bounded symbolic operations', () => {
    it('differentiates and simplifies safely', () => {
        const result=symbolicMath({operation:'differentiate',expression:'x^3+sin(x)'});
        expect(scalar(result.expression,['x'])({x:2})).toBeCloseTo(12+Math.cos(2));
        expect(symbolicMath({operation:'simplify',expression:'x+x'}).expression).toBe('2 * x');
    });
    it('solves linear and quadratic equations including repeated and nonreal roots', () => {
        expect(symbolicMath({operation:'solve',expression:'2*x+1=0'}).approximate).toEqual([-0.5]);
        expect(symbolicMath({operation:'solve',expression:'x^2=2'}).approximate).toEqual([-Math.sqrt(2),Math.sqrt(2)]);
        expect(symbolicMath({operation:'solve',expression:'(x-1)^2=0'}).approximate).toEqual([1]);
        expect(symbolicMath({operation:'solve',expression:'x^2+1=0'}).solutionSet).toBe('no real solutions');
        expect(symbolicMath({operation:'solve',expression:'x=x'}).solutionSet).toBe('all real numbers');
    });
    it('rejects unsupported or unsafe problems honestly', () => {
        for(const expression of ['sin(x)=0','x^3=1','1/x=2','import("a")','{x<0:x,0}']) expect(()=>symbolicMath({operation:'solve',expression})).toThrow();
    });
});

import { compileInequality, buildInequalityRegion } from './inequalityRegions.js';
it('supports piecewise operands in regions and solid inequalities', () => {
    const region=compileInequality('y<={x<0:-x,x>=0:x}');
    expect(region.contains(-2,1)).toBe(true); expect(region.contains(2,3)).toBe(false);
    expect(buildInequalityRegion('y<={x<0:-x,x>=0:x}',{xMin:-3,xMax:3,yMin:-3,yMax:3},16).polygons.length).toBeGreaterThan(0);
    const solid=compileSurface('z<={x<0:-x,x>=0:x}');
    expect(solid.inequality).toBe(true); expect(solid.evaluate(-2,0,1)).toBeLessThan(0);
    expect(buildSurface('z<={x<0:-x,x>=0:x}',2,8).length).toBeGreaterThan(0);
});
