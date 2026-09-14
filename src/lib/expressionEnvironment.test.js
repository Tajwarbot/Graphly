import { describe, it, expect } from 'vitest';
import { expressionEnvironment, parameterDefinition } from './expressionEnvironment.js';
import { samplePlot2D } from './plot2D.js';
import { compileSurface } from './surfaceEngine.js';
describe('shared parameters', () => {
    it('resolves out of order dependencies and updates both dimensions', () => {
        const env = expressionEnvironment([{equation:'b=2*a'}, {equation:'a=3'}]);
        expect(env.values.b).toBe(6);
        expect(compileSurface(env.expand('z=b*x')).evaluate(2,0)).toBe(12);
        const points = samplePlot2D(env.expand('y=a*x'), {xMin:-1,xMax:1,yMin:-5,yMax:5});
        expect(points.filter(p=>Number.isFinite(p.y)).every(p=>Math.abs(p.y-3*p.x)<1e-8)).toBe(true);
    });
    it('rejects cycles and duplicates without arbitrary evaluation', () => {
        expect(() => expressionEnvironment([{equation:'a=b'},{equation:'b=a'}]).expand('a*x')).toThrow(/circular/);
        expect(() => expressionEnvironment([{equation:'a=1'},{equation:'a=2'}]).expand('a*x')).toThrow(/Duplicate/);
        expect(expressionEnvironment([{equation:'a=import(1)'}]).errors.a).toBeTruthy();
    });
    it('does not turn coordinates or parametric variables into sliders', () => {
        for (const name of ['x','y','z','u','v','t','pi','e']) expect(parameterDefinition(`${name}=2`)).toBeNull();
    });
});

import { prepareScene, prepare2DScene, parameterPatch } from './sceneTools.js';
it('validates complete scenes against named parameters before creating layers', () => {
    expect(prepareScene([{equation:'a=2'},{equation:'x^2+y^2+z^2=a^2'}])).toHaveLength(2);
    expect(prepare2DScene([{equation:'y=a*x'}],{},[{equation:'a=2'}])).toHaveLength(1);
    expect(() => prepareScene([{equation:'a=b'},{equation:'b=a'}])).toThrow(/circular/);
    expect(() => prepare2DScene([{equation:'a=2'},{equation:'y=missing*x'}])).toThrow();
});
it('parameter changes preserve valid bounds and reject invalid tool requests', () => {
    const patch=parameterPatch({equation:'a=2'},3,{min:0,max:5,step:.2});
    expect(patch.equation).toBe('a=3'); expect(patch.parameterSettings.step).toBe(.2);
    expect(() => parameterPatch({equation:'z=x'},3)).toThrow();
    expect(() => parameterPatch({equation:'a=2'},9,{min:0,max:5})).toThrow();
});
