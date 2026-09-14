import { it, expect } from 'vitest';
import { expressionEnvironment, isDefinition } from './expressionEnvironment.js';
import { prepare2DScene, prepareScene } from './sceneTools.js';
import { scalar } from './expressionSyntax.js';
it('reuses nested functions and parameter coefficients in both dimensions',()=>{
 const env=expressionEnvironment([{equation:'a=2'},{equation:'f(t)=a*t^2'},{equation:'g(u,v)=f(u)+sin(v)'}]);
 expect(env.errors).toEqual({});
 expect(scalar(env.expand('g(x,y)'))({x:3,y:0})).toBe(18);
 expect(isDefinition('f(t)=t^2')).toBe(true);
 expect(prepareScene([{equation:'f(t)=t^2'},{equation:'z=f(x)+f(y)'}])).toHaveLength(2);
});
it('rejects recursive, undefined, and unsafe function definitions',()=>{
 for(const equation of ['f(t)=f(t)','f(t)=bad(t)','f(t)=import(t)']) expect(Object.keys(expressionEnvironment([{equation}]).errors)).toHaveLength(1);
 const env=expressionEnvironment([{equation:'f(t)=t^2'}]);expect(()=>env.expand('f(1,2)')).toThrow(/arguments/);
});
it('broadcasts lists in lockstep, supports ranges and one-based indexing',()=>{
 const env=expressionEnvironment([{equation:'a=[1,2,3]'},{equation:'b=[3...5]'}]);
 expect(env.expandAll('y=a*x+b')).toEqual(['y=(1)*x+(3)','y=(2)*x+(4)','y=(3)*x+(5)']);
 expect(env.expand('a[2]')).toBe('(2)');
 expect(env.expandAll('y=[1,2]*x')).toHaveLength(2);
 expect(()=>env.expandAll('y=a+[1,2]')).toThrow(/equal lengths/);
 expect(()=>env.expand('a[0]')).toThrow(/one-based/);
 expect(prepare2DScene([{equation:'a=[1,2,3]'},{equation:'y=a*x'}])).toHaveLength(2);
});
it('bounds list workload and rejects injected internal names',()=>{
 expect(()=>expressionEnvironment([]).expandAll('y=[1...1000]*x')).toThrow(/100/);
 expect(()=>expressionEnvironment([]).expandAll('listInternal0')).toThrow(/Reserved/);
});
it('graphs a matching-dimensional function definition while retaining reuse',()=>{
 const env=expressionEnvironment([{equation:'f(t)=t^2'},{equation:'g(u,v)=u+v'}]);
 expect(env.plotExpressions('f(t)=t^2',2)).toHaveLength(1);
 expect(env.plotExpressions('g(u,v)=u+v',3)).toHaveLength(1);
 expect(env.plotExpressions('f(t)=t^2',3)).toEqual([]);
});
it('keeps function arguments separate from callee global coefficients',()=>{
 const env=expressionEnvironment([{equation:'a=2'},{equation:'f(t)=a*t'},{equation:'g(a)=f(a)+a'}]);
 expect(scalar(env.expand('g(3)'),[])({})).toBe(9);
});
