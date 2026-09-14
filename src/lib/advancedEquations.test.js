import { describe,it,expect,vi } from 'vitest';
import { parseIntentLocally } from './chatIntent.js';
import { buildSurface,compileSurface } from './surfaceEngine.js';
import { samplePlot2D } from './plot2D.js';
import { compileParametric,parametricCurve } from './expressionSyntax.js';
import { compileInequality } from './inequalityRegions.js';
describe('equations beyond explicit functions',()=>{
 it('renders repeated zero factors instead of losing the plane',()=>{
  const vertices=buildSurface('(z-1.3)^2=0',3,12); expect(vertices.length).toBeGreaterThan(0);
  for(let i=2;i<vertices.length;i+=3)expect(vertices[i]).toBeCloseTo(1.3,3);
 });
 it('clips a surface to chained coordinate restrictions',()=>{
  const vertices=buildSurface('z=x+y {-1<x<1}{y>0}',3,12); expect(vertices.length).toBeGreaterThan(0);
  for(let i=0;i<vertices.length;i+=3){expect(vertices[i]).toBeGreaterThanOrEqual(-1.0001);expect(vertices[i]).toBeLessThanOrEqual(1.0001);expect(vertices[i+1]).toBeGreaterThanOrEqual(-.0001);}
 });
 it('samples a 3D parametric torus across specified parameter domains',()=>{
  const equation='((3+cos(v))*cos(u),(3+cos(v))*sin(u),sin(v)){0<=u<=2*pi}{0<=v<=2*pi}';
  expect(compileParametric(equation).ranges.u[1]).toBeCloseTo(2*Math.PI);
  const vertices=buildSurface(equation,5,16);expect(vertices.length).toBeGreaterThan(0);
  for(let i=0;i<vertices.length;i+=3)expect((Math.hypot(vertices[i],vertices[i+1])-3)**2+vertices[i+2]**2).toBeCloseTo(1,4);
 });
 it('samples helix curves and points without requiring z= syntax',()=>{
  const points=parametricCurve('(cos(t),sin(t),t){0<t<4*pi}',3);
  expect(points.filter(Boolean).length).toBeGreaterThan(500);expect(points.at(-1)[2]).toBeCloseTo(4*Math.PI);
  expect(parametricCurve('(1,2,3)',3)).toEqual([[1,2,3]]);
 });
 it('plots 2D tuples, vertical equations and restrictions',()=>{
  const bounds={xMin:-5,xMax:5,yMin:-5,yMax:5};
  expect(samplePlot2D('(3*cos(t),2*sin(t)){0<t<2*pi}',bounds).filter(p=>p.x!==null).length).toBeGreaterThan(500);
  const vertical=samplePlot2D('x=2',bounds).filter(p=>p.x!==null);expect(vertical.length).toBeGreaterThan(0);expect(vertical.every(p=>Math.abs(p.x-2)<.001)).toBe(true);
  const restricted=samplePlot2D('y=x²{-1<x<1}',bounds).filter(p=>p.y!==null);expect(restricted.every(p=>p.x>=-1.0001&&p.x<=1.0001)).toBe(true);
 });
 it('builds a bounded volume for 3D inequalities and chained regions',()=>{
  expect(buildSurface('x^2+y^2+z^2<=4',3,12).length).toBeGreaterThan(0);
  const slab=compileSurface('-1<z<1');expect(slab.evaluate(0,0,0)).toBeLessThan(0);expect(slab.evaluate(0,0,2)).toBeGreaterThan(0);
  const restricted=compileInequality('x^2+y^2<=4{x>0}');expect(restricted.contains(1,0)).toBe(true);expect(restricted.contains(-1,0)).toBe(false);
 });
 it('rejects executable and undefined parameter syntax',()=>{
  expect(()=>compileParametric('(import(1),v,u)')).toThrow();expect(()=>compileSurface('z=x{window>0}')).toThrow();
 });
});

it('routes tuple curves and dependent-variable equations correctly offline',()=>{
 const execute=vi.fn(()=>({success:true,message:'Added'}));
 parseIntentLocally('plot (cos(t),sin(t),t){0<t<2*pi}',execute);
 expect(execute.mock.calls[0][0]).toBe('switchTo3D');
 parseIntentLocally('plot y=y^2+x',execute);
 expect(execute.mock.calls[1]).toEqual(['plotImplicitEquation',{expression:'y=y^2+x'}]);
});
