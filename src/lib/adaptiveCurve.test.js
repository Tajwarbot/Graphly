import { it,expect } from 'vitest';
import { samplePlot2D } from './plot2D.js';
import { parametricCurve } from './expressionSyntax.js';
import { buildSurface } from './surfaceEngine.js';
it('breaks finite piecewise jumps rather than drawing a connecting line',()=>{
 const points=samplePlot2D('y={x<0:0,1}',{xMin:-1,xMax:1,yMin:-1,yMax:2});
 for(let i=1;i<points.length;i++) if(points[i].y!==null&&points[i-1].y!==null)expect(Math.abs(points[i].y-points[i-1].y)).toBeLessThan(.1);
});
it('does not join parametric poles',()=>{
 const points=parametricCurve('(t,1/t){-1<t<1}',2,31);
 expect(points.some(p=>p===null)).toBe(true);
});
it('does not bridge a step surface with slanted triangles',()=>{
 const vertices=buildSurface('z={x<0:0,1}',2,8);
 expect(vertices.length).toBeGreaterThan(0);
 for(let i=0;i<vertices.length;i+=9) expect(Math.max(vertices[i+2],vertices[i+5],vertices[i+8])-Math.min(vertices[i+2],vertices[i+5],vertices[i+8])).toBeLessThan(.1);
});
