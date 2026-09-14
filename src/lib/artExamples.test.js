import { describe, expect, it, vi } from 'vitest';
import { ART_EXAMPLES, artExampleRequest } from './artExamples.js';
import { samplePlot2D } from './plot2D.js';
import { buildSurface } from './surfaceEngine.js';
import { parametricCurve } from './expressionSyntax.js';
import { parseIntentLocally } from './chatIntent.js';
describe('sampled math art recipes', () => {
 it('samples all 2D artwork layers as finite visible curves', () => {
  for (const demo of [ART_EXAMPLES.butterfly,ART_EXAMPLES.flower]) for(const layer of demo.layers){
   const points=samplePlot2D(layer.equation,demo.bounds).filter(p=>p.x!==null&&p.y!==null);
   expect(points.length).toBeGreaterThan(100);
   expect(points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))).toBe(true);
   expect(Math.max(...points.map(p=>p.x))-Math.min(...points.map(p=>p.x))).toBeGreaterThan(0.1);
  }
 });
 it('renders a nondegenerate ribbon and a closed trefoil centreline',()=>{
  const vertices=buildSurface(ART_EXAMPLES.knot.layers[0].equation,4,24);
  expect(vertices.length).toBeGreaterThan(1000); expect([...vertices].every(Number.isFinite)).toBe(true);
  const points=parametricCurve(ART_EXAMPLES.knot.layers[1].equation,3).filter(Boolean);
  expect(points.length).toBeGreaterThan(100);
  expect(Math.hypot(...points[0].map((v,i)=>v-points.at(-1)[i]))).toBeLessThan(1e-8);
  expect(Math.max(...points.map(p=>p[2]))).toBeGreaterThan(.59);
  expect(Math.min(...points.map(p=>p[2]))).toBeLessThan(-.59);
 });
 it('uses atomic scene tools and does not plot explanations',()=>{
  const execute=vi.fn(()=>({success:true,message:'Accepted'}));
  parseIntentLocally('draw a butterfly',execute);expect(execute.mock.calls[0][0]).toBe('add2DScene');
  parseIntentLocally('create a trefoil ribbon',execute);expect(execute.mock.calls[1][0]).toBe('addScene');
  expect(artExampleRequest('explain a butterfly')).toBeNull();
 });
 it('preserves failed tool outcomes',()=>{
  const result=parseIntentLocally('draw a flower',()=>({success:false,message:'Invalid layer'}));
  expect(result.actions[0].success).toBe(false);expect(result.text).toBe('Invalid layer');
 });
});
