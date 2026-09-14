import { describe, it, expect, vi } from 'vitest';
import { runChatTools } from './chatToolLoop.js';
import { parsePointTable, validateAttachment } from './chatAttachments.js';
import { prepareScene } from './sceneTools.js';
import { parseIntentLocally } from './chatIntent.js';
import { buildSurface, buildSurfaceNormals } from './surfaceEngine.js';

describe('compositional assistant',()=>{
 it('sends failures back so the model can repair a formula',async()=>{
  const response = calls=>({functionCalls:()=>calls,text:()=> 'Done'});
  const chat={sendMessage:vi.fn().mockResolvedValueOnce({response:response([{name:'switchTo3D',args:{expression:'bad'}}])}).mockResolvedValueOnce({response:response([{name:'switchTo3D',args:{expression:'z=x'}}])}).mockResolvedValueOnce({response:response([])})};
  const exec=vi.fn().mockReturnValueOnce({success:false,message:'Invalid equation'}).mockReturnValueOnce({success:true,message:'Added',layerId:'a'});
  const result=await runChatTools(chat,'planes',exec);
  expect(result.actions.map(a=>a.success)).toEqual([false,true]);
  expect(chat.sendMessage.mock.calls[1][0][0].functionResponse.response.message).toBe('Invalid equation');
 });
 it('keeps applied mutations without replay after network failure',async()=>{
  const chat={sendMessage:vi.fn().mockResolvedValueOnce({response:{functionCalls:()=>[{name:'plot',args:{}}]}}).mockRejectedValueOnce(new Error('offline'))};
  const exec=vi.fn(()=>({success:true,message:'Added'}));
  const result=await runChatTools(chat,'go',exec); expect(exec).toHaveBeenCalledTimes(1); expect(result.actions).toHaveLength(1);
 });
 it('constructs two intersecting planes offline',()=>{
  const exec=vi.fn(()=>({success:true,message:'Added'})); parseIntentLocally('plot 2 planes intersecting',exec);
  expect(exec.mock.calls.map(c=>c[1].expression)).toEqual(['z = x','z = -x']);
  exec.mockClear(); parseIntentLocally('explain two intersecting planes',exec); expect(exec).not.toHaveBeenCalled();
 });
 it('validates an entire scene before generating additions',()=>{
  expect(()=>prepareScene([{equation:'z=x'},{equation:'bad'}])).toThrow();
  const scene=prepareScene([{equation:'z=x'},{equation:'z=-x'}]); expect(scene[0].id).not.toBe(scene[1].id);
 });
});
describe('file inputs',()=>{
 it('supports quoted columns and rejects missing/nonfinite data',()=>{
  expect(parsePointTable('"x","y"\n1,2\n3,4')).toEqual([{x:1,y:2},{x:3,y:4}]);
  expect(()=>parsePointTable('x,y\n,2')).toThrow(); expect(()=>parsePointTable('x,y\n1,Infinity')).toThrow();
  expect(parsePointTable('x\ty\n1\t2','\t')).toEqual([{x:1,y:2}]);
 });
 it('rejects oversized and unsupported uploads before reading',()=>{
  expect(()=>validateAttachment({name:'a.csv',type:'text/csv',size:300000})).toThrow();
  expect(()=>validateAttachment({name:'app.exe',type:'',size:10})).toThrow();
 });
});
it('gradient normals on a sphere are unit length and point outwards',()=>{
 const vertices=buildSurface('x^2+y^2+z^2=4',3,12); const normals=buildSurfaceNormals('x^2+y^2+z^2=4',vertices);
 for(let i=0;i<vertices.length;i+=3){expect(Math.hypot(...normals.slice(i,i+3))).toBeCloseTo(1,5); expect(vertices[i]*normals[i]+vertices[i+1]*normals[i+1]+vertices[i+2]*normals[i+2]).toBeGreaterThan(1.9);}
});
