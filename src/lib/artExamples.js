// Original compositions from standard parametric techniques; no contest art copied.
// https://help.desmos.com/hc/en-us/articles/4406809622541-Math-Art
// https://help.desmos.com/hc/en-us/articles/4406906208397-Parametric-Equations
const radius = '(exp(cos(t))-2*cos(4*t)-sin(t/12)^5)';
export const ART_EXAMPLES = {
 butterfly: {dimension:'2d',layers:[{name:'Butterfly wings',equation:`(sin(t)*${radius},cos(t)*${radius}){0<=t<=12*pi}`,color:'#7c3aed'}],bounds:{xMin:-4,xMax:4,yMin:-4,yMax:4}},
 flower: {dimension:'2d',layers:[
  {name:'Outer petals',equation:'(2*cos(5*t)*cos(t),2*cos(5*t)*sin(t)){0<=t<=2*pi}',color:'#db2777'},
  {name:'Inner petals',equation:'(1.1*cos(5*t)*cos(t+0.3),1.1*cos(5*t)*sin(t+0.3)){0<=t<=2*pi}',color:'#f59e0b'},
  {name:'Stem',equation:'(0.15*sin(t),-t){0<=t<=4}',color:'#15803d'},
  {name:'Leaf',equation:'(0.6+0.65*cos(t),-2+0.3*sin(t)){0<=t<=2*pi}',color:'#16a34a'}
 ],bounds:{xMin:-3,xMax:3,yMin:-4.5,yMax:2.5}},
 knot: {dimension:'3d',layers:[
  {name:'Trefoil ribbon',equation:'((2+(0.6+v)*cos(3*u))*cos(2*u),(2+(0.6+v)*cos(3*u))*sin(2*u),(0.6+v)*sin(3*u)){0<=u<=2*pi}{-0.12<=v<=0.12}',color:'#2563eb',opacity:0.9},
  {name:'Knot centreline',equation:'((2+0.6*cos(3*t))*cos(2*t),(2+0.6*cos(3*t))*sin(2*t),0.6*sin(3*t)){0<=t<=2*pi}',color:'#f59e0b'}
 ],bounds:{xMin:-3.5,xMax:3.5,yMin:-3.5,yMax:3.5,zMin:-2,zMax:2},camera:{position:[7,-9,6],target:[0,0,0]}}
};
export function artExampleRequest(prompt) {
 if (!/\b(plot|draw|create|make|build|graph|show)\b/i.test(prompt)) return null;
 const key = /\bbutterfly\b/i.test(prompt)?'butterfly':/\b(torus knot|trefoil|knot ribbon)\b/i.test(prompt)?'knot':/\b(flower|rosette)\b/i.test(prompt)?'flower':null;
 return key ? ART_EXAMPLES[key] : null;
}
