import { parse } from 'mathjs';

const cache = new Map();
export function compileSurface(input) {
    const raw = String(input || '').trim().replace(/²/g, '^2').replace(/³/g, '^3').replace(/−/g, '-').replace(/\bln\s*\(/g, 'log(');
    if (!raw) throw new Error('Enter an equation or expression.');
    if (cache.has(raw)) return cache.get(raw);
    const parts = raw.split('=');
    if (parts.length > 2) throw new Error('Use one equality sign per equation.');
    const implicit = parts.length === 2 && !(parts[0].trim() === 'z' && !/\bz\b/.test(parts[1]));
    const expression = implicit ? `(${parts[0]})-(${parts[1]})` : parts.length === 2 ? parts[1] : raw;
    const ast = parse(expression);
    ast.traverse(node => {
        if (node.isAssignmentNode || node.isFunctionAssignmentNode || node.isBlockNode || node.isAccessorNode || node.isArrayNode || node.isObjectNode || node.isConditionalNode || node.isRangeNode || (node.isOperatorNode && !['+', '-', '*', '/', '^'].includes(node.op))) throw new Error('Only mathematical expressions are supported.');
        if (node.isSymbolNode && !['x', 'y', 'z', 'pi', 'e', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'sqrt', 'cbrt', 'abs', 'exp', 'log', 'log10', 'log2', 'ln', 'min', 'max', 'floor', 'ceil', 'round', 'sign'].includes(node.name)) throw new Error(`Unknown symbol: ${node.name}`);
    });
    if (!implicit && /\bz\b/.test(expression)) throw new Error('Use an equation such as x^2 + y^2 + z^2 = 9.');
    const compiled = ast.compile();
    const fn = { implicit, evaluate(x, y, z = 0) {
        try { const value = compiled.evaluate({ x, y, z }); return typeof value === 'number' && Number.isFinite(value) ? value : NaN; }
        catch { return NaN; }
    }};
    if (cache.size >= 200) cache.delete(cache.keys().next().value);
    cache.set(raw, fn);
    return fn;
}

// Output is in mathematical coordinates; Three.js coordinate mapping happens in the view.
export function buildSurface(input, range = 5, resolution = 36) {
    const fn = compileSurface(input);
    const n = Math.max(8, Math.min(64, Math.round(resolution)));
    const bounds = typeof range === 'number' ? {xMin:-range,xMax:range,yMin:-range,yMax:range,zMin:-range,zMax:range} : {...range, zMin:range.zMin ?? -5, zMax:range.zMax ?? 5};
    for(const axis of ['x','y','z']) if(!Number.isFinite(bounds[`${axis}Min`]) || !Number.isFinite(bounds[`${axis}Max`]) || bounds[`${axis}Min`] >= bounds[`${axis}Max`]) throw new Error('Surface bounds must be finite and increasing.');
    const minimum=[bounds.xMin,bounds.yMin,bounds.zMin];
    const steps=[(bounds.xMax-bounds.xMin)/n,(bounds.yMax-bounds.yMin)/n,(bounds.zMax-bounds.zMin)/n];
    const step=Math.min(...steps);
    const heightLimit=Math.max(Math.abs(bounds.zMin),Math.abs(bounds.zMax))*4;
    const vertices = [];
    const push = (a,b,c) => {
        if (fn.implicit) {
            const u=b.map((v,i)=>v-a[i]), v=c.map((w,i)=>w-a[i]);
            const normal=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
            const length=Math.hypot(...normal);
            if(length<1e-12) return;
            const mid=a.map((x,i)=>(x+b[i]+c[i])/3);
            const epsilon=step*0.01;
            const plus=fn.evaluate(...mid.map((x,i)=>x+normal[i]/length*epsilon));
            const minus=fn.evaluate(...mid.map((x,i)=>x-normal[i]/length*epsilon));
            if (plus<minus) [b,c]=[c,b];
        }
        for(const p of [a,b,c]) vertices.push(...p);
    };
    if (!fn.implicit) {
        const points = [];
        for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
            const x = bounds.xMin + i * steps[0], y = bounds.yMin + j * steps[1];
            points.push([x, y, fn.evaluate(x, y)]);
        }
        const triangle = (a, b, c) => {
            if (![a,b,c].every(p => Number.isFinite(p[2]) && Math.abs(p[2]) <= heightLimit)) return;
            // Reject large jumps rather than connecting across poles.
            if (Math.max(a[2],b[2],c[2]) - Math.min(a[2],b[2],c[2]) > heightLimit / 2) return;
            push(a,b,c);
        };
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
            const k = i*(n+1)+j, a=points[k], b=points[k+n+1], c=points[k+1], d=points[k+n+2];
            triangle(a,b,c); triangle(b,d,c);
        }
    } else {
        const stride=n+1, values=new Float64Array(stride**3);
        const idx=(i,j,k)=>(i*stride+j)*stride+k;
        for(let i=0;i<=n;i++) for(let j=0;j<=n;j++) for(let k=0;k<=n;k++) values[idx(i,j,k)]=fn.evaluate(minimum[0]+i*steps[0],minimum[1]+j*steps[1],minimum[2]+k*steps[2]);
        const corners=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
        const tetrahedra=[[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]];
        for(let i=0;i<n;i++) for(let j=0;j<n;j++) for(let k=0;k<n;k++) {
            const points=corners.map(([a,b,c])=>[minimum[0]+(i+a)*steps[0],minimum[1]+(j+b)*steps[1],minimum[2]+(k+c)*steps[2]]);
            const v=corners.map(([a,b,c])=>values[idx(i+a,j+b,k+c)]);
            if (!v.every(Number.isFinite) || v.every(x=>x>=0) || v.every(x=>x<0)) continue;
            const edge=(a,b)=>{
                let lo=0,hi=1;
                // Refine the root and reject discontinuities masquerading as zero crossings.
                for(let t=0;t<12;t++) {
                    const m=(lo+hi)/2, p=points[a].map((x,d)=>x+m*(points[b][d]-x));
                    const f=fn.evaluate(...p);
                    if (!Number.isFinite(f)) return null;
                    if ((f<0)===(v[a]<0)) lo=m; else hi=m;
                }
                const m=(lo+hi)/2,p=points[a].map((x,d)=>x+m*(points[b][d]-x));
                if(Math.abs(fn.evaluate(...p)) > Math.max(1,Math.abs(v[a]),Math.abs(v[b]))*0.01) return null;
                return p;
            };
            for(const tet of tetrahedra) {
                const inside=tet.filter(a=>v[a]<0), outside=tet.filter(a=>v[a]>=0);
                if(!inside.length || !outside.length) continue;
                if(inside.length===1 || outside.length===1) {
                    const one=inside.length===1?inside:outside, many=inside.length===1?outside:inside;
                    const tri=many.map(b=>edge(one[0],b)); if(tri.every(Boolean)) push(...tri);
                } else {
                    const [a,b]=inside,[c,d]=outside;
                    const p=[edge(a,c),edge(a,d),edge(b,c),edge(b,d)];
                    if(p.every(Boolean)) {push(p[0],p[1],p[2]);push(p[1],p[3],p[2]);}
                }
            }
        }
    }
    return new Float32Array(vertices);
}
