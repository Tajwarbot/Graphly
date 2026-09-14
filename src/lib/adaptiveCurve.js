/** Refine finite curves by midpoint error; break unresolved jumps instead of joining them. */
export function adaptiveCurve(evaluate, min, max, intervals=200, budget=12000) {
    const output=[];
    const finite=p=>Array.isArray(p)&&p.every(Number.isFinite);
    const emit=p=>{if(output.length<budget)output.push(p);};
    function segment(a,b,pa,pb,depth) {
        if(output.length>=budget)return;
        const mid=(a+b)/2, pm=evaluate(mid);
        if(!finite(pa)||!finite(pb)||!finite(pm)){emit(null);emit(finite(pb)?pb:null);return;}
        const error=Math.hypot(...pm.map((v,i)=>v-(pa[i]+pb[i])/2));
        const scale=Math.max(1,...pa.map(Math.abs),...pb.map(Math.abs));
        if(error>0.002*scale) {
            if(depth<8){segment(a,mid,pa,pm,depth+1);segment(mid,b,pm,pb,depth+1);return;}
            emit(null);
        }
        emit(pb);
    }
    let a=min,pa=evaluate(a);emit(finite(pa)?pa:null);
    for(let i=1;i<=intervals&&output.length<budget;i++) {
        const b=min+(max-min)*i/intervals,pb=evaluate(b);
        segment(a,b,pa,pb,0);a=b;pa=pb;
    }
    return output;
}
