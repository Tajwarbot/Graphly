import { scalar, normalizeMath, splitEquation, splitConditions } from './expressionSyntax.js';
const reserved = new Set(['x','y','z','t','u','v','e','pi','tau','constructor','prototype','__proto__','sin','cos','tan','sqrt','abs','log','exp','min','max']);
export function functionDefinition(equation) {
    const parts=splitEquation(normalizeMath(equation));
    if(parts.length!==2) return null;
    const match=parts[0].trim().match(/^([A-Za-z][A-Za-z0-9_]*)\(([^()]*)\)$/);
    if(!match || reserved.has(match[1])) return null;
    const args=match[2].split(',').map(s=>s.trim());
    if(!args.length || args.length>3 || new Set(args).size!==args.length || args.some(a=>!/^\w+$/.test(a)||['constructor','prototype','__proto__'].includes(a))) throw new Error('Use one to three distinct function arguments.');
    return {name:match[1],args,expression:parts[1]};
}
export function parameterDefinition(equation) {
    const parts=splitEquation(normalizeMath(equation));
    if(parts.length!==2 || !/^[A-Za-z][A-Za-z0-9_]*$/.test(parts[0].trim()) || reserved.has(parts[0].trim())) return null;
    return {name:parts[0].trim(),expression:parts[1]};
}
export function isDefinition(equation) { try { return !!(functionDefinition(equation)||parameterDefinition(equation)); } catch { return true; } }
function listLiteral(text) {
    if(!text.startsWith('[')||!text.endsWith(']')) return null;
    const body=text.slice(1,-1), range=body.match(/^(.+?)\.\.\.(.+)$/);
    if(range) {
        const start=scalar(range[1],[])({}), end=scalar(range[2],[])({});
        if(!Number.isInteger(start)||!Number.isInteger(end)||Math.abs(end-start)>99) throw new Error('List ranges need integer endpoints and at most 100 items.');
        return Array.from({length:Math.abs(end-start)+1},(_,i)=>String(start+i*(end>=start?1:-1)));
    }
    const items=splitConditions(body);
    if(!body.trim()||items.length>100) throw new Error('Lists need 1–100 items.');
    return items;
}
export function expressionEnvironment(layers) {
    const definitions=new Map(), functions=new Map(), values={}, lists=new Map(), errors=Object.create(null);
    for(const layer of layers) {
        let definition;
        try { definition=functionDefinition(layer.equation)||parameterDefinition(layer.equation); } catch(error) {errors[layer.id || 'definition']=error.message;continue;}
        if(!definition) continue;
        if(definitions.has(definition.name)||functions.has(definition.name)) errors[definition.name]='Duplicate definition.';
        (definition.args?functions:definitions).set(definition.name,definition);
    }
    let argumentId=0;
    function inline(text,stack=[]) {
        if(text.length>24000 || stack.length>16) throw new Error('Function expansion is too complex.');
        let output='',cursor=0;
        const calls=/\b([A-Za-z][A-Za-z0-9_]*)\s*\(/g;
        for(let match;(match=calls.exec(text));) {
            const definition=functions.get(match[1]); if(!definition) continue;
            if(stack.includes(definition.name)) throw new Error('Recursive function definitions are not supported.');
            let end=calls.lastIndex,depth=1;
            for(;end<text.length && depth;end++){if(text[end]==='(')depth++;if(text[end]===')')depth--;}
            if(depth) throw new Error('Close the function arguments.');
            const args=splitConditions(text.slice(calls.lastIndex,end-1)).map(arg=>inline(arg,stack));
            if(args.length!==definition.args.length) throw new Error(`${definition.name} needs ${definition.args.length} arguments.`);
            const aliases=new Map(definition.args.map(name=>[name,`functionArgumentInternal${argumentId++}`]));
            const bindings=new Map(definition.args.map((name,i)=>[aliases.get(name),args[i]]));
            // Rename bound arguments before expanding callees, so a caller's argument
            // cannot capture a free global parameter inside another function.
            let body=definition.expression.replace(/\b[A-Za-z][A-Za-z0-9_]*\b/g,name=>aliases.get(name)||name);
            body=inline(body,[...stack,definition.name]);
            body=body.replace(/\b[A-Za-z][A-Za-z0-9_]*\b/g,name=>bindings.has(name)?`(${bindings.get(name)})`:name);
            output+=text.slice(cursor,match.index)+`(${body})`;cursor=end;calls.lastIndex=end;
        }
        const expanded=output+text.slice(cursor);
        if(expanded.length>24000)throw new Error('Function expansion is too complex.');
        return expanded;
    }
    const pending=new Set(definitions.keys());
    for(let pass=0;pass<definitions.size;pass++) for(const name of pending) {
        if(errors[name])continue;
        try {
            const text=inline(definitions.get(name).expression).trim(), items=listLiteral(text);
            const evaluate=expr=>scalar(expr,[...definitions.keys()])(values);
            const result=items?items.map(evaluate):evaluate(text);
            if(items?result.every(Number.isFinite):Number.isFinite(result)) {
                if(items)lists.set(name,result);else values[name]=result;
                pending.delete(name);
            }
        } catch(error){errors[name]=error.message;}
    }
    for(const name of pending) errors[name] ||= 'Undefined or circular parameter dependency.';
    for(const [name,definition] of functions) {
        try { scalar(inline(definition.expression,[name]),[...definition.args,...definitions.keys()]); } catch(error) {errors[name]=error.message;}
    }
    function expandAll(equation) {
        if (/(listInternal|functionArgumentInternal)\d/.test(equation)) throw new Error('Reserved internal symbol.');
        let text=inline(normalizeMath(equation));
        const vectors=[];
        text=text.replace(/\b[A-Za-z][A-Za-z0-9_]*\b/g,name=>{
            if(errors[name])throw new Error(`${name}: ${errors[name]}`);
            if(lists.has(name)){const index=vectors.length;vectors.push(lists.get(name));return `listInternal${index}`;}
            return Object.hasOwn(values,name)?`(${values[name]})`:name;
        });
        // One-based constant list indexing is handled before broadcasting.
        text=text.replace(/listInternal(\d+)\[([^\]]+)\]/g,(_,index,position)=>{
            const n=scalar(position,[])({}); if(!Number.isInteger(n)||n<1||n>vectors[index].length)throw new Error('List index is outside the one-based range.');
            return `(${vectors[index][n-1]})`;
        });
        text=text.replace(/\[[^\]]*\]/g,literal=>{
            const vector=listLiteral(literal).map(item=>scalar(item,[])({}));
            if(!vector.every(Number.isFinite))throw new Error('Lists need finite numeric entries.');
            const index=vectors.length;vectors.push(vector);return `listInternal${index}`;
        });
        const used=[...text.matchAll(/listInternal(\d+)/g)].map(m=>vectors[Number(m[1])]);
        const length=used[0]?.length || 1;
        if(used.some(vector=>vector.length!==length))throw new Error('Combined lists must have equal lengths.');
        return Array.from({length},(_,i)=>text.replace(/listInternal(\d+)/g,(_,index)=>`(${vectors[Number(index)][i]})`));
    }
    return {values,lists,errors,expandAll,plotExpressions(equation,dimension) {
        const definition=functionDefinition(equation);
        if(definition) {
            if(errors[definition.name])throw new Error(errors[definition.name]);
            if(definition.args.length!==dimension-1) return [];
            return expandAll(`${definition.name}(${dimension===2?'x':'x,y'})`);
        }
        if(parameterDefinition(equation))return [];
        return expandAll(equation);
    },expand(equation){const variants=expandAll(equation);if(variants.length!==1)throw new Error('Use list-aware plotting for this expression.');return variants[0];}};
}
