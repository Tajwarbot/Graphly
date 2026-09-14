import { tupleComponents, splitRestrictions, splitEquation, splitConditions } from '../lib/expressionSyntax.js';
import { parse } from 'mathjs';

// React creates MathML nodes directly: expression text never becomes HTML.
function renderNode(node, key = 'root') {
    if (node.isConstantNode) return <mn key={key}>{String(node.value)}</mn>;
    if (node.isSymbolNode) return <mi key={key}>{node.name === 'pi' ? 'π' : node.name}</mi>;
    if (node.isParenthesisNode) return <mrow key={key}><mo>(</mo>{renderNode(node.content)}<mo>)</mo></mrow>;
    if (node.isOperatorNode) {
        const args = node.args.map((arg, i) => renderNode(arg, i));
        if (node.op === '^') return <msup key={key}>{args}</msup>;
        if (node.op === '/') return <mfrac key={key}>{args}</mfrac>;
        const op = ({ '*': '·', '<=': '≤', '>=': '≥', '==': '=' })[node.op] || node.op;
        return <mrow key={key}>{args.length === 1 ? <><mo>{op}</mo>{args}</> : args.flatMap((arg, i) => i ? [<mo key={`op${i}`}>{op}</mo>, arg] : [arg])}</mrow>;
    }
    if (node.isFunctionNode) {
        if (node.fn.name === 'sqrt') return <msqrt key={key}>{renderNode(node.args[0])}</msqrt>;
        return <mrow key={key}><mi>{node.fn.name}</mi><mo>(</mo>{node.args.flatMap((arg, i) => i ? [<mo key={`comma${i}`}>,</mo>, renderNode(arg, i)] : [renderNode(arg, i)])}<mo>)</mo></mrow>;
    }
    throw new Error('Display fallback');
}
function renderExpression(expression, key = 'expression') {
    const text = expression.trim();
    const {base,restrictions} = splitRestrictions(text);
    if (restrictions.length) return <mrow key={key}>{renderExpression(base)}{restrictions.map((r,i)=><mrow key={i}><mo>{'{'} </mo>{renderExpression(r)}<mo>{'}'}</mo></mrow>)}</mrow>;
    const sides = splitEquation(text);
    if (sides.length === 2) return <mrow key={key}>{renderExpression(sides[0])}<mo>=</mo>{renderExpression(sides[1])}</mrow>;
    if (text.startsWith('{') && text.endsWith('}')) {
        const branches = splitConditions(text.slice(1,-1));
        return <mrow key={key}><mo stretchy="true">{'{'}</mo><mtable columnalign="left left">{branches.map((branch,i)=>{
            let depth=0,colon=-1;
            for(let j=0;j<branch.length;j++){if('({'.includes(branch[j]))depth++;if(')}'.includes(branch[j]))depth--;if(branch[j]===':'&&depth===0){colon=j;break;}}
            return <mtr key={i}><mtd>{renderExpression(colon<0?branch:branch.slice(colon+1))}</mtd><mtd>{colon<0?<mtext>otherwise</mtext>:renderExpression(branch.slice(0,colon))}</mtd></mtr>;
        })}</mtable></mrow>;
    }
    const tuple=tupleComponents(text);
    if(tuple) return <mrow key={key}><mo>(</mo>{tuple.flatMap((part,i)=>i?[<mo key={`comma${i}`}>,</mo>,renderExpression(part,i)]:[renderExpression(part,i)])}<mo>)</mo></mrow>;
    return renderNode(parse(text),key);
}
export function MathExpression({ expression, className = '' }) {
    let content;
    try { content=renderExpression(String(expression)); }
    catch { content=<mtext>{expression}</mtext>; }
    return <math className={className} aria-label={expression} style={{fontSize:'1em'}}><mrow>{content}</mrow></math>;
}
