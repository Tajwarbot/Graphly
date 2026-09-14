import { derivative, simplify, parse } from 'mathjs';
import { scalar, normalizeMath, splitEquation } from './expressionSyntax.js';

/** Bounded scalar symbolic operations, not a general computer algebra solver. */
export function symbolicMath({ operation, expression, variable = 'x' }) {
    if (!/^[a-zA-Z]$/.test(variable)) throw new Error('Choose a single-letter variable.');
    const input = normalizeMath(expression);
    if (input.length > 1000 || !input) throw new Error('Use 1–1,000 characters for symbolic calculations.');
    if (/[{}]/.test(input)) throw new Error('Symbolic piecewise calculations are not supported yet.');
    const parts = splitEquation(input);
    if (parts.length > 2) throw new Error('Use one equality sign.');
    const source = parts.length === 2 ? `(${parts[0]})-(${parts[1]})` : input;
    const ast = parse(source);
    // Symbolic constants are intentionally limited to single letters. This keeps
    // the plotting parser's allowlist in force while allowing expressions such
    // as a*x^2 + b*y to be simplified or partially differentiated.
    const symbols = new Set([variable]);
    const builtins = new Set(['pi', 'e', 'tau', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'sqrt', 'cbrt', 'abs', 'exp', 'log', 'log10', 'log2', 'min', 'max', 'floor', 'ceil', 'round', 'sign']);
    ast.traverse((node) => {
        if (!node.isSymbolNode) return;
        if (/^[a-zA-Z]$/.test(node.name)) symbols.add(node.name);
        else if (!builtins.has(node.name))
            throw new Error('Use single-letter symbolic constants.');
    });
    scalar(source, [...symbols]); // Same AST allowlist as the plotting engine, before any CAS operation.
    if (operation === 'differentiate' || operation === 'derivative') {
        if (parts.length !== 1) throw new Error('Differentiate an expression, not an equation.');
        return { operation: 'differentiate', variable, expression: derivative(source, variable).toString() };
    }
    if (operation === 'simplify') {
        if (parts.length !== 1) throw new Error('Simplify an expression, not an equation.');
        return { operation, variable, expression: simplify(source).toString(), note: 'Simplification may remove removable singularities; retain the original domain.' };
    }
    if (operation === 'integrate' || operation === 'integral') {
        if (parts.length !== 1) throw new Error('Integrate an expression, not an equation.');
        const maxDegree = 8;
        const add = (a, b, sign = '+') => Array.from({ length: Math.max(a.length, b.length) }, (_, i) =>
            simplify(`(${a[i] || '0'}) ${sign} (${b[i] || '0'})`).toString());
        const multiply = (a, b) => {
            if (a.length + b.length - 2 > maxDegree) throw new Error(`Integration supports polynomials of degree at most ${maxDegree}.`);
            const result = Array(a.length + b.length - 1).fill('0');
            a.forEach((left, i) => b.forEach((right, j) => {
                result[i + j] = simplify(`(${result[i + j]}) + ((${left}) * (${right}))`).toString();
            }));
            return result;
        };
        const power = (base, exponent) => {
            let result = ['1'];
            for (let i = 0; i < exponent; i++) result = multiply(result, base);
            return result;
        };
        function polynomial(node) {
            if (node.isParenthesisNode) return polynomial(node.content);
            if (node.isConstantNode) return [node.toString()];
            if (node.isSymbolNode) return node.name === variable ? ['0', '1'] : [node.toString()];
            if (!node.isOperatorNode) throw new Error('Integration supports polynomial expressions only.');
            const left = polynomial(node.args[0]);
            if (node.args.length === 1) return node.op === '-' ? left.map(value => simplify(`-(${value})`).toString()) : left;
            const right = polynomial(node.args[1]);
            if (node.op === '+') return add(left, right);
            if (node.op === '-') return add(left, right, '-');
            if (node.op === '*') return multiply(left, right);
            if (node.op === '/') {
                if (right.length !== 1) throw new Error('Integration supports polynomial expressions only.');
                return left.map(value => simplify(`(${value}) / (${right[0]})`).toString());
            }
            if (node.op === '^') {
                if (!node.args[1].isConstantNode || !Number.isInteger(Number(node.args[1].value)) || Number(node.args[1].value) < 0)
                    throw new Error('Integration supports non-negative integer polynomial powers only.');
                return power(left, Number(node.args[1].value));
            }
            throw new Error('Integration supports polynomial expressions only.');
        }
        const coefficients = polynomial(ast);
        const terms = coefficients.map((coefficient, degree) => {
            const factor = simplify(`(${coefficient}) / (${degree + 1})`).toString();
            const exponent = degree + 1;
            return `(${factor}) * ${variable}${exponent === 1 ? '' : `^${exponent}`}`;
        });
        return {
            operation: 'integrate',
            variable,
            expression: simplify(terms.join(' + ')).toString(),
            note: 'This is an antiderivative; add an arbitrary constant C. It is valid on the original expression’s domain.'
        };
    }
    if (operation !== 'solve') throw new Error('Choose simplify, differentiate, integrate, or solve.');
    // Extract coefficients structurally; never fit sampled values or accidentally
    // interpret a trigonometric expression as a polynomial.
    const add = (a, b, sign = 1) => Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] || 0) + sign * (b[i] || 0));
    const multiply = (a, b) => {
        if (a.length + b.length > 4) throw new Error('Solving supports polynomials of degree at most two.');
        const c = Array(a.length + b.length - 1).fill(0);
        a.forEach((x, i) => b.forEach((y, j) => { c[i + j] += x * y; })); return c;
    };
    function coefficients(node) {
        if (node.isParenthesisNode) return coefficients(node.content);
        if (node.isConstantNode) return [Number(node.value)];
        if (node.isSymbolNode && node.name === variable) return [0, 1];
        if (node.isSymbolNode) throw new Error('Solving requires numeric polynomial coefficients.');
        if (!node.isOperatorNode) throw new Error('Solving supports only linear and quadratic polynomials.');
        const a = coefficients(node.args[0]);
        if (node.args.length === 1) return node.op === '-' ? a.map(v => -v) : a;
        const b = coefficients(node.args[1]);
        if (node.op === '+') return add(a, b);
        if (node.op === '-') return add(a, b, -1);
        if (node.op === '*') return multiply(a, b);
        if (node.op === '/' && b.length === 1 && b[0] !== 0) return a.map(v => v / b[0]);
        if (node.op === '^' && b.length === 1 && [0, 1, 2].includes(b[0])) return b[0] === 0 ? [1] : b[0] === 1 ? a : multiply(a, a);
        throw new Error('Solving supports polynomial degree ≤ 2 with constant nonzero denominators.');
    }
    const c = coefficients(ast);
    if (!c.every(Number.isFinite)) throw new Error('Coefficients exceed the supported numeric range.');
    while (c.length > 1 && c.at(-1) === 0) c.pop();
    if (c.length === 1) return { operation, variable, solutions: [], solutionSet: c[0] === 0 ? 'all real numbers' : 'no solutions' };
    const exact = value => simplify(value).toString();
    if (c.length === 2) return { operation, variable, solutions: [exact(`-(${c[0]})/(${c[1]})`)], approximate: [-c[0] / c[1]] };
    const [constant, b, a] = c, discriminant = b * b - 4 * a * constant;
    if (!Number.isFinite(discriminant)) throw new Error('Discriminant exceeds the supported numeric range.');
    if (discriminant < 0) return { operation, variable, solutions: [], solutionSet: 'no real solutions', note: 'Complex roots are not supported.' };
    const signs = discriminant === 0 ? [1] : [-1, 1];
    return { operation, variable, solutions: signs.map(sign => `(-(${b}) ${sign < 0 ? '-' : '+'} sqrt(${discriminant})) / (${2 * a})`), approximate: signs.map(sign => (-b + sign * Math.sqrt(discriminant)) / (2 * a)) };
}
