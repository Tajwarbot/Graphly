import { compileParametric, splitRestrictions, compileRestrictions, allowedBy, scalar, splitEquation } from './expressionSyntax.js';
import { compileInequality, isInequality } from './inequalityRegions.js';

export const calculateNiceTicks = (min, max, maxTicks = 8) => {
    if (typeof min !== 'number' || typeof max !== 'number') return [];
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
    if (min === max) return [];
    if (min > max) {
        const tmp = min;
        min = max;
        max = tmp;
    }
    const range = max - min;
    if (range <= 0 || !Number.isFinite(range)) return [min];
    const roughStep = range / Math.max(1, maxTicks - 1);
    if (!Number.isFinite(roughStep) || roughStep <= 0) return [min];

    const exponent = Math.floor(Math.log10(roughStep));
    if (!Number.isFinite(exponent)) return [min];

    const fraction = roughStep / Math.pow(10, exponent);
    let niceFraction;
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;

    const step = niceFraction * Math.pow(10, exponent);
    if (!Number.isFinite(step) || step <= 0) return [min];

    // Precision boundary check: if step is smaller than IEEE-754 precision at min
    if (min + step === min || max - step === max) {
        return [min, max];
    }

    const start = Math.ceil(min / step) * step;
    const end = Math.floor(max / step) * step;
    const ticks = [];
    const epsilon = step * 1e-6;
    const maxAllowedTicks = 50;
    let count = 0;

    for (let t = start; t <= end + epsilon && count < maxAllowedTicks; t += step) {
        // Avoid floating point inaccuracies like 0.30000000000000004
        const cleanT = exponent < 0 && exponent > -12 
            ? parseFloat(t.toFixed(Math.abs(exponent) + 2)) 
            : t;
        ticks.push(cleanT);
        count++;
    }
    return ticks;
};

export const formatNumber = (num) => {
    if (typeof num !== 'number') return num;
    if (!Number.isFinite(num)) return isNaN(num) ? '—' : String(num);
    if (num === 0) return 0;
    const abs = Math.abs(num);
    if (abs >= 10000 || abs < 0.001) {
        return num.toExponential(2);
    }
    return parseFloat(num.toFixed(3));
};

export const formatEquationNumber = (num) => {
    if (typeof num !== 'number' || !Number.isFinite(num)) return String(num);
    if (Math.abs(num) < 0.001 && num !== 0) return num.toExponential(2);
    return num.toFixed(3);
};

export const getRegressionParams = (points, type = 'linear') => {
    const validPoints = points.filter(p => p && Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x);
    const n = validPoints.length;
    if (validPoints.length < 2 || validPoints[0].x === validPoints[n - 1].x) return null;

    let r2 = null;

    const calculateR2 = (predictedY, actualY) => {
        const yMean = actualY.reduce((a, b) => a + b, 0) / actualY.length;
        const ssRes = actualY.reduce((sum, y, i) => sum + Math.pow(y - predictedY[i], 2), 0);
        const ssTot = actualY.reduce((sum, y) => sum + Math.pow(y - yMean, 2), 0);
        return ssTot === 0 ? (ssRes === 0 ? 1 : 0) : 1 - (ssRes / ssTot);
    };

    const det3x3 = (m) => {
        return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };

    if (type === 'linear') {
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        validPoints.forEach(p => { sumX += p.x; sumY += p.y; sumXY += p.x * p.y; sumX2 += p.x * p.x; });
        const denominator = n * sumX2 - sumX * sumX;
        if (denominator <= 0 || !Number.isFinite(denominator)) return null;
        const slope = (n * sumXY - sumX * sumY) / denominator;
        const intercept = (sumY - slope * sumX) / n;
        if (![slope, intercept].every(Number.isFinite)) return null;

        const preds = validPoints.map(p => slope * p.x + intercept);
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, slope, intercept, r2, equation: `y = ${formatEquationNumber(slope)}x + ${formatEquationNumber(intercept)}` };
    }
    else if (type === 'quadratic') {
        let s00 = n, s10 = 0, s20 = 0, s30 = 0, s40 = 0, s01 = 0, s11 = 0, s21 = 0;
        validPoints.forEach(p => {
            const x = p.x; const y = p.y;
            s10 += x; s20 += x * x; s30 += x * x * x; s40 += x * x * x * x;
            s01 += y; s11 += x * y; s21 += x * x * y;
        });
        const M = [[s00, s10, s20], [s10, s20, s30], [s20, s30, s40]];
        const Det = det3x3(M);
        if (Det === 0 || !Number.isFinite(Det)) return null;
        const c = det3x3([[s01, s10, s20], [s11, s20, s30], [s21, s30, s40]]) / Det;
        const b = det3x3([[s00, s01, s20], [s10, s11, s30], [s20, s21, s40]]) / Det;
        const a = det3x3([[s00, s10, s01], [s10, s20, s11], [s20, s30, s21]]) / Det;

        if (![a, b, c].every(Number.isFinite)) return null;
        const preds = validPoints.map(p => a * p.x * p.x + b * p.x + c);
        r2 = calculateR2(preds, validPoints.map(p => p.y));

        return { type, a, b, c, r2, equation: `y = ${formatEquationNumber(a)}x² + ${formatEquationNumber(b)}x + ${formatEquationNumber(c)}` };
    }
    else if (type === 'exponential') {
        const v = validPoints.filter(p => p.y > 0);
        if (v.length < 2 || v[0].x === v[v.length - 1].x) return null;
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        const N = v.length;
        v.forEach(p => {
            const lny = Math.log(p.y);
            sumX += p.x; sumY += lny; sumXY += p.x * lny; sumX2 += p.x * p.x;
        });
        const denominator = N * sumX2 - sumX * sumX;
        if (denominator <= 0 || !Number.isFinite(denominator)) return null;
        const b = (N * sumXY - sumX * sumY) / denominator;
        const a = Math.exp((sumY - b * sumX) / N);

        if (![a, b].every(Number.isFinite)) return null;
        const preds = v.map(p => a * Math.exp(b * p.x));
        r2 = calculateR2(preds, v.map(p => p.y));

        return { type, a, b, r2, equation: `y = ${formatEquationNumber(a)}e^(${formatEquationNumber(b)}x)` };
    }
    else if (type === 'power') {
        const v = validPoints.filter(p => p.x > 0 && p.y > 0);
        if (v.length < 2 || v[0].x === v[v.length - 1].x) return null;
        let sumlnX = 0, sumlnY = 0, sumlnXlnY = 0, sumlnX2 = 0;
        const N = v.length;
        v.forEach(p => {
            const lnx = Math.log(p.x);
            const lny = Math.log(p.y);
            sumlnX += lnx; sumlnY += lny; sumlnXlnY += lnx * lny; sumlnX2 += lnx * lnx;
        });
        const denominator = N * sumlnX2 - sumlnX * sumlnX;
        if (denominator <= 0 || !Number.isFinite(denominator)) return null;
        const b = (N * sumlnXlnY - sumlnX * sumlnY) / denominator;
        const a = Math.exp((sumlnY - b * sumlnX) / N);

        if (![a, b].every(Number.isFinite)) return null;
        const preds = v.map(p => a * Math.pow(p.x, b));
        r2 = calculateR2(preds, v.map(p => p.y));

        return { type, a, b, r2, equation: `y = ${formatEquationNumber(a)}x^${formatEquationNumber(b)}` };
    }
    else if (type === 'logarithmic') {
        const v = validPoints.filter(p => p.x > 0);
        if (v.length < 2 || v[0].x === v[v.length - 1].x) return null;
        let sumLnX = 0, sumY = 0, sumLnXY = 0, sumLnX2 = 0;
        const N = v.length;
        v.forEach(p => {
            const lnx = Math.log(p.x);
            sumLnX += lnx; sumY += p.y; sumLnXY += lnx * p.y; sumLnX2 += lnx * lnx;
        });
        const denominator = N * sumLnX2 - sumLnX * sumLnX;
        if (denominator <= 0 || !Number.isFinite(denominator)) return null;
        const b = (N * sumLnXY - sumLnX * sumY) / denominator;
        const a = (sumY - b * sumLnX) / N;

        if (![a, b].every(Number.isFinite)) return null;
        const preds = v.map(p => a + b * Math.log(p.x));
        r2 = calculateR2(preds, v.map(p => p.y));

        return { type, a, b, r2, equation: `y = ${formatEquationNumber(a)} + ${formatEquationNumber(b)}ln(x)` };
    }
    return null;
};

export const generateTrendlineData = (params, xMin, xMax, yMinData, yMaxData) => {
    if (!params || xMin === 'auto' || xMax === 'auto') return [];
    const points = [];
    const resolution = 150;
    const step = (xMax - xMin) / (resolution - 1);
    const yRange = Math.abs(yMaxData - yMinData) || 10;
    const SAFE_MAX = yMaxData + (yRange * 5);
    const SAFE_MIN = yMinData - (yRange * 5);

    for (let i = 0; i < resolution; i++) {
        const x = xMin + (i * step);
        let y = null;
        if (params.type === 'linear') y = params.slope * x + params.intercept;
        else if (params.type === 'quadratic') y = params.a * x * x + params.b * x + params.c;
        else if (params.type === 'exponential') y = params.a * Math.exp(params.b * x);
        else if (params.type === 'power') y = params.a * Math.pow(x, params.b);
        else if (params.type === 'logarithmic') {
            if (x > 0) y = params.a + params.b * Math.log(x);
        }

        if (y !== null && !isNaN(y)) {
            if (y > SAFE_MAX) y = SAFE_MAX;
            if (y < SAFE_MIN) y = SAFE_MIN;
            points.push({ x, y });
        }
    }
    return points;
};

export const calculateStats = (data, xKey, yKey) => {
    if (!data) return { meanX: 0, meanY: 0, stdDevX: 0, stdDevY: 0, n: 0 };
    const validData = data.filter(d => Number.isFinite(parseFloat(d[xKey])) && Number.isFinite(parseFloat(d[yKey])));
    const n = validData.length;
    if (n === 0) return { meanX: 0, meanY: 0, stdDevX: 0, stdDevY: 0, n: 0 };

    const sumX = validData.reduce((acc, val) => acc + parseFloat(val[xKey]), 0);
    const sumY = validData.reduce((acc, val) => acc + parseFloat(val[yKey]), 0);
    const meanX = sumX / n;
    const meanY = sumY / n;

    const sumSqDiffX = validData.reduce((acc, val) => acc + Math.pow(parseFloat(val[xKey]) - meanX, 2), 0);
    const sumSqDiffY = validData.reduce((acc, val) => acc + Math.pow(parseFloat(val[yKey]) - meanY, 2), 0);

    const stdDevX = Math.sqrt(sumSqDiffX / n);
    const stdDevY = Math.sqrt(sumSqDiffY / n);

    return { meanX, meanY, stdDevX, stdDevY, n };
};

// ==========================================
// AST PARSER & EVALUATOR
// ==========================================

const FUNCTIONS = {
    sin: Math.sin, cos: Math.cos, tan: Math.tan,
    asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
    sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
    sec: x => 1 / Math.cos(x),
    csc: x => 1 / Math.sin(x),
    cot: x => 1 / Math.tan(x),
    exp: Math.exp,
    ln: Math.log,
    log: Math.log10,
    log10: Math.log10,
    log2: Math.log2,
    sqrt: Math.sqrt,
    cbrt: Math.cbrt,
    abs: Math.abs,
    ceil: Math.ceil,
    floor: Math.floor,
    round: Math.round,
    sign: Math.sign,
    max: Math.max,
    min: Math.min
};

const CONSTANTS = {
    pi: Math.PI,
    e: Math.E,
    tau: Math.PI * 2
};

export class ParseError extends Error {
    constructor(message, position) {
        super(`${message} at position ${position}`);
        this.position = position;
        this.name = 'ParseError';
    }
}

export function tokenize(input) {
    const tokens = [];
    let i = 0;
    while (i < input.length) {
        let char = input[i];
        if (/\s/.test(char)) {
            i++;
            continue;
        }

        if (/[a-zA-Zθ]/.test(char)) {
            let start = i;
            while (i < input.length && /[a-zA-Z0-9θ_]/.test(input[i])) {
                i++;
            }
            tokens.push({ type: 'IDENT', value: input.substring(start, i), pos: start });
            continue;
        }

        if (/[0-9.]/.test(char)) {
            let start = i;
            let dotCount = char === '.' ? 1 : 0;
            let hasExp = false;
            i++;
            while (i < input.length) {
                let c = input[i];
                if (c === '.') {
                    dotCount++;
                    if (dotCount > 1) throw new ParseError("Invalid number format", i);
                    i++;
                } else if ((c === 'e' || c === 'E') && !hasExp) {
                    hasExp = true;
                    i++;
                    if (input[i] === '+' || input[i] === '-') i++;
                } else if (/[0-9]/.test(c)) {
                    i++;
                } else {
                    break;
                }
            }
            tokens.push({ type: 'NUMBER', value: parseFloat(input.substring(start, i)), pos: start });
            continue;
        }

        if ('+-*/^'.includes(char)) {
            tokens.push({ type: 'OP', value: char, pos: i });
            i++;
            continue;
        }

        if (char === '(') {
            tokens.push({ type: 'LPAREN', value: '(', pos: i });
            i++;
            continue;
        }
        if (char === ')') {
            tokens.push({ type: 'RPAREN', value: ')', pos: i });
            i++;
            continue;
        }
        if (char === ',') {
            tokens.push({ type: 'COMMA', value: ',', pos: i });
            i++;
            continue;
        }

        throw new ParseError(`Invalid character '${char}'`, i);
    }

    const processed = [];
    for (let i = 0; i < tokens.length; i++) {
        const curr = tokens[i];
        if (processed.length > 0) {
            const prev = processed[processed.length - 1];
            const prevIsValue = prev.type === 'NUMBER' || prev.type === 'RPAREN' || 
                (prev.type === 'IDENT' && !Object.prototype.hasOwnProperty.call(FUNCTIONS, prev.value));
            const currIsValueOrParen = curr.type === 'NUMBER' || curr.type === 'IDENT' || curr.type === 'LPAREN';
            
            if (prevIsValue && currIsValueOrParen) {
                processed.push({ type: 'OP', value: '*', pos: curr.pos, implicit: true });
            }
        }
        processed.push(curr);
    }

    processed.push({ type: 'EOF', value: '', pos: input.length });
    return processed;
}

const PRECEDENCE = {
    '+': 10,
    '-': 10,
    '*': 20,
    '/': 20,
    '^': 30
};

export function parse(tokens) {
    let current = 0;

    function peek() {
        return tokens[current];
    }

    function consume(type) {
        if (peek().type === type) {
            return tokens[current++];
        }
        throw new ParseError(`Expected ${type} but got ${peek().type}`, peek().pos);
    }

    function parsePrimary() {
        const token = peek();

        if (token.type === 'NUMBER') {
            consume('NUMBER');
            return { type: 'number', value: token.value };
        }

        if (token.type === 'IDENT') {
            consume('IDENT');
            if (peek().type === 'LPAREN') {
                consume('LPAREN');
                const args = [];
                if (peek().type !== 'RPAREN') {
                    args.push(parseExpression(0));
                    while (peek().type === 'COMMA') {
                        consume('COMMA');
                        args.push(parseExpression(0));
                    }
                }
                consume('RPAREN');
                return { type: 'call', name: token.value, args, pos: token.pos };
            } else {
                if (Object.prototype.hasOwnProperty.call(CONSTANTS, token.value)) {
                    return { type: 'constant', name: token.value };
                }
                const validVars = ['x', 'y', 't', 'r', 'θ'];
                if (!validVars.includes(token.value)) {
                    throw new ParseError(`Unknown identifier '${token.value}'`, token.pos);
                }
                return { type: 'variable', name: token.value };
            }
        }

        if (token.type === 'LPAREN') {
            consume('LPAREN');
            const expr = parseExpression(0);
            consume('RPAREN');
            return expr;
        }

        if (token.type === 'OP' && token.value === '-') {
            consume('OP');
            const expr = parseExpression(25);
            return { type: 'unary', op: '-', expr };
        }
        if (token.type === 'OP' && token.value === '+') {
            consume('OP');
            return parseExpression(25);
        }

        throw new ParseError(`Unexpected token ${token.value || token.type}`, token.pos);
    }

    function parseExpression(minPrecedence) {
        let left = parsePrimary();

        while (peek().type === 'OP') {
            const opToken = peek();
            const prec = PRECEDENCE[opToken.value];
            if (prec === undefined || prec < minPrecedence) {
                break;
            }
            
            consume('OP');
            const nextMinPrec = opToken.value === '^' ? prec : prec + 1;
            const right = parseExpression(nextMinPrec);

            left = { type: 'binary', op: opToken.value, left, right };
        }

        return left;
    }

    const ast = parseExpression(0);
    if (peek().type !== 'EOF') {
        throw new ParseError(`Unexpected token ${peek().value}`, peek().pos);
    }
    return ast;
}

export function compileAST(ast) {
    if (ast.type === 'number') {
        const val = ast.value;
        return () => val;
    }
    if (ast.type === 'constant') {
        const val = CONSTANTS[ast.name];
        return () => val;
    }
    if (ast.type === 'variable') {
        const name = ast.name;
        return (scope) => {
            const val = scope[name];
            if (val === undefined) throw new ParseError(`Undefined variable ${name}`, 0);
            return val;
        };
    }
    if (ast.type === 'unary') {
        const exprFn = compileAST(ast.expr);
        if (ast.op === '-') return (scope) => -exprFn(scope);
        return exprFn;
    }
    if (ast.type === 'binary') {
        const leftFn = compileAST(ast.left);
        const rightFn = compileAST(ast.right);
        const op = ast.op;
        if (op === '+') return (s) => leftFn(s) + rightFn(s);
        if (op === '-') return (s) => leftFn(s) - rightFn(s);
        if (op === '*') return (s) => leftFn(s) * rightFn(s);
        if (op === '/') return (s) => leftFn(s) / rightFn(s);
        if (op === '^') return (s) => Math.pow(leftFn(s), rightFn(s));
    }
    if (ast.type === 'call') {
        const fn = FUNCTIONS[ast.name];
        if (!fn) throw new ParseError(`Unknown function ${ast.name}`, ast.pos || 0);
        const argFns = ast.args.map(compileAST);
        return (scope) => {
            const args = argFns.map(f => f(scope));
            return fn(...args);
        };
    }
}

const compileCache = new Map();

export function compileEquation(equation) {
    if (compileCache.has(equation)) {
        return compileCache.get(equation);
    }
    const tokens = tokenize(typeof equation === 'string' ? equation.replace(/²/g, '^2').replace(/³/g, '^3').replace(/−/g, '-') : equation);
    const ast = parse(tokens);
    const fn = compileAST(ast);
    
    if (compileCache.size > 500) {
        const keys = compileCache.keys();
        for (let i = 0; i < 250; i++) {
            compileCache.delete(keys.next().value);
        }
    }
    compileCache.set(equation, fn);
    return fn;
}

const mathjsCache = new Map();

export function compileMathFunction(expression) {
    if (!expression || typeof expression !== 'string') return null;
    const raw = expression.trim().replace(/²/g, '^2').replace(/³/g, '^3').replace(/−/g, '-');
    try {
        const parametric=compileParametric(raw,2);
        if(parametric){const fn=t=>parametric.point({t});fn.type='parametric';return fn;}
        const {base,restrictions}=splitRestrictions(raw);
        if(isInequality(base)){const region=compileInequality(base);const fn=(x,y)=>region.contains(x,y);fn.type='inequality';return fn;}
        if(restrictions.length){const original=compileMathFunction(base);if(!original)return null;const fields=compileRestrictions(restrictions,['x','y']);const fn=(x,y)=>{const result=original(x,y);return allowedBy(fields,{x,y:original.type==='explicit'?result:y})?result:NaN;};fn.type=original.type;return fn;}
    } catch {return null;}
    if (!raw) return null;

    if (mathjsCache.has(raw)) {
        return mathjsCache.get(raw);
    }

    try {
        let isImplicit = false;
        let evaluateExpr = raw;

        if (splitEquation(raw).length > 1) {
            const parts = splitEquation(raw);
            if (parts.length !== 2) return null;
            if (parts.length === 2) {
                const left = parts[0].trim();
                const right = parts[1].trim();
                // Check if it's explicit y = f(x)
                if ((left === 'y' || left === 'f(x)') && !/\by\b/.test(right)) {
                    isImplicit = false;
                    evaluateExpr = right;
                } else {
                    // Implicit equation: L(x, y) - (R(x, y)) = 0
                    isImplicit = true;
                    evaluateExpr = `(${left}) - (${right})`;
                }
            }
        } else {
            // No '=': if it contains 'y' as a standalone variable, treat as implicit F(x, y) = 0
            if (/\by\b/.test(raw)) {
                isImplicit = true;
                evaluateExpr = raw;
            } else {
                isImplicit = false;
                evaluateExpr = raw;
            }
        }

        const compiled = { evaluate: scalar(evaluateExpr,['x','y']) };

        // Callable function with attached metadata
        const callable = isImplicit
            ? (x, y) => {
                try {
                    const val = compiled.evaluate({ x, y });
                    return typeof val === 'number' && Number.isFinite(val) ? val : NaN;
                } catch {
                    return NaN;
                }
            }
            : (x) => {
                try {
                    const val = compiled.evaluate({ x });
                    return typeof val === 'number' && Number.isFinite(val) ? val : NaN;
                } catch {
                    return NaN;
                }
            };

        callable.type = isImplicit ? 'implicit' : 'explicit';
        callable.rawExpression = raw;
        callable.evaluateExpr = evaluateExpr;
        callable.fn = callable;

        if (mathjsCache.size > 200) {
            const keys = mathjsCache.keys();
            for (let i = 0; i < 100; i++) {
                mathjsCache.delete(keys.next().value);
            }
        }
        mathjsCache.set(raw, callable);
        return callable;
    } catch {
        return null;
    }
}

export const generateFunctionPoints = (equation, xMin = -10, xMax = 10, resolution = 200) => {
    try {
        const points = [];
        if (equation.includes('{')) {
            const fn = compileMathFunction(equation);
            if (!fn || fn.type !== 'explicit') throw new Error('Enter an explicit piecewise function.');
            const count = Math.min(2000, Math.max(20, resolution));
            for (let i = 0; i <= count; i++) {
                const x = xMin + (xMax - xMin) * i / count, y = fn(x);
                points.push({ x, y: Number.isFinite(y) ? y : null });
            }
            return points;
        }
        // Preserve the bounded expression parser and its positional errors.
        // Strip only an explicit dependent-variable prefix, never an implicit equation.
        let expression = equation;
        const explicit = typeof equation === 'string' && equation.match(/^\s*(?:y|f\(x\))\s*=\s*(.+)$/);
        if (explicit && !/[=]|\by\b/.test(explicit[1])) expression = explicit[1];
        const fn = compileEquation(expression);
        
        if (xMin === xMax) {
            const y = fn({ x: xMin });
            if (Number.isFinite(y)) {
                points.push({ x: xMin, y });
            } else {
                points.push({ x: xMin, y: null });
            }
            return points;
        }

        let prevY = null;
        const dx = (xMax - xMin) / resolution;

        for (let i = 0; i <= resolution; i++) {
            const x = xMin + i * dx;
            let y = fn({ x });

            if (!Number.isFinite(y)) {
                points.push({ x, y: null });
                prevY = null;
                continue;
            }

            if (prevY !== null) {
                const dy = y - prevY;
                const slope = dy / dx;

                const signFlip = (y > 0 && prevY < 0) || (y < 0 && prevY > 0);
                if (signFlip && Math.abs(dy) > 20) {
                    points.push({ x: x - dx / 2, y: null });
                } else if (Math.abs(slope) > 1e6 && Math.abs(dy) > 50) {
                    points.push({ x: x - dx / 2, y: null });
                }
            }

            points.push({ x, y });
            prevY = y;
        }
        return points;
    } catch (e) {
        const arr = [];
        arr.error = e;
        return arr;
    }
};
