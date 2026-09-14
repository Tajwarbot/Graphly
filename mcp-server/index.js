#!/usr/bin/env node

/**
 * Graphly MCP Server
 * Exposes plotting tools via standard Model Context Protocol (MCP) JSON-RPC over stdio.
 * Generates shareable Graphly links that users click to view interactive graphs.
 */

import readline from 'readline';
import { createSessionStore, startBridge } from './session.js';

const BASE_URL = process.env.GRAPHLY_BASE_URL || "https://graphly.netlify.app";

function encodeState(state) {
    const jsonStr = JSON.stringify(state);
    return Buffer.from(jsonStr).toString('base64');
}

function getShareableUrl(state) {
    const encoded = encodeState(state);
    return `${BASE_URL}/?state=${encodeURIComponent(encoded)}`;
}

const store = createSessionStore();
let bridgePromise;
const sessionResult = async graph => {
    bridgePromise ||= startBridge(store, BASE_URL).catch(error => { bridgePromise = undefined; throw error; });
    const bridge = await bridgePromise;
    const url = bridge.url(graph.id);
    return { content: [{ type: 'text', text: `Graph accepted (revision ${graph.revision}). Open the live viewer: ${url}. Rendering is not verified by the server.` }], structuredContent: { graph, url } };
};
const graphId = { type: 'string', description: 'ID returned by create_graph' };
const expressionId = { type: 'string', description: 'Stable ID; omit to append a new expression, reuse only for an intentional edit' };
const tool = (name, description, properties, required) => ({ name, description, inputSchema: { type: 'object', properties, required, additionalProperties: false } });
const SESSION_TOOLS = [
    tool('create_graph', 'Create a live graph session, retained for the lifetime of this server. Open its URL once, then update by graph_id.', { dimension: { type: 'string', enum: ['2d', '3d'] }, title: { type: 'string' } }, []),
    tool('upsert_expression', 'Append an equation to a live graph. Reuse expression_id only when editing an existing equation. 3D supports implicit equations such as x^2+y^2+z^2=9.', { graph_id: graphId, expression_id: expressionId, equation: { type: 'string' }, color: { type: 'string', pattern: '^#[0-9a-fA-F]{6}$' }, visible: { type: 'boolean' } }, ['graph_id', 'equation']),
    tool('remove_expression', 'Remove one expression from a live graph.', { graph_id: graphId, expression_id: expressionId }, ['graph_id', 'expression_id']),
    tool('export_graph', 'Create a durable share link containing every expression and the view. No local session token is included.', { graph_id: graphId }, ['graph_id']),
    tool('get_graph', 'Read authoritative graph state. Accepted means stored, not proof of a successful render.', { graph_id: graphId }, ['graph_id']),
    tool('set_view', 'Update graph bounds or 3D camera. Camera uses mathematical x/y/z coordinates.', { graph_id: graphId, bounds: { type: 'object', properties: Object.fromEntries(['xMin','xMax','yMin','yMax','zMin','zMax'].map(k => [k, { type: 'number' }])), required: ['xMin','xMax','yMin','yMax'], additionalProperties: false }, camera: { type: 'object', properties: Object.fromEntries(['position','target'].map(k => [k, { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 }])), required: ['position','target'], additionalProperties: false } }, ['graph_id'])
];

// Tool definitions schema
const TOOLS = [...SESSION_TOOLS,
    {
        name: "plot_function",
        description: "Generate a shareable Graphly URL to view an explicit 2D mathematical curve y = f(x).",
        inputSchema: {
            type: "object",
            properties: {
                expression: {
                    type: "string",
                    description: "The mathematical expression to plot, e.g. 'sin(x)', 'x^2 - 4', '1/x', '2x + 1'"
                },
                xMin: { type: "number", description: "Optional minimum X bound" },
                xMax: { type: "number", description: "Optional maximum X bound" }
            },
            required: ["expression"]
        }
    },
    {
        name: "plot_data_table",
        description: "Generate a shareable Graphly URL to view and analyze a tabular dataset with scatter plotting and regression.",
        inputSchema: {
            type: "object",
            properties: {
                name: {
                    type: "string",
                    description: "Name of the dataset, e.g. 'Experimental Growth'"
                },
                rows: {
                    type: "array",
                    description: "Array of coordinate objects [{ x: 1, y: 2 }, { x: 2, y: 4 }]",
                    items: {
                        type: "object",
                        properties: {
                            x: { type: "number" },
                            y: { type: "number" }
                        },
                        required: ["x", "y"]
                    }
                }
            },
            required: ["rows"]
        }
    },
    {
        name: "plot_3d_surface",
        description: "Generate a shareable Graphly URL for a 3D explicit surface or implicit equation such as x^2+y^2+z^2=9. Requires the updated viewer.",
        inputSchema: {
            type: "object",
            properties: {
                expression: {
                    type: "string",
                    description: "An explicit expression or complete implicit equation, e.g. 'sin(x)*cos(y)' or 'x^2/9+y^2/4+z^2=1'"
                }
            },
            required: ["expression"]
        }
    }
];

// Handle MCP Tool Calls
async function callTool(name, args) {
    if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Tool arguments must be an object');
    if (name === 'create_graph') return sessionResult(store.create(args));
    if (name === 'upsert_expression') return sessionResult(store.upsert(args));
    if (name === 'remove_expression') return sessionResult(store.remove(args));
    if (name === 'export_graph') {
        const graph = store.get(args.graph_id);
        const state = { mode: graph.dimension === '2d' ? 'function' : '3d', expressions: graph.expressions, view: graph.view, title: graph.title };
        const url = getShareableUrl(state);
        return { content: [{ type: 'text', text: `Graph snapshot: ${url}` }], structuredContent: { state, url } };
    }
    if (name === 'get_graph') return sessionResult(store.get(args.graph_id));
    if (name === 'set_view') return sessionResult(store.setView(args));
    if (['plot_function', 'plot_3d_surface'].includes(name) && (typeof args.expression !== 'string' || !args.expression.trim() || args.expression.length > 4096)) throw new Error('Provide a nonempty expression (maximum 4096 characters)');
    if (name === 'plot_data_table' && (!Array.isArray(args.rows) || args.rows.length > 10000 || !args.rows.every(row => row && Number.isFinite(row.x) && Number.isFinite(row.y)))) throw new Error('rows must contain at most 10000 finite numeric x/y points');
    if (name === 'plot_function' && (args.xMin !== undefined || args.xMax !== undefined) && (!Number.isFinite(args.xMin) || !Number.isFinite(args.xMax) || args.xMin >= args.xMax)) throw new Error('Provide finite xMin < xMax');
    if (name === "plot_function") {
        const state = {
            mode: "function",
            expression: args.expression,
            viewportBounds: args.xMin !== undefined && args.xMax !== undefined ? {
                xMin: args.xMin,
                xMax: args.xMax,
                yMin: args.xMin,
                yMax: args.xMax
            } : undefined
        };
        const url = getShareableUrl(state);
        return {
            content: [
                {
                    type: "text",
                    text: `Graphly 2D Plot Created: [View Plot in Graphly](${url})\nDirect URL: ${url}`
                }
            ]
        };
    }

    if (name === "plot_data_table") {
        const state = {
            mode: "data",
            name: args.name || "Dataset",
            rows: args.rows
        };
        const url = getShareableUrl(state);
        return {
            content: [
                {
                    type: "text",
                    text: `Graphly Data Table Created with ${args.rows.length} points: [View Scatter & Regression in Graphly](${url})\nDirect URL: ${url}`
                }
            ]
        };
    }

    if (name === "plot_3d_surface") {
        const state = {
            mode: "3d",
            expression: args.expression
        };
        const url = getShareableUrl(state);
        return {
            content: [
                {
                    type: "text",
                    text: `Graphly 3D Surface Created (${args.expression}): [View 3D Surface in Graphly](${url})\nDirect URL: ${url}`
                }
            ]
        };
    }

    throw new Error(`Tool not found: ${name}`);
}

// JSON-RPC stdio loop
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

let queue = Promise.resolve();
const send = message => process.stdout.write(JSON.stringify(message) + '\n');
async function handle(line) {
    if (!line.trim()) return;
    let msg;
    try { msg = JSON.parse(line); }
    catch { send({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } }); return; }
    if (!msg || msg.jsonrpc !== '2.0' || typeof msg.method !== 'string') { send({ jsonrpc: '2.0', id: msg?.id ?? null, error: { code: -32600, message: 'Invalid request' } }); return; }
    const { id, method, params } = msg;
    if (id === undefined) return;
    let result;
    if (method === 'initialize') result = { protocolVersion: '2024-11-05', capabilities: { tools: {} }, serverInfo: { name: 'graphly-mcp-server', version: '1.1.0' } };
    else if (method === 'ping') result = {};
    else if (method === 'tools/list') result = { tools: TOOLS };
    else if (method === 'tools/call') {
        try { result = await callTool(params?.name, params?.arguments || {}); }
        catch (error) { result = { isError: true, content: [{ type: 'text', text: error.message }] }; }
    } else { send({ jsonrpc: '2.0', id, error: { code: -32601, message: `Method not found: ${method}` } }); return; }
    send({ jsonrpc: '2.0', id, result });
}
rl.on('line', line => { queue = queue.then(() => handle(line)).catch(error => process.stderr.write(`${error.message}\n`)); });
rl.on('close', () => { queue.then(async () => { if (bridgePromise) (await bridgePromise).close(); }).catch(() => {}); });
