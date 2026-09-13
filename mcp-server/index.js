#!/usr/bin/env node

/**
 * Graphly MCP Server
 * Exposes plotting tools via standard Model Context Protocol (MCP) JSON-RPC over stdio.
 * Generates shareable Graphly links that users click to view interactive graphs.
 */

import readline from 'readline';

const BASE_URL = process.env.GRAPHLY_BASE_URL || "https://graphly.netlify.app";

function encodeState(state) {
    const jsonStr = JSON.stringify(state);
    return Buffer.from(jsonStr).toString('base64');
}

function getShareableUrl(state) {
    const encoded = encodeState(state);
    return `${BASE_URL}/?state=${encodeURIComponent(encoded)}`;
}

// Tool definitions schema
const TOOLS = [
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
        description: "Generate a shareable Graphly URL to view an interactive 3D surface z = f(x, y) with OrbitControls.",
        inputSchema: {
            type: "object",
            properties: {
                expression: {
                    type: "string",
                    description: "The 3D surface expression in terms of x and y, e.g. 'sin(x) * cos(y)', '(x^2 - y^2)/4'"
                }
            },
            required: ["expression"]
        }
    }
];

// Handle MCP Tool Calls
function callTool(name, args) {
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
                    text: `Graphly 3D Surface Created (z = ${args.expression}): [View 3D Surface in Graphly](${url})\nDirect URL: ${url}`
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

rl.on('line', (line) => {
    if (!line.trim()) return;

    try {
        const msg = JSON.parse(line);
        const { id, method, params } = msg;

        // MCP Handshake
        if (method === 'initialize') {
            const response = {
                jsonrpc: "2.0",
                id,
                result: {
                    protocolVersion: "2024-11-05",
                    capabilities: {
                        tools: {}
                    },
                    serverInfo: {
                        name: "graphly-mcp-server",
                        version: "1.0.0"
                    }
                }
            };
            process.stdout.write(JSON.stringify(response) + '\n');
            return;
        }

        // List Tools
        if (method === 'tools/list') {
            const response = {
                jsonrpc: "2.0",
                id,
                result: {
                    tools: TOOLS
                }
            };
            process.stdout.write(JSON.stringify(response) + '\n');
            return;
        }

        // Call Tool
        if (method === 'tools/call') {
            const { name, arguments: toolArgs } = params;
            const toolResult = callTool(name, toolArgs || {});
            const response = {
                jsonrpc: "2.0",
                id,
                result: toolResult
            };
            process.stdout.write(JSON.stringify(response) + '\n');
            return;
        }

        // Default response for notifications or unhandled methods
        if (id !== undefined) {
            process.stdout.write(JSON.stringify({
                jsonrpc: "2.0",
                id,
                error: { code: -32601, message: `Method not found: ${method}` }
            }) + '\n');
        }
    } catch (e) {
        process.stderr.write(`Error processing line: ${e.message}\n`);
    }
});
