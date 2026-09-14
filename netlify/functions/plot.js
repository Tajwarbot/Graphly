/**
 * Netlify Serverless Function for Graphly AI Actions (ChatGPT & Claude Web)
 * Endpoint: POST /api/plot or GET /api/plot
 */

export async function handler(event) {
    const corsHeaders = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Content-Type': 'application/json'
    };

    // Handle CORS preflight
    if (event.httpMethod === 'OPTIONS') {
        return {
            statusCode: 200,
            headers: corsHeaders,
            body: ''
        };
    }

    try {
        let data = {};
        if (event.httpMethod === 'POST') {
            try {
                data = JSON.parse(event.body || '{}');
            } catch {
                data = {};
            }
        } else {
            data = event.queryStringParameters || {};
        }

        const mode = data.mode || (data.rows ? 'data' : 'function');
        const expression = data.expression || data.expr || data.fn || '';
        const name = data.name || (mode === '3d' ? '3D Surface' : 'Function Plot');
        const rows = data.rows || undefined;

        const stateObj = {
            mode,
            expression: expression || undefined,
            name,
            rows
        };

        const jsonStr = JSON.stringify(stateObj);
        const encodedState = Buffer.from(jsonStr).toString('base64');
        const baseUrl = 'https://graphly.netlify.app';
        const url = `${baseUrl}/?state=${encodeURIComponent(encodedState)}`;
        const previewUrl = expression
            ? `${baseUrl}/?mode=${mode}&fn=${encodeURIComponent(expression)}`
            : url;

        return {
            statusCode: 200,
            headers: corsHeaders,
            body: JSON.stringify({
                success: true,
                url,
                previewUrl,
                message: `Graphly ${mode.toUpperCase()} plot generated successfully.`
            })
        };
    } catch (err) {
        return {
            statusCode: 500,
            headers: corsHeaders,
            body: JSON.stringify({
                success: false,
                error: err.message
            })
        };
    }
}
