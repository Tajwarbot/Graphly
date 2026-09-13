/**
 * Graphly Shareable Plot URL Generator Endpoint
 * Compatible with Netlify Functions, Vercel Serverless, and Node.js Express.
 */

export function encodeGraphState(state) {
    const jsonStr = JSON.stringify(state);
    return Buffer.from(jsonStr).toString('base64');
}

export function generatePlotUrl(state, baseUrl = "https://graphly.netlify.app") {
    const encoded = encodeGraphState(state);
    return `${baseUrl}/?state=${encodeURIComponent(encoded)}`;
}

/**
 * Serverless handler
 */
export default async function handler(req, res) {
    try {
        const data = req.method === 'POST' ? req.body : req.query;

        if (!data || Object.keys(data).length === 0) {
            const errResp = { success: false, error: "Missing plot specification. Provide mode and expression or rows." };
            if (res?.status) return res.status(400).json(errResp);
            return errResp;
        }

        const state = {
            mode: data.mode || 'function',
            expression: data.expression || data.expr || undefined,
            rows: data.rows || undefined,
            name: data.name || undefined,
            viewportBounds: data.viewportBounds || undefined
        };

        const shareableUrl = generatePlotUrl(state);

        const responsePayload = {
            success: true,
            url: shareableUrl,
            state
        };

        if (res?.status) {
            return res.status(200).json(responsePayload);
        }
        return responsePayload;
    } catch (err) {
        const errorPayload = { success: false, error: err.message };
        if (res?.status) return res.status(500).json(errorPayload);
        return errorPayload;
    }
}
