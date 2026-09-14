import http from 'node:http';
import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';

const text = (value, name) => {
    if (typeof value !== 'string' || !value.trim() || value.length > 4096) throw new Error(`${name} must be a nonempty string (maximum 4096 characters)`);
    return value.trim();
};
export function createSessionStore() {
    const graphs = new Map();
    const listeners = new Set();
    function get(id) { const graph = graphs.get(id); if (!graph) throw new Error('Unknown graph_id'); return structuredClone(graph); }
    function update(id, fn) {
        const graph = get(id); fn(graph); graph.revision++; graph.status = 'accepted';
        graphs.set(id, graph); for (const listener of listeners) listener(structuredClone(graph)); return get(id);
    }
    return {
        get,
        subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
        create({ dimension = '3d', title = 'Graphly graph' } = {}) {
            if (!['2d', '3d'].includes(dimension)) throw new Error('dimension must be 2d or 3d');
            if (graphs.size >= 100) throw new Error('Session limit reached (100 graphs). Restart the server to clear sessions.');
            const graph = { id: randomUUID(), dimension, title: text(title, 'title'), expressions: [], view: {}, revision: 0, status: 'accepted' };
            graphs.set(graph.id, graph); return get(graph.id);
        },
        upsert({ graph_id, expression_id, equation, color, visible = true }) {
            equation = text(equation, 'equation');
            if (expression_id !== undefined) expression_id = text(expression_id, 'expression_id');
            if (typeof visible !== 'boolean') throw new Error('visible must be boolean');
            if (color !== undefined && !/^#[0-9a-f]{6}$/i.test(color)) throw new Error('color must be a six digit hex color');
            return update(graph_id, graph => {
                const id = expression_id || randomUUID();
                const index = graph.expressions.findIndex(item => item.id === id);
                if (index < 0 && graph.expressions.length >= 100) throw new Error('Expression limit reached (100)');
                const item = { ...(index < 0 ? {} : graph.expressions[index]), id, equation, visible, ...(color ? { color } : {}) };
                if (index < 0) graph.expressions.push(item); else graph.expressions[index] = item;
            });
        },
        remove({ graph_id, expression_id }) {
            return update(graph_id, graph => { if (!graph.expressions.some(e => e.id === expression_id)) throw new Error('Unknown expression_id'); graph.expressions = graph.expressions.filter(e => e.id !== expression_id); });
        },
        setView({ graph_id, bounds, camera }) {
            if (!bounds && !camera) throw new Error('Provide bounds or camera');
            if (bounds) {
                for (const axis of ['x', 'y', ...(bounds.zMin !== undefined || bounds.zMax !== undefined ? ['z'] : [])]) {
                    if (!Number.isFinite(bounds[`${axis}Min`]) || !Number.isFinite(bounds[`${axis}Max`]) || bounds[`${axis}Min`] >= bounds[`${axis}Max`]) throw new Error(`Invalid ${axis} bounds`);
                }
                bounds = Object.fromEntries(Object.entries(bounds).filter(([key]) => /^(x|y|z)(Min|Max)$/.test(key)));
            }
            if (camera) {
                for (const key of ['position', 'target']) if (!Array.isArray(camera[key]) || camera[key].length !== 3 || !camera[key].every(Number.isFinite)) throw new Error('Camera needs finite position and target triples');
                camera = { position: camera.position, target: camera.target };
                if (camera.position.every((v, i) => v === camera.target[i])) throw new Error('Camera position must differ from target');
            }
            return update(graph_id, graph => { graph.view = { ...graph.view, ...(bounds ? { bounds } : {}), ...(camera ? { camera } : {}) }; });
        }
    };
}

export async function startBridge(store, baseUrl) {
    const origin = new URL(baseUrl).origin;
    const token = randomBytes(32).toString('hex');
    const clients = new Set();
    const server = http.createServer((req, res) => {
        if (req.headers.origin !== origin) { res.writeHead(403); res.end('Origin denied'); return; }
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Vary', 'Origin');
        if (req.method === 'OPTIONS') {
            res.setHeader('Access-Control-Allow-Methods', 'GET');
            res.setHeader('Access-Control-Allow-Private-Network', 'true');
            res.writeHead(204); res.end(); return;
        }
        const url = new URL(req.url, 'http://127.0.0.1');
        const candidate = Buffer.from(url.searchParams.get('token') || '');
        const secret = Buffer.from(token);
        if (candidate.length !== secret.length || !timingSafeEqual(candidate, secret)) { res.writeHead(403); res.end('Invalid session token'); return; }
        if (req.method !== 'GET' || url.pathname !== '/events') { res.writeHead(404); res.end(); return; }
        let graph;
        try { graph = store.get(url.searchParams.get('graph')); } catch { res.writeHead(404); res.end('Unknown graph'); return; }
        if (clients.size >= 32) { res.writeHead(429); res.end('Too many viewers'); return; }
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
        const send = value => res.write(`data: ${JSON.stringify(value)}\n\n`);
        send(graph); clients.add(res);
        const unsubscribe = store.subscribe(next => { if (next.id === graph.id) send(next); });
        const heartbeat = setInterval(() => res.write(': heartbeat\n\n'), 15000);
        req.on('close', () => { clearInterval(heartbeat); unsubscribe(); clients.delete(res); });
    });
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    return {
        url(graphId) {
            const url = new URL(baseUrl);
            url.hash = `graphlySession=${encodeURIComponent(JSON.stringify({ endpoint: `http://127.0.0.1:${server.address().port}`, token, graphId }))}`;
            return url.href;
        },
        close() { for (const client of clients) client.end(); server.close(); }
    };
}
