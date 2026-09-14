import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import { createSessionStore, startBridge } from './session.js';

test('append, edit, isolate, remove and validate graph state atomically', () => {
    const store = createSessionStore();
    const graph = store.create();
    store.upsert({ graph_id: graph.id, expression_id: 'sphere', equation: 'x^2+y^2+z^2=9' });
    store.upsert({ graph_id: graph.id, equation: 'z=0' });
    const edited = store.upsert({ graph_id: graph.id, expression_id: 'sphere', equation: 'x^2+y^2+z^2=4' });
    assert.equal(edited.expressions.length, 2);
    assert.equal(edited.revision, 3);
    edited.expressions.length = 0;
    assert.equal(store.get(graph.id).expressions.length, 2);
    assert.throws(() => store.setView({ graph_id: graph.id, bounds: { xMin: 5, xMax: 1, yMin: -1, yMax: 1 } }));
    assert.equal(store.get(graph.id).revision, 3);
    assert.equal(store.remove({ graph_id: graph.id, expression_id: 'sphere' }).expressions.length, 1);
    assert.equal(store.create().expressions.length, 0);
});

test('bridge enforces capability and exact origin; streams revisions', async t => {
    const store = createSessionStore();
    const graph = store.create();
    const bridge = await startBridge(store, 'http://localhost:5173');
    t.after(() => bridge.close());
    const session = JSON.parse(new URLSearchParams(new URL(bridge.url(graph.id)).hash.slice(1)).get('graphlySession'));
    const url = `${session.endpoint}/events?graph=${graph.id}&token=${session.token}`;
    assert.equal((await fetch(url)).status, 403);
    assert.equal((await fetch(url, { headers: { Origin: 'http://evil.example' } })).status, 403);
    assert.equal((await fetch(url.replace(session.token, 'bad'), { headers: { Origin: 'http://localhost:5173' } })).status, 403);
    const abort = new AbortController();
    t.after(() => abort.abort());
    const response = await fetch(url, { headers: { Origin: 'http://localhost:5173' }, signal: abort.signal });
    assert.equal(response.status, 200);
    const reader = response.body.getReader();
    const initial = new TextDecoder().decode((await reader.read()).value);
    assert.match(initial, /"revision":0/);
    store.upsert({ graph_id: graph.id, equation: 'z=x^2+y^2' });
    const next = new TextDecoder().decode((await reader.read()).value);
    assert.match(next, /"revision":1/);
    await reader.cancel();
});

test('stdio handles handshake, invalid tool arguments and malformed JSON without hanging', async () => {
    const child = spawn(process.execPath, [new URL('./index.js', import.meta.url).pathname], { stdio: ['pipe', 'pipe', 'pipe'] });
    const messages = [];
    child.stderr.on('data', data => process.stderr.write(data));
    readline.createInterface({ input: child.stdout }).on('line', line => messages.push(JSON.parse(line)));
    const done = new Promise(resolve => child.on('close', resolve));
    child.stdin.end([
        JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize' }),
        JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'plot_function', arguments: {} } }),
        'invalid',
        JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'tools/list' })
    ].join('\n'));
    await done;
    assert.equal(messages[0].result.serverInfo.version, '1.1.0');
    assert.equal(messages[1].result.isError, true);
    assert.equal(messages[2].error.code, -32700);
    assert(messages[3].result.tools.some(tool => tool.name === 'upsert_expression'));
});

test('export preserves multiple equations and view without local credentials', async t => {
    const child = spawn(process.execPath, [new URL('./index.js', import.meta.url).pathname], { stdio: ['pipe', 'pipe', 'pipe'] });
    t.after(() => child.kill());
    const pending = new Map();
    readline.createInterface({ input: child.stdout }).on('line', line => {
        const message = JSON.parse(line);
        pending.get(message.id)?.(message.result);
        pending.delete(message.id);
    });
    let nextId = 0;
    const call = (name, args) => new Promise(resolve => {
        const id = ++nextId;
        pending.set(id, resolve);
        child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } }) + '\n');
    });
    const created = await call('create_graph', { dimension: '3d', title: 'Ellipsoid and plane' });
    assert.equal(created.isError, undefined);
    const graph_id = created.structuredContent.graph.id;
    await call('upsert_expression', { graph_id, equation: 'x^2/9+y^2/4+z^2=1' });
    await call('upsert_expression', { graph_id, equation: 'z=0' });
    await call('set_view', { graph_id, bounds: { xMin: -4, xMax: 4, yMin: -4, yMax: 4, zMin: -2, zMax: 2 } });
    const exported = await call('export_graph', { graph_id });
    const url = new URL(exported.structuredContent.url);
    assert.equal(url.hash, '');
    const state = JSON.parse(Buffer.from(url.searchParams.get('state'), 'base64').toString());
    assert.equal(state.mode, '3d');
    assert.equal(state.expressions.length, 2);
    assert.equal(state.title, 'Ellipsoid and plane');
    assert.equal(state.view.bounds.zMin, -2);
    assert.equal(JSON.stringify(state).includes('token'), false);
    assert.equal(JSON.stringify(state).includes('endpoint'), false);
    assert.deepEqual(state, exported.structuredContent.state);
});
