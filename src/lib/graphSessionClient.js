/** Subscribe to an explicitly opened local MCP graph. The capability token stays in the URL fragment. */
export function connectGraphSession({ onGraph, onError = () => {} }) {
    const params = new URLSearchParams(window.location.hash.slice(1));
    const raw = params.get('graphlySession');
    if (!raw) return () => {};
    let source;
    try {
        const session = JSON.parse(raw);
        const endpoint = new URL(session.endpoint);
        if (endpoint.protocol !== 'http:' || endpoint.hostname !== '127.0.0.1' || !endpoint.port || endpoint.username || endpoint.password || !/^[a-f0-9]{64}$/.test(session.token) || typeof session.graphId !== 'string') throw new Error('Invalid local graph session link');
        endpoint.pathname = '/events';
        endpoint.search = new URLSearchParams({ token: session.token, graph: session.graphId });
        source = new EventSource(endpoint.href);
        let revision = -1;
        source.onmessage = event => {
            try {
                const graph = JSON.parse(event.data);
                if (graph.id !== session.graphId || !['2d', '3d'].includes(graph.dimension) || !Array.isArray(graph.expressions) || !Number.isInteger(graph.revision)) throw new Error('Invalid graph snapshot');
                if (graph.revision <= revision) return;
                onGraph(graph); revision = graph.revision;
            } catch (error) { onError(error); }
        };
        source.onerror = () => onError(new Error('Live graph disconnected. Keep the MCP server running and allow local network access; the viewer will retry.'));
    } catch (error) { onError(error); }
    return () => source?.close();
}
