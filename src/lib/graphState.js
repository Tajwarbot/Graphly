export const GRAPH_COLORS = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED', '#0891B2'];
const uniqueId = () => globalThis.crypto?.randomUUID?.() ?? `graph-${Date.now()}-${Math.random().toString(36).slice(2)}`;
export function makeSurface(equation, index = 0) {
    return { id: uniqueId(), name: `Surface ${index + 1}`, equation, color: GRAPH_COLORS[index % GRAPH_COLORS.length], visible: true, opacity: 1, wireframeMode: 'solid' };
}
export function appendDataset(graph, dataset) {
    const current = graph ?? { title: 'Untitled Graph', datasets: [], globalConfig: { showGrid: true, enableZoom: true, xAxisLabel: 'X', yAxisLabel: 'Y', aspectRatio: 'auto', showLabels: false }, annotations: [], createdAt: new Date().toISOString() };
    return { ...current, datasets: [...(current.datasets ?? []), { ...dataset, color: GRAPH_COLORS[(current.datasets?.length ?? 0) % GRAPH_COLORS.length] }] };
}
export function makeDataset({ equation, name, rows }) {
    return { id: uniqueId(), name: name || equation, equation, data: rows ?? [], visible: true, config: { type: equation ? 'function' : 'scatter', xKey: 'x', yKey: 'y', showTrendline: !equation, trendlineType: 'linear', trendlineColor: '#ef4444' } };
}
