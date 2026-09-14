import { buildSurface } from './surfaceEngine.js';
self.onmessage = ({ data }) => {
    const results = data.surfaces.map(surface => {
        try { return { id: surface.id, vertices: buildSurface(surface.equation, data.range, data.resolution) }; }
        catch (error) { return { id: surface.id, error: error.message }; }
    });
    self.postMessage(results, results.filter(r => r.vertices).map(r => r.vertices.buffer));
};
