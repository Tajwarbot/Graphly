import {
    buildSurface,
    buildSurfaceNormals,
    compileSurface
} from './surfaceEngine.js';
import { parametricCurve } from './expressionSyntax.js';
self.onmessage = ({ data }) => {
    const results = data.surfaces.map((surface) => {
        try {
            const spec = compileSurface(surface.equation);
            if (spec.parametric && spec.parametric.kind !== 'surface') {
                const points = parametricCurve(surface.equation, 3),
                    segments = [];
                if (spec.parametric.kind === 'point')
                    segments.push(...points.filter(Boolean).flat());
                else
                    for (let i = 1; i < points.length; i++)
                        if (points[i - 1] && points[i])
                            segments.push(...points[i - 1], ...points[i]);
                return {
                    id: surface.id,
                    vertices: new Float32Array(segments),
                    kind: spec.parametric.kind
                };
            }
            const vertices = buildSurface(
                surface.equation,
                data.range,
                data.resolution
            );
            return {
                id: surface.id,
                vertices,
                normals:
                    spec.parametric || spec.inequality
                        ? undefined
                        : buildSurfaceNormals(surface.equation, vertices),
                inequality: spec.inequality
            };
        } catch (error) {
            return { id: surface.id, error: error.message };
        }
    });
    self.postMessage(
        results,
        results.flatMap((r) =>
            [r.vertices?.buffer, r.normals?.buffer].filter(Boolean)
        )
    );
};
