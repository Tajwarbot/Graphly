import { compileSurface } from './surfaceEngine.js';
import { makeSurface } from './graphState.js';
export function validateView(view = {}) {
    if (view.bounds)
        for (const axis of ['x', 'y', 'z']) {
            const min = view.bounds[`${axis}Min`],
                max = view.bounds[`${axis}Max`];
            if (axis === 'z' && min === undefined && max === undefined)
                continue;
            if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max)
                throw new Error('View bounds must be finite and increasing.');
        }
    if (view.camera) {
        for (const key of ['position', 'target'])
            if (
                !Array.isArray(view.camera[key]) ||
                view.camera[key].length !== 3 ||
                !view.camera[key].every(Number.isFinite)
            )
                throw new Error(
                    'Camera position and target need three finite coordinates.'
                );
        if (view.camera.position.every((n, i) => n === view.camera.target[i]))
            throw new Error('Camera position must differ from its target.');
    }
    return view;
}
export function validateStyle(style) {
    const result = {};
    if (style.color !== undefined) {
        if (!/^#[0-9a-f]{6}$/i.test(style.color))
            throw new Error('Use a six-digit hex color.');
        result.color = style.color;
    }
    if (style.opacity !== undefined) {
        if (
            !Number.isFinite(style.opacity) ||
            style.opacity < 0.1 ||
            style.opacity > 1
        )
            throw new Error('Opacity must be between 0.1 and 1.');
        result.opacity = style.opacity;
    }
    if (style.visible !== undefined) {
        if (typeof style.visible !== 'boolean')
            throw new Error('Visibility must be true or false.');
        result.visible = style.visible;
    }
    if (style.name !== undefined) {
        if (
            typeof style.name !== 'string' ||
            !style.name.trim() ||
            style.name.length > 100
        )
            throw new Error('Names need 1–100 characters.');
        result.name = style.name.trim();
    }
    return result;
}
export function prepareScene(surfaces, view = {}) {
    if (!Array.isArray(surfaces) || !surfaces.length || surfaces.length > 12)
        throw new Error('A scene needs 1–12 surfaces.');
    validateView(view);
    return surfaces.map((surface, i) => {
        compileSurface(surface.equation);
        return {
            ...makeSurface(surface.equation, i),
            ...validateStyle(surface)
        };
    });
}
