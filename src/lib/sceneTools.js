import { compileSurface } from './surfaceEngine.js';
import { samplePlot2D } from './plot2D.js';
import { expressionEnvironment, parameterDefinition, isDefinition } from './expressionEnvironment.js';
import { makeDataset, makeSurface } from './graphState.js';
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
export function validateExpressions(layers, dimension, bounds) {
    const environment = expressionEnvironment(layers);
    const errors = Object.entries(environment.errors);
    if (errors.length) throw new Error(errors.map(([name,error]) => `${name}: ${error}`).join(" "));
    for (const layer of layers) {
        if (!layer.equation || isDefinition(layer.equation)) continue;
        for(const equation of environment.expandAll(layer.equation)) {
            if (dimension === 3) compileSurface(equation);
            else samplePlot2D(equation, bounds || {xMin:-10,xMax:10,yMin:-10,yMax:10});
        }
    }
    return environment;
}
export function prepareScene(surfaces, view = {}, existing = []) {
    if (!Array.isArray(surfaces) || !surfaces.length || surfaces.length > 12)
        throw new Error('A scene needs 1–12 surfaces.');
    validateView(view);
    validateExpressions([...existing,...surfaces], 3, view.bounds);
    return surfaces.map((surface, i) => {
        if (typeof surface.equation !== "string" || !surface.equation.trim()) throw new Error("Provide an equation for each layer.");
        return {
            ...makeSurface(surface.equation, i),
            ...validateStyle(surface)
        };
    });
}

export function prepare2DScene(layers, view = {}, existing = []) {
    if (!Array.isArray(layers) || !layers.length || layers.length > 12) throw new Error('A scene needs 1–12 layers.');
    validateView(view);
    if (layers.some(layer => typeof layer.equation !== 'string' || !layer.equation.trim())) throw new Error('Provide an equation for each layer.');
    validateExpressions([...existing,...layers],2,view.bounds);
    return layers.map(layer => ({...makeDataset({equation:layer.equation,name:layer.name}),...validateStyle(layer)}));
}
export function parameterPatch(layer, value, settings = {}) {
    const definition = parameterDefinition(layer?.equation);
    if (!definition) throw new Error('This layer is not a parameter definition.');
    if (value === undefined) value = Number(definition.expression);
    if (!Number.isFinite(value)) throw new Error('Parameter value must be finite.');
    const extent = Math.max(10,Math.abs(value));
    const next = {...{min:-extent,max:extent,step:0.01},...layer.parameterSettings,...settings};
    if (![next.min,next.max,next.step].every(Number.isFinite) || next.min >= next.max || next.step <= 0) throw new Error('Parameter bounds must increase and step must be positive.');
    if (value < next.min || value > next.max) throw new Error('Parameter value must lie within its slider bounds.');
    return {equation:`${definition.name}=${value}`,parameterSettings:next};
}
