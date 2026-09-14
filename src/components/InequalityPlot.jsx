import { useMemo, useId } from 'react';
import { usePlotArea } from 'recharts';
import {
    buildInequalityRegion,
    regionToSvgPath,
    isInequality,
    compileInequality
} from '../lib/inequalityRegions.js';
import { generateImplicitSegments } from '../lib/implicitContours.js';
import { splitRestrictions } from '../lib/expressionSyntax.js';
export function InequalityPlot({ datasets, bounds }) {
    const area = usePlotArea(),
        id = useId();
    const key = JSON.stringify(
        datasets
            .filter((d) => d.visible && d.config.type === 'function')
            .flatMap(({ id, equation, plotEquation, plotEquations, color }) => (plotEquations || [plotEquation ?? equation]).map((item,i)=>({id:`${id}-${i}`,equation:item,color})))
    );
    const domainKey = JSON.stringify(bounds);
    const regions = useMemo(
        () =>
            JSON.parse(key).flatMap((item) => {
                try {
                    const { base } = splitRestrictions(item.equation);
                    if (!isInequality(base)) return [];
                    const range = JSON.parse(domainKey),
                        region = buildInequalityRegion(
                            item.equation,
                            range,
                            48
                        );
                    const constraints = compileInequality(
                        item.equation
                    ).constraints;
                    const boundaries = region.boundaries.map((b) => ({
                        ...b,
                        segments: generateImplicitSegments(
                            b.expression,
                            {
                                x: [range.xMin, range.xMax],
                                y: [range.yMin, range.yMax]
                            },
                            72
                        ).filter(([a, b]) =>
                            constraints.every(
                                (c) =>
                                    c.evaluate(
                                        (a[0] + b[0]) / 2,
                                        (a[1] + b[1]) / 2
                                    ) <= 1e-3
                            )
                        )
                    }));
                    return [{ ...item, region, boundaries }];
                } catch {
                    return [];
                }
            }),
        [key, domainKey]
    );
    if (!area) return null;
    const project = ([x, y]) => [
        area.x + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * area.width,
        area.y +
            area.height -
            ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * area.height
    ];
    return (
        <g pointerEvents="none">
            <defs>
                <clipPath id={id}>
                    <rect
                        x={area.x}
                        y={area.y}
                        width={area.width}
                        height={area.height}
                    />
                </clipPath>
            </defs>
            <g clipPath={`url(#${id})`}>
                {regions.map((item) => (
                    <g key={item.id}>
                        <path
                            d={regionToSvgPath(item.region, project)}
                            fill={item.color}
                            fillOpacity={0.16}
                        />
                        {item.boundaries.map((b, i) => (
                            <path
                                key={i}
                                d={b.segments
                                    .map(
                                        ([a, c]) =>
                                            `M${project(a).join(',')}L${project(c).join(',')}`
                                    )
                                    .join('')}
                                stroke={item.color}
                                strokeWidth={1.5}
                                strokeDasharray={b.strict ? '5 4' : undefined}
                                fill="none"
                            />
                        ))}
                    </g>
                ))}
            </g>
        </g>
    );
}
