import { useEffect, useRef, useState, useId } from 'react';
import { parameterDefinition } from '../lib/expressionEnvironment.js';
export function ParameterSlider({ equation, onChange, settings = {}, onSettingsChange }) {
    const [playing,setPlaying] = useState(false);
    const latest = useRef({equation,onChange,settings});
    const inputId = useId();
    useEffect(() => { latest.current = {equation,onChange,settings}; }, [equation,onChange,settings]);
    useEffect(() => {
        if (!playing) return;
        const stop = () => { if (document.hidden) setPlaying(false); };
        document.addEventListener('visibilitychange',stop);
        const timer = setInterval(() => {
            const {equation:current,onChange:change,settings:options} = latest.current;
            const parameter = parameterDefinition(current);
            if (!parameter || !Number.isFinite(Number(parameter.expression))) {setPlaying(false);return;}
            const min = options.min ?? -10, max = options.max ?? 10, step = options.step ?? .01;
            const value = Number(parameter.expression);
            const increment = Math.max(step,(max-min)/40);
            const next = value+increment > max ? min : Math.min(max,min+Math.round((value+increment-min)/step)*step);
            change(`${parameter.name}=${Number(next.toPrecision(12))}`);
        },250);
        return () => { clearInterval(timer); document.removeEventListener('visibilitychange',stop); };
    },[playing]);
    const definition = parameterDefinition(equation);
    if (!definition || !/^\s*[+-]?(?:\d+\.?\d*|\.\d+)\s*$/.test(definition.expression)) return null;
    const value = Number(definition.expression);
    const extent = Math.max(10, Math.ceil(Math.abs(value)));
    const min = Number.isFinite(settings.min) ? settings.min : -extent;
    const max = Number.isFinite(settings.max) ? settings.max : extent;
    const step = Number.isFinite(settings.step) && settings.step > 0 ? settings.step : 0.01;
    const update = (key, raw) => {
        if (!raw.trim()) return;
        const next = {...{min,max,step}, [key]:Number(raw)};
        if (!Number.isFinite(next[key]) || next.min >= next.max || next.step <= 0) return;
        onSettingsChange?.(next);
    };
    return <div className="flex flex-col gap-2 py-2 text-xs font-mono">
        <label htmlFor={inputId}>{definition.name} = {value}</label>
        <button type="button" aria-label={`${playing ? 'Pause' : 'Play'} ${definition.name}`} aria-pressed={playing}
            className="min-h-11 border border-neutral-300 rounded bg-white hover:bg-neutral-100"
            onClick={() => {if (!playing && onSettingsChange) onSettingsChange({min,max,step}); setPlaying(!playing);}}>{playing ? 'Pause' : 'Play'}</button>
        <input id={inputId} type="range" min={min} max={max} step={step} value={value}
            aria-label={`${definition.name} parameter`} className="w-full min-h-11 accent-blue-600"
            onChange={event => onChange(`${definition.name}=${event.target.value}`)} />
        {onSettingsChange && <div className="grid grid-cols-3 gap-2">
            {['min','max','step'].map(key => <label key={key} className="min-w-0">{key}
                <input type="number" aria-label={`${definition.name} ${key}`} key={`${key}-${settings[key]}`} defaultValue={{min,max,step}[key]}
                    className="w-full min-h-11 border border-neutral-300 rounded px-2 bg-white" step="any"
                    onBlur={event => update(key,event.target.value)} onKeyDown={event => {if(event.key === 'Enter') event.currentTarget.blur();}} />
            </label>)}
        </div>}
    </div>;
}
