import { afterEach, expect, it, vi } from 'vitest';
import { startSurfaceJob } from './surfaceJob.js';
afterEach(() => vi.useRealTimers());
function setup() {
    vi.useFakeTimers();
    const workers = [];
    const options = {
        createWorker: vi.fn(() => { const w = { postMessage: vi.fn(), terminate: vi.fn() }; workers.push(w); return w; }),
        payload: { surfaces: [] }, onStart: vi.fn(), onResult: vi.fn(), onError: vi.fn()
    };
    const cancel = startSurfaceJob(options);
    return { workers, options, cancel };
}
it('does not create a worker before debounce and cancels obsolete edits', () => {
    const { cancel, options } = setup(); cancel(); vi.runAllTimers();
    expect(options.createWorker).not.toHaveBeenCalled(); expect(options.onStart).not.toHaveBeenCalled();
});
it('recovers from a startup failure with a fresh worker and settles once', () => {
    const { workers, options } = setup(); vi.advanceTimersByTime(140);
    workers[0].onerror(); workers[1].onmessage({ data: ['mesh'] }); vi.runAllTimers();
    expect(options.onStart).toHaveBeenCalledTimes(1);
    expect(options.onResult).toHaveBeenCalledWith(['mesh']); expect(options.onError).not.toHaveBeenCalled();
    expect(workers.every(w => w.terminate.mock.calls.length === 1)).toBe(true);
});
it('ends loading after repeated startup failures without restarting it', () => {
    const { workers, options } = setup(); vi.advanceTimersByTime(140);
    workers[0].onerror(); workers[1].onerror(); vi.runAllTimers();
    expect(options.onStart).toHaveBeenCalledTimes(1); expect(options.onError).toHaveBeenCalledTimes(1);
    expect(options.createWorker).toHaveBeenCalledTimes(2);
});
it('terminates a hung build and reports a timeout', () => {
    const { workers, options } = setup(); vi.advanceTimersByTime(30140);
    expect(options.onError).toHaveBeenCalledWith(expect.stringContaining('too long'));
    expect(workers[0].terminate).toHaveBeenCalledTimes(1);
});
it('ignores late results after cancellation', () => {
    const { workers, options, cancel } = setup(); vi.advanceTimersByTime(140);
    const late = workers[0].onmessage; cancel(); late({ data: [] }); vi.runAllTimers();
    expect(options.onResult).not.toHaveBeenCalled(); expect(options.onError).not.toHaveBeenCalled();
});
