/** One cancellable build. Worker startup failures get one fresh-worker retry. */
export function startSurfaceJob({ createWorker, payload, onStart, onResult, onError, delay = 140, timeout = 30000 }) {
    let worker, deadline, stopped = false, attempts = 0;
    const release = () => {
        clearTimeout(deadline);
        if (worker) {
            worker.onmessage = worker.onerror = worker.onmessageerror = null;
            worker.terminate();
            worker = undefined;
        }
    };
    const fail = (message, retry = true) => {
        if (stopped) return;
        release();
        if (retry && attempts < 2) { launch(); return; }
        stopped = true;
        onError(message);
    };
    const launch = () => {
        attempts++;
        try {
            worker = createWorker();
            worker.onmessage = ({ data }) => {
                if (stopped) return;
                stopped = true;
                release();
                onResult(data);
            };
            worker.onerror = () => fail('The 3D renderer could not start. Retry rendering; if it persists, reload the page.');
            worker.onmessageerror = () => fail('The 3D renderer returned unreadable data. Retry rendering.');
            deadline = setTimeout(() => fail('Rendering took too long. Try tighter bounds or a lower mesh density, then retry.', false), timeout);
            worker.postMessage(payload);
        } catch {
            fail('The 3D renderer could not start. Retry rendering; if it persists, reload the page.');
        }
    };
    const debounce = setTimeout(() => { if (!stopped) { onStart(); launch(); } }, delay);
    return () => { stopped = true; clearTimeout(debounce); release(); };
}
