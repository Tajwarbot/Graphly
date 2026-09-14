import { useEffect } from 'react';

export function useModalFocus(ref, onClose) {
    useEffect(() => {
        const previous = document.activeElement;
        const element = ref.current;
        const selector = 'button:not(:disabled), a[href], input:not(:disabled), textarea:not(:disabled), select:not(:disabled), [tabindex="0"]';
        element?.querySelector(selector)?.focus();
        const onKey = event => {
            if (event.key === 'Escape') { event.preventDefault(); onClose(); }
            if (event.key !== 'Tab') return;
            const items = [...(element?.querySelectorAll(selector) || [])];
            const first = items[0];
            const last = items[items.length - 1];
            if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
            else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
        };
        element?.addEventListener('keydown', onKey);
        return () => { element?.removeEventListener('keydown', onKey); previous?.focus(); };
    }, [ref, onClose]);
}
