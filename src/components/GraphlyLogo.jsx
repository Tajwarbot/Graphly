/** A coordinate trace turns through a G: geometry and the Graphly initial share one mark. */
export function GraphlyLogo({ size = 30, showText = true, className = '' }) {
    return <span className={`inline-flex items-center gap-2.5 ${className}`}>
        <svg width={size} height={size} viewBox="0 0 48 48" fill="none" aria-hidden="true" className="shrink-0">
            <path d="M10 8V38H41" stroke="#a1a1aa" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M37 14C33 9 25 8 19 12C11 18 12 30 19 34C26 38 36 34 37 26H26" stroke="#2563eb" strokeWidth="4.2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="37" cy="14" r="3" fill="#2563eb" />
            <circle cx="26" cy="26" r="3" fill="#2563eb" />
        </svg>
        {showText && <span className="text-base font-semibold tracking-tight text-neutral-900">Graphly</span>}
    </span>;
}
