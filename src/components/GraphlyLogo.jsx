/** User-supplied Graphly mark, shared by navigation and homepage. */
export function GraphlyLogo({ size = 30, showText = true, className = '' }) {
    return <span className={`inline-flex items-center gap-2.5 ${className}`}>
        <img src="/graphly-logo.png" width={size} height={size} alt="" aria-hidden="true" className="shrink-0 object-contain" />
        {showText && <span className="text-base font-bold tracking-tight text-neutral-900">Graphly</span>}
    </span>;
}
