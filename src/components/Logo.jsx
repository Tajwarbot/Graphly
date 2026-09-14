import { GraphlyLogo } from './GraphlyLogo';

export function LogoIcon({ size = 26, className = '' }) {
    return <GraphlyLogo size={size} showText={false} className={className} />;
}

export function Logo({ size = 26, showText = true, className = '' }) {
    return <GraphlyLogo size={size} showText={showText} className={className} />;
}
