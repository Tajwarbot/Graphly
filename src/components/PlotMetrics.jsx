import { useEffect } from 'react';
import { usePlotArea } from 'recharts';
export function PlotMetrics({ onMeasure }) {
    const area = usePlotArea();
    useEffect(() => {
        if (area?.width > 0 && area?.height > 0)
            onMeasure({ width: area.width, height: area.height });
    }, [area?.width, area?.height, onMeasure]);
    return null;
}
