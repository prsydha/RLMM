import React, { useId } from 'react'

export default function Sparkline({ data, color = '#00f2fe', width = '100%', height = 60, maxItems = 50 }) {
    const gradientId = useId()

    if (!data || data.length < 2) {
        return (
            <div style={{
                width,
                height,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(255,255,255,0.02)',
                borderRadius: '4px',
                color: '#666',
                fontSize: '0.7rem',
                border: '1px dashed rgba(255,255,255,0.1)'
            }}>
                WAITING FOR DATA...
            </div>
        )
    }

    // Normalize data
    const values = data.slice(-maxItems)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const range = max - min || 1

    // Calculate points: 0..100 for x and y
    const points = values.map((val, i) => {
        const x = (i / (values.length - 1)) * 100
        // Invert Y because SVG 0 is top
        const y = 100 - ((val - min) / range * 100)
        return `${x},${y}`
    }).join(' ')

    // Create area polygon points (start at bottom left, go to points, end at bottom right)
    const areaPoints = `0,100 ${points} 100,100`

    return (
        <div className="sparkline-container" style={{ width, height, position: 'relative' }}>
            <svg
                width="100%"
                height="100%"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                style={{ overflow: 'visible' }}
            >
                <defs>
                    <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} stopOpacity="0.4" />
                        <stop offset="100%" stopColor={color} stopOpacity="0" />
                    </linearGradient>
                </defs>

                {/* Area fill */}
                <polygon
                    points={areaPoints}
                    fill={`url(#${gradientId})`}
                />

                {/* Line */}
                <polyline
                    points={points}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
            </svg>

            {/* Min/Max Labels */}
            <div style={{ position: 'absolute', top: -5, right: 0, fontSize: '9px', color: '#888', fontFamily: 'monospace' }}>
                {max.toFixed(2)}
            </div>
            <div style={{ position: 'absolute', bottom: -5, right: 0, fontSize: '9px', color: '#888', fontFamily: 'monospace' }}>
                {min.toFixed(2)}
            </div>
        </div>
    )
}
