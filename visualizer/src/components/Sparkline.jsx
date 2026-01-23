
import React from 'react'

export default function Sparkline({ data, color = '#00f2fe', width = '100%', height = 60, maxItems = 50 }) {
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
                color: '#444',
                fontSize: '0.7rem'
            }}>
                Gathering data...
            </div>
        )
    }

    // Normalize data
    const values = data.slice(-maxItems)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const range = max - min || 1

    // Calculate points
    const points = values.map((val, i) => {
        const x = (i / (values.length - 1)) * 100
        // Invert Y because SVG 0 is top
        const y = 100 - ((val - min) / range * 100)
        return `${x},${y}`
    }).join(' ')

    return (
        <div className="sparkline-container" style={{ width, height, position: 'relative' }}>
            <svg
                width="100%"
                height="100%"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                style={{ overflow: 'visible' }}
            >
                {/* Gradient definition */}
                <defs>
                    <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor={color} stopOpacity="0.5" />
                        <stop offset="100%" stopColor={color} stopOpacity="0" />
                    </linearGradient>
                </defs>

                {/* Area fill */}
                <polygon
                    points={`0,100 ${points} 100,100`}
                    fill={`url(#gradient-${color})`}
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
            <div style={{ position: 'absolute', top: 0, right: 0, fontSize: '0.6rem', color: '#666' }}>
                {max.toFixed(1)}
            </div>
            <div style={{ position: 'absolute', bottom: 0, right: 0, fontSize: '0.6rem', color: '#666' }}>
                {min.toFixed(1)}
            </div>
        </div>
    )
}
