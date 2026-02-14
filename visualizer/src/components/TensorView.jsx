import React from 'react'

export default function TensorView({ action }) {
    if (!action || !action.u) return null

    return (
        <div className="tensor-view-panel">
            <div className="tensor-header-row">
                <h3 className="tensor-header">TENSOR FACTORS</h3>
                <div className="tensor-badge">RANK {action.u.length}</div>
            </div>

            <div className="factors-container">
                <FactorHeatmap label="U (Left)" data={action.u} color="#00f2fe" />
                <div className="multiply-symbol">×</div>
                <FactorHeatmap label="V (Right)" data={action.v} color="#b224ef" />
                <div className="multiply-symbol">×</div>
                <FactorHeatmap label="W (Out)" data={action.w} color="#43e97b" />
            </div>
            <style>{`
            .tensor-view-panel {
                width: 100%;
            }
            .tensor-header-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                padding-bottom: 10px;
            }
            .tensor-header {
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #94a3b8;
                font-weight: 700;
            }
            .tensor-badge {
                font-size: 10px;
                background: rgba(255,255,255,0.1);
                padding: 2px 8px;
                border-radius: 4px;
                color: #fff;
            }
            .factors-container {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                overflow-x: auto;
                padding-bottom: 10px;
            }
            .multiply-symbol {
                font-size: 20px;
                color: rgba(255,255,255,0.2);
                font-weight: 300;
            }
        `}</style>
        </div>
    )
}

function FactorHeatmap({ label, data, color }) {
    if (!data || data.length === 0) return null

    let gridData = data
    if (data.length > 0 && !Array.isArray(data[0])) {
        gridData = [data]
    }

    const rows = gridData.length
    const cols = gridData[0].length

    return (
        <div className="heatmap-wrapper">
            <div className="heatmap-label" style={{ color: color, textShadow: `0 0 10px ${color}40` }}>{label}</div>
            <div className="heatmap-grid" style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${cols}, 1fr)`,
                gap: '2px',
                background: 'rgba(0,0,0,0.5)',
                padding: '4px',
                borderRadius: '4px',
                border: `1px solid ${color}30`
            }}>
                {gridData.map((row, i) => (
                    row.map((val, j) => {
                        const opacity = Math.abs(val)
                        const bgColor = val !== 0 ? color : 'transparent'

                        return (
                            <div key={`${i}-${j}`} className="heatmap-cell" style={{
                                width: '24px',
                                height: '24px',
                                background: bgColor,
                                opacity: val !== 0 ? (val < 0 ? 0.6 : 0.9) : 0.1,
                                border: val < 0 ? `1px solid ${color}` : '1px solid rgba(255,255,255,0.05)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '10px',
                                color: '#fff',
                                fontWeight: 'bold',
                                borderRadius: '2px',
                                transition: 'all 0.2s ease'
                            }} title={`[${i},${j}] = ${val}`}>
                                {val !== 0 ? Math.round(val) : ''}
                            </div>
                        )
                    })
                ))}
            </div>
            <style>{`
            .heatmap-wrapper {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .heatmap-label {
                font-size: 11px;
                margin-bottom: 8px;
                font-weight: 600;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            .heatmap-cell:hover {
                transform: scale(1.2);
                z-index: 10;
                box-shadow: 0 0 10px rgba(0,0,0,0.5);
            }
        `}</style>
        </div>
    )
}
