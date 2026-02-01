
import React from 'react'

export default function TensorView({ action }) {
    if (!action || !action.u) return null

    // U, V, W are arrays of shape [Rank, Dim]
    // Ideally we want to show them side by side

    return (
        <div className="tensor-view-panel">
            <h3 className="tensor-header">Tensor Factors (Rank Components)</h3>
            <div className="factors-container">
                <FactorHeatmap label="U (Left)" data={action.u} color="#00f2fe" />
                <div className="multiply-symbol">×</div>
                <FactorHeatmap label="V (Right)" data={action.v} color="#b224ef" />
                <div className="multiply-symbol">×</div>
                <FactorHeatmap label="W (Out)" data={action.w} color="#43e97b" />
            </div>
            <style>{`
            .tensor-view-panel {
                background: rgba(42, 43, 56, 0.3);
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                border: 1px solid rgba(255,255,255,0.05);
            }
            .tensor-header {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #888;
                margin-bottom: 15px;
                text-align: center;
            }
            .factors-container {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 15px;
                overflow-x: auto;
            }
            .multiply-symbol {
                font-size: 1.5rem;
                color: #555;
            }
        `}</style>
        </div>
    )
}

function FactorHeatmap({ label, data, color }) {
    // Data is array of [Rank, Vector] or similar list of lists
    if (!data || data.length === 0) return null

    // Determine dimensions and normalize to 2D
    let gridData = data
    // Check if first element is not an array (i.e. it's a number) -> 1D array
    if (data.length > 0 && !Array.isArray(data[0])) {
        gridData = [data]
    }

    const rows = gridData.length
    const cols = gridData[0].length

    return (
        <div className="heatmap-wrapper">
            <div className="heatmap-label" style={{ color: color }}>{label}</div>
            <div className="heatmap-grid" style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${cols}, 1fr)`,
                gap: '1px',
                background: '#000'
            }}>
                {gridData.map((row, i) => (
                    row.map((val, j) => {
                        // Normalize val for opacity/color
                        // Expecting -1, 0, 1 mostly
                        const opacity = Math.abs(val)
                        const bgColor = val !== 0 ? color : '#1a1b26'

                        return (
                            <div key={`${i}-${j}`} style={{
                                width: '20px',
                                height: '20px',
                                background: bgColor,
                                opacity: val !== 0 ? (val < 0 ? 0.7 : 1) : 1,
                                border: val < 0 ? `1px solid ${color}` : '1px solid rgba(255,255,255,0.1)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '0.65rem',
                                color: '#fff',
                                fontWeight: 'bold'
                            }} title={`[${i},${j}] = ${val}`}>
                                {val !== 0 ? val : ''}
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
                font-size: 0.7rem;
                margin-bottom: 5px;
                font-weight: bold;
            }
            .heatmap-grid {
                border: 1px solid #333;
                padding: 2px;
            }
        `}</style>
        </div>
    )
}
