import React from 'react'

export default function TensorView({ action }) {
    if (!action || !action.u) return null

    // Calculate sparsity statistics
    const calculateSparsity = (vec) => {
        const zeros = vec.filter(v => v === 0).length
        return (zeros / vec.length) * 100
    }

    const uSparsity = calculateSparsity(action.u)
    const vSparsity = calculateSparsity(action.v)
    const wSparsity = calculateSparsity(action.w)
    const avgSparsity = (uSparsity + vSparsity + wSparsity) / 3

    return (
        <div className="tensor-view-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h3 className="tensor-header">🎯 Current Action Vectors</h3>
                <div style={{
                    padding: '4px 10px',
                    background: 'rgba(178, 36, 239, 0.2)',
                    borderRadius: '8px',
                    fontSize: '0.7rem',
                    color: '#b224ef'
                }}>
                    {avgSparsity.toFixed(0)}% Sparse
                </div>
            </div>

            <div className="factors-container">
                <FactorHeatmap label="U (Matrix A)" data={action.u} color="#00f2fe" sparsity={uSparsity} />
                <div className="multiply-symbol">⊗</div>
                <FactorHeatmap label="V (Matrix B)" data={action.v} color="#b224ef" sparsity={vSparsity} />
                <div className="multiply-symbol">⊗</div>
                <FactorHeatmap label="W (Output C)" data={action.w} color="#43e97b" sparsity={wSparsity} />
            </div>

            {/* Show tensor decomposition formula */}
            <div style={{
                marginTop: '15px',
                padding: '10px',
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '6px',
                fontSize: '0.75rem',
                color: '#888',
                textAlign: 'center',
                fontFamily: 'monospace'
            }}>
                T = <span style={{ color: '#00f2fe' }}>u</span> ⊗ <span style={{ color: '#b224ef' }}>v</span> ⊗ <span style={{ color: '#43e97b' }}>w</span>
                <div style={{ fontSize: '0.65rem', marginTop: '5px', opacity: 0.7 }}>
                    Rank-1 Tensor Decomposition
                </div>
            </div>

            <style>{`
            .tensor-view-panel {
                background: rgba(42, 43, 56, 0.4);
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .tensor-header {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #fff;
                margin: 0;
            }
            .factors-container {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                overflow-x: auto;
                padding: 10px 0;
            }
            .multiply-symbol {
                font-size: 2rem;
                color: #666;
                font-weight: 300;
            }
        `}</style>
        </div>
    )
}

function FactorHeatmap({ label, data, color, sparsity }) {
    if (!data || data.length === 0) return null

    // Determine dimensions and normalize to 2D
    let gridData = data
    if (data.length > 0 && !Array.isArray(data[0])) {
        gridData = [data]
    }

    const rows = gridData.length
    const cols = gridData[0].length

    return (
        <div className="heatmap-wrapper">
            <div className="heatmap-label" style={{ color: color }}>
                {label}
            </div>
            <div style={{ fontSize: '0.6rem', color: '#666', marginBottom: '5px' }}>
                {sparsity.toFixed(0)}% zeros
            </div>
            <div className="heatmap-grid" style={{
                display: 'grid',
                gridTemplateColumns: `repeat(${cols}, 1fr)`,
                gap: '2px',
                background: '#0a0a15',
                padding: '4px',
                borderRadius: '4px'
            }}>
                {gridData.map((row, i) => (
                    row.map((val, j) => {
                        // Color coding: positive = solid color, negative = bordered, zero = dark
                        const bgColor = val > 0 ? color : val < 0 ? '#1a1b26' : '#0a0a15'
                        const borderColor = val < 0 ? color : 'rgba(255,255,255,0.1)'

                        return (
                            <div key={`${i}-${j}`} style={{
                                width: '24px',
                                height: '24px',
                                background: bgColor,
                                opacity: val !== 0 ? 1 : 0.3,
                                border: `2px solid ${borderColor}`,
                                borderRadius: '2px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '0.75rem',
                                color: val === 0 ? '#444' : '#fff',
                                fontWeight: 'bold',
                                transition: 'all 0.2s ease',
                                cursor: 'help'
                            }}
                                title={`Position [${i},${j}] = ${val}\n${val > 0 ? 'Positive' : val < 0 ? 'Negative' : 'Zero'}`}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.transform = 'scale(1.15)'
                                    e.currentTarget.style.zIndex = '10'
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.transform = 'scale(1)'
                                    e.currentTarget.style.zIndex = '1'
                                }}
                            >
                                {val !== 0 ? (val > 0 ? '+' : '–') : '0'}
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
                font-size: 0.75rem;
                margin-bottom: 3px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .heatmap-grid {
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            }
        `}</style>
        </div>
    )
}
