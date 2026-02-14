import React from 'react'
import Sparkline from './Sparkline'

export default function ControlPanel({
  connected,
  stats,
  bestMetrics,
  lastSuccess,
  rewardHistory,
  residualHistory
}) {
  return (
    <div className="control-panel">
      {/* Connection Status */}
      <div className={`section-title`}>SYSTEM STATUS</div>
      <div className="stat-card" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></div>
          <span className="stat-value" style={{ fontSize: '13px' }}>
            {connected ? 'CONNECTED TO SERVER' : 'SEARCHING FOR SIGNAL...'}
          </span>
        </div>
      </div>

      {/* Rewards Graph */}
      <div className="control-section">
        <div className="section-title">REWARD TREND</div>
        <div className="graph-container">
          <Sparkline data={rewardHistory} color={stats.reward > 0 ? '#43e97b' : '#ff6b6b'} height={80} />
        </div>
      </div>

      {/* Residual Graph */}
      <div className="control-section">
        <div className="section-title">RESIDUAL NORM</div>
        <div className="graph-container">
          <Sparkline
            data={residualHistory || []}
            color={stats.residual < 1 ? '#43e97b' : '#ff6b6b'}
            height={80}
          />
        </div>
      </div>

      {/* Sparsity & Time */}
      <div className="control-section">
        <div className="section-title">METRICS</div>
        <div className="stats-grid">
          {/* Sparsity */}
          {stats.sparsity !== undefined && (
            <div className="stat-card" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <span className="stat-label">ACTION SPARSITY</span>
              <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{
                  width: `${stats.sparsity * 100}%`,
                  height: '100%',
                  background: 'var(--accent)',
                  boxShadow: '0 0 10px var(--accent)'
                }} />
              </div>
              <span className="stat-value" style={{ fontSize: '12px', alignSelf: 'flex-end' }}>{(stats.sparsity * 100).toFixed(1)}%</span>
            </div>
          )}

          <div className="stat-card">
            <span className="stat-label">LAST ACTION</span>
            <span className="stat-value" style={{
              color: stats.valid ? 'var(--success)' : 'var(--error)',
            }}>
              {stats.valid ? 'VALID' : 'INVALID'}
            </span>
          </div>

          <div className="stat-card">
            <span className="stat-label">ELAPSED TIME</span>
            <span className="stat-value highlight">
              {stats.elapsed ? stats.elapsed.toFixed(1) + 's' : '0.0s'}
            </span>
          </div>
        </div>
      </div>

      {/* Latest Hit / Best Solution */}
      <div className="control-section">
        <div className="section-title">INSIGHTS</div>

        {/* Best Solution Found */}
        <div className={`insight-card best ${bestMetrics ? 'active' : ''}`}>
          <div className="insight-label">BEST SOLUTION</div>
          {bestMetrics ? (
            <div className="insight-content">
              <div className="rank-display">RANK {bestMetrics.rank}</div>
              <div className="step-display">/ {bestMetrics.steps} STEPS</div>
              <div className="episode-ref">EP {bestMetrics.episode}</div>
            </div>
          ) : (
            <div className="insight-content placeholder">
              NO DATA...
            </div>
          )}
        </div>

        {/* Latest Hit */}
        <div className={`insight-card latest ${lastSuccess ? 'active' : ''}`}>
          <div className="insight-label">LATEST SUCCESS</div>
          {lastSuccess ? (
            <div className="insight-content">
              <div className="rank-display">RANK {lastSuccess.rank}</div>
              <div className="step-display">/ {lastSuccess.steps} STEPS</div>
              <div className="episode-ref">EP {lastSuccess.episode}</div>
            </div>
          ) : (
            <div className="insight-content placeholder">
              SEARCHING...
            </div>
          )}
        </div>
      </div>

      <style>{`
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #666;
            box-shadow: 0 0 5px #666;
            transition: all 0.3s;
        }
        .status-dot.connected {
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }
        .status-dot.disconnected {
            background: var(--error);
            box-shadow: 0 0 10px var(--error);
        }
        
        .graph-container {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .insight-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 12px;
            border-left: 2px solid #444;
            transition: all 0.3s ease;
        }
        .insight-card.best.active { 
            border-left-color: var(--success); 
            background: rgba(0, 210, 255, 0.05);
        }
        .insight-card.latest.active { 
            border-left-color: var(--primary); 
            background: rgba(0, 242, 254, 0.05);
        }
        
        .insight-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 8px;
            font-weight: 700;
        }
        
        .insight-content {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .insight-content.placeholder {
            font-size: 12px;
            color: #555;
            font-style: italic;
        }
        
        .rank-display {
            font-size: 14px;
            font-weight: 800;
            color: #fff;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .step-display {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .episode-ref {
            font-size: 10px;
            color: #666;
            margin-left: auto;
            background: rgba(255,255,255,0.05);
            padding: 2px 6px;
            border-radius: 4px;
        }
      `}</style>
    </div>
  )
}
