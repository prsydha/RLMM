
import React from 'react'
import Sparkline from './Sparkline'

export default function ControlPanel({
  connected,
  stats,
  bestMetrics,
  lastSuccess,
  rewardHistory
  
}) {
  return (
    <div className="control-panel">
      {/* Connection Status */}
      <div className="status">
        <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></div>
        <span>{connected ? 'Connected to Server' : 'Waiting for connection...'}</span>
      </div>

      {/* Rewards Graph */}
      <div className="control-section">
        <h3>Reward Trend</h3>
        <div style={{ marginBottom: '10px' }}>
          <Sparkline data={rewardHistory} color={stats.reward > 0 ? '#43e97b' : '#ff6b6b'} height={80} />
        </div>
      </div>

      {/* Sparsity & Time */}
      <div className="control-section">
        <h3>Training Metrics</h3>
        <div className="stats-panel">
          {/* Sparsity */}
          {stats.sparsity !== undefined && (
            <div className="stat-item">
              <span className="stat-label">Action Sparsity</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{ width: '60px', height: '4px', background: '#333', borderRadius: '2px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${stats.sparsity * 100}%`,
                    height: '100%',
                    background: '#b224ef'
                  }} />
                </div>
                <span className="stat-value">{(stats.sparsity * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}

          <div className="stat-item">
            <span className="stat-label">Time Elapsed</span>
            <span className="stat-value highlight">
              {stats.elapsed ? stats.elapsed.toFixed(1) + 's' : '0.0s'}
            </span>
          </div>
        </div>
      </div>

      {/* Latest Hit / Best Solution */}
      <div className="control-section">
        <h3>Insights</h3>

        {/* Best Solution Found */}
        <div className="insight-card best">
          <div className="insight-label">Best Solution Found</div>
          {bestMetrics ? (
            <div className="insight-content">
              <div className="rank-display">Rank {bestMetrics.rank}</div>
              <div className="step-display">in {bestMetrics.steps} steps</div>
              <div className="episode-ref">Ep {bestMetrics.episode}</div>
            </div>
          ) : (
            <div className="insight-content placeholder">
              No solutions yet...
            </div>
          )}
        </div>

        {/* Latest Hit */}
        <div className="insight-card latest">
          <div className="insight-label">Latest Hit (Last Success)</div>
          {lastSuccess ? (
            <div className="insight-content">
              <div className="rank-display">Rank {lastSuccess.rank}</div>
              <div className="step-display">in {lastSuccess.steps} steps</div>
              <div className="episode-ref">Ep {lastSuccess.episode}</div>
            </div>
          ) : (
            <div className="insight-content placeholder">
              Searching...
            </div>
          )}
        </div>
      </div>

      {/* Live Stats */}
      <div className="control-section">
        <h3>Live Stats</h3>
        <div className="stats-panel">
          {/* Combined compact stats */}
          <div className="stat-grid">
            <div className="stat-box">
              <div className="label">Step</div>
              <div className="value">{stats.step}</div>
            </div>
            <div className="stat-box">
              <div className="label">Episode</div>
              <div className="value">{stats.episode || 0}</div>
            </div>
            <div className="stat-box">
              <div className="label">Resid.</div>
              <div className="value tiny">
                {typeof stats.residual === 'number' ? stats.residual.toExponential(1) : '0'}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .insight-card {
            background: rgba(42, 43, 56, 0.5);
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #666;
        }
        .insight-card.best { border-left-color: #43e97b; }
        .insight-card.latest { border-left-color: #00f2fe; }
        
        .insight-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #888;
            margin-bottom: 5px;
        }
        
        .insight-content {
            display: flex;
            align-items: baseline;
            gap: 8px;
        }
        
        .rank-display {
            font-size: 1.2rem;
            font-weight: bold;
            color: #fff;
        }
        
        .step-display {
            font-size: 0.9rem;
            color: #aaa;
        }
        
        .episode-ref {
            font-size: 0.8rem;
            color: #666;
            margin-left: auto;
        }
        
        .stat-value.highlight {
            font-size: 1.2rem;
            color: #00f2fe;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
        }
        .stat-box {
            text-align: center;
            background: rgba(255,255,255,0.03);
            padding: 8px;
            border-radius: 4px;
        }
        .stat-box .label { font-size: 0.7rem; color: #888; margin-bottom: 4px; }
        .stat-box .value { font-size: 1rem; font-weight: bold; color: #fff; }
        .stat-box .value.tiny { font-size: 0.85rem; }
      `}</style>
    </div>
  )
}
