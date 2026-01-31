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
      <div className="status">
        <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></div>
        <span>{connected ? 'Connected to Server' : 'Waiting for connection...'}</span>
      </div>

      {/* Rewards Graph */}
      <div className="control-section">
        <h3>📈 Reward Trend (Last 50)</h3>
        <div style={{ marginBottom: '10px' }}>
          <Sparkline data={rewardHistory} color={stats.reward > 0 ? '#43e97b' : '#ff6b6b'} height={80} />
        </div>
        {rewardHistory.length > 0 && (
          <div style={{ fontSize: '0.75rem', color: '#888', display: 'flex', justifyContent: 'space-between' }}>
            <span>Min: {Math.min(...rewardHistory).toFixed(2)}</span>
            <span>Avg: {(rewardHistory.reduce((a, b) => a + b, 0) / rewardHistory.length).toFixed(2)}</span>
            <span>Max: {Math.max(...rewardHistory).toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Residual Graph */}
      <div className="control-section">
        <h3>📉 Residual Norm (Last 50)</h3>
        <div style={{ marginBottom: '10px' }}>
          <Sparkline
            data={residualHistory || []}
            color={stats.residual < 1 ? '#43e97b' : '#ff6b6b'}
            height={80}
          />
        </div>
        {residualHistory.length > 0 && (
          <div style={{ fontSize: '0.75rem', color: '#888', display: 'flex', justifyContent: 'space-between' }}>
            <span>Min: {Math.min(...residualHistory).toFixed(4)}</span>
            <span>Avg: {(residualHistory.reduce((a, b) => a + b, 0) / residualHistory.length).toFixed(4)}</span>
            <span>Max: {Math.max(...residualHistory).toFixed(4)}</span>
          </div>
        )}
      </div>

      {/* Performance Metrics */}
      <div className="control-section">
        <h3>⚡ Performance</h3>
        <div className="stats-panel">
          {/* Sparsity */}
          {stats.sparsity !== undefined && (
            <div className="stat-item">
              <span className="stat-label">Action Sparsity</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  flex: 1,
                  height: '6px',
                  background: 'rgba(255,255,255,0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${stats.sparsity * 100}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #b224ef, #00f2fe)',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
                <span className="stat-value">{(stats.sparsity * 100).toFixed(1)}%</span>
              </div>
            </div>
          )}

          <div className="stat-item">
            <span className="stat-label">⏱️ Time Elapsed</span>
            <span className="stat-value highlight">
              {stats.elapsed ? formatTime(stats.elapsed) : '0.0s'}
            </span>
          </div>

          <div className="stat-item">
            <span className="stat-label">🎯 Matrix Size</span>
            <span className="stat-value">
              {stats.matrixSize || '2×2'}
            </span>
          </div>
        </div>
      </div>

      {/* Latest Hit / Best Solution */}
      <div className="control-section">
        <h3>🏆 Achievements</h3>

        {/* Best Solution Found */}
        <div className="insight-card best">
          <div className="insight-label">🥇 Best Solution Found</div>
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
          <div className="insight-label">⚡ Latest Success</div>
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

        {/* Efficiency Comparison */}
        {bestMetrics && (
          <div className="insight-card efficiency">
            <div className="insight-label">💡 Efficiency vs Naive (2×2 = 8 ops)</div>
            <div className="insight-content">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%' }}>
                <div style={{ flex: 1 }}>
                  <div style={{
                    fontSize: '0.8rem',
                    color: '#888',
                    marginBottom: '5px'
                  }}>
                    {bestMetrics.rank} ops = {((1 - bestMetrics.rank / 8) * 100).toFixed(1)}% better
                  </div>
                  <div style={{
                    height: '8px',
                    background: 'rgba(255,255,255,0.1)',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${(1 - bestMetrics.rank / 8) * 100}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #43e97b, #00f2fe)',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Live Stats Grid */}
      <div className="control-section">
        <h3>📊 Current State</h3>
        <div className="stats-panel">
          <div className="stat-grid">
            <div className="stat-box">
              <div className="label">Global Step</div>
              <div className="value">{stats.step || 0}</div>
            </div>
            <div className="stat-box">
              <div className="label">Episode</div>
              <div className="value">{stats.episode || 0}</div>
            </div>
            <div className="stat-box">
              <div className="label">Rank Used</div>
              <div className="value">{stats.rank || 0}</div>
            </div>
            <div className="stat-box">
              <div className="label">Reward</div>
              <div className={`value ${stats.reward > 0 ? 'positive' : 'negative'}`}>
                {typeof stats.reward === 'number' ? stats.reward.toFixed(2) : '0.00'}
              </div>
            </div>
            <div className="stat-box">
              <div className="label">Residual</div>
              <div className="value tiny">
                {typeof stats.residual === 'number' ? stats.residual.toFixed(4) : '0.0000'}
              </div>
            </div>
            <div className="stat-box">
              <div className="label">Total Ops</div>
              <div className="value">{stats.totalOps || 0}</div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .insight-card {
            background: rgba(42, 43, 56, 0.5);
            padding: 14px;
            borderRadius: 8px;
            margin-bottom: 12px;
            border-left: 3px solid #666;
            transition: all 0.3s ease;
        }
        .insight-card:hover {
            background: rgba(42, 43, 56, 0.7);
            transform: translateX(2px);
        }
        .insight-card.best { border-left-color: #43e97b; }
        .insight-card.latest { border-left-color: #00f2fe; }
        .insight-card.efficiency { border-left-color: #b224ef; }
        
        .insight-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #888;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .insight-content {
            display: flex;
            align-items: baseline;
            gap: 10px;
        }
        
        .rank-display {
            font-size: 1.3rem;
            font-weight: bold;
            color: #fff;
        }
        
        .step-display {
            font-size: 0.95rem;
            color: #aaa;
        }
        
        .episode-ref {
            font-size: 0.85rem;
            color: #666;
            margin-left: auto;
        }
        
        .stat-value.highlight {
            font-size: 1.3rem;
            color: #00f2fe;
            font-weight: bold;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .stat-box {
            text-align: center;
            background: rgba(255,255,255,0.03);
            padding: 12px 8px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.05);
            transition: all 0.2s ease;
        }
        .stat-box:hover {
            background: rgba(255,255,255,0.06);
            border-color: rgba(0, 242, 254, 0.3);
        }
        .stat-box .label { 
            font-size: 0.7rem; 
            color: #888; 
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-box .value { 
            font-size: 1.1rem; 
            font-weight: bold; 
            color: #fff; 
        }
        .stat-box .value.tiny { font-size: 0.9rem; }
        .stat-box .value.positive { color: #43e97b; }
        .stat-box .value.negative { color: #ff6b6b; }
      `}</style>
    </div>
  )
}

function formatTime(seconds) {
  if (!seconds) return '0.0s'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${mins}m`
}
