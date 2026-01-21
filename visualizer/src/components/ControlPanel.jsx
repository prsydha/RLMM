import React from 'react'

export default function ControlPanel({ 
  running, 
  paused, 
  onStart, 
  onPause, 
  onReset,
  speed,
  onSpeedChange,
  connected,
  stats
}) {
  return (
    <div className="control-panel">
      {/* Connection Status */}
      <div className="status">
        <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></div>
        <span>{connected ? 'Connected to Server' : 'Demo Mode (No Server)'}</span>
      </div>

      {/* Main Controls */}
      <div className="control-section">
        <h3>Playback</h3>
        <div className="controls">
          <button className="btn-start" onClick={onStart} disabled={running && !paused}>
            {running && !paused ? 'Running...' : 'Start'}
          </button>
          <button className="btn-pause" onClick={onPause} disabled={!running}>
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button className="btn-reset" onClick={onReset}>
            Reset
          </button>
        </div>
      </div>

      {/* Speed Control */}
      <div className="control-section">
        <h3>Animation Speed</h3>
        <div className="slider-container">
          <div className="slider-label">
            <span>Slower</span>
            <span>{speed.toFixed(1)}x</span>
            <span>Faster</span>
          </div>
          <input 
            type="range" 
            min="0.1" 
            max="3" 
            step="0.1" 
            value={speed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            className="slider"
          />
        </div>
      </div>

      {/* Statistics */}
      <div className="control-section">
        <h3>Statistics</h3>
        <div className="stats-panel">
          <div className="stat-item">
            <span className="stat-label">Current Step</span>
            <span className="stat-value">{stats.step}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total Operations</span>
            <span className="stat-value">{stats.totalOps}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Elapsed Time</span>
            <span className="stat-value">{stats.elapsed}s</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Matrix Size</span>
            <span className="stat-value">{stats.matrixSize}</span>
          </div>
        </div>
      </div>

      {/* Info */}
      <div className="control-section">
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          <p><strong>How it works:</strong></p>
          <p>Watch in real-time as matrix multiplication happens. Highlighted cubes show the current operation.</p>
        </div>
      </div>
    </div>
  )
}
