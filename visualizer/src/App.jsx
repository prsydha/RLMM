import React, { useState, useCallback, useRef, useEffect } from 'react'
import TrainingMonitor from './components/TrainingMonitor'
import ControlPanel from './components/ControlPanel'
import TerminalPanel from './components/TerminalPanel'
import TensorView from './components/TensorView'
import MatrixVisualizer from './components/MatrixVisualizer'

export default function App() {
  const [connected, setConnected] = useState(false)
  const [logs, setLogs] = useState([])
  const [show3D, setShow3D] = useState(true)

  // Track stats
  const [stats, setStats] = useState({
    step: 0,
    totalOps: 8,
    elapsed: 0,
    matrixSize: '2×2',
    episode: 0,
    rank: 0,
    residual: 0,
    reward: 0,
    sparsity: 0,
    valid: true,
    status: 'SEARCHING',
    action: null
  })

  // Track History
  const [bestMetrics, setBestMetrics] = useState(null)
  const [lastSuccess, setLastSuccess] = useState(null)
  const [rewardHistory, setRewardHistory] = useState([])
  const [residualHistory, setResidualHistory] = useState([])

  const addLog = useCallback((msg, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 })
    setLogs(prev => [...prev.slice(-99), { time, message: msg, type }])
  }, [])

  // Update stats
  const handleStatsUpdate = useCallback((newStats) => {
    setStats(prev => ({ ...prev, ...newStats }))

    if (newStats.reward !== undefined) {
      setRewardHistory(prev => {
        const updated = [...prev, newStats.reward]
        if (updated.length > 50) return updated.slice(-50)
        return updated
      })
    }

    if (newStats.residual !== undefined) {
      setResidualHistory(prev => {
        const updated = [...prev, newStats.residual]
        if (updated.length > 50) return updated.slice(-50)
        return updated
      })
    }
  }, [])

  // Best solution tracking
  useEffect(() => {
    if (stats.residual !== undefined && stats.residual < 0.01) {
      const newHit = {
        rank: stats.rank,
        steps: stats.step,
        episode: stats.episode,
        time: stats.elapsed
      }

      setLastSuccess(newHit)

      setBestMetrics(prev => {
        if (!prev) return newHit
        if (newHit.rank < prev.rank) return newHit
        if (newHit.rank === prev.rank && newHit.steps < prev.steps) return newHit
        return prev
      })
    }
  }, [stats.residual, stats.rank, stats.step, stats.episode, stats.elapsed])

  return (
    <div className="app">
      <header className="app-header">
        <h1>TensorMind</h1>
        <div className="status-bar">
          <button
            className="btn-secondary"
            style={{ padding: '4px 12px', fontSize: '11px' }}
            onClick={() => setShow3D(!show3D)}
          >
            {show3D ? 'HIDE 3D VIEW' : 'SHOW 3D VIEW'}
          </button>
          <span className={`status-indicator ${connected ? 'online' : 'offline'}`}>
            {connected ? 'ONLINE' : 'OFFLINE'}
          </span>
          <span className="version">v3.0.0-CYBER</span>
        </div>
      </header>

      <div className="main-layout">
        <div className="left-panel">
          <div className="panel-glass visualizer-container" style={{ display: 'flex', flexDirection: 'column', position: 'relative' }}>

            {/* 3D Matrix Visualizer Layer */}
            {show3D && (
              <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}>
                <MatrixVisualizer
                  running={connected}
                  speed={1}
                  onStatsUpdate={() => { }}
                  onConnectionChange={() => { }}
                  onLog={() => { }}
                />
              </div>
            )}

            {/* Top Half: Monitor Overlay */}
            <div style={{ flex: 1, zIndex: 1, pointerEvents: 'none' }}>
              <div style={{ width: '100%', height: '100%', padding: '20px' }}>
                <TrainingMonitor
                  onStatsUpdate={handleStatsUpdate}
                  onConnectionChange={setConnected}
                  onLog={addLog}
                  compact={true}
                />
              </div>
            </div>

            {/* Bottom Half: Tensor Visualizer */}
            {stats.action && (
              <div style={{ padding: '20px', background: 'rgba(5,5,10,0.8)', borderTop: '1px solid rgba(255,255,255,0.1)', zIndex: 2 }}>
                <TensorView action={stats.action} />
              </div>
            )}
          </div>

          <div className="panel-glass terminal-panel">
            <div className="terminal-header">
              <div className="mac-dots"><span></span><span></span><span></span></div>
              <span style={{ opacity: 0.5 }}>TERMINAL // AGENT_LOGS</span>
            </div>
            <TerminalPanel logs={logs} />
          </div>
        </div>

        <div className="right-panel panel-glass">
          <ControlPanel
            connected={connected}
            stats={stats}
            bestMetrics={bestMetrics}
            lastSuccess={lastSuccess}
            rewardHistory={rewardHistory}
            residualHistory={residualHistory}
          />
        </div>
      </div>
    </div>
  )
}
