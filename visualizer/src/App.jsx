
import React, { useState, useCallback, useRef, useEffect } from 'react'
import TrainingMonitor from './components/TrainingMonitor'
import ControlPanel from './components/ControlPanel'
import TerminalPanel from './components/TerminalPanel'
import TensorView from './components/TensorView'

export default function App() {
  const [connected, setConnected] = useState(false)
  const [logs, setLogs] = useState([])

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
        <h1>⚡ Matrix Core</h1>
        <div className="status-bar">
          <span className={`status-indicator ${connected ? 'online' : 'offline'}`}>
            {connected ? 'ONLINE' : 'OFFLINE'}
          </span>
          <span className="version">v2.1.0-NEON</span>
        </div>
      </header>

      <div className="main-layout">
        <div className="left-panel">
          <div className="visualizer-container" style={{ display: 'flex', flexDirection: 'column' }}>
            {/* Top Half: Monitor */}
            <div style={{ flex: 1, minHeight: '300px' }}>
              <TrainingMonitor
                onStatsUpdate={handleStatsUpdate}
                onConnectionChange={setConnected}
                onLog={addLog}
              />
            </div>

            {/* Bottom Half: Tensor Visualizer */}
            {stats.action && (
              <div style={{ padding: '20px', background: '#0d0e12', borderTop: '1px solid #333' }}>
                <TensorView action={stats.action} />
              </div>
            )}
          </div>
          <TerminalPanel logs={logs} />
        </div>

        <div className="right-panel">
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
