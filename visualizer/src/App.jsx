import React, { useState, useCallback } from 'react'
import MatrixVisualizer from './components/MatrixVisualizer'
import ControlPanel from './components/ControlPanel'
import TerminalPanel from './components/TerminalPanel'

export default function App() {
  const [running, setRunning] = useState(false)
  const [paused, setPaused] = useState(false)
  const [resetKey, setResetKey] = useState(0)
  const [speed, setSpeed] = useState(1.0)
  const [connected, setConnected] = useState(false)
  const [logs, setLogs] = useState([])
  const [stats, setStats] = useState({
    step: 0,
    totalOps: 8,
    elapsed: 0,
    matrixSize: '2×2'
  })

  const addLog = useCallback((msg, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 })
    setLogs(prev => [...prev.slice(-99), { time, message: msg, type }])
  }, [])

  const handleReset = () => {
    setRunning(false)
    setPaused(false)
    setResetKey(k => k + 1)
    setStats({ step: 0, totalOps: 8, elapsed: 0, matrixSize: '2×2' })
    setLogs([])
    addLog('System reset initiated.', 'system')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>⚡ Matrix Core</h1>
        <div className="status-bar">
          <span className={`status-indicator ${connected ? 'online' : 'offline'}`}>
            {connected ? 'ONLINE' : 'OFFLINE'}
          </span>
          <span className="version">v2.0.4-NEON</span>
        </div>
      </header>

      <div className="main-layout">
        <div className="left-panel">
          <div className="visualizer-container">
            {/* Pass addLog as detailLogger */}
            <MatrixVisualizer
              key={resetKey}
              running={running}
              paused={paused}
              speed={speed}
              onStatsUpdate={setStats}
              onConnectionChange={setConnected}
              onLog={addLog}
            />
          </div>
          <TerminalPanel logs={logs} />
        </div>

        <div className="right-panel">
          <ControlPanel
            running={running}
            paused={paused}
            speed={speed}
            connected={connected}
            stats={stats}
            onStart={() => {
              setRunning(true);
              setPaused(false);
              addLog('Process started. Allocating tensors...', 'success');
            }}
            onPause={() => {
              setPaused(p => !p);
              addLog(paused ? 'Process resumed.' : 'Process paused by user.', 'warning');
            }}
            onReset={handleReset}
            onSpeedChange={setSpeed}
          />
        </div>
      </div>
    </div>
  )
}
