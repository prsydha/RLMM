import React, { useEffect, useRef, useState } from 'react'

export default function TrainingMonitor({
  running,
  paused,
  onStatsUpdate,
  onConnectionChange,
  onLog,
  compact = false
}) {
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const isMountedRef = useRef(false)
  const [currentMetrics, setCurrentMetrics] = useState(null)

  useEffect(() => {
    isMountedRef.current = true

    const connect = () => {
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return
      }

      try {
        const ws = new WebSocket('ws://localhost:8765')
        wsRef.current = ws

        ws.onopen = () => {
          if (!isMountedRef.current) {
            ws.close()
            return
          }
          console.log('✅ Connected to Training Server')
          onConnectionChange(true)
          if (onLog) onLog('Connected to Training Server', 'system')
        }

        ws.onmessage = (event) => {
          if (!isMountedRef.current) return
          try {
            const data = JSON.parse(event.data)

            if (data.type === 'log') {
              if (onLog) onLog(data.message, data.level || 'info')
            } else if (data.type === 'stats') {
              if (onStatsUpdate) onStatsUpdate(data.data)
            } else if (data.type === 'step') {
              setCurrentMetrics({
                episode: data.episode,
                reward: data.reward,
                residual: data.residual,
                rank: data.rank,
                step: data.step_count
              })

              if (onStatsUpdate) {
                onStatsUpdate({
                  step: data.global_step,
                  episode: data.episode,
                  reward: data.reward,
                  residual: data.residual,
                  rank: data.rank,
                  totalOps: data.rank,
                  elapsed: data.elapsed,
                  sparsity: data.sparsity,
                  valid: data.action_valid,
                  status: data.status,
                  action: data.action,
                  learning_rate: data.learning_rate,
                  loss: data.loss,
                  policy_loss: data.policy_loss,
                  value_loss: data.value_loss
                })
              }
              if (onLog) {
                onLog(`Ep ${data.episode} Step ${data.step_count}: R=${data.reward.toFixed(4)} Rank=${data.rank} Res=${data.residual.toFixed(2)}`, 'info')
              }
            }
          } catch (e) {
            console.error('Error parsing message', e)
          }
        }

        ws.onclose = () => {
          if (isMountedRef.current) {
            console.log('❌ Disconnected')
            onConnectionChange(false)
            wsRef.current = null
            reconnectTimeoutRef.current = setTimeout(connect, 3000)
          }
        }

        ws.onerror = (err) => {
          console.error('WebSocket error', err)
        }

      } catch (err) {
        if (isMountedRef.current) {
          reconnectTimeoutRef.current = setTimeout(connect, 3000)
        }
      }
    }

    connect()

    return () => {
      isMountedRef.current = false
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
        wsRef.current = null
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [onStatsUpdate, onConnectionChange, onLog])

  if (!currentMetrics) {
    return (
      <div className="monitor-container empty">
        <div className="loader"></div>
        <div className="monitor-status">WAITING FOR SIGNAL...</div>
        <style>{`
                 .monitor-container.empty {
                     display: flex;
                     flex-direction: column;
                     align-items: center;
                     justify-content: center;
                     height: 100%;
                     opacity: 0.7;
                 }
                 .loader {
                     width: 40px; height: 40px;
                     border: 2px solid rgba(0,242,254,0.3);
                     border-top-color: #00f2fe;
                     border-radius: 50%;
                     animation: spin 1s linear infinite;
                     margin-bottom: 15px;
                 }
                 .monitor-status {
                     font-size: 12px;
                     letter-spacing: 2px;
                     color: #00f2fe;
                 }
                 @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
             `}</style>
      </div>
    )
  }

  return (
    <div className="monitor-container">
      <div className="monitor-header">
        <div className="live-dot"></div>
        <span>LIVE TRAINING // AGENT_INTERCEPT</span>
      </div>

      <div className="metrics-grid">
        <MetricCard label="EPISODE" value={currentMetrics.episode} />
        <MetricCard label="STEP" value={currentMetrics.step} />

        <MetricCard
          label="REWARD"
          value={currentMetrics.reward.toFixed(4)}
          highlight
          color={currentMetrics.reward > 0 ? '#43e97b' : '#ff6b6b'}
        />

        <MetricCard
          label="RESIDUAL"
          value={currentMetrics.residual.toFixed(2)}
          color="#f0932b"
        />

        <MetricCard
          label="RANK"
          value={currentMetrics.rank}
          color="#00f2fe"
        />

        <MetricCard
          label="STATUS"
          value={currentMetrics.step > 0 ? 'LEARNING' : 'IDLE'}
          color={currentMetrics.step > 0 ? '#43e97b' : '#666'}
        />
      </div>

      <style>{`
        .monitor-container {
            pointer-events: auto; /* enable interaction */
            background: rgba(0,0,0,0.3);
            backdrop-filter: blur(5px);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            max-width: 400px;
        }
        .monitor-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 10px;
            letter-spacing: 2px;
            color: #00f2fe;
            margin-bottom: 20px;
        }
        .live-dot {
            width: 6px; height: 6px;
            background: #ef4444;
            border-radius: 50%;
            box-shadow: 0 0 10px #ef4444;
            animation: pulse 2s infinite;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
      `}</style>
    </div>
  )
}

function MetricCard({ label, value, color, highlight }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
    }}>
      <div style={{ fontSize: '10px', color: '#888', marginBottom: '4px', letterSpacing: '1px' }}>{label}</div>
      <div style={{
        fontSize: highlight ? '24px' : '18px',
        fontWeight: '700',
        color: color || '#fff',
        fontFamily: "'JetBrains Mono', monospace",
        textShadow: color ? `0 0 15px ${color}40` : 'none'
      }}>
        {value}
      </div>
    </div>
  )
}
