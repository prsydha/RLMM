
import React, { useEffect, useRef, useState } from 'react'

export default function TrainingMonitor({
  running,
  paused,
  onStatsUpdate,
  onConnectionChange,
  onLog
}) {
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const isMountedRef = useRef(false)
  const [currentMetrics, setCurrentMetrics] = useState(null)

  useEffect(() => {
    isMountedRef.current = true

    const connect = () => {
      // Prevent multiple connections
      if (wsRef.current) {
        if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
          return
        }
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

            // Handle different message types
            if (data.type === 'log') {
              if (onLog) onLog(data.message, data.level || 'info')
            } else if (data.type === 'stats') {
              if (onStatsUpdate) onStatsUpdate(data.data)
            } else if (data.type === 'step') {
              // Update local state for visualization
              setCurrentMetrics({
                episode: data.episode,
                reward: data.reward,
                residual: data.residual,
                rank: data.rank,
                step: data.step_count
              })

              // Update parent state
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
            // Only auto-reconnect if we are still mounted
            reconnectTimeoutRef.current = setTimeout(connect, 3000)
          }
        }

        ws.onerror = (err) => {
          console.error('WebSocket error', err)
          // wsRef.current = null // Let onclose handle this
        }

      } catch (err) {
        if (isMountedRef.current) {
          console.error('Connection failed', err)
          reconnectTimeoutRef.current = setTimeout(connect, 3000)
        }
      }
    }

    connect()

    return () => {
      isMountedRef.current = false
      if (wsRef.current) {
        // Remove listener to prevent onclose triggering reconnect
        wsRef.current.onclose = null
        wsRef.current.close()
        wsRef.current = null
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }
    }
  }, [onStatsUpdate, onConnectionChange, onLog])

  return (
    <div className="training-monitor" style={{
      padding: '30px',
      color: '#e0e0e0',
      fontFamily: "'Roboto Mono', monospace",
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      background: 'radial-gradient(circle at center, #1a1a2e 0%, #0a0a15 100%)',
      border: '1px solid rgba(0, 242, 254, 0.1)',
      borderRadius: '12px',
      boxShadow: '0 0 20px rgba(0,0,0,0.5)'
    }}>
      <div style={{ marginBottom: 'auto', textAlign: 'center' }}>
        <h2 style={{ color: '#00f2fe', textShadow: '0 0 10px rgba(0,242,254,0.5)' }}>RUNNING AGENT</h2>
        <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>AWAITING STEPS FROM TRAIN.PY</div>
      </div>

      {currentMetrics ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', width: '100%' }}>
          <MetricCard label="EPISODE" value={currentMetrics.episode} color="#fff" />
          <MetricCard label="STEP" value={currentMetrics.step} color="#fff" />

          <MetricCard
            label="REWARD"
            value={currentMetrics.reward.toFixed(4)}
            color={currentMetrics.reward > 0 ? '#43e97b' : '#ff6b6b'}
            large
          />

          <MetricCard
            label="RESIDUAL"
            value={currentMetrics.residual.toFixed(2)}
            color="#f0932b"
          />

          <MetricCard
            label="RANK USED"
            value={currentMetrics.rank}
            color="#00f2fe"
          />

          <MetricCard
            label="STATUS"
            value={currentMetrics.status || 'SEARCHING'}
            color={currentMetrics.status === 'SOLVED' ? '#43e97b' : currentMetrics.status === 'PARTIAL' ? '#f0932b' : '#666'}
          />
        </div>
      ) : (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="loader" style={{
            width: '40px', height: '40px',
            border: '3px solid rgba(0,242,254,0.3)',
            borderTopColor: '#00f2fe',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      <div style={{ marginTop: 'auto', opacity: 0.5, fontSize: '0.8rem', textAlign: 'center' }}>
        LIVE TRAINING CONNECTION ACTIVE
      </div>
    </div>
  )
}

function MetricCard({ label, value, color, large, fullWidth }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.05)',
      padding: '15px',
      borderRadius: '8px',
      gridColumn: fullWidth ? 'span 2' : 'span 1',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center'
    }}>
      <div style={{ fontSize: '0.8rem', color: '#888', marginBottom: '5px', letterSpacing: '1px' }}>{label}</div>
      <div style={{
        fontSize: large ? '2.5rem' : '1.8rem',
        fontWeight: 'bold',
        color: color || '#fff',
        fontFamily: "'Outfit', sans-serif"
      }}>
        {value}
      </div>
    </div>
  )
}
