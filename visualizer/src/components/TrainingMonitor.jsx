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
  const [epochData, setEpochData] = useState(null)
  const [estimatedTime, setEstimatedTime] = useState(null)
  const [successHistory, setSuccessHistory] = useState([])

  // Calculate time estimate
  useEffect(() => {
    if (currentMetrics && epochData) {
      const successRate = epochData.success_rate || 0
      const avgEpochsPerSuccess = successRate > 0 ? 1 / successRate : Infinity

      if (avgEpochsPerSuccess < Infinity) {
        // Rough estimate: avg episodes per epoch * avg steps per episode * avg time per step
        const avgTimePerEpoch = epochData.elapsed / (epochData.epoch || 1)
        const estimatedSecondsToNextHit = avgEpochsPerSuccess * avgTimePerEpoch
        setEstimatedTime(estimatedSecondsToNextHit)
      }
    }
  }, [currentMetrics, epochData])

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
                step: data.step_count,
                global_step: data.global_step,
                sparsity: data.sparsity
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
                  action: data.action
                })
              }
              if (onLog) {
                onLog(`Ep ${data.episode} Step ${data.step_count}: R=${data.reward.toFixed(4)} Rank=${data.rank} Res=${data.residual.toFixed(4)}`, 'info')
              }
            } else if (data.type === 'epoch') {
              // New: epoch-level data
              setEpochData({
                epoch: data.epoch,
                success_rate: data.success_rate,
                elapsed: data.elapsed,
                reward_mode: data.reward_mode,
                gradient_norm: data.gradient_norm,
                temperature: data.temperature,
                epsilon: data.epsilon,
                best_rank: data.best_rank
              })

              // Track success history
              if (data.epoch_successes > 0) {
                setSuccessHistory(prev => [...prev.slice(-20), data.epoch_successes])
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

  const formatTime = (seconds) => {
    if (!seconds || seconds === Infinity) return '--'
    if (seconds < 60) return `${seconds.toFixed(0)}s`
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`
    return `${(seconds / 3600).toFixed(1)}h`
  }

  return (
    <div className="training-monitor" style={{
      padding: '25px',
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
      {/* Header */}
      <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ color: '#00f2fe', textShadow: '0 0 10px rgba(0,242,254,0.5)', margin: 0, fontSize: '1.5rem' }}>
          LIVE TRAINING
        </h2>
        {epochData && (
          <div style={{
            background: epochData.reward_mode === 'dense' ? 'rgba(67, 233, 123, 0.2)' : 'rgba(178, 36, 239, 0.2)',
            padding: '4px 12px',
            borderRadius: '12px',
            fontSize: '0.75rem',
            fontWeight: 'bold',
            color: epochData.reward_mode === 'dense' ? '#43e97b' : '#b224ef',
            border: `1px solid ${epochData.reward_mode === 'dense' ? '#43e97b' : '#b224ef'}`
          }}>
            {epochData.reward_mode?.toUpperCase() || 'SPARSE'}
          </div>
        )}
      </div>

      {currentMetrics ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px', flex: 1 }}>
          {/* Top Row: Primary Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '15px' }}>
            <MetricCard
              label="EPISODE"
              value={currentMetrics.episode}
              color="#fff"
              icon="📊"
            />
            <MetricCard
              label="STEP"
              value={currentMetrics.step}
              color="#fff"
              icon="🔢"
            />
            <MetricCard
              label="GLOBAL"
              value={currentMetrics.global_step}
              color="#00f2fe"
              icon="🌐"
            />
          </div>

          {/* Second Row: Reward & Residual */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <MetricCard
              label="REWARD"
              value={currentMetrics.reward.toFixed(3)}
              color={currentMetrics.reward > 0 ? '#43e97b' : '#ff6b6b'}
              large
              icon={currentMetrics.reward > 0 ? '✅' : '❌'}
            />
            <MetricCard
              label="RESIDUAL"
              value={currentMetrics.residual < 0.0001 ? currentMetrics.residual.toFixed(6) : currentMetrics.residual.toFixed(4)}
              color={currentMetrics.residual < 1 ? '#43e97b' : '#f0932b'}
              large
              icon="📉"
            />
          </div>

          {/* Third Row: Rank & Sparsity */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <MetricCard
              label="RANK USED"
              value={currentMetrics.rank}
              color="#00f2fe"
              icon="🎯"
            />
            <MetricCard
              label="SPARSITY"
              value={`${(currentMetrics.sparsity * 100).toFixed(1)}%`}
              color="#b224ef"
              icon="✨"
            />
          </div>

          {/* Epoch Info & Time Estimate */}
          {epochData && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: estimatedTime ? '1fr 1fr' : '1fr',
              gap: '15px',
              marginTop: '10px'
            }}>
              <div style={{
                background: 'rgba(0, 242, 254, 0.1)',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid rgba(0, 242, 254, 0.3)'
              }}>
                <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>EPOCH {epochData.epoch}</div>
                <div style={{ fontSize: '0.9rem', color: '#00f2fe' }}>
                  Success Rate: <span style={{ fontWeight: 'bold' }}>{(epochData.success_rate * 100).toFixed(1)}%</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: '#aaa', marginTop: '3px' }}>
                  Best: Rank {epochData.best_rank}
                </div>
              </div>

              {estimatedTime && estimatedTime < 3600 && (
                <div style={{
                  background: 'rgba(67, 233, 123, 0.1)',
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(67, 233, 123, 0.3)',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center'
                }}>
                  <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '5px' }}>⏱️ NEXT HIT ETA</div>
                  <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#43e97b' }}>
                    {formatTime(estimatedTime)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Gradient Health Indicator */}
          {epochData && epochData.gradient_norm !== undefined && (
            <div style={{
              background: 'rgba(255,255,255,0.05)',
              padding: '10px',
              borderRadius: '6px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ fontSize: '0.75rem', color: '#888' }}>GRADIENT NORM</span>
              <span style={{
                fontSize: '0.9rem',
                fontWeight: 'bold',
                color: epochData.gradient_norm > 10 ? '#ff6b6b' : epochData.gradient_norm < 0.01 ? '#f0932b' : '#43e97b'
              }}>
                {epochData.gradient_norm.toFixed(4)}
                {epochData.gradient_norm > 10 && ' ⚠️ EXPLODING'}
                {epochData.gradient_norm < 0.01 && ' ⚠️ VANISHING'}
              </span>
            </div>
          )}
        </div>
      ) : (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px' }}>
          <div className="loader" style={{
            width: '50px', height: '50px',
            border: '4px solid rgba(0,242,254,0.2)',
            borderTopColor: '#00f2fe',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <div style={{ fontSize: '0.9rem', opacity: 0.7, textAlign: 'center' }}>
            AWAITING TRAINING DATA...<br />
            <span style={{ fontSize: '0.75rem' }}>Connecting to train.py</span>
          </div>
          <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      <div style={{ marginTop: '15px', opacity: 0.4, fontSize: '0.7rem', textAlign: 'center' }}>
        WEBSOCKET LIVE CONNECTION • PORT 8765
      </div>
    </div>
  )
}

function MetricCard({ label, value, color, large, icon }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.05)',
      padding: '12px',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      border: '1px solid rgba(255,255,255,0.1)',
      transition: 'all 0.3s ease'
    }}>
      <div style={{ fontSize: '0.7rem', color: '#888', marginBottom: '3px', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '5px' }}>
        {icon && <span>{icon}</span>}
        {label}
      </div>
      <div style={{
        fontSize: large ? '2.2rem' : '1.6rem',
        fontWeight: 'bold',
        color: color || '#fff',
        fontFamily: "'Outfit', sans-serif"
      }}>
        {value}
      </div>
    </div>
  )
}
