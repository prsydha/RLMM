import React, { useEffect, useRef } from 'react'

export default function TerminalPanel({ logs }) {
    const endRef = useRef(null)

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [logs])

    return (
        <div className="terminal-panel">
            <div className="terminal-header">
                <span className="terminal-dot red"></span>
                <span className="terminal-dot yellow"></span>
                <span className="terminal-dot green"></span>
                <span className="terminal-title">user@rl-matrix-core:~</span>
            </div>
            <div className="terminal-content">
                {logs.length === 0 && (
                    <div className="log-entry system">
                        <span className="timestamp">{new Date().toLocaleTimeString()}</span>
                        <span className="message">System initialized. Waiting for GPU tasks...</span>
                    </div>
                )}
                {logs.map((log, i) => (
                    <div key={i} className={`log-entry ${log.type}`}>
                        <span className="timestamp">[{log.time}]</span>
                        <span className="prompt">$</span>
                        <span className="message">{log.message}</span>
                    </div>
                ))}
                <div ref={endRef} />
            </div>
        </div>
    )
}
