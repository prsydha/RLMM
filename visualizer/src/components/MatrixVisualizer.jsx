import React, { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

/**
 * MatrixVisualizer - Enhanced Three.js visualization with:
 * - OrbitControls for camera manipulation
 * - Text labels on cubes showing values
 * - WebSocket support with auto-reconnect
 * - Demo mode fallback
 */
export default function MatrixVisualizer({
  running,
  paused,
  speed = 1.0,
  onStatsUpdate,
  onConnectionChange,
  onLog
}) {
  const mountRef = useRef(null)
  const [step, setStep] = useState(0)
  const startTimeRef = useRef(null)
  const beamsRef = useRef([]) // Store active beams

  // Matrix state: initialize 2x2 demo
  const [A, setA] = useState([[1, 0], [0, 1]])
  const [B, setB] = useState([[1, 1], [0, 1]])
  const [C, setC] = useState([[0, 0], [0, 0]])

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    // FIX: Ensure we have dimensions from parent if mount has none yet
    const width = mount.clientWidth || mount.parentElement.clientWidth || window.innerWidth
    const height = mount.clientHeight || mount.parentElement.clientHeight || window.innerHeight

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x050508)
    scene.fog = new THREE.Fog(0x050508, 10, 50)

    // Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000)
    camera.position.set(0, 10, 20)
    camera.lookAt(0, 0, 0)

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.2
    mount.appendChild(renderer.domElement)

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = 5
    controls.maxDistance = 30
    controls.maxPolarAngle = Math.PI / 2
    controls.target.set(0, 0, 0)

    // Enhanced Lighting for Glass
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8) // Brighter ambient for glass
    scene.add(ambientLight)

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.0)
    mainLight.position.set(5, 10, 7)
    mainLight.castShadow = true
    scene.add(mainLight)

    // Rim light for edges
    const rimLight = new THREE.PointLight(0x00f2fe, 1, 20)
    rimLight.position.set(0, 5, -5)
    scene.add(rimLight)

    const fillLight = new THREE.DirectionalLight(0x667eea, 0.3)
    fillLight.position.set(-5, 5, -5)
    scene.add(fillLight)

    const backLight = new THREE.PointLight(0xf5576c, 0.5)
    backLight.position.set(0, 3, -8)
    scene.add(backLight)

    // Grid Helper
    const gridHelper = new THREE.GridHelper(40, 40, 0x2a2b38, 0x151620)
    gridHelper.position.y = -2
    scene.add(gridHelper)

    // Constants for cube layout
    const gap = 0.5
    const size = 1.0

    // Data structures to track meshes and their target states
    const meshes = { A: [], B: [], C: [] }
    const targetColors = { A: [], B: [], C: [] }
    const targetScales = { A: [], B: [], C: [] }

    /**
     * Create a text sprite for displaying values on cubes
     */
    function createTextSprite(text, color = '#ffffff') {
      const canvas = document.createElement('canvas')
      const context = canvas.getContext('2d')
      // High resolution canvas
      canvas.width = 512
      canvas.height = 512

      context.fillStyle = color
      // Large font for downscaling
      context.font = 'Bold 280px "Roboto Mono", "Courier New", monospace'
      context.textAlign = 'center'
      context.textBaseline = 'middle'

      // Add slight shadow for better contrast against glass
      context.shadowColor = 'rgba(0,0,0,0.8)'
      context.shadowBlur = 10
      context.shadowOffsetX = 4
      context.shadowOffsetY = 4

      context.fillText(text, 256, 256)

      const texture = new THREE.CanvasTexture(canvas)
      texture.minFilter = THREE.LinearFilter // Better scaling

      const spriteMaterial = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 1.0,
        depthTest: false // Ensure text overlays glass
      })
      const sprite = new THREE.Sprite(spriteMaterial)
      // Scale down to physical size
      sprite.scale.set(0.6, 0.6, 1)
      return sprite
    }

    /**
     * Create a grid of matrix cubes with labels
     */
    function createGrid(matrix, label, xOffset, color) {
      const rows = matrix.length
      const cols = matrix[0].length
      // Center the grid: 
      // Width = cols * size + (cols-1) * gap
      // Left = xOffset - Width/2 + size/2
      const gridWidth = cols * size + (cols - 1) * gap
      const gridHeight = rows * size + (rows - 1) * gap

      const startX = xOffset - gridWidth / 2 + size / 2
      const startY = gridHeight / 2 - size / 2

      for (let i = 0; i < rows; ++i) {
        for (let j = 0; j < cols; ++j) {
          const x = startX + j * (size + gap)
          const y = startY - i * (size + gap)
          const z = 0

          // Enhanced TENSOR Material: Glass + Wireframe
          const geometry = new THREE.BoxGeometry(size, size, size)

          // 1. Physical Glass Material
          const material = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(color),
            metalness: 0.1,
            roughness: 0.05, // Very smooth
            transmission: 0.6, // Glass-like transparency
            thickness: 0.5, // Volume
            ior: 1.5, // Refraction
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide
          })

          const mesh = new THREE.Mesh(geometry, material)
          mesh.position.set(x, y, z)
          mesh.castShadow = true
          mesh.receiveShadow = true

          // 2. Wireframe / Edges
          const edges = new THREE.EdgesGeometry(geometry)
          const lineMaterial = new THREE.LineBasicMaterial({
            color: new THREE.Color(color).multiplyScalar(1.5), // Bright neon edge
            transparent: true,
            opacity: 0.8,
            linewidth: 2
          })
          const wireframe = new THREE.LineSegments(edges, lineMaterial)
          mesh.add(wireframe)

          // Add text label with render order to appear on top
          const value = matrix[i][j]
          const textSprite = createTextSprite(value.toString(), '#ffffff')
          textSprite.position.z = 0 // Center inside (or slightly front)
          textSprite.renderOrder = 1 // Ensure it renders after glass
          mesh.add(textSprite)
          mesh.userData.textSprite = textSprite
          mesh.userData.value = value

          scene.add(mesh)
          meshes[label].push(mesh)
          targetColors[label].push(new THREE.Color(color))
          targetScales[label].push(new THREE.Vector3(1, 1, 1))
        }
      }

      // Add matrix label
      const labelSprite = createTextSprite(label, color)
      labelSprite.position.set(xOffset, startY + (size + gap), 0)
      labelSprite.scale.set(1.5, 1.5, 1)
      scene.add(labelSprite)
    }

    // Create matrices with distinct neon colors
    // Layout: A (Left) -> B (Middle) -> C (Right)
    createGrid(A, 'A', -9, 0x00f2fe) // Cyan
    createGrid(B, 'B', -2, 0xb224ef) // Purple
    createGrid(C, 'C', 7, 0x43e97b)  // Green

    let highlightTimeouts = []
    let animationId = null
    let elapsedTime = 0

    /**
     * Smoothly animate a cube highlight
     */
    function highlight(mesh, color = 0xffaa00, duration = 600) {
      const idx = meshes.A.indexOf(mesh) !== -1 ? meshes.A.indexOf(mesh) :
        meshes.B.indexOf(mesh) !== -1 ? meshes.B.indexOf(mesh) :
          meshes.C.indexOf(mesh)

      const label = meshes.A.includes(mesh) ? 'A' :
        meshes.B.includes(mesh) ? 'B' : 'C'

      if (idx === -1) return

      const originalColor = label === 'A' ? 0x00f2fe :
        label === 'B' ? 0xb224ef : 0x43e97b

      const highlightColor = 0xffffff

      targetColors[label][idx].setHex(highlightColor)
      targetScales[label][idx].set(1.3, 1.3, 1.3)

      const timeout = setTimeout(() => {
        targetColors[label][idx].setHex(originalColor)
        targetScales[label][idx].set(1, 1, 1)
      }, duration / speed)

      highlightTimeouts.push(timeout)
    }

    /**
     * Draw a laser beam between two points
     */
    function createBeam(startPos, endPos, color = 0x00f2fe) {
      if (!scene) return

      const points = [startPos, endPos]
      const geometry = new THREE.BufferGeometry().setFromPoints(points)
      const material = new THREE.LineBasicMaterial({
        color: color,
        linewidth: 2,
        transparent: true,
        opacity: 0.8
      })

      const line = new THREE.Line(geometry, material)
      scene.add(line)

      // Store reference to remove later
      beamsRef.current.push({ mesh: line, createdAt: Date.now() })
    }

    /**
     * Update cube value and text sprite
     */
    function updateCubeValue(mesh, newValue) {
      if (!mesh || !mesh.userData.textSprite) return

      // Remove old sprite
      mesh.remove(mesh.userData.textSprite)
      mesh.userData.textSprite.material.map.dispose()
      mesh.userData.textSprite.material.dispose()

      // Create new sprite (High-Res)
      const formattedValue = Number.isInteger(newValue) ? newValue.toString() : newValue.toFixed(1)
      const textSprite = createTextSprite(formattedValue, '#ffffff')
      textSprite.position.z = 0
      textSprite.renderOrder = 1
      mesh.add(textSprite)
      mesh.userData.textSprite = textSprite
      mesh.userData.value = newValue
    }

    /**
     * Animation loop with smooth interpolation
     */
    function animate() {
      animationId = requestAnimationFrame(animate)

      controls.update()

      // Smoothly interpolate colors and scales
      Object.keys(meshes).forEach(label => {
        meshes[label].forEach((mesh, idx) => {
          // Color lerp
          mesh.material.color.lerp(targetColors[label][idx], 0.1)

          // Scale lerp
          mesh.scale.lerp(targetScales[label][idx], 0.15)

          // Gentle rotation for visual interest
          mesh.rotation.y += 0.002
        })
      })

      // Update elapsed time
      if (running && !paused && startTimeRef.current) {
        elapsedTime = ((Date.now() - startTimeRef.current) / 1000).toFixed(1)
      }

      // Fade out beams
      const now = Date.now()
      beamsRef.current = beamsRef.current.filter(beam => {
        const age = now - beam.createdAt
        if (age > 600 / speed) {
          beam.mesh.geometry.dispose()
          beam.mesh.material.dispose()
          scene.remove(beam.mesh)
          return false
        } else {
          beam.mesh.material.opacity = 1 - (age / (600 / speed))
          return true
        }
      })

      renderer.render(scene, camera)
    }

    // WebSocket and demo logic
    let ws = null
    let demoInterval = null
    let reconnectTimeout = null
    const reconnectDelay = 3000

    function connectWebSocket() {
      try {
        ws = new WebSocket('ws://localhost:8765')

        ws.onopen = () => {
          console.log('✅ WebSocket connected')
          onConnectionChange(true)
        }

        ws.onmessage = (e) => {
          try {
            const payload = JSON.parse(e.data)
            handleMessage(payload)
          } catch (err) {
            console.warn('Invalid payload', err)
          }
        }

        ws.onclose = () => {
          console.log('❌ WebSocket disconnected')
          onConnectionChange(false)

          // Auto-reconnect
          if (running) {
            reconnectTimeout = setTimeout(connectWebSocket, reconnectDelay)
          }

          // Start demo if running (with delay to avoid instant loop)
          if (running && !demoInterval) {
            setTimeout(() => {
              if (running && !demoInterval) startDemo()
            }, 2000)
          }
        }

        ws.onerror = (err) => {
          console.error('WebSocket error:', err)
        }
      } catch (err) {
        console.error('Failed to connect WebSocket:', err)
        onConnectionChange(false)
        if (running) startDemo()
      }
    }

    /**
     * Generate and run demo animation
     */
    function startDemo() {
      if (demoInterval) return

      // Randomize matrices for next run (values -1, 0, 1)
      const getRand = () => Math.floor(Math.random() * 3) - 1
      const newA = [[getRand(), getRand()], [getRand(), getRand()]]
      const newB = [[getRand(), getRand()], [getRand(), getRand()]]

      // Update state and meshes
      setA(newA)
      setB(newB)
      setC([[0, 0], [0, 0]]) // Reset C

      // Update cubes text/value
      // Note: We need to update existing meshes or force re-render. 
      // Simplest is to update values directly on meshes.
      // But meshes are created in useEffect dependent on [running]. 
      // Actually we should just update the values in meshes.

      const flatA = newA.flat()
      meshes.A.forEach((m, i) => updateCubeValue(m, flatA[i]))

      const flatB = newB.flat()
      meshes.B.forEach((m, i) => updateCubeValue(m, flatB[i]))

      meshes.C.forEach(m => updateCubeValue(m, 0))

      const seq = []
      const rows = newA.length
      const cols = newB[0].length
      const inner = newA[0].length

      // Generate multiplication sequence
      for (let i = 0; i < rows; ++i) {
        for (let j = 0; j < cols; ++j) {
          for (let k = 0; k < inner; ++k) {
            seq.push({
              step: seq.length + 1,
              A_index: [i, k],
              B_index: [k, j],
              C_index: [i, j],
              A_value: A[i][k],
              B_value: B[k][j],
              C_matrix: null
            })
          }
        }
      }

      let s = 0
      const baseInterval = 800

      demoInterval = setInterval(() => {
        if (!running || paused) return
        if (s >= seq.length) {
          clearInterval(demoInterval)
          demoInterval = null
          return
        }

        const payload = seq[s]
        const [i, j] = payload.C_index
        const val = payload.A_value * payload.B_value
        const newC = C.map(row => [...row])
        newC[i][j] += val
        payload.C_matrix = newC

        handleMessage(payload)
        s++
      }, baseInterval / speed)

      if (onLog) onLog(`Demo sequence started: ${rows}x${cols} Matrix Multiply`, 'info')
    }

    function stopDemo() {
      if (demoInterval) {
        clearInterval(demoInterval)
        demoInterval = null
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
        reconnectTimeout = null
      }
    }

    /**
     * Handle incoming multiplication step
     */
    /**
     * Handle incoming messages
     */
    function handleMessage(data) {
      // 1. Handle Training Step (Rank-1 Update)
      if (data.type === 'step') {
        setStep(data.global_step)
        const { u, v, w } = data.action

        // Helper to update a matrix grid based on a vector
        const updateGrid = (label, vector, colorHex) => {
          if (!vector || !meshes[label]) return

          meshes[label].forEach((mesh, idx) => {
            const val = vector[idx] || 0
            updateCubeValue(mesh, val)

            // Highlight non-zero values
            if (val !== 0) {
              const color = new THREE.Color(colorHex)
              targetColors[label][idx].copy(color)
              targetScales[label][idx].set(1.2, 1.2, 1.2)

              // Reset after a short delay
              setTimeout(() => {
                targetColors[label][idx].setHex(colorHex)
                targetScales[label][idx].set(1, 1, 1)
              }, 200)
            } else {
              // Dim zero values
              targetColors[label][idx].setHex(0x333333)
            }
          })
        }

        // Update A (Cyan), B (Purple), C (Green)
        updateGrid('A', u, 0x00f2fe)
        updateGrid('B', v, 0xb224ef)
        updateGrid('C', w, 0x43e97b)

        // Draw beams for interactions 
        // Logic: Connect any active element in A/B to the active elements in C
        // To reduce clutter, we can connect "center of mass" of A's update to center of mass of C's update?
        // Or connect all active A -> all active C. Since tensors are sparse (rank-1 mostly zeros), 
        // this shouldn't be too many lines. Let's try direct connections.

        const activeA = []
        meshes.A.forEach((m, i) => { if (Math.abs(u[i]) > 0.1) activeA.push(m) })

        const activeB = []
        meshes.B.forEach((m, i) => { if (Math.abs(v[i]) > 0.1) activeB.push(m) })

        const activeC = []
        meshes.C.forEach((m, i) => { if (Math.abs(w[i]) > 0.1) activeC.push(m) })

        // 1. Beams from A (Cyan) -> C (Green)
        activeA.forEach(aMesh => {
          activeC.forEach(cMesh => {
            // slightly perturbed start/end or curved lines would look cool, 
            // but straight lines with glow texture might be enough.
            createBeam(aMesh.position, cMesh.position, 0x00f2fe)
          })
        })

        // 2. Beams from B (Purple) -> C (Green)
        activeB.forEach(bMesh => {
          activeC.forEach(cMesh => {
            createBeam(bMesh.position, cMesh.position, 0xb224ef)
          })
        })

        // Stats
        onStatsUpdate({
          step: data.global_step,
          rank: data.rank,
          reward: data.reward,
          status: data.status
        })

        return
      }

      // 2. Handle Demo/Legacy Message (Element-wise)
      if (data.A_index) {
        setStep(data.step)

        const [ai, ak] = data.A_index
        const [bk, bj] = data.B_index
        const [ci, cj] = data.C_index

        const aMeshIdx = ai * A[0].length + ak
        const bMeshIdx = bk * B[0].length + bj
        const cMeshIdx = ci * C[0].length + cj

        const aMesh = meshes.A[aMeshIdx]
        const bMesh = meshes.B[bMeshIdx]
        const cMesh = meshes.C[cMeshIdx]

        // Highlight operands and result
        if (aMesh) {
          highlight(aMesh, 0xffffff, 500)
          // Draw beam from A to C
          if (cMesh) createBeam(aMesh.position, cMesh.position, 0x00f2fe)
        }
        if (bMesh) {
          highlight(bMesh, 0xffffff, 500)
          // Draw beam from B to C
          if (cMesh) createBeam(bMesh.position, cMesh.position, 0xb224ef)
        }
        if (cMesh) highlight(cMesh, 0xffffff, 600)

        // Emit Log
        const logMsg = `MAC: C[${ci},${cj}] += ${data.A_value.toFixed(2)} * ${data.B_value.toFixed(2)}`
        if (onLog) onLog(logMsg, 'calc')

        // Update C matrix
        if (data.C_matrix) {
          setC(data.C_matrix)

          // Update all C cube values and colors
          const flat = data.C_matrix.flat()
          const maxVal = Math.max(...flat.map(Math.abs), 1)

          meshes.C.forEach((m, idx) => {
            const val = flat[idx]
            updateCubeValue(m, val)

            // Color by magnitude (using Green gradient)
            const intensity = Math.min(1, Math.abs(val) / maxVal)
            // Interpolate between dark green and bright lime/white
            const color = new THREE.Color(0x43e97b)
            if (val !== 0) {
              color.lerp(new THREE.Color(0xffffff), intensity * 0.5)
            } else {
              color.multiplyScalar(0.3)
            }
            targetColors.C[idx].copy(color)
          })
        }

        // Update stats
        onStatsUpdate({
          step: data.step,
          totalOps: A.length * B[0].length * A[0].length,
          elapsed: elapsedTime,
          matrixSize: `${A.length}×${A[0].length}`
        })
      }
    }

    // Start logic based on running state
    const runWatcher = setInterval(() => {
      if (running && !paused) {
        if (!startTimeRef.current) {
          startTimeRef.current = Date.now()
        }
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          if (!demoInterval) {
            connectWebSocket()
          }
        }
      } else if (!running) {
        stopDemo()
        if (ws) ws.close()
        startTimeRef.current = null
        elapsedTime = 0
        // Clean up beams
        beamsRef.current.forEach(beam => {
          scene.remove(beam.mesh)
          beam.mesh.geometry.dispose()
          beam.mesh.material.dispose()
        })
        beamsRef.current = []
      }
    }, 300)

    // Start animation loop
    animate()

    // Handle window resize
    const handleResize = () => {
      const newWidth = mount.clientWidth || mount.parentElement.clientWidth || window.innerWidth
      const newHeight = mount.clientHeight || mount.parentElement.clientHeight || window.innerHeight
      camera.aspect = newWidth / newHeight
      camera.updateProjectionMatrix()
      renderer.setSize(newWidth, newHeight)
    }
    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      clearInterval(runWatcher)
      stopDemo()
      highlightTimeouts.forEach(t => clearTimeout(t))

      if (ws) ws.close()

      controls.dispose()
      mount.removeChild(renderer.domElement)

      // Dispose all meshes
      Object.values(meshes).flat().forEach(m => {
        if (m.userData.textSprite) {
          m.userData.textSprite.material.map.dispose()
          m.userData.textSprite.material.dispose()
        }
        m.geometry.dispose()
        m.material.dispose()
        scene.remove(m)
      })

      // Clean up beams
      beamsRef.current.forEach(beam => {
        if (beam.mesh) {
          scene.remove(beam.mesh)
          beam.mesh.geometry.dispose()
          beam.mesh.material.dispose()
        }
      })
      beamsRef.current = []

      renderer.dispose()
    }
  }, [running, paused, speed])

  return (
    <div className="visualizer" style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div className="canvas" ref={mountRef} style={{ width: '100%', height: '100%' }} />
      <div className="overlay-info">
        <div className="info-badge step-display">
          Step {step}
        </div>
      </div>
      <style>{`
        .visualizer canvas {
            display: block;
            width: 100% !important;
            height: 100% !important;
        }
        .overlay-info {
            position: absolute;
            bottom: 20px;
            right: 20px;
            pointer-events: none;
        }
        .info-badge {
            background: rgba(0,0,0,0.5);
            padding: 8px 12px;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            color: #fff;
            font-size: 12px;
            font-family: monospace;
        }
      `}</style>
    </div>
  )
}
