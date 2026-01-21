# ⚡ Matrix Multiplication Visualizer

A beautiful, real-time 3D visualization of matrix multiplication powered by Three.js, React, and WebSocket connectivity.

## ✨ Features

- **🎨 Stunning 3D Visualization**: Watch matrix operations come to life with smooth animations
- **🎮 Interactive Controls**: OrbitControls for 360° camera manipulation
- **📊 Real-time Stats**: Track steps, operations, and performance metrics
- **⚙️ Adjustable Speed**: Control animation speed from 0.1x to 3x
- **🔌 WebSocket Integration**: Connect to backend servers for live algorithm streaming
- **🎭 Demo Mode**: Automatic fallback with generated data when server unavailable
- **📱 Responsive Design**: Glassmorphism UI with modern gradients and effects
- **🎯 Formula Display**: See the current operation in mathematical notation

## 🚀 Quick Start

### Installation

```bash
cd visualizer
npm install
```

### Development

```bash
npm run dev
```

Open your browser at [http://localhost:5173](http://localhost:5173)

### Production Build

```bash
npm run build
npm run preview
```

## 🎮 Usage

### Controls

- **Start**: Begin the matrix multiplication animation
- **Pause/Resume**: Pause and resume the animation
- **Reset**: Clear all progress and reset to initial state
- **Speed Slider**: Adjust animation speed (0.1x - 3.0x)
- **Mouse Drag**: Rotate the 3D view (OrbitControls)
- **Mouse Scroll**: Zoom in/out

### WebSocket Integration

The visualizer automatically attempts to connect to `ws://localhost:8765`. If no server is available, it falls back to demo mode.

#### Expected WebSocket Message Format

```json
{
  "step": 1,
  "A_index": [0, 1],
  "B_index": [1, 0],
  "C_index": [0, 0],
  "A_value": 2,
  "B_value": 3,
  "C_matrix": [[6, 0], [0, 0]]
}
```

**Field Descriptions:**
- `step`: Current operation number
- `A_index`: [row, col] in matrix A being used
- `B_index`: [row, col] in matrix B being used
- `C_index`: [row, col] in result matrix C being updated
- `A_value`: Value from matrix A
- `B_value`: Value from matrix B
- `C_matrix`: Current state of result matrix C

### Demo Mode

When WebSocket connection fails, the visualizer generates a sample 2×2 matrix multiplication:
- Matrix A: `[[1,2],[3,4]]`
- Matrix B: `[[5,6],[7,8]]`
- Result C: Computed step-by-step

## 🎨 Visual Features

### Color Coding
- **Purple (#667eea)**: Matrix A cubes
- **Pink (#f5576c)**: Matrix B cubes
- **Blue (#4facfe)**: Result Matrix C cubes
- **Orange (#ff9500)**: Currently selected operands
- **Green (#00ff88)**: Result being updated

### Animations
- Smooth color transitions using linear interpolation
- Scale animations on cube interactions
- Gentle rotation for visual interest
- Value updates with sprite text labels

## 📁 Project Structure

```
visualizer/
├── src/
│   ├── components/
│   │   ├── MatrixVisualizer.jsx  # Main 3D visualization
│   │   ├── ControlPanel.jsx       # UI controls & stats
│   │   └── Cube.jsx               # Reusable cube component
│   ├── App.jsx                    # Root component
│   ├── main.jsx                   # Entry point
│   └── styles.css                 # Modern CSS with glassmorphism
├── index.html                      # HTML template
├── vite.config.js                 # Vite configuration
└── package.json                   # Dependencies
```

## 🔧 Tech Stack

- **React 18**: Modern hooks-based components
- **Three.js**: 3D graphics library
- **Vite**: Lightning-fast build tool
- **OrbitControls**: Camera manipulation
- **WebSocket**: Real-time data streaming

## 🎯 Integration with RL Pipeline

To integrate with your GPU RL pipeline:

1. Create a WebSocket server that emits step data
2. Connect the server to `agent_to_kernel.py` output
3. Stream multiplication steps as they happen
4. The visualizer will automatically display them in real-time

Example Python WebSocket server:

```python
import asyncio
import websockets
import json

async def handler(websocket, path):
    # Send multiplication steps
    for step in generate_steps():
        await websocket.send(json.dumps(step))
        await asyncio.sleep(0.5)

start_server = websockets.serve(handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

## 📊 Performance

- Smooth 60 FPS animations
- Efficient memory management with proper cleanup
- Shadow mapping for realistic lighting
- Optimized for modern browsers

## 🎓 Learning Resources

- **Three.js Docs**: [threejs.org/docs](https://threejs.org/docs/)
- **React Hooks**: [react.dev/reference/react](https://react.dev/reference/react)
- **WebSocket API**: [developer.mozilla.org/en-US/docs/Web/API/WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)

## 🤝 Contributing

This visualizer is part of the RLMM (Reinforcement Learning Matrix Multiplication) project. See the main repository for contribution guidelines.

## 📝 License

Part of the RLMM project. See main repository for license information.

---

**Built with ❤️ for visualizing GPU algorithm optimization**
