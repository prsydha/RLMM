#!/usr/bin/env python3
"""
Simple WebSocket server for testing the Matrix Visualizer.
Streams a 2x2 matrix multiplication sequence to connected clients.

Usage:
    python server_demo.py

Then connect the visualizer at http://localhost:5173
"""

import asyncio
import websockets
import json
import time

# Sample matrices for demo
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[0, 0], [0, 0]]

def generate_multiplication_steps():
    """Generate all steps for matrix multiplication"""
    steps = []
    rows_a = len(A)
    cols_a = len(A[0])
    cols_b = len(B[0])
    
    step_count = 1
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                # Compute partial result
                result[i][j] += A[i][k] * B[k][j]
                
                # Create step payload
                step = {
                    "step": step_count,
                    "A_index": [i, k],
                    "B_index": [k, j],
                    "C_index": [i, j],
                    "A_value": A[i][k],
                    "B_value": B[k][j],
                    "C_matrix": [row[:] for row in result]  # Deep copy
                }
                steps.append(step)
                step_count += 1
    
    return steps

async def handle_client(websocket, path):
    """Handle a single WebSocket client connection"""
    client_addr = websocket.remote_address
    print(f"✅ Client connected: {client_addr}")
    
    try:
        # Generate multiplication steps
        steps = generate_multiplication_steps()
        
        # Send each step with a delay
        for step in steps:
            message = json.dumps(step)
            await websocket.send(message)
            print(f"📤 Sent step {step['step']}: C[{step['C_index'][0]},{step['C_index'][1]}] = {step['A_value']} × {step['B_value']}")
            await asyncio.sleep(0.8)  # 800ms between steps
        
        print(f"✔️  Completed sending {len(steps)} steps")
        
        # Keep connection open
        await websocket.wait_closed()
        
    except websockets.exceptions.ConnectionClosed:
        print(f"❌ Client disconnected: {client_addr}")
    except Exception as e:
        print(f"⚠️  Error: {e}")

async def main():
    """Start the WebSocket server"""
    server = await websockets.serve(handle_client, "localhost", 8765)
    print("🚀 WebSocket server started on ws://localhost:8765")
    print("📊 Matrix A:", A)
    print("📊 Matrix B:", B)
    print("⏳ Waiting for connections...\n")
    
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped")
