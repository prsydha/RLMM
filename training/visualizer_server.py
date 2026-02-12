
import asyncio
import json
import threading
import websockets
import time

class VisualizerServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.thread = None
        self.running = False
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def runner():
            print(f"Visualizer Server starting on ws://{self.host}:{self.port}")
            async with websockets.serve(self._handler, self.host, self.port):
                print(f"Visualizer Server running...")
                while self.running:
                    await asyncio.sleep(0.1)

        try:
            self.loop.run_until_complete(runner())
        except Exception as e:
            print(f"Visualizer Server Error: {e}")
        finally:
            self.loop.close()
        
    async def _handler(self, websocket, *args):
        self.clients.add(websocket)
        print(f"Visualizer client connected: {websocket.remote_address}")
        try:
            # Keep connection open
            async for message in websocket:
                pass # Ignore incoming messages
        except Exception as e:
            print(f"Visualizer client error: {e}")
        finally:
            self.clients.remove(websocket)
            print(f"Visualizer client disconnected: {websocket.remote_address}")
            
    def broadcast(self, data):
        if not self.clients:
            return
            
        # Convert numpy types to native python types for JSON serialization
        import numpy as np
        
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
            
        try:
            message = json.dumps(data, default=convert, allow_nan=False)
        except ValueError:
            # Handle NaN/Infinity by replacing with null or 0
            # simplejson can do ignore_nan, but standard json raises ValueError if allow_nan=False
            # Let's do a manual cleanup if needed, or just allow_nan=True (default) but JS hates it.
            # Safe strategy: allow_nan=True creates invalid JSON for JS. 
            # We must clean it. 
            # For now, let's just log error and return
            print("Visualizer Broadcast Error: Data contains NaN or Infinity")
            return

        # Schedule the send in the event loop
        if self.loop and self.loop.is_running():
            print(f"Visualizer: Broadcasting to {len(self.clients)} clients") 
            asyncio.run_coroutine_threadsafe(self._broadcast_message(message), self.loop)
        else:
            print("Visualizer Error: Loop is not running")

    async def _broadcast_message(self, message):
        if self.clients:
            results = await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
            for res in results:
                if isinstance(res, Exception):
                    print(f"Visualizer Send Error: {res}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
