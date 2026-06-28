
import time
import logging
from training.visualizer_server import VisualizerServer

# Config logging
logging.basicConfig(level=logging.INFO)

print("Starting Test Server...")
server = VisualizerServer()
server.start()

print("Server started. Navigate to http://localhost:5173 to test connection.")
print("Run for 60 seconds...")

try:
    for i in range(60):
        server.broadcast({
            "type": "log", 
            "message": f"Test Heartbeat {i}", 
            "level": "info"
        })
        
        # Simulate stats update
        server.broadcast({
            "type": "stats",
            "data": {
                "step": i,
                "episode": 1,
                "reward": 0.0,
                "residual": 1.0 / (i + 1),
                "rank": 7,
                "elapsed": i
            }
        })
        
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")

server.stop()
print("Done.")
