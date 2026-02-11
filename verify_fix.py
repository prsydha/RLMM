
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from agent.mcts_agent import MCTSAgent
    print("Successfully imported MCTSAgent")
    
    # Mock objects
    class MockModel:
        def __call__(self, x):
            return None, None
            
    class MockEnv:
        pass
        
    agent = MCTSAgent(model=MockModel(), env=MockEnv(), temperature=0.5)
    print(f"Successfully instantiated MCTSAgent with temperature: {agent.temperature}")
    
    agent_default = MCTSAgent(model=MockModel(), env=MockEnv())
    print(f"Successfully instantiated MCTSAgent with default temperature: {agent_default.temperature}")
    
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
