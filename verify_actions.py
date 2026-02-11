
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from agent.mcts_agent import MCTSAgent
    
    # Create a mock agent with necessary attributes
    class MockModel: pass
    class MockEnv: pass
    
    agent = MCTSAgent(model=MockModel(), env=MockEnv(), temperature=1.0)
    
    # Mock prob logits (12 heads, 3 actions each)
    # n_heads = 12, actions=3
    logits = [torch.randn(1, 3) for _ in range(12)]
    
    print("Testing _sample_actions...")
    actions = agent._sample_actions(logits, k=5)
    
    print(f"Successfully sampled {len(actions)} actions")
    for action, prob in actions:
        print(f"Action type: {type(action)}, Prob: {prob:.4f}")
        if len(action) != 12:
            raise ValueError(f"Action length mismatch: expected 12, got {len(action)}")
            
    print("Test passed!")

except Exception as e:
    print(f"Test Failed: {e}")
    # Print full traceback
    import traceback
    traceback.print_exc()
    sys.exit(1)
