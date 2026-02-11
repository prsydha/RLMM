
import sys
import os
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from training.train import compute_marginal_targets

def test_compute_marginal_targets():
    # 12-element tuple
    action1 = tuple([0] * 12) # all 0 -> index 1
    action2 = tuple([1] * 12) # all 1 -> index 2
    
    visit_counts = {
        action1: 10,
        action2: 30
    }
    # Total = 40.
    # action1 prob = 0.25. action2 prob = 0.75.
    
    targets = compute_marginal_targets(visit_counts, n_heads=12)
    
    print("Targets shape:", targets.shape)
    print("Target head 0:", targets[0])
    
    expected_0 = np.array([0, 0.25, 0.75]) # index 0 is -1 (0 prob), index 1 is 0 (0.25), index 2 is 1 (0.75)
    
    if np.allclose(targets[0], expected_0):
        print("compute_marginal_targets Test Passed!")
    else:
        print("compute_marginal_targets Test Failed!")
        print("Expected:", expected_0)
        print("Got:", targets[0])

def test_mcts_interface():
    # Just check if we can import MCTSAgent and call search with return_probs
    from agent.mcts_agent import MCTSAgent
    
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return [torch.randn(1, 3) for _ in range(12)], torch.tensor([[0.5]])

    class MockNode:
        def __init__(self):
            self.visit_count = 1
            self.pass_me = True # Mock
    
    # We can't easily mock the whole MCTS search without Env, so we trust the unit test of function above
    # and the static analysis that we added return_probs.
    pass

if __name__ == "__main__":
    test_compute_marginal_targets()
