import torch
import numpy as np
import math
from utils.tree_node import TreeNode

class MCTSAgent:
    def __init__(self, model, env, n_simulations=50, cpuct=1.0, device='cpu'):
        self.model = model
        self.env = env # copy of the environment for simulations
        self.n_simulations = n_simulations
        self.cpuct = cpuct # exploration constant
        self.device = device
        self.action_map = [-1, 0, 1]

    def search(self, root_state):
        """
        runs MCTS simulations starting from the current state.
        returns the best action found.
        """
        root = TreeNode(root_state, prior=1.0)

        # 1. expansion of root node (initialize root)
        self._expand_node(root)

        # 2. simulation loop
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            # A. Selection
            while not node.is_leaf():
                action, node = self._select_child(node)
                search_path.append(node)

            # B. Expansion and Evaluation
            # if the node represents a terminal state(solved), value is high
            # otherwise, use the neural network to evaluate
            value = self._evaluate_and_expand(node)

            # C. Backpropagation
            self._backpropagate(search_path, value)
        
        # 3. select best move (greedy with respect to visit count)
        # return the action with the most visits
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action
    
    def _select_child(self, node):
        """
        selects the child with the highest Upper Confidence Bound (UCB) score.
        Addition: adds a 'sparsity bonus to the score'
        """
        best_score = -float('inf') # negative infinity
        best_action = None
        best_child = None

        for action, child in node.children.items():
            # "quality" value : "Based on my actual experience so far, how much do I like this move?"
            # Q(s,a)=∑V/N(s,a)​
            # Experience / Success
            q_value = child.value()

            # the u-value : Upper confidence bound value
            # represents how much 'potential' or 'uncertainty' there is about this move
            # U(s,a) = cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # where,
            # Cpuct​ : a constant that controls how aggressive the exploration is. (PUCT stands for: Predictor + Upper Confidence bound applied to Trees.)
            # P(s,a) : the prior probability of selecting action a in state s (from neural network)
            # N(s) : total visit count for parent node s
            # N(s,a) : visit count for child node (s,a)
            # Potential / Curiosity
            u_value = self.cpuct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)

            # sparsity bonus
            # we add a small bonus if the action vector is sparse (i.e., has more zeros)
            # action is a tuple of 12 ints so count how many are 0.
            zeros_count = action.count(0)
            sparsity_bonus = 0.05 * (zeros_count / len(action))

            score = q_value + u_value + sparsity_bonus

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
        