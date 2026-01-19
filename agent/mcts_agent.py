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
    
    def _evaluate_and_expand(self, node):
        """
        uses the neural network to evaluate the node and expand it.
        returns the value of the node.
        """
         # prepare state for network

         # standard "Dense" (linear) layers in neural networks expect a flat vector as input
        state_tensor = torch.FloatTensor(node.state.flatten()).unsqueeze(0).to(self.device) # flatten the multidim array, convert to a PyTorch tensor with high-precision decimals to calculate gradients, add fake batch dimension, move to device( gpu or cpu)

        with torch.no_grad(): # disable gradient calculation for inference
            policy_logits, value = self.model(state_tensor)

        value = value.item() # get scalar value from tensor

        # check if this state solves the environment
        # in a real run, we might simulate 1 step of environment here
        norm = np.linalg.norm(node.state)
        if norm < 1e-5:
            return 1.0 # solved!
        
        # expansion: we cannot add all possible actions due to combinatorial explosion
        # so we sample the top K most probable actions from the policy head
        top_k_actions = self._sample_actions(policy_logits, k=10)

        # instantiate child nodes for each action
        for action in top_k_actions:
            if action not in node.children:
                # calculate the next state (residual)
                u, v, w = self._parse_action(action)

                # we need a temporary copy of the environment to simulate the step
                # for simplicity, we can implement the tensor update math directly:
                update = np.einsum('i, j, k -> ijk', u, v, w)
                next_state = node.state - update

                # create child node with prior probability from policy head
                # Prior is product of probabilities from the 12 heads? 
                # Roughly, we can just assign uniform or derived prior.
                child = TreeNode(next_state, parent=node, prior=1.0/len(top_k_actions))  ####### to be refined 
                node.children[action] = child
                
        return value
    
    def _backpropagate(self, search_path, value):
        """
        backpropagates the value up the search path, updating visit counts and value sums.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # for two-player games, we would invert the value here
            # value = -value
            # for single-player optimization, value stays positive
    

    def _sample_actions(self, logits, k=10):
        """
        Constructs k valid actions by sampling from the factorized logits.
        """
        actions = []
        # convert logits to probabilities using softmax
        probs = [torch.softmax(l, dim = 1) for l in logits]

        for _ in range(k):
            action_list = []
            for head_prob in probs:
                # sample index based on probabilities
                idx = torch.multinomial(head_prob, 1).item()
                val = self.action_map[idx]
                action_list.append(val)
            actions.append(tuple(action_list))
        return list(set(actions)) # return unique actions only
        
    def _parse_action(self, action_tuple):
        # Convert tuple of 12 ints to u, v, w arrays
        u = np.array(action_tuple[0:4])
        v = np.array(action_tuple[4:8])
        w = np.array(action_tuple[8:12])
        return u, v, w

