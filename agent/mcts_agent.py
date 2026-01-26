import torch
import numpy as np
import math
from utils.tree_node import TreeNode

class MCTSAgent:
    def __init__(self, model, env, n_simulations=50, cpuct=1.0, device='cpu', temperature=1.0):
        self.model = model
        self.env = env # copy of the environment for simulations
        self.n_simulations = n_simulations
        self.cpuct = cpuct # exploration constant
        self.device = device
        self.temperature = temperature
        self.action_map = [-1, 0, 1]

    def search(self, root_state, add_noise=True):
        """
        runs MCTS simulations starting from the current state.
        returns the best action found.
        """
        root = TreeNode(root_state, prior=1.0)

        # 1. expansion of root node (initialize root)
        self._evaluate_and_expand(root)
        
        # Add Dirichlet noise to root for exploration (AlphaZero technique)
        if add_noise and len(root.children) > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([0.3] * len(actions))
            for i, action in enumerate(actions):
                root.children[action].prior = 0.75 * root.children[action].prior + 0.25 * noise[i]

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
        
        # 3. select best move with temperature-based sampling
        if self.temperature < 0.01:
            # Greedy: select most visited
            best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        else:
            # Sample based on visit counts with temperature
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count for a in actions])
            # Apply temperature
            visits_temp = visits ** (1.0 / self.temperature)
            probs = visits_temp / visits_temp.sum()
            # Sample an index, then get the action
            action_idx = np.random.choice(len(actions), p=probs)
            best_action = actions[action_idx]
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
         # Optimize: reuse tensor creation to minimize CPU-GPU transfers
        state_flat = node.state.flatten()
        
        # Check if already a tensor on correct device
        if isinstance(state_flat, torch.Tensor):
            state_tensor = state_flat.unsqueeze(0)
        else:
            # Create tensor directly on device (faster than .to(device))
            state_tensor = torch.tensor(state_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

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
        # Increase k for better exploration, especially early in training
        top_k_actions = self._sample_actions(policy_logits, k=30)

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
    

    def _sample_actions(self, logits, k=30):
        """
        Constructs k valid actions by sampling from the factorized logits.
        Ensures diversity by mixing sampling strategies.
        Uses heuristics to prefer sparse actions.
        """
        actions = set()
        # convert logits to probabilities using softmax with temperature
        probs = [torch.softmax(l / self.temperature, dim=1) for l in logits]

        # Strategy 1: Sample from distribution (exploration)    
        for _ in range(k):
            action_list = []
            for head_prob in probs:
                # sample index based on probabilities (happens on GPU)
                idx = torch.multinomial(head_prob, 1).item()
                val = self.action_map[idx]
                action_list.append(val)
            actions.add(tuple(action_list))
        
        # Strategy 2: Add some greedy actions (exploitation) - use GPU
        for _ in range(k // 3):
            action_list = []
            for head_prob in probs:
                idx = torch.argmax(head_prob, dim=1).item()
                val = self.action_map[idx]
                action_list.append(val)
            actions.add(tuple(action_list))
        
        # Strategy 3: Add random exploration with bias toward non-zero
        for _ in range(k // 4):
            action_list = []
            for _ in range(len(logits)):
                val = np.random.choice(self.action_map, p=[0.4, 0.2, 0.4])
                action_list.append(val)
            actions.add(tuple(action_list))
        
        # Strategy 4: Heuristic sparse actions (one-hot and two-hot patterns)
        # These are more likely to reduce residual effectively
        for _ in range(k // 4):
            action_list = [0] * len(logits)
            # Randomly set 1-3 positions to non-zero
            num_nonzero = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            positions = np.random.choice(len(logits), size=num_nonzero, replace=False)
            for pos in positions:
                action_list[pos] = np.random.choice([-1, 1])
            actions.add(tuple(action_list))
            
        return list(actions)
        
    def _parse_action(self, action_tuple):
        # Convert tuple of 12 ints to u, v, w arrays
        u = np.array(action_tuple[0:4])
        v = np.array(action_tuple[4:8])
        w = np.array(action_tuple[8:12])
        return u, v, w

