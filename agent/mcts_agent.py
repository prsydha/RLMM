import torch
import numpy as np
import math
from utils.tree_node import TreeNode
from utils.encode_input import encode_state

VIRTUAL_LOSS = 3.0

class MCTSAgent:
    def __init__(self, model, env, n_simulations=100, cpuct=1.0, device='cpu'):
        self.model = model
        self.env = env # copy of the environment for simulations
        self.n_simulations = n_simulations
        self.cpuct = cpuct # exploration constant
        self.device = device
        self.action_map = [-1, 0, 1]

    def search(self, root_state, return_probs=False):
        """
        runs MCTS simulations starting from the current state.
        If return_probs is True, returns (best_action, visit_distribution_map).
        """
        # 1. expansion of root node (initialize root)
        root = TreeNode(root_state, prior=1.0)

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

        if return_probs:
            # return a dictionary: {action_tuple: visit_count}
            # this represnts the full "posterior" distribution found by MCTS
            visit_counts = {action: child.visit_count for action, child in root.children.items()}
            return best_action, visit_counts
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
            if child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
            else:
                q_value = 0.0

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
            # zeros_count = action.count(0)
            # sparsity_bonus = 0.05 * (zeros_count / len(action))

            score = q_value + u_value  # + sparsity_bonus

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

        # check if node is terminal
        if hasattr(node, 'is_terminal') and node.is_terminal:
            return node.value_sum / max(1, node.visit_count) # return stored value
        
        # check if this state solves the environment
        # in a real run, we might simulate 1 step of environment here
        norm = np.linalg.norm(node.state)
        if norm < 1e-5:
            node.is_terminal = True # mark as terminal to avoid re-evaluation
            return 1.0 # solved!
        
        encoded = encode_state(node.state)
        
         # prepare state for network
         # standard "Dense" (linear) layers in neural networks expect a flat vector as input
        state_tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device) # flatten the multidim array, convert to a PyTorch tensor with high-precision decimals to calculate gradients, add fake batch dimension, move to device( gpu or cpu)

        with torch.no_grad(): # disable gradient calculation for inference
            policy_logits, value = self.model(state_tensor)

        value = value.item() # get scalar value from tensor
        
        # expansion: we cannot add all possible actions due to combinatorial explosion
        # so we sample the K most probable actions from the policy head
        top_k_actions = self._sample_actions(policy_logits, k=10)

        # instantiate child nodes for each action
        for action, action_prob in top_k_actions:
            if action not in node.children:
                # calculate the next state (residual)
                u, v, w = self._parse_action(action)

                # we need a temporary copy of the environment to simulate the step
                # for simplicity, we can implement the tensor update math directly:
                update = np.einsum('i, j, k -> ijk', u, v, w)
                next_state = node.state - update

                # create child node with prior probability from policy head
                # Prior is product of probabilities from the 12 heads (joint distribution)
                child = TreeNode(next_state, parent=node, prior=action_prob)  # refined
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
    

    def _sample_actions(self, logits, k=15):
        """
        Samples k actions and calculates their joint probabilities.
        Returns: List of tuples: [(action_tuple, joint_probability), ...]
        """
        sampled_results = []
        # convert logits to probabilities using softmax (shape: [batch=1, num_actions])
        probs = [torch.softmax(l, dim = 1) for l in logits]

        # Determine vector length (4 for 2x2, 9 for 3x3)
        L = len(logits) // 3

        attempts = 0
        max_attempts = k * 10 # Safety break to prevent infinite loops

        while len(sampled_results) < k and attempts < max_attempts:
            attempts += 1
            action_list = []
            joint_prob = 1.0

            for head_prob in probs:
                # head_prob shape: (1, n_actions)
                # sample 1 index based on probability distribution
                idx_tensor = torch.multinomial(head_prob, 1)
                idx = idx_tensor.item()

                # Get the actual probability of choosing this specific index
                prob_of_idx = head_prob[0, idx].item()

                # map index to action value (-1, 0, 1)
                val = self.action_map[idx]

                action_list.append(val)
                joint_prob *= prob_of_idx # product of probabilities for joint distribution

            # split into u, v, w vectors
            u = action_list[0:L]
            v = action_list[L:2*L]
            w = action_list[2*L:3*L]

            # validation: ensure that u, v, w are not all zero vectors
            is_u_zero = sum(abs(x) for x in u) == 0
            is_v_zero = sum(abs(x) for x in v) == 0
            is_w_zero = sum(abs(x) for x in w) == 0

            if is_u_zero or is_v_zero or is_w_zero:
                continue # Reject and try again
            
            sampled_results.append((tuple(action_list), joint_prob))

        # deduplication scheme: take the first occurrence
        # here we just use a dict(keeps only unique keys) to keep unique actions and their most recent probability
        unique_actions = {act: prob for act, prob in sampled_results}
        return unique_actions.items()
        
    def _parse_action(self, action_tuple):
        # Convert tuple of 12 ints to u, v, w arrays
        u = np.array(action_tuple[0:4])
        v = np.array(action_tuple[4:8])
        w = np.array(action_tuple[8:12])
        return u, v, w
    
    def batch_search(self, root_state, simulations=100, batch_size=8):
        root = TreeNode(root_state, prior=1.0)
        
        # Expand root first to populate children
        self._evaluate_and_expand(root) 
        
        # Add Dirichlet noise to root (optional, good for training)
        self._add_dirichlet_noise(root)

        num_batches = simulations // batch_size
        
        for _ in range(num_batches):
            leaves = []
            search_paths = []
            
            # --- PHASE 1: SELECTION (Parallel Descent) ---
            for _ in range(batch_size):
                node = root
                path = [node]
                
                # Traverse until we hit a leaf (unexpanded node) or terminal state
                while node.is_expanded():
                    # Pick best child using UCB
                    # IMPORTANT: UCB calculation must see the temporary Virtual Loss!
                    action, node = self._select_child(node)
                    path.append(node)
                    
                    # APPLY VIRTUAL LOSS IMMEDIATELY
                    # We pretend we visited this node and it was a "loss"
                    node.visit_count += 1
                    node.value_sum -= VIRTUAL_LOSS 
                
                search_paths.append(path)
                leaves.append(node)
                
            # --- PHASE 2: BATCH EVALUATION ---
            # Collect states from all leaves that need expansion
            # (Some leaves might be terminal, so we filter)
            expandable_leaves = [(i, node) for i, node in enumerate(leaves) 
                                 if not node.is_expanded() and not self._is_terminal(node)]
            
            if expandable_leaves:
                indices, nodes = zip(*expandable_leaves)
                
                # Prepare batch for GPU
                states = [torch.FloatTensor(n.state) for n in nodes]
                state_batch = torch.stack(states).to(self.device)
                
                # ONE big forward pass instead of 8 small ones
                with torch.no_grad():
                    policy_logits, values = self.model(state_batch)
                
                # Expand all nodes
                for i, idx in enumerate(indices):
                    node = leaves[idx]
                    probs = torch.softmax(policy_logits[i], dim=0)
                    est_value = values[i].item()
                    
                    # Create children
                    self._expand_node_from_probs(node, probs)
                    
                    # Store the REAL value to backpropagate
                    # We store it in the path list or a temporary variable
                    search_paths[idx][-1].temp_value = est_value

            # --- PHASE 3: BACKPROPAGATION & CORRECTION ---
            for i, path in enumerate(search_paths):
                leaf = path[-1]
                
                # Determine the value to propagate
                # If it was terminal, calculate reward. If expanded, use NN value.
                if hasattr(leaf, 'temp_value'):
                    value = leaf.temp_value
                    del leaf.temp_value # Clean up
                else:
                    # It was terminal or we couldn't evaluate
                    value = self._get_terminal_value(leaf)

                for node in reversed(path):
                    # 1. REMOVE VIRTUAL LOSS (Undo the "lie")
                    node.visit_count -= 1
                    node.value_sum += VIRTUAL_LOSS
                    
                    # 2. ADD REAL UPDATE
                    node.visit_count += 1
                    node.value_sum += value
                    
        return self._select_best_action(root)

