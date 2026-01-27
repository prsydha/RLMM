# defining standard node class to store the statistics for the search

import numpy as np

class TreeNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent

        self.children = {} # map from action to TreeNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior # probability from neural network

        self.is_terminal = False # track if this node has been solved

        self.sparsity_score = 0.0 # track sparsity of the move leading here

    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count 
    
    