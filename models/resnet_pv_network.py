import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ResidualBlock(nn.Module):
    """Residual block with layer normalization for stable training."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = F.gelu(self.fc1(x))  # GELU activation (smoother than ReLU)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual  # Skip connection


class PolicyValueNet(nn.Module):
    '''
    A neural network that outputs both policy logits and a value estimate.
    
    Improvements over vanilla architecture:
    - Residual connections for better gradient flow
    - Layer normalization for stable training
    - GELU activation (smoother gradients)
    - Deeper policy heads with attention-like mechanism
    - Separate value stream with its own processing
    '''

    def __init__(self, input_dim=config.INPUT_DIM,
                 hidden_dim=config.HIDDEN_DIM,
                 n_heads=config.N_HEADS,
                 n_actions=3,
                 n_residual_blocks=4,
                 dropout=0.1):
        super(PolicyValueNet, self).__init__()
        
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        # --- Input projection ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        
        # --- Shared body with residual blocks ---
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_residual_blocks)
        ])
        
        # --- Policy head ---
        # Deeper policy processing before the final heads
        self.policy_trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Separate heads for u, v, w (4 components each for 2x2)
        # Using slightly deeper heads for better action discrimination
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_actions)
            ) for _ in range(n_heads)
        ])
        
        # --- Value head ---
        # Separate value stream for better value estimation
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_ln(x)
        x = F.gelu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy stream
        policy_features = self.policy_trunk(x)
        policy_logits = [head(policy_features) for head in self.policy_heads]
        
        # Value stream (uses shared features, not policy features)
        value = torch.tanh(self.value_trunk(x))
        
        return policy_logits, value
    
    def get_action_probs(self, x):
        """Get action probabilities (for inference)."""
        policy_logits, value = self.forward(x)
        probs = [F.softmax(logits, dim=-1) for logits in policy_logits]
        return probs, value