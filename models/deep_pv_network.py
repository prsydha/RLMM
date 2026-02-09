import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# Residual Block: a common building block in deep networks that helps with training very deep architectures by allowing gradients to flow through skip connections. It is a structural design where the input to a layer is added directly to its output, allowing information to bypass certain layers.

# In traditional neural network, a layer tries to learn a direct mapping while in a residual block, the layers only learn the residual( the difference) needed to improve the current representation.

# How it helps mitigate the vanishing gradient problem: vanishing gradients occur when gradients become very small during backpropagation, making it difficult for the network to learn. Because of the +x skip connection, the gradient can flow directly through the network without being diminished by multiple layers of transformations, allowing for more effective training of deeper networks.

class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # LayerNorm ( Layer Normalization ) is a technique to keep the neural network stable by ensuring that the inputs to each layer have a consistent distribution( mean and variance ). Without normalization, numbers inside the network can become too large or too small, making training difficult.
        # LayerNorm performs three steps for each input: 1. Centering: subtract the mean of the inputs. 2. Scaling: divide by the standard deviation. 3. Learned Shift: multiplies the result by a learned weight (γ) and adds a learned bias (β).

        self.bn1 = nn.LayerNorm(hidden_dim) # LayerNorm is often better for MLPs than BatchNorm
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.fc1(x))) # GELU is the modern ReLU
        out = self.bn2(self.fc2(out))
        out += residual # The "Skip" connection
        return F.gelu(out)

class DeepTensorNet(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, hidden_dim=512, num_res_blocks=6, n_heads=config.N_HEADS):
        super().__init__()
        
        self.n_heads = n_heads
        
        # 1. Input Projection
        # We assume input is the flattened tensor.
        self.start_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 2. The Deep Torso (Multiple Residual Blocks)
        # Stacking 4-8 of these gives the net "room to think"
        self.backbone = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        
        # 3. Policy Head - Split into N_HEADS separate linear layers
        # Each head outputs 3 possibilities (-1, 0, 1)
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3) # 3 actions: -1, 0, 1
            ) for _ in range(n_heads)
        ])
        
        # 4. Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1), # Continuous prediction
            nn.Tanh() # Force output to -1 to 1 range
        )

    def forward(self, x):
        # Flatten if necessary
        x = self.start_block(x)
        x = self.backbone(x)
            
        # Heads
        # Return a list of tensors, one for each head
        policy_logits = [head(x) for head in self.policy_heads]
        value = self.value_head(x)
        
        return policy_logits, value