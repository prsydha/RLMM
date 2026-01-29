import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block: a common building block in deep networks that helps with training very deep architectures by allowing gradients to flow through skip connections. It is a structural design where the input to a layer is added directly to its output, allowing information to bypass certain layers.

# In traditional neural network, a layer tries to learn a direct mapping while in a residual block, the layers only learn the residual( the difference) needed to improve the current representation.

# How it helps mitigate the vanishing gradient problem: vanishing gradients occur when gradients become very small during backpropagation, making it difficult for the network to learn. Because of the +x skip connection, the gradient can flow directly through the network without being diminished by multiple layers of transformations, allowing for more effective training of deeper networks.

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)

        # LayerNorm ( Layer Normalization ) is a technique to keep the neural network stable by ensuring that the inputs to each layer have a consistent distribution( mean and variance ). Without normalization, numbers inside the network can become too large or too small, making training difficult.
        # LayerNorm performs three steps for each input: 1. Centering: subtract the mean of the inputs. 2. Scaling: divide by the standard deviation. 3. Learned Shift: multiplies the result by a learned weight (γ) and adds a learned bias (β).

        self.bn1 = nn.LayerNorm(dim) # LayerNorm is often better for MLPs than BatchNorm
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.fc1(x))) # GELU is the modern ReLU
        out = self.bn2(self.fc2(out))
        out += residual # The "Skip" connection
        return F.gelu(out)

class DeepTensorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_blocks=6):
        super().__init__()
        
        # 1. Input Projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.LayerNorm(hidden_dim)
        
        # 2. The Deep Torso (Multiple Residual Blocks)
        self.torso = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 3. Policy Head
        # Output is 27 indices (for 2x2) with 3 possibilities each (-1, 0, 1)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 27 * 3) # Raw logits for cross-entropy
        )
        
        # 4. Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1) # Continuous prediction
        )

    def forward(self, x):
        # Flatten if necessary
        x = x.view(x.size(0), -1)
        
        # Torso
        x = F.gelu(self.input_bn(self.input_layer(x)))
        for block in self.torso:
            x = block(x)
            
        # Heads
        policy_logits = self.policy_head(x).view(-1, 27, 3)
        value = self.value_head(x)
        
        return policy_logits, value