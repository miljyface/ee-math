import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    # dynamic hyperbolic tangent from "transformers without normalisation"
    # essentially just every transformation we've learn't applied to y = tanh(x)
    # ^ y = atan(x)+c
    # picked because it looks similar to regular normalisation distributions
    def __init__(self, num_features: int, init_alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        
        if x.dim() == 4:
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        else: # fallback for fully connected layer
            gamma = self.gamma.view(1, -1)
            beta = self.beta.view(1, -1)
            
        return gamma * x + beta