import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    """
    Dynamic Tanh (DyT) normalization layer from "Transformers without Normalization"
    Formula: DyT(x) = γ * tanh(α * x) + β (Transformation of tanh)
    
    Args:
        num_features: Number of features (channels)
        init_alpha: Initial value for α (default: 0.5 for non-LLM models)
    """
    def __init__(self, num_features: int, init_alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W) or (B, C)
        x = torch.tanh(self.alpha * x)
        
        # Reshape gamma and beta for broadcasting
        if x.dim() == 4:  # Conv layer: (B, C, H, W)
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        else:  # FC layer: (B, C)
            gamma = self.gamma.view(1, -1)
            beta = self.beta.view(1, -1)
            
        return gamma * x + beta