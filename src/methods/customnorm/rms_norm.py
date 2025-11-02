import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in LLaMA and modern language models
    """
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4: 
            mean_square = x.pow(2).mean(dim=1, keepdim=True)
            x = x / torch.sqrt(mean_square + self.eps)
            return x * self.gamma.view(1, -1, 1, 1)
        else:
            mean_square = x.pow(2).mean(dim=1, keepdim=True)
            x = x / torch.sqrt(mean_square + self.eps)
            return x * self.gamma.view(1, -1)