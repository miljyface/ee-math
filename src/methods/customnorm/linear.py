import torch
import torch.nn as nn
from typing import List

class PiecewiseLinear(nn.Module):
    # very jagged sine wave
    def __init__(self, num_features: int, breakpoints: List[float] = [-2.0, -1.0, 1.0, 2.0]):
        super().__init__()
        self.breakpoints = sorted(breakpoints)
        self.num_regions = len(breakpoints) + 1
        
        self.slopes = nn.Parameter(torch.ones(self.num_regions, num_features))
        self.intercepts = nn.Parameter(torch.zeros(self.num_regions, num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[1])
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.num_regions):
            if i == 0:
                mask = x_flat < self.breakpoints[0]
            elif i == self.num_regions - 1:
                mask = x_flat >= self.breakpoints[-1]
            else:
                mask = (x_flat >= self.breakpoints[i-1]) & (x_flat < self.breakpoints[i])
            
            output = torch.where(mask, 
                                self.slopes[i] * x_flat + self.intercepts[i],
                                output)
        
        return output.reshape(original_shape)