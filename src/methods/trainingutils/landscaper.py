import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader

class LossLandscapeAnalyzer:
    # analyzes loss landscape geometry using filter-normalized random directions
    # based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
    
    def __init__(self, model: nn.Module, criterion: nn.Module, dataloader: DataLoader, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
    
    def get_filter_normalized_directions(self, num_directions: int = 2) -> List[Dict]:  
        directions = []
        for _ in range(num_directions):
            direction = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    d = torch.randn_like(param)
                    if d.numel() > 1:
                        d = d / (d.norm() + 1e-10) * param.data.norm()
                    
                    direction[name] = d
            
            directions.append(direction)
        return directions
    
    def compute_loss_2d(self, direction1: Dict, direction2: Dict,
                        alpha_range: Tuple[float, float] = (-1, 1),
                        beta_range: Tuple[float, float] = (-1, 1),
                        num_points: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        original_params = {name: param.data.clone()
                          for name, param in self.model.named_parameters()
                          if param.requires_grad}
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
        betas = np.linspace(beta_range[0], beta_range[1], num_points)
        alpha_grid, beta_grid = np.meshgrid(alphas, betas)
        loss_grid = np.zeros_like(alpha_grid)
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data = (original_params[name] +
                                    alpha * direction1[name] +
                                    beta * direction2[name])
                
                loss = self._evaluate_loss()
                loss_grid[j, i] = loss
                
                print(f"[Landscaper] Grid point ({i},{j}): α={alpha:.2f}, β={beta:.2f}, loss={loss:.4f}")
    
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = original_params[name]
        
        return alpha_grid, beta_grid, loss_grid
    
    def _evaluate_loss(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 10:
                    # minimum since my laptop sucks
                    break
        
        return total_loss / num_batches