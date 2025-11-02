import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from pyhessian import hessian

class LossLandscapeAnalyzer:
    """
    Analyzes loss landscape geometry using filter-normalized random directions
    Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
    """
    
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
                    break
        
        return total_loss / num_batches
    
    def compute_hessian_eigenvalues(self, num_eigenvalues: int = 5) -> Dict[str, np.ndarray]:
        print(f"[Landscaper] Computing Hessian eigenvalues (top {num_eigenvalues})...")
        self.model.to(self.device)
        
        # MOVING data to correct device for PyHessian
        class DeviceLoader:
            def __init__(self, dataloader, device):
                self.dataloader = dataloader
                self.device = device
            def __iter__(self):
                for x, y in self.dataloader:
                    yield x.to(self.device), y.to(self.device)
            def __len__(self):
                return len(self.dataloader)
        dl_device = DeviceLoader(self.dataloader, self.device)

        # Create Hessian computation object
        hessian_comp = hessian(
            self.model,
            self.criterion,
            dataloader=dl_device,
            cuda=True
        )

        # Compute top eigenvalues (largest magnitude)
        top_eigenvalues, _top_eigenvectors = hessian_comp.eigenvalues(
            top_n=num_eigenvalues
        )

        # Compute trace (sum of all eigenvalues - indicator of sharpness)
        print("[Landscaper] Computing Hessian trace...")
        trace = hessian_comp.trace()

        # Compute density (eigenvalue distribution)
        print("[Landscaper] Computing eigenvalue density...")
        density_eigen, density_weight = hessian_comp.density()

        results = {
            'top_eigenvalues': np.array(top_eigenvalues),
            'max_eigenvalue': np.array([top_eigenvalues[0]]) if top_eigenvalues else np.array([1.0]),
            'min_eigenvalue': np.array([top_eigenvalues[-1]]) if len(top_eigenvalues) > 1 else np.array([0.1]),
            'trace': np.array([trace]),
            'density_eigen': np.array(density_eigen),
            'density_weight': np.array(density_weight)
        }
        # Compute sharpness metric (condition number approximation)
        if len(top_eigenvalues) > 1:
            condition_number = abs(top_eigenvalues[0] / top_eigenvalues[-1]) if top_eigenvalues[-1] != 0 else np.inf
            results['condition_number'] = condition_number
            print(f"[Landscaper] Condition number (λ_max/λ_min): {condition_number:.2f}")
            print(f"[Landscaper] Top eigenvalues: {top_eigenvalues}")
            print(f"[Landscaper] Trace: {trace:.4f}")

        return results
