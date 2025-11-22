
import torch
import torch.nn as nn
from typing import Optional
from methods.customnorm.dyt import DynamicTanh
from methods.customnorm.linear import PiecewiseLinear
from methods.customnorm.rms_norm import RMSNorm

class MNISTNet(nn.Module):
    # standard cnn architecture for MNIST
    # MNIST is easy, 4 layers should be overkill
    def __init__(self, norm_type: str = 'none', num_classes: int = 10):
        super().__init__()
        self.norm_type = norm_type
        
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.norm1 = self._make_norm_layer(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = self._make_norm_layer(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm3 = self._make_norm_layer(64)
        
        # fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.norm4 = self._make_norm_layer(128, is_conv=False)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def _make_norm_layer(self, num_features: int, is_conv: bool = True) -> Optional[nn.Module]:
        if self.norm_type == 'none':
            return nn.Identity()
        elif self.norm_type == 'batch':
            return nn.BatchNorm2d(num_features) if is_conv else nn.BatchNorm1d(num_features)
        elif self.norm_type == 'layer':
            return nn.GroupNorm(1, num_features) if is_conv else nn.LayerNorm(num_features)
        elif self.norm_type == 'group':
            # use 8 groups for group norm
            return nn.GroupNorm(min(8, num_features), num_features)
        elif self.norm_type == 'rms':
            return RMSNorm(num_features)
        elif self.norm_type == 'dyt':
            return DynamicTanh(num_features, init_alpha=0.5)
        elif self.norm_type == 'piecewise':
            return PiecewiseLinear(num_features)
        else:
            raise ValueError(f"Unknown normalization type: {self.norm_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        if self.norm3 is not None:
            x = self.norm3(x)
        x = self.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        if self.norm4 is not None:
            x = self.norm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x