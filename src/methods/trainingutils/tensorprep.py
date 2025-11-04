import torch
import numpy as np
from torch.utils.data import TensorDataset
from .dataloader import MnistDataloader
from config import train_image_path, train_label_path, test_image_path, test_label_path


def prep_tensors():
    # load and preprocess MNIST data with proper normalization.
    
    mnist_dataloader = MnistDataloader(train_image_path, train_label_path, test_image_path, test_label_path)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # Convert to numpy arrays & Normalise
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    train_mean = x_train.mean()
    train_std = x_train.std()
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std
    
    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Ensure images are [batch_size, 1, 28, 28]
    if x_train.ndim == 3:  # [N, 28, 28]
        x_train = x_train.unsqueeze(1)  # [N, 1, 28, 28]
    elif x_train.ndim == 4 and x_train.shape[1] == 28:  # [N, 28, 28, 1]
        x_train = x_train.permute(0, 3, 1, 2)  # to [N, 1, 28, 28]
    
    if x_test.ndim == 3:
        x_test = x_test.unsqueeze(1)
    elif x_test.ndim == 4 and x_test.shape[1] == 28:
        x_test = x_test.permute(0, 3, 1, 2)
    
    print(f"Training set shape: {x_train.shape}, range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Test set shape: {x_test.shape}, range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
