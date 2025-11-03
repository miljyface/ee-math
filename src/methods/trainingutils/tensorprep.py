import torch
import numpy as np
from torch.utils.data import TensorDataset
from .dataloader import MnistDataloader

#train_image_path = '/home/lguan/文档/ee-math/rawdata/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte' 
#train_label_path = '/home/lguan/文档/ee-math/rawdata/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
#test_image_path = '/home/lguan/文档/ee-math/rawdata/train-images-idx3-ubyte/train-images-idx3-ubyte'
#test_label_path = '/home/lguan/文档/ee-math/rawdata/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

train_image_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte' 
train_label_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
test_image_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/train-images-idx3-ubyte/train-images-idx3-ubyte'
test_label_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

def prep_tensors():
    mnist_dataloader = MnistDataloader(train_image_path, train_label_path, test_image_path, test_label_path)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # Create datasets and dataloaders
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Ensure images are [batch_size, 1, 28, 28]
    if x_train.ndim == 3:  # [N, 28, 28]
        x_train = x_train.unsqueeze(1)  # [N, 1, 28, 28]
    elif x_train.ndim == 4 and x_train.shape[1] == 28:  # [N, 28, 28, 1]
        x_train = x_train.permute(0, 3, 1, 2)           # to [N, 1, 28, 28]

    if x_test.ndim == 3:
        x_test = x_test.unsqueeze(1)
    elif x_test.ndim == 4 and x_test.shape[1] == 28:
        x_test = x_test.permute(0, 3, 1, 2)

    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)