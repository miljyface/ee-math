import torch
from pathlib import Path

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("mps")

#OUTPUT_DIR = Path("/home/rguan/documents/ee-math/output")
OUTPUT_DIR = Path("/Users/guanrong/Desktop/ee_shit/ee-math/output")

# for shanghai box
#train_image_path = '/home/lguan/文档/ee-math/rawdata/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte' 
#train_label_path = '/home/lguan/文档/ee-math/rawdata/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
#test_image_path = '/home/lguan/文档/ee-math/rawdata/train-images-idx3-ubyte/train-images-idx3-ubyte'
#test_label_path = '/home/lguan/文档/ee-math/rawdata/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

# for mac
test_image_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte' 
test_label_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
train_image_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/train-images-idx3-ubyte/train-images-idx3-ubyte'
train_label_path = '/Users/guanrong/Desktop/ee_shit/ee-math/rawdata/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

# for home gpus /home/rguan/documents/ee-math/rawdata
#test_image_path = '/home/rguan/documents/ee-math/rawdata/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte' 
#test_label_path = '/home/rguan/documents/ee-math/rawdata/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
#train_image_path = '/home/rguan/documents/ee-math/rawdata/train-images-idx3-ubyte/train-images-idx3-ubyte'
#train_label_path = '/home/rguan/documents/ee-math/rawdata/train-labels-idx1-ubyte/train-labels-idx1-ubyte'