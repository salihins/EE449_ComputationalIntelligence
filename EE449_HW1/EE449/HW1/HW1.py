import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np
from sklearn.model_selection import train_test_split
import utils
import json
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt



# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


# training set
train_data = torchvision.datasets.FashionMNIST("./data", train = True, download = True,
  transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)]))
# test set
test_data = torchvision.datasets.FashionMNIST("./data", train = False,
  transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,),(0.5,),)]))

train_set, valid_set = train_test_split(train_data, test_size=0.1, stratify = train_data.targets)

train_generator = torch.utils.data.DataLoader(train_set, batch_size = 50, shuffle = True)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = 50, shuffle = False)
valid_generator = torch.utils.data.DataLoader(valid_set, batch_size = 50, shuffle = True)


