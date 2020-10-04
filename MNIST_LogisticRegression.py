import torch as torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
# Import nn.functional
import torch.nn.functional as F

# the 28×28 sized images will be 784 pixel input values
totalPixelsInputValues=784

# load dataset from keras
(x_train, y_train), (x_test, y_test)= mnist.load_data()

# Filtered dataset with class values 3 and 5 for Vid: V00933455
train_filter_class= np.where((y_train==4) | (y_train==5))
test_filter_class= np.where((y_test==4) | (y_test==5))
x_train, y_train = x_train[train_filter_class], y_train[train_filter_class]
x_test, y_test = x_test[test_filter_class], y_test[test_filter_class]

x_train = x_train.reshape(x_train.shape[0],28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0],28, 28, 1).astype('float32') / 255

print('x_train', len(x_train), x_train.shape)
print('y_train', len(y_train), y_train.shape)
print('x_test', len(x_test))
# print('x_train', x_train[0])

# create tensor for torch
X_Train = torch.from_numpy(x_train)
Y_Train = torch.from_numpy(y_train)
X_Test = torch.from_numpy(x_test)
Y_Test = torch.from_numpy(y_test)

# Filtered dataset with class values 3 and 5 for Vid: V00933455
""" train_filter_class= torch.where((Y_Train==5) | (Y_Train==3))
test_filter_class= torch.where((Y_Test==5) | (Y_Test==3))
x_train, y_train = X_Train[train_filter_class], Y_Train[train_filter_class]
x_test, y_test = X_Test[test_filter_class], Y_Test[test_filter_class] """

# reshaping train and test data
""" x_train = x_train.reshape(x_train.shape[0],28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0],28, 28, 1).astype('float32') / 255 """
#reshaping train and test data
""" x_train = x_train.reshape(x_train.shape[0],784)
x_test = x_test.reshape(x_test.shape[0],784) """

# normalize inputs from 0-255 to 0-1 - TODO
""" x_train = x_train / 255
x_test = x_test / 255 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)"""

print('AFTER RESHAPE')
print('X_Train', len(X_Train))
print('X_Test', len(X_Test))
print('X_Train', X_Train)
train_ds = TensorDataset(X_Train, X_Test)
print('train_ds', len(train_ds))
# Define data loader
batch_size = 1000
epoch_num = 15
lr = 0.001
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
print('train_ds', len(train_ds))
next(iter(train_dl))

# Define model
model = nn.Linear(784, 10)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define loss function
loss_fn = nn.MarginRankingLoss()
# loss = loss_fn(model(x_train), x_test)

# Loop through dataset in defined batches
def execute(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        print('train_dl', len(train_dl))
        print('train_dl', train_dl)
        for xb,yb in train_dl:
            print('xb,yb', xb,yb)
            # reset gradient
            opt.zero_grad()
            # generate predictions
            pred = model(xb)
            # forward pass
            loss = loss_fn(pred, yb)
            # backward pass and update
            loss.backward()
            opt.step() # updating weights
    print('Training loss: ', loss_fn(model(x_train), x_test))

execute(100, model, loss_fn, optimizer)

# Generate predictions
preds = model(x_train)
preds

# Compare with targets
x_test




