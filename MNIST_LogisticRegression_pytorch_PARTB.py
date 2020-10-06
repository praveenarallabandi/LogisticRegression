import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
# Plotting libraries
import matplotlib.pyplot as plt
import numpy as np


# 1.Loading and Normalizing MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         shuffle=False)

# 2.Define a Convolutional Neural Network - Logistic Regression
class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc1 = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def predict(self, x):
        # a function to predict the labels of a batch of inputs 
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        x = F.softmax(self.forward(x), dim=1)
        return x

    def accuracy(self, x, y):
        # a function to calculate the accuracy of label prediction for a batch of inputs
        #   x: a batch of inputs - images
        #   y: the true labels associated with x
        correct = 0
        prediction = self.predict(x)
        maxs, indices = torch.max(prediction, 1)
        correct += (indices == y).sum().item()
        # compare original and predicted class data
        acc = 100 * torch.sum(torch.eq(indices.float(), y.float()).float())/y.size()[0]
        return acc.cpu().data, correct

# the 28×28 sized images will be 784 pixel input values
numberOfFeatures = 784
numberOfClasses = 10
batch_per_ep = 5000
net = LogisticRegression(numberOfFeatures, numberOfClasses)

# 3. Define a Loss function and optimizer 
# Use logistic loss & class one-hot encode combined together
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the CNN
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0

    for step, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # Filtered dataset with class values 3 and 5 for Vid: V00933455
        train_filter_class= np.where((labels==3) | (labels==5))
        inputs, labels = inputs[train_filter_class], labels[train_filter_class]

        inputs = inputs.view(inputs.shape[0], -1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print loss statistics
        running_loss += loss.item()
        if not step%10:    # print every 10 mini-batches
            # Average measure of loss
            training_loss = running_loss / 10
            print('epoch:%d, iteration:%5d training average running loss: %.3f' %(epoch + 1, step + 1, training_loss))
            running_loss = 0.0
            plt.plot(training_loss,step,'bo')
            plt.title('trainingriskvalue vs iterations')

print('Finished Training')
plt.show()
print("Training Plotting Done!")

# 5. save the trained model
PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

# 6. Now test the trained model with test data
# load the trained model from saved path
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
error_calc = 0
with torch.no_grad():
    running_loss = 0.0
    for step, data in enumerate(testloader, 0):
        inputs, labels = data

        # Filtered dataset with class values 4 and 5 for Vid: V00933455
        train_filter_class= np.where((labels==3) | (labels==5))
        inputs, labels = inputs[train_filter_class], labels[train_filter_class]

        inputs = inputs.view(inputs.shape[0], -1)
        outputs = net(inputs)
        accu, corr = net.accuracy(inputs, labels)
        correct += corr
        total += labels.size(0)

        # print loss statistics
        running_loss += loss.item()
        if not step % 10:    # print every 10 mini-batches
            training_loss = running_loss / 10
            print('iteration:%5d testdata average running loss: %.3f' %(step + 1, training_loss))
            running_loss = 0.0
            plt.plot(training_loss, step, 'ro')
            plt.title('testsetriskvalue vs iterations')
            error_calc = ((100/total) * (total - correct))
            # Need to change this TODO
            plt.plot(error_calc, step, 'bo')
            plt.title('testseterror vs iterations')

print('Total testset images: %s correctly identified images: %s Error: %s%% Accuracy : %d%%'% (total, correct, ((100/total) * (total - correct)), accu))
plt.show()
print("Plotting Done!")
