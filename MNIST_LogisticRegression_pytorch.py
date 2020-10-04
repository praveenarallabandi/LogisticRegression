import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
# Plotting libraries
import visdom
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np # using only plotting in imshow method

# Initialize plotting library
class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )

# General utlity method
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
    # Initialize the visualization environment
    # vis = Visualizations()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
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
        if not i%10:    # print every 10 mini-batches
            training_loss = running_loss / 10
            print('[%d, %5d] training loss: %.3f' %(epoch + 1, i + 1, training_loss))
            running_loss = 0.0
            # vis.plot_loss(np.mean(running_loss), i)
        
print('Finished Training')

# 5. save the trained model
PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

# 6. Now test the trained model with test data
# load the trained model from saved path
net.load_state_dict(torch.load(PATH))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(images.shape[0], -1)
        outputs = net(images)
        accu, corr = net.accuracy(images, labels)
        correct += corr
        total += labels.size(0)
        # print loss statistics
        """ running_loss += loss.item()
        if not step%10:    # print every 10 mini-batches
            print('[%d, %5d] testdata loss: %.3f' %(epoch + 1, step + 1, running_loss / 10))
            running_loss = 0.0
            vis.plot_loss(np.mean(running_loss), step) """

print('Total Accuracy of the network on the total test dataset images: %s correctly identified images: %s is : %d%%'% (total, correct, accu))