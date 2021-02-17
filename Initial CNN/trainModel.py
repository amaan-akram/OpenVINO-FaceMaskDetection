import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import time

start = time.time()

# Directories FOR DATASETS
HQ_RMFD = '../DATASETS/HQ-RMFD'
RMFD = '../DATASETS/RMFD'
SMFD = '../DATASETS/SMFD'
ALL_DATA = '../DATASETS/ALL_DATA'

#CHANGE THIS VARIABLE TO TRAIN WITH DIFFERENT DATASETS ABOVE
trainDIR = HQ_RMFD

#TEST DATASET
testDIR = '../DATASETS/test'

# device setup - cuda
device = ("cuda")

# tensors image data and normalize tensors to range of [1,-1]
transformsForTrain = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transformsForTest = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainSet = torchvision.datasets.ImageFolder(root=trainDIR, transform=transformsForTrain)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True)

testSet = torchvision.datasets.ImageFolder(root=testDIR, transform=transformsForTest)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=True)

classes = ("mask", "no_mask")


# Defining a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


def showImg(img):  #
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(testLoader)
images, labels = dataiter.next()

# print images
showImg(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

images = images.to(device)
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

end = time.time()
print("run time:",  end - start , "s")
"""
images, labels = trainSet[2000]
# show images
showImg(images)
print(labels)

"""
