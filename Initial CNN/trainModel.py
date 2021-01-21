import os
import numpy as np
import torch
import glob
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

DATADIR = 'data'
TESTDIR = 'testData'

batch_size = 64

imageTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data = ImageFolder(DATADIR, transform=imageTransform)
testData = ImageFolder(TESTDIR, transform=imageTransform)
classes = data.class_to_idx

print(data)
"""
image, label = data[2267]
print('Image Size:', image.shape)
print('Label:', d.classes[label])
datasetLabels = d.class_to_idx

plt.imshow(image.permute(1, 2, 0))
plt.show()
"""

trainDL = DataLoader(data, batch_size=batch_size, pin_memory=True)
testDL = DataLoader(testData, batch_size=batch_size, pin_memory=True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Custom Data Loader which transfers the data to cuda graphics card, or RAM whichever available, and then returns the data
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
trainDL = DeviceDataLoader(trainDL, device)
testDL = DeviceDataLoader(testDL, device)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3), stride=1, padding=1),  # 3x224x224 => 8x224x224
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 64, (3, 3), padding=1),  # 8x112x112 => 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=1),  # 64x56x56 => 128x56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x56x56 => 128x24x24
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Flatten()
        )

        self.linear = nn.Sequential(nn.Linear(4608, 2048, bias=True),
                                    nn.Dropout(0.5),
                                    nn.ReLU(),
                                    nn.Linear(2048, 100, bias=True),
                                    nn.Dropout(0.5),
                                    nn.ReLU(),
                                    nn.Linear(100, 2, bias=True))

    def training_step(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        return loss

    def forward(self, batch):
        out = self.network(batch)
        out = self.linear(out)
        return out

    def validation_step(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        _, y_pred = torch.max(out, dim=1)
        label = label.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        accuracy = accuracy_score(label, y_pred)
        precision = recall_score(label, y_pred, average='micro')
        print('Confusion Matrix for Evaluation:')
        print(confusion_matrix(label, y_pred))
        return {'val_loss': loss.detach(), 'val_accuracy': torch.Tensor([accuracy]),
                'precision': torch.Tensor([precision])}

    def validation_epoch_end(self, outputs):
        val_loss = [x['val_loss'] for x in outputs]
        val_loss_n = torch.stack(val_loss).mean()
        val_score = [x['val_accuracy'] for x in outputs]
        val_score_n = torch.stack(val_score).mean()
        precision = [x['precision'] for x in outputs]
        precision = torch.stack(precision).mean().item()
        return {'val_loss': val_loss_n, 'val_score': val_score_n, 'precision': precision}

    def epoch_end(self, epoch, result):
        print('Epoch {}: train_loss: {:.4f} val_loss: {:.4f} val_score: {:.4f} precision: {}'.format(epoch, result[
            'train_loss'], result['val_loss'], result['val_score'], result['precision']))

model = CNN()
model.to(device)





""""
device = torch.device('cuda')  # change to cpu if cuda is unavailable

# training data and testing data
train_path = 'data'
test_path = 'testData'

root=pathlib.Path(train_path)
classes=sorted(['mask', 'no_mask'])

# normalise images
transformer = transforms.Compose([
    transforms.Resize((150,150)),
    # Add image augmentation code here
    transforms.ToTensor(),  # convert from numpy to tenser
    transforms.Normalize([0.5, 0.5, 0.5],  # converts from range 0-1 to -1 to 1
                         [0.5, 0.5, 0.5])
])

# Dataloader
trainData_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=256, shuffle=True
)
#print(trainData_loader.dataset)

testData_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=256, shuffle=True
)
#print(testData_loader.dataset)


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(in_features=32 * 75 * 75, out_features=num_classes)

        # feed forward

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 32 * 75 * 75)  # reshape matrix for fully connected layer

        return output


model = ConvNet(num_classes=2).to(device)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()
epoch_num = 10

train_count = len(glob.glob(train_path + '/**/*.*'))
test_count = len(glob.glob(test_path + '/**/*.*'))

accuracy = 0.0


for epoch in range(epoch_num):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(trainData_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    model.eval()

    test_accuracy = 0.0

    for i, (images, labels) in enumerate(testData_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    # Save the best model
    if test_accuracy > accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy

print(train_count, test_count)
"""
