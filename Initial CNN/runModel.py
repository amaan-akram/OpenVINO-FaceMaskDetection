import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2

pred_path = 'pred'
train_path = 'data'
root=pathlib.Path(train_path)
classes=['mask', 'no_mask']


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


checkpoint = torch.load('best_checkpoint.model')
model = ConvNet(num_classes=2)
model.load_state_dict(checkpoint)
model.eval()

transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    # Add image augmentation code here
    transforms.ToTensor(),  # convert from numpy to tenser
    transforms.Normalize([0.5, 0.5, 0.5],  # converts from range 0-1 to -1 to 1
                         [0.5, 0.5, 0.5])
])


def prediction(img_path, transfermer):
    image = Image.open(img_path)
    image_tensor = transfermer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.numpy().argmax()

    pred = classes[index]
    return pred

images_path=glob.glob(pred_path + '/*.*')

pred_dict={}
for i in images_path:
    pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)

print(pred_dict)