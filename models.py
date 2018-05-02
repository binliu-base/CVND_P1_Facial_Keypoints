## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64 , 3)  

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)        

        self.conv5 = nn.Conv2d(128, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)

        self.conv2_drop = nn.Dropout2d(p=0.5)                
        self.pool = nn.MaxPool2d(2, 2)        

        self.fc1 = nn.Linear(128 * 17 * 17, 1024)
        self.fc2 = nn.Linear(1024, 136)  
        

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()

        x = self.conv2_drop(self.pool(F.relu(self.conv2(F.relu(self.conv1(x))))))
        x = self.conv2_drop(self.pool(F.relu(self.conv4(F.relu(self.conv3(x))))))
        x = self.conv2_drop(self.pool(F.relu(self.conv6(F.relu(self.conv5(x))))))        
        x = x.view(-1, 128 * 17 * 17)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
            
        return x
