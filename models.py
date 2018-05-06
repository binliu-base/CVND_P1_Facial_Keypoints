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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64 , 3)  
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)        

        self.bn1 = nn.BatchNorm2d(32)           
        self.bn2 = nn.BatchNorm2d(64)                   
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)        
        self.f1_bn = nn.BatchNorm1d(1000)                                                              
        self.f2_bn = nn.BatchNorm1d(1000)                                                                      
        # self.conv2_drop = nn.Dropout2d(p=0.5)                
        self.conv2_drop = nn.Dropout2d(p=0.5, inplace=True)                        
        self.pool = nn.MaxPool2d(2, 2)     

        self.fc1 = nn.Linear(256 * 8 * 8, 1000)
        # self.fc2 = nn.Linear(3000, 1000)          
        self.fc3 = nn.Linear(1000, 136)  

        

    def forward(self, x):

        x = x.to(self.device)
        x = self.conv2_drop(self.pool(F.leaky_relu(self.bn1(self.conv1(x)))))
        x = self.conv2_drop(self.pool(F.leaky_relu(self.bn2(self.conv2(x)))))
        x = self.conv2_drop(self.pool(F.leaky_relu(self.bn3(self.conv3(x)))))
        x = self.conv2_drop(self.pool(F.leaky_relu(self.bn4(self.conv4(x)))))                        
                
        x = x.view(-1, 256 * 8 * 8)
        x = F.leaky_relu(self.f1_bn(self.fc1(x)))
        # x = F.leaky_relu(self.f2_bn(self.fc2(x)))
        x = self.fc3(x)
            
        return x
