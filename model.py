import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F


class face_keypoints_network(nn.Module):
    
    def __init__(self):
        super(face_keypoints_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 600)
        self.fc3 = nn.Linear(600, 136)
        
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 112
        x = self.pool(F.relu(self.conv2(x))) # 56
        x = self.pool(F.relu(self.conv3(x))) # 28
        x = self.pool(F.relu(self.conv4(x))) # 14
        x = self.pool(F.relu(self.conv5(x))) # 7
        
        x = x.view(-1, 256 * 7 * 7)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x