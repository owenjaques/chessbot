import torch.nn as nn
import torch.nn.functional as F
import torch

# http://cs231n.stanford.edu/reports/2015/pdfs/ConvChess.pdf

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 4)
        self.conv2 = nn.Conv2d(64, 128, 4)
        self.conv3 = nn.Conv2d()
        self.pool = nn.MaxPool2d((4, 4))
        self.dropout = nn.Dropout(0.05)
        self.fc1 = nn.Linear( 128, 64)
        self.fc2 = nn.Linear( 64, 10)


    def forward(self, x):
        # apply convolutional layers and activation functions
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)

        # flatten the tensor
        x = x.view(x.size(0), -1)

        # apply fully connected layers and dropout
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)

        return x
