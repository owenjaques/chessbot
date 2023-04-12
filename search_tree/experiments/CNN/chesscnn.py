import torch.nn as nn
import torch.nn.functional as F
import torch

class ChessCNN1(nn.Module):
    def __init__(self):
        super(ChessCNN1, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)  # add padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        # apply convolutional layers and activation functions
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # flatten the tensor
        x = x.view(-1, 64 * 2 * 2)

        # apply fully connected layers and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


class ChessCNN2(nn.Module):
    def __init__(self):
        super(ChessCNN2, self).__init__()
        self.conv1 = nn.LazyConv2d(64, 3)
        self.conv2 = nn.LazyConv2d(128, 3)
        self.conv3 = nn.LazyConv2d(128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.LazyLinear( 128)
        self.fc2 = nn.LazyLinear( 128)
        self.fc3 = nn.LazyLinear( 1)

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
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChessCNN3(nn.Module):
    def __init__(self):
        super(ChessCNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 1)


    def forward(self, x):
        # apply convolutional layers and activation functions
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        #x = F.relu(self.conv3(x))
        #x = self.pool(x)
        #x = self.dropout(x)

        # flatten the tensor
        x = x.view(x.size(0), -1)
        #x = x.flatten(start_dim=1)

        # apply fully connected layers and dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)

        return x


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=1, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        # apply convolutional layers and activation functions with pooling
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        x = F.relu(self.conv3(x))
        #x = self.pool(x)

        # flatten the tensor
        x = x.view(x.size(0), -1)

        # apply fully connected layers and dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.sigmoid(x)

        return x.squeeze()
    
