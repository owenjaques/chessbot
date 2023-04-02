"""
to-do:
accuracy function is likely wrong
"""


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ConvNet(nn.Module):
    def __init__(self, num_filters=64, kernel_size=3, dropout_prob=0.5):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.fc1 = nn.Linear(num_filters * 16, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.num_filters * 16)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
    

class ConvNetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class HyperparameterSearch:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def run(self):
        param_grid = {
            'num_filters': [32, 64, 128],
            'kernel_size': [3, 5, 7],
            'dropout_prob': [0.25, 0.5, 0.75],
            'lr': [0.001, 0.01, 0.1]
        }
        model = ConvNet()
        train_dataset = ConvNetDataset(self.X_train, self.y_train)
        val_dataset = ConvNetDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=3, verbose=1)
        grid_search.fit(train_loader)
        best_params = grid_search.best_params_
        best_model = ConvNet(**best_params)
        best_score = grid_search.best_score_

        return best_model, best_score
