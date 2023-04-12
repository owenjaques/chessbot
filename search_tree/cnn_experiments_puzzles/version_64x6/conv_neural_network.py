# class to initialize the neural network and train it on the data set
# the model input is a 1-hot vector of length 64*2 where the first 64 elements are the white pieces and the second 64 elements are the black pieces
# the model output is a 1-hot vector of length 64*64 where each element is the probability of making that move
# the model is trained on the data set of 1-hot vectors of length 64*2 + 64*64
# the model is trained to predict the probability of making each move given the current board state
# the model structure is taken as a argument passed to the class constructor
#
# Pytorch implementation of a convolutional neural network

# rough model structure. File needs to be redone eventually for better organization

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import time
import random
import os
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import glob

import sys


class ConvNetArgs(nn.Module):
    def __init__(self, model_structure):
        super(ConvNetArgs, self).__init__()
        self.model_structure = model_structure
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(1, self.model_structure[0], 3, 1, 1))
        for i in range(len(self.model_structure) - 1):
            self.conv_layers.append(nn.Conv2d(self.model_structure[i], self.model_structure[i + 1], 3, 1, 1))
        self.fc_layers.append(nn.Linear(self.model_structure[-1] * 8 * 8, 4096))
        self.fc_layers.append(nn.Linear(4096, 4096))
        self.fc_layers.append(nn.Linear(4096, 64 * 64))

    def forward(self, x):
        x = x.view(-1, 2, 8, 8)
        for i in range(len(self.model_structure)):
            x = F.relu(self.conv_layers[i](x))
        x = x.view(-1, self.model_structure[-1] * 8 * 8)
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)
        return x
    
# class ConvNet(nn.Module): that has default model structure
class CovNetDefault(nn.Module):
    def __init__(self):
        super(CovNetDefault, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 64 * 64)

    def forward(self, x):
        #print(x.shape)
        x = x.view(-1, 6, 8, 8)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = x.view(-1, 64 * 8 * 8)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.sigmoid(self.fc3(x))
        #print(x.shape)
        x = self.fc4(x)
        #print(x.shape)

        x = x.view(-1, 64 * 64)

        return x



class ChessConvNet:
    def __init__(self, model_structure, load_model=False, model_path="version_64x8/model/"):
        self.model_structure = model_structure
        # if load_model is True, load the model from the model_path and continue training it rather than creating a new model
        self.model_file_name = model_path
        if load_model and model_path is not None:
            self.model = self.load_model_optimizer()
        else:
            model = self.create_model()
            if model is not None:
                print("Model created successfully")
            else:
                print("Model creation failed")
                sys.exit(1)
            self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = self.loss_function.cuda()
        self.loss_history = []
        self.accuracy_history = []
        self.batch_size = 32
        self.num_epochs = 1
        self.data_set = None
        self.data_loader = None
        self.validation_set = None
        self.validation_loader = None
        self.version = 0
        


    def create_model(self):
        model = None
        if torch.cuda.is_available():
            if self.model_structure is None:
                model = CovNetDefault().cuda()
            else:
                model = ConvNetArgs(self.model_structure).cuda()
        else:
            if self.model_structure is None:
                model = CovNetDefault()
            else:
                model = ConvNetArgs(self.model_structure)
        return model

    def train(self, data, validation=None, return_history=False):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Using GPU")
            self.data_set = data_utils.TensorDataset(torch.from_numpy(data[:, :384]).float().cuda(), torch.from_numpy(data[:, 384:]).float().cuda())
        else:
            self.data_set = data_utils.TensorDataset(torch.from_numpy(data[:, :384]).float(), torch.from_numpy(data[:, 384:]).float())
        self.data_loader = data_utils.DataLoader(self.data_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        if validation is not None:
            if torch.cuda.is_available():
                self.validation_set = data_utils.TensorDataset(torch.from_numpy(validation[:, :384]).float().cuda(), torch.from_numpy(validation[:, 384:]).float().cuda())
            else:
                self.validation_set = data_utils.TensorDataset(torch.from_numpy(validation[:, :384]).float(), torch.from_numpy(validation[:, 384:]).float())
            self.validation_loader = data_utils.DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # check if loss history is empty, if not, then we are continuing training on a model that was already trained
        if len(self.loss_history) == 0:
            self.loss_history = []
            self.accuracy_history = []
        for epoch in range(self.num_epochs):
            start_time = time.time()
            for i, (board_image, move) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                output = self.model(board_image)
                loss = self.loss_function(output, move)
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, i + 1, len(self.data_loader), loss.item()))
                    self.loss_history.append(loss.item())
                    self.accuracy_history.append(self.test_accuracy())
            end_time = time.time()
            print('Time elapsed for epoch {} is {} seconds'.format(epoch + 1, end_time - start_time))
            #if epoch % 10 == 0:
            #    self.save_model_optimizer()

        if return_history:
            return self.loss_history, self.accuracy_history

    # test accuracy by running the model on the validation set and comparing the output to the actual move
    # should be considered accurate if the move is in the top 5 moves
    def test_accuracy(self):
        # if validation set is None, check the accuracy on the training set
        data_loader = None
        if self.validation_set is None:
            data_loader = self.data_loader
        else:
            data_loader = self.validation_loader
        if data_loader is None:
            print("Error: No data loader in test_accuracy")
            return 0
        correct = 0
        correct_5 = 0
        total = 0
        for board_image, move in data_loader:
            output = self.model(board_image)
            #_, predicted = torch.argmax(output.data, 1)
            #probs = torch.softmax(output, dim=1)
            probs = output
            best_moves = torch.topk(probs, 5, dim=1)
            best_move = torch.argmax(probs, dim=1)
            target = torch.argmax(move, dim=1)
            #if total == 0:
                #print("Best move: {}".format(best_move))
                #print("Target: {}".format(target))
                #print(len(best_move))
            for i in range(len(best_move)):
                if target[i] == best_move[i]:
                    correct += 1
                if target[i] in best_moves.indices[i]:
                    correct_5 += 1

            total += move.size(0)
        print('Accuracy of the network on the {} best move: {} %'.format(total, 100 * correct / total))
        print('Accuracy of the network on the {} best move being in top 5: {} %'.format(total, 100 * correct_5 / total))
        return 100 * correct / total
    
    # save the model and optimizer state to continue training later, also save the loss and accuracy history
    def save_model_optimizer(self):
        model_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }
        # if model_filename is a directory, then save the model in that directory with the default name
        if os.path.isdir(self.model_file_name):
            self.model_file_name = os.path.join(self.model_file_name, "chess_conv_net_model_8_"  +  str(self.version))
        # if model has already been saved, then append the time to the end of the file name
        #if os.path.isfile(self.model_file_name):
            #self.model_file_name = self.model_file_name + "_" + str(time.time())
        with open(self.model_file_name, 'wb') as f:
            torch.save(model_state, f)

    # load the model and optimizer state to continue training later, also load the loss and accuracy history
    def load_model_optimizer(self):
        # if model_filename is a directory, then load the model from that directory with the default name and most recent time
        if os.path.isdir(self.model_file_name):
            model_files = [f for f in os.listdir(self.model_file_name) if os.path.isfile(os.path.join(self.model_file_name, f))]
            model_files.sort()
            if len(model_files) == 0:
                print("Error: No model files in directory")
                print("Making new model with name: {}".format("chess_conv_net_model_"))
                return self.create_model()
            self.model_file_name = os.path.join(self.model_file_name, model_files[-1])
        with open(self.model_file_name, 'rb') as f:
            model_state = torch.load(f)
            self.model.load_state_dict(model_state['model'])
            self.optimizer.load_state_dict(model_state['optimizer'])
            self.loss_history = model_state['loss_history']
            self.accuracy_history = model_state['accuracy_history']


    # the fuck was I thinking?...
    def predict_move(self, board_image):
        board_image = board_image.reshape(1, 128)

        board_image = torch.from_numpy(board_image).float().cuda()
        output = self.model(board_image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
    
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('10s of batches')
        plt.ylabel('loss')
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.accuracy_history)
        plt.xlabel('10s of batches')
        plt.ylabel('accuracy')
        plt.show()


# for each chunk of data, train the model on it and save the model and optimizer state 
def chunk_trainer():
    # for each chunk in data directory
    try:
        # need to change this to the validation data
        with open("version_64x6/validation/valid_2.npz", "rb") as f:
            validation = np.load(f, allow_pickle=True)["data"]
    except:
        validation = None

    version = 0

    # skip the first few chunks if you want to start training from a later chunk (savepoint)
    # currently at: 2 chunks processed
    skip = 0
    epochs = 3

    # "version_64x2/data, version_64x2/data_first_move", "version_64x2/data_second_move"
    data_dirs = ["version_64x6/data"]

    for i in range(epochs):
        for data_dir in data_dirs:
            # load 1 datafile at a time
            for data_file in os.listdir(data_dir):
                if skip > 0:
                    skip -= 1
                    continue
                print("Loading data from {}".format(data_file))
                data = None
                with open(os.path.join(data_dir, data_file), "rb") as f:
                    data = np.load(f, allow_pickle=True)["data"]
                if data is None:
                    print("No data to train on: exiting")
                    sys.exit(1)
                # load the model and optimizer state from the previous chunk if it exists
                model = ChessConvNet(None, model_path="version_64x6/model/")
                model.load_model_optimizer()
                model.optimizer.param_groups[0]['lr'] = 0.001
                if torch.cuda.is_available():
                    model.loss_function = model.loss_function.cuda()
                # train the model on the data
                model.train(data, validation=validation)
                model.version = version
                model.save_model_optimizer()
                print("Model saved to {}".format(model.model_file_name))

    
if __name__ == "__main__":
    sys.path.append("..")

    model_structure = None
    continue_training = True

    if continue_training:
        chunk_trainer()


