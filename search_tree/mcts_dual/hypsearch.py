import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchsummary import summary
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class ConvNetHyperparamSearch:
    def __init__(self, dataset, num_classes, device):
        self.dataset = dataset
        self.num_classes = num_classes
        self.device = device

    def search(self, num_epochs=10, batch_size=64, learning_rate=0.001):
        # Define a range of hyperparameters to search over
        channels = [16, 32, 64]
        kernel_sizes = [3, 5]
        paddings = [1, 2]
        dilations = [1, 2]

        # Initialize a dictionary to store the best hyperparameters and their corresponding accuracy
        best_hyperparams = {}
        best_accuracy = 0.0

        # Loop over all possible combinations of hyperparameters
        for combo in itertools.product(channels, kernel_sizes, paddings, dilations):
            # Unpack the hyperparameters
            channel1, channel2, channel3 = combo[0], combo[0]*2, combo[0]*4
            kernel_size1, kernel_size2, kernel_size3 = combo[1], combo[1], combo[1]
            padding1, padding2, padding3 = combo[2], combo[2], combo[2]
            dilation1, dilation2, dilation3 = combo[3], combo[3], combo[3]

            # Define the model
            model = nn.Sequential(
                nn.Conv2d(1, channel1, kernel_size=kernel_size1,
                          padding=padding1, dilation=dilation1),
                nn.ReLU(),
                nn.Conv2d(channel1, channel2, kernel_size=kernel_size2,
                          padding=padding2, dilation=dilation2),
                nn.ReLU(),
                nn.Conv2d(channel2, channel3, kernel_size=kernel_size3,
                          padding=padding3, dilation=dilation3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(channel3 * 6 * 6, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes)
            )

            # Move the model to the specified device
            model.to(self.device)

            # Define the loss function and optimizer
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Split the dataset into training and validation sets
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size])

            # Create data loaders for the training and validation sets
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False)

            # Train the model for the specified number of epochs
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    try:
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    except:
                        continue

                model.eval()
                total_correct = 0
                try: 
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs, labels = inputs.to(
                                self.device), labels.to(self.device)
                            outputs = model(inputs)
                            _, predictions = torch.max(outputs, 1)
                            total_correct += torch.sum(predictions == labels)

                    accuracy = total_correct.item() / len(val_dataset)
                    print(
                        f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy*100:.2f}%')
                except:
                    continue
            # Check if the model's accuracy is the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparams = {
                    'channel1': channel1, 
                    'channel2': channel2,
                    'channel3': channel3,
                    'kernel_size1': kernel_size1,
                    'kernel_size2': kernel_size2,
                    'kernel_size3': kernel_size3,
                    'padding1': padding1,
                    'padding2': padding2,
                    'padding3': padding3,
                    'dilation1': dilation1,
                    'dilation2': dilation2,
                    'dilation3': dilation3
                }

        print(f'Best accuracy: {best_accuracy*100:.2f}%')
        print(f'Best hyperparameters: {best_hyperparams}')


class ChessDataset(Dataset):
    def __init__(self):
        # Load your data and preprocess if necessary
        pass

    def __getitem__(self, idx):
        # Return a tuple (input, label)
        pass

    def __len__(self):
        # Return the length of the dataset
        pass


class ConvNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ConvNet, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Calculate the shape of the output from the convolutional layers
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.conv3(self.conv2(self.conv1(x)))
            self.feature_size = x.view(1, -1).shape[1]

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, output_shape)

    def forward(self, x):
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the convolutional layers
        x = x.view(-1, self.feature_size)

        # Apply fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



if __name__ == '__main__':
    # Import the MNIST dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    dataset = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    # Initialize a ConvNetHyperparamSearch object
    search = ConvNetHyperparamSearch(dataset, num_classes=10, device='cpu')

    # Search for the best hyperparameters
    search.search(num_epochs=10, batch_size=64, learning_rate=0.001)