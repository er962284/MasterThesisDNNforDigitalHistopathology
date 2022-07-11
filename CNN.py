import torch
from torchvision import datasets, transforms
import torch.nn.init
import torch.nn as nn
from torch.autograd import Variable

# Define Global Variables
BATCH_SIZE = 32
KEEP_PROB = 1  # Used in Dropout layer
# Load training data
train = datasets.MNIST(root='MNIST_data/',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)
# Load test dataset
test = datasets.MNIST(root='MNIST_data/',
                      train=False,
                      transform=transforms.ToTensor(),
                      download=True)
# to see train and test dataset
# print(train, '\n\n', test)
# train dataset loader
train_data_loader = torch.utils.data.DataLoader(
    dataset=train,
    batch_size=BATCH_SIZE,
    shuffle=True)
# test dataset loader
test_data_loader = torch.utils.data.DataLoader(
    dataset=test,
    batch_size=BATCH_SIZE,
    shuffle=True)

# To visualize image from dataset
import matplotlib.pyplot as plt
import numpy as np

for i in train_data_loader:
    image = np.array(i[0][0])
    print(image.shape)
    image = np.transpose(image, (1, 2, 0)).astype(np.float32)
    plt.imshow(image)
    plt.xlabel(i[1][0])
    break  # Since we want to see only one image


# Create Convolutional model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # input image channels =1
        # input image size = 28, 28

        # First Convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - KEEP_PROB)
        )

        # Second Convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - KEEP_PROB)
        )

        # Third Convolutional layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=1 - KEEP_PROB)
        )
        # Fully connected layers
        # input = 4*4*128 = 2048
        self.lin_layer1 = nn.Linear(2048, 625, bias=True)
        nn.init.xavier_uniform(self.lin_layer1.weight)

        # Fully connected layer
        self.layer4 = nn.Sequential(
            self.lin_layer1,
            nn.ReLU(),
            nn.Dropout(p=1 - KEEP_PROB)
        )

        self.lin_layer2 = nn.Linear(in_features=625, out_features=10, bias=True)

        nn.init.kaiming_uniform_(self.lin_layer2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # Flatten
        out = out.view(out.size(0), -1)
        out = self.lin_layer1(out)
        out = self.lin_layer2(out)


model = Model()
# Learning rate
lr = 0.001
# Criterion
criterion = torch.nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# model training function
def train_model(model, optimizer, data_loader, loss_module, num_epochs):
    """
    Train a model on the training set of FashionMNIST
    Inputs:
        net - Object of BaseNetwork
        model_name - (str) Name of the model, used for creating the checkpoint names
        max_epochs - Number of epochs we want to (maximally) train for
        patience - If the performance on the validation set has not improved for #patience epochs, we stop training early
        batch_size - Size of batches used in training
        overwrite - Determines how to handle the case when there already exists a checkpoint. If True, it will be overwritten. Otherwise, we skip training.
    """
    model.train()
    for epoch in range(num_epochs):
        for i, (data_inputs, data_labels) in enumerate(train_data_loader):

            X = Variable(data_inputs)
            y = Variable(data_labels)

            optimizer.zero_grad()

            # forward propogation
            hypothesis = model(X)

            # Caculate the loss
            cost = loss_module(hypothesis, y)

            # Perform backpropogation
            cost.backward()

            # Update the parameters
            optimizer.step()

            if i % 500 == 0:
                print('Epoch: ', epoch, 'Batch: ', i, 'Loss: ', cost)


train_model(model=model, optimizer=optimizer, data_loader=train_data_loader, loss_module=criterion, num_epochs=15)


# Model Evaluation function
def eval_model(model, data_loader):
    """
    Test a model on a specified dataset.
    Inputs:
        model - Trained model of type BaseNetwork
        data_loader - DataLoader object of the dataset to test on (validation or test)
    """
    model.eval()
    true_preds, count = 0., 0
    for imgs, labels in data_loader:
        with torch.no_grad():
            preds = model(imgs).argmax(dim=-1)
            true_preds += (preds == labels).sum().item()
            count += labels.shape[0]
    test_acc = true_preds / count
    return test_acc


eval_model(model, test_data_loader)