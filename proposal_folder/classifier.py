import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        #self.conv1 = nn.Conv1d(4, 8, kernel_size=3)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3)
        self.conv4 = nn.Conv1d(16, 32, kernel_size=3)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=3)
        #self.conv5 = nn.Conv1d(128, 256, kernel_size=3)

        self.fc1 = nn.Linear(4032, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
    def shape(self, x):
        print(x.shape)

    def forward(self, x):
        #x = x.squeeze(2)
        #x = self.pool(self.relu(self.conv1(x)))
        #self.shape(x)
        x = self.pool(self.relu(self.conv2(x)))
        #self.shape(x)
        x = self.pool(self.relu(self.conv3(x)))
        #self.shape(x)
        x = self.pool(self.relu(self.conv4(x)))
        #self.shape(x)
        x = self.pool(self.relu(self.conv5(x)))
        #self.shape(x)
        x = x.view(4, -1)
        #self.shape(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout2(x)
        x = self.fc6(x)
        x = F.softmax(x, dim=1)
        return x
