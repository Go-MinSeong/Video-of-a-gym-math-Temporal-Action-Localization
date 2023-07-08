import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv1 = nn.Conv1d(8, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)  # 배치 정규화 추가
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)  # 배치 정규화 추가
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)  # 배치 정규화 추가
        self.conv4 = nn.Conv1d(128, 512, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(512)  # 배치 정규화 추가
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(1024)  # 배치 정규화 추가
        self.conv6 = nn.Conv1d(1024, 2048, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(2048)  # 배치 정규화 추가

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(61440, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.pool(self.relu(self.bn5(self.conv5(x))))
        x = self.pool(self.relu(self.bn6(self.conv6(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        x = F.softmax(x, dim=1)
        return x


class ConvNett(nn.Module):
    def __init__(self):
        super(ConvNett, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv1 = nn.Conv1d(8, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)  # 배치 정규화 추가
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)  # 배치 정규화 추가
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(128)  # 배치 정규화 추가
        self.conv4 = nn.Conv1d(128, 512, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(512)  # 배치 정규화 추가
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(1024)  # 배치 정규화 추가
        self.conv6 = nn.Conv1d(1024, 2048, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(2048)  # 배치 정규화 추가

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(61440, 16384)
        self.fc2 = nn.Linear(16384, 8192)
        self.fc3 = nn.Linear(8192, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.fc9(x)
        x = F.softmax(x, dim=1)
        return x


    def shape(self, x):
        print(x.shape)
