import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, n_classes, base_channels=64, dropout_p=-1, **kwargs):
        super(SimpleCNN, self).__init__()
        hidden_channels = [base_channels * k for k in [1, 2, 4, 4, 8]]

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 256

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[1], hidden_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 128

        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 64
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_channels[2], hidden_channels[3], kernel_size=3),
            nn.BatchNorm2d(hidden_channels[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 32

        self.layer5 = nn.Sequential(
            nn.Conv2d(hidden_channels[3], hidden_channels[4], kernel_size=3), # 10
            nn.BatchNorm2d(hidden_channels[4]), 
            nn.ReLU(),
            nn.Conv2d(hidden_channels[4], hidden_channels[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[4]),
            nn.ReLU(),) # 32
        
        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout3 = nn.Dropout(dropout_p)
            self.dropout4 = nn.Dropout(dropout_p)
            self.dropout5 = nn.Dropout(dropout_p)
        else:
            self.dropout3 = nn.Identity()
            self.dropout4 = nn.Identity()
            self.dropout5 = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(hidden_channels[4], n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.dropout4(x)
        x = self.layer5(x)
        x = self.dropout5(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    