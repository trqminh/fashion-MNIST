import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*2*2, 120),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, self.num_classes),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128*2*2)
        x = self.fc1(x)
        x = x.view(-1, 120)
        x = self.fc2(x)
        x = x.view(-1, 84)
        x = self.fc3(x)

        return x
