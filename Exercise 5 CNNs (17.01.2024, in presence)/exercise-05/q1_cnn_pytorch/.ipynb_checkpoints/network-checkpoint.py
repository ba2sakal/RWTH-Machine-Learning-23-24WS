import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,24,5,1)
        self.conv2 = nn.Conv2d(24,32,5,1)
        self.conv3 = nn.Conv2d(32,50,5,1)
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(50*3*3,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,3)

    def forward(self, x, label):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 50*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        logits = F.softmax(x, dim= 1)
        loss = F.cross_entropy(logits, target = label)

        return loss, logits
