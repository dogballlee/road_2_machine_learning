import torch
import torch.nn
import torch.optim
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, 16*5*5)
        logits = self.fc_unit(x)
        return  logits

def main():
    net = LeNet5()
    temp = torch.randn(2, 3, 32, 32)
    out = net(temp)
    print("lenet_out:", out.shape)

if __name__ == "__main__":
    main()