
import torch.nn as nn

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.BatchNorm2d(o),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Net(nn.Module):
    def __init__(self, n=50):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128)
        )
        self.head = nn.Linear(128, n)

    def forward(self, x):
        return self.head(
            self.features(x).mean(dim=(2, 3))
        )

