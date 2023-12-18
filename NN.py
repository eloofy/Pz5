import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, in_channels, image_size, num_classes):
        super(SimpleNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(32 * image_size * image_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)
