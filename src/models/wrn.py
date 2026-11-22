import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class WRN16_2(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        widen_factor = 2
        n = 2
        init_channels = 16

        self.conv1 = nn.Conv2d(
            3,
            init_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            init_channels,
            init_channels * widen_factor,
            n,
            1,
        )
        self.layer2 = self._make_layer(
            init_channels * widen_factor,
            32 * widen_factor,
            n,
            2,
        )
        self.layer3 = self._make_layer(
            32 * widen_factor,
            64 * widen_factor,
            n,
            2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * widen_factor, num_classes)
        self._initialize_weights()

    def _make_layer(self, in_planes, out_planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_planes, out_planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_planes, out_planes, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits