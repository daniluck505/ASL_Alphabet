import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.conv0 = nn.Conv2d(nc, nc, 3, padding=1)
        self.norm0 = nn.BatchNorm2d(nc)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(nc, nc, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(nc)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(x + out)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.act = nn.GELU()
        self.conv0 = nn.Conv2d(nc, nc//4, kernel_size=1, padding=0)
        self.norm0 = nn.BatchNorm2d(nc//4)
        self.conv1 = nn.Conv2d(nc//4, nc//4, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(nc//4)
        self.conv2 = nn.Conv2d(nc//4, nc, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv0(x)
        out = self.norm0(out)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(x + out)
        return out


class ResStack(nn.Module):
    def __init__(self, nc, num_blocks, block_type):
        super().__init__()
        stack = []
        for i in range(num_blocks):
            if block_type == 'bottleneck':
                stack.append(BottleneckBlock(nc))
            elif block_type == 'resblock':
                stack.append(ResBlock(nc))

        self.blocks = nn.Sequential(*stack)

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    def __init__(self, in_nc, nc, out_nc, num_blocks, block_type):
        super().__init__()
        self.conv0 = nn.Conv2d(in_nc, nc, 3, padding=1)
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(2,2)


        self.layer1 = ResStack(nc, num_blocks, block_type)
        self.conv1 = nn.Conv2d(nc, 2*nc, 3, padding=1, stride=2)
        self.layer2 = ResStack(2*nc, num_blocks, block_type)
        self.conv2 = nn.Conv2d(2*nc, 4*nc, 3, padding=1, stride=2)
        self.layer3 = ResStack(4*nc, num_blocks, block_type)
        self.conv3 = nn.Conv2d(4*nc, 4*nc, 3, padding=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4*nc, out_nc)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.conv1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out