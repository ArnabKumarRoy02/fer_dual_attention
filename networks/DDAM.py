import os
import torch
import torch.nn as nn
from torch.nn import Module
from networks import MixFeatureNet


class Linear_Block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_Block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class DDAMNet(Module):
    def __init__(self, num_classes=7, num_heads=2, pretrained=True):
        super(DDAMNet, self).__init__()
        net = MixFeatureNet.MixFeatureNet()

        if pretrained:
            net = torch.load(os.path.join('./pretrained', "MFN_msceleb.pth"))

        self.features = nn.Sequential(*list(net.children())[:-4])
        self.num_heads = num_heads
        
        for i in range(num_heads):
            setattr(self, 'cat_head%d' %(i), CoordinateAttentionHead())

        self.linear = Linear_Block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()
        self.fc = nn.Linear(512, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.features(x)
        heads = []

        for i in range(self.num_heads):
            heads.append(getattr(self, 'cat_head%d' %i)(x))
        head_out = heads
        y = heads[0]

        for i in range(1, self.num_heads):
            y = torch.max(y, heads[i])

        y = x*y
        y = self.linear(y)
        y = self.flatten(y)
        out = self.fc(y)

        return out, x, head_out
    

class h_sigmoid(Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    

class h_swish(Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    

class CoordinateAttentionHead(Module):
    def __init__(self):
        super().__init__()
        self.CoordinateAttention = CoordinateAttention(512, 512)
    
    def forward(self, x):
        ca = self.CoordinateAttention(x)
        return ca
    

class CoordinateAttention(Module):
    def __init__(self, inp, out, groups=32):
        super(CoordinateAttention, self).__init__()
        self.Linear_h = Linear_Block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))
        self.Linear_w = Linear_Block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))

        midP = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, midP, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(midP)
        self.conv2 = nn.Conv2d(midP, out, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(midP, out, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_Block(out, out, groups=out, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.Linear_h(x)
        x_w = self.Linear_w(x)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = x_w * x_h

        return y