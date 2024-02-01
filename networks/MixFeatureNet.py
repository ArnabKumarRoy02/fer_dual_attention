import torch
import torch.nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

def l2Norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Conv_Block(Module):
    def __init__(self, in_c, out_c, kernel = (1, 1), stride = (1, 1), padding = (0, 0), groups = 1):
        super(Conv_Block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x
    

class Linear_Block(Module):
    def __init__(self, in_c, out_c, kernel = (1, 1), stride = (1, 1), padding = (0, 0), groups = 1):
        super(Linear_Block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class DepthWise(Module):
    def __init__(self, in_c, out_c, kernel = (3, 3), stride = (2, 2), padding = (1, 1), groups = 1, residual = False):
        super(DepthWise, self).__init__()
        self.conv = Conv_Block(in_c, out_c=groups, kernel = (1, 1), stride = (1, 1), padding = (0, 0))
        self.conv_dw = Conv_Block(groups, groups, groups=groups, kernel=kernel, stride=stride, padding=padding)
        self.project = Linear_Block(groups, out_c, kernel= (1, 1), padding= (0, 0), stride= (1, 1))

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else: 
            output = x
        return output