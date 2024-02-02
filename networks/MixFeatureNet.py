import torch
import torch.nn as nn
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
    

class Swish(Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}


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
    
class swish(Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class CoordinateAttention(Module):
    def __init__(self, input, output, groups= 32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        midP = max(8, input // groups)

        self.conv1 = nn.Conv2d(input, midP, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(midP)
        self.conv2 = nn.Conv2d(midP, output, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(midP, output, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

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

        y = identity * x_w * x_h
        return y
    

class MixedDepthwiseConv(Module):
    def __init__(self, channels, kernel_size, split_out_channels, stride):
        super(MixedDepthwiseConv, self).__init__()
        self.num_groups = len(kernel_size)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()

        for i in range(self.num_groups):
            self.mixed_depthwise_conv.append(Conv2d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size[i],
                stride=stride,
                padding=kernel_size[i] // 2,
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = BatchNorm2d(channels)
        self.prelu = PReLU(channels)

    def forward(self, x):
        if self.num_groups == 1:
            return self.mixed_depthwise_conv[0](x)
        
        x_split = torch.split(x, self.split_channels, 1)
        x = [conv(t) for conv, t, in zip(self.mixed_depthwise_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x
    

class MixDepthWise(Module):
    def __init__(self, in_c, out_c, residual= False, kernel = (3, 3), stride= (2, 2), padding= (1, 1), groups= 1, kernel_size= [3, 5, 7], split_out_channels= [64, 32, 32]):
        super(MixDepthWise, self).__init__()
        self.conv = Conv_Block(in_c, out_c=groups, kernel= (1, 1), padding= (0, 0), stride= (1, 1))
        self.conv_dw = MixedDepthwiseConv(channels=groups, kernel_size=kernel_size, split_out_channels=split_out_channels, stride=stride)
        self.CA = CoordinateAttention(groups, groups)
        self.project = Linear_Block(groups, out_c, kernel= (1, 1), padding= (0, 0), stride= (1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.CA(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output
    

class Residual(Module):
    def __init__(self, channel, num_blocks, groups, kernel= (3, 3), stride= (1, 1), padding= (1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(DepthWise(channel, channel, kernel= kernel, stride=stride, padding= padding, groups= groups, residual= True))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)
    

class MixResidual(Module):
    def __init__(self, channel, num_blocks, groups, kernel= (3, 3), stride= (1, 1), padding= (1, 1), kernel_size= [3, 5], split_out_channels= [64, 64]):
        super(MixResidual, self).__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(MixDepthWise(channel, channel, kernel=kernel, stride=stride, padding=padding, groups=groups, kernel_size=kernel_size, split_out_channels=split_out_channels, residual=True))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)