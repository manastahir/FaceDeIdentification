import torch
import torch.nn as nn
from collections import OrderedDict

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class DepthwiseSeparableConv2DBlock(nn.Module):
    def __init__(self, nin, nout, ksize, stride, block_no):
        super(DepthwiseSeparableConv2DBlock, self).__init__()

        self.block = nn.Sequential(OrderedDict([
        (f'depthwise_{block_no}', nn.Conv2d(nin, nout, kernel_size=ksize, padding=1, stride=stride, groups=nin, bias=False)),
        (f'pointwise_{block_no}',  nn.Conv2d(nout, nout, kernel_size=1, bias=False)),
        (f'instanceNorm_{block_no}', nn.InstanceNorm2d(nout, affine=True))
        ]))
    def forward(self, x):
        return self.block(x)


class UpScale(nn.Module):
    def __init__(self, in_channels, feature_shape, block_no):
        super(UpScale, self).__init__()
        out_channels = int(2*in_channels)
        
        self.block = nn.Sequential(OrderedDict([
                     (f'conv2d_up_{block_no}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1,
                                                         bias=False)),
                     (f'instanceNorm_up_{block_no}' , nn.InstanceNorm2d(out_channels, affine=True)),
                     (f'activ_up_{block_no}', nn.LeakyReLU(0.1)),
                     (f'reshape_up_{block_no}', Reshape(-1, int(in_channels/2), int(feature_shape*2), int(feature_shape*2)))
        ]))

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, block_no):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
                      (f'cov2d_1_res_{block_no}', nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                                            bias=False)),
                      (f'activ_res_{block_no}', nn.LeakyReLU(0.2)),
                      (f'cov2d_2_res_{block_no}', nn.Conv2d(in_channels, in_channels, kernel_size=3,  padding=1, 
                                                            bias=False))
        ]))

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        return x    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super(ConvBlock, self).__init__()
        
        self.do_norm = norm
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2,
                                                            bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activ = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        if (self.do_norm):
            x = self.norm(x)
        x = self.activ(x)

        return x