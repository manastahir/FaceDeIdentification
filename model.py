import torch
import torch.nn as nn
from collections import OrderedDict

from modules import Reshape, DepthwiseSeparableConv2DBlock, UpScale, ResidualBlock, ConvBlock

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def get_device_type(self) -> str or Exception:
        msg = f'get_device_type has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor or Exception:
        msg = f'forward has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    def batch_forward(self, x: torch.Tensor, batch: int = 25) -> torch.Tensor:
        return torch.cat(
            tuple(self.forward(x[i:i+batch]).cpu() for i in range(0, x.shape[0], batch)), 0)

    def get_loss(self, labels: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor or Exception:
        msg = f'get_loss has not been implemented for {self.__class__.__name__}'
        raise NotImplementedError(msg)

    
class Generator(Network):
      def __init__(self, **kwargs):
            super(Generator, self).__init__(**kwargs)

            self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)

            self.strided_conv1 = DepthwiseSeparableConv2DBlock(32,  64,   3, 2, 1)  #128
            self.strided_conv2 = DepthwiseSeparableConv2DBlock(64,  128,  3, 2, 2)  #64
            self.strided_conv3 = DepthwiseSeparableConv2DBlock(128, 256,  3, 2, 3)  #32
            self.strided_conv4 = DepthwiseSeparableConv2DBlock(256, 512,  3, 2, 4)  #16
            self.strided_conv5 = DepthwiseSeparableConv2DBlock(512, 1024, 3, 2, 5)  #8

            self.encoder_dense = nn.Linear(1024*8*8, 1024, bias=False)
          
            self.decoder_dense = nn.Linear(1024+2048, 1024*4*4, bias=False)

            self.upscale_1 = UpScale(1024, 4,  1)
            self.upscale_2 = UpScale(512,  8,  2)
            self.upscale_3 = UpScale(256,  16, 3)
            self.upscale_4 = UpScale(128,  32, 4)
            self.upscale_5 = UpScale(64,   64, 5)
            self.upscale_6 = UpScale(32,   128,6)
          
            self.residual_1 = ResidualBlock(512, 1)
            self.residual_2 = ResidualBlock(256, 2)
            self.residual_3 = ResidualBlock(128, 3)
            self.residual_4 = ResidualBlock(64,  4)
            self.residual_5 = ResidualBlock(32,  5)
            self.residual_6 = ResidualBlock(16,  6)

            self.img_conv = nn.Conv2d(16,  3, kernel_size=3, padding=1, bias=False)
            self.mask_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False)
      
      def forward(self, inputs):
            x, resnet_vec = inputs
            
            #Encoder
            x = self.conv2d_1(x)          #256
            x = self.strided_conv1(x)     #128
            x = self.strided_conv2(x)     #64
            x = self.strided_conv3(x)     #32
            x = self.strided_conv4(x)     #16
            x = self.strided_conv5(x)     #8
          
            x = self.encoder_dense(nn.Flatten()(x))
            x = torch.cat([x, resnet_vec.view(-1, 2048)], axis=1)
          
            #Decoder
            x = self.decoder_dense(x)
            x = self.upscale_1(Reshape(-1, 1024, 4, 4)(x))
            x = self.residual_1(x)
            
            x = self.upscale_2(x)
            x = self.residual_2(x)
            
            x = self.upscale_3(x)
            x = self.residual_3(x)
            
            x = self.upscale_4(x)
            x = self.residual_4(x)
            
            x = self.upscale_5(x)
            x = self.residual_5(x)
            
            x = self.upscale_6(x)
            x = self.residual_6(x)

            img = torch.nn.Tanh()(self.img_conv(x))
            mask = torch.nn.Sigmoid()(self.mask_conv(x))
            
            return img,mask

class Discriminator(Network):
    
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.conv1 = ConvBlock(3,   32, norm=False)
        
        self.conv2 = ConvBlock(32,  64)

        self.conv3 = ConvBlock(64,  128)

        self.conv4 = ConvBlock(128, 256)

        self.conv5 =  nn.Conv2d(256, 1, kernel_size=16, bias=False) 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Sigmoid()(self.conv5(x))

        return x.view(-1, 1)   

class AdvserialAutoEncoder(Network):
    def __init__(self, generator, discriminator, **kwargs):
        super(AdvserialAutoEncoder, self).__init__(**kwargs)
        
        self.generator = generator
        self.discriminator = discriminator
    
    def forward(self, inputs):
        source, real, resnet_vec, lam = inputs
        
        raw, mask = self.generator([source, resnet_vec])
        masked = raw*mask +(1-mask)*source
        
        x = lam*real + (1-lam)*masked
        
        yhat = self.discriminator(x)

        return [raw, mask, masked, yhat]