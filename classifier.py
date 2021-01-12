import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet50_ft

class ft_model(nn.Module):
    def __init__(self, weights_path):
        super(ft_model, self).__init__()

        self.backbone = resnet50_ft(weights_path)
    
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=True)
        x = x*255
        return self.backbone(x)