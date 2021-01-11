
import torch
import torchvision
import torch.nn as nn

from resnet import resnet50_ft

class ft_model(nn.Module):
    def __init__(self, weights_path):
        super(ft_model, self).__init__()

        self.backbone = resnet50_ft(weights_path)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor()
    ])
        
    def transform_inputs(self, x):
        b = x.size()[0]
        xs = torch.empty(b, 3, 224, 224, device='cuda')
        for i in range(b):
            xs[i] = self.transform(255 * x[i])
        return xs
    
    def forward(self, x):
        x = self.transform_inputs(x)
        return self.backbone(x)