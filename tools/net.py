## Neural network based in Torch. 
## Requires torch==1.5.0 torchvision==0.6.0, use !pip install if needed.

import torch
import torchvision
import numpy as np
import torch.nn as nn

VGG19 = torchvision.models.vgg19(pretrained = True)

def summary(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

summary(VGG19)