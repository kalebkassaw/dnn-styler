## Neural network based in Torch. 
## Requires torch==1.5.0 torchvision==0.6.0, use !pip install if needed.

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy

from tools import image, loss

layers = ['conv_1_1', 'conv_1_2',
          'conv_2_1', 'conv_2_2',
          'conv_3_1', 'conv_3_2', 'conv_3_3', 'conv_3_4',
          'conv_4_1', 'conv_4_2', 'conv_4_3', 'conv_4_4',
          'conv_5_1', 'conv_5_2', 'conv_5_3', 'conv_5_4']

class Styler(nn.Module):
    layer_dict = {
        'conv_1_1':  0, 'conv_1_2':  2,
        'conv_2_1':  5, 'conv_2_2':  7,
        'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16,
        'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25,
        'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34,
        }

    layer_dict_flip = {value:key for key, value in layer_dict.items()}
    im_mean  = [0.485, 0.456, 0.406]
    im_stdev = [0.229, 0.224, 0.225]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __init__(self, content_im, style_im, im_size, content_layers, style_layers):
        super(Styler, self).__init__()
        norm = image.Normalize(Styler.im_mean, Styler.im_stdev).to(Styler.device)
        VGG19 = torchvision.models.vgg19(pretrained = True)
        VGG19 = VGG19.to(Styler.device)

        assert content_im.size() == style_im.size(), "Content and style images should have the same aspect ratio."
        self.inshape = content_im.size()
        self.model = nn.Sequential(norm)

        ldf = Styler.layer_dict_flip
        self.content_losses = []
        self.style_losses = []

        for name, layer in VGG19.features.named_children():
            try:
                self.model.add_module(name, layer)
                if ldf[int(name)] in content_layers:
                    t = self.model(content_im).detach()
                    l = loss.ContentLoss(t)
                    self.model.add_module('content_{}'.format(name), l)
                    self.content_losses.append(l)
                if ldf[int(name)] in style_layers:
                    t = self.model(style_im).detach()
                    l = loss.StyleLoss(t)
                    self.style_losses.append(l)
                    self.model.add_module('style_{}'.format(name), l)
            except KeyError:
                if isinstance(layer, nn.ReLU): self.model.add_module(name, nn.ReLU(inplace=False))
                else: self.model.add_module(name, layer)
                continue
        
    def forward(self, x):
        return self.model(x)

    def summary(self):
        for name, child in self.model.named_children():
            print(name, type(child))