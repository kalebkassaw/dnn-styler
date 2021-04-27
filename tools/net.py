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

'''layers = ['relu_1_1', 'relu_1_2',
          'relu_2_1', 'relu_2_2',
          'relu_3_1', 'relu_3_2', 'relu_3_3', 'relu_3_4',
          'relu_4_1', 'relu_4_2', 'relu_4_3', 'relu_4_4',
          'relu_5_1', 'relu_5_2', 'relu_5_3', 'relu_5_4']'''
'''layers = ['conv_1', 'conv_2',
          'conv_3', 'conv_4',
          'conv_5', 'conv_6', 'conv_7', 'conv_8',
          'conv_9', 'conv_10', 'conv_11', 'conv_12',
          'conv_13', 'conv_14', 'conv_15', 'conv_16']'''
          
class Styler(nn.Module):
    layer_dict = {
        'conv_1_1':  0, 'conv_1_2':  2,
        'conv_2_1':  5, 'conv_2_2':  7,
        'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16,
        'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25,
        'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34,
        'relu_1_1':  1, 'relu_1_2':  3,
        'relu_2_1':  6, 'relu_2_2':  8,
        'relu_3_1': 11, 'relu_3_2': 13, 'relu_3_3': 15, 'relu_3_4': 17,
        'relu_4_1': 20, 'relu_4_2': 22, 'relu_4_3': 24, 'relu_4_4': 26,
        'relu_5_1': 29, 'relu_5_2': 31, 'relu_5_3': 33, 'relu_5_4': 35,
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

        #'''
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
        #'''
        '''
        i = 0  # increment every time we see a conv
        for n, layer in VGG19.features.named_children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = self.model(content_im).detach()
                content_loss = loss.ContentLoss(target)
                self.model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = self.model(style_im).detach()
                style_loss = loss.StyleLoss(target_feature)
                self.model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], loss.ContentLoss) or isinstance(self.model[i], loss.StyleLoss):
                break

        self.model = self.model[:(i + 1)]
        #'''

    def forward(self, x):
        return self.model(x)

    def summary(self):
        for name, child in self.model.named_children():
            print(name, type(child))