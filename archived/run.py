import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from copy import deepcopy

from tools import net, image, loss

def run_styler(net, input_image, epochs, loss_ratio):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = input_image.to(device)
    x.requires_grad = True
    optimizer = optim.LBFGS([x.requires_grad_()], lr=1.0)#Adam([x], lr=0.1)

    def closure():
        optimizer.zero_grad()
        x.data.clamp_(1e-3, 1)
        y = net(x)

        content_loss = 0
        style_loss = 0

        for c in net.content_losses: content_loss += c.loss
        for s in net.style_losses: style_loss += s.loss

        if(style_loss > 1e3):
            print('Overflow, style_loss = %.3f' % (style_loss))
            # style_loss = 0.1
            return np.inf
            
        style_loss /= loss_ratio
        style_loss /= len(net.style_losses)
        loss = content_loss + style_loss
        
        #print("Content loss: {}, style loss: {}".format(content_loss, style_loss))
        loss.backward()
                
        return loss
        
    best_loss = np.inf

    for i in range(epochs):
        l = optimizer.step(closure)
        print('Epoch: %d \t Loss: %.3f' % (i+1, l))
        if l > best_loss * 2: 
            x = x.clone().to(device) + torch.randn(x.shape) * 1e-3
            continue
        elif l < best_loss: best_loss = l
        
    x.data.clamp_(0, 1)
    return x