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
    
    #x = torch.rand(net.inshape).to(device)
    #x = torch.ones(input_image.shape).to(device)
    x = input_image.to(device)
    x.requires_grad = True
    #optimizer = optim.LBFGS([x], max_iter=epochs)#Adam([x], lr=0.1)
    optimizer = optim.LBFGS([x.requires_grad_()], lr=1.0)

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
            style_loss = 0.1
            
        style_loss /= loss_ratio
        loss = (content_loss + style_loss)/len(net.style_losses)
        
        #print("Content loss: {}, style loss: {}".format(content_loss, style_loss))
        loss.backward()
                
        return loss
        
    for i in range(epochs):
        l = optimizer.step(closure)
        print('Epoch: %d \t Loss: %.3f' % (i+1, l))
        if(l == np.inf):
            break
        
    x.data.clamp(0, 1)
    return x