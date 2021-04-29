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
import time

from tools import net, image, loss

def run_styler(net, input_image, epochs, loss_ratio, device):        
    x = input_image.to(device)
    x.requires_grad = True
    lr       = 0.1
    #steps    = (np.array([0.25, 0.5, 0.75, 0.9]) * epochs).astype(int)
    #steps    = (np.array([0.5, 0.9]) * epochs).astype(int)
    lr_decay = 0.3

    #optimizer = optim.LBFGS([x.requires_grad_()], lr=lr)
    optimizer = optim.Adam([x.requires_grad_()], lr=lr)

    def closure():
        optimizer.zero_grad()
        x.data.clamp_(0, 1)
        y = net(x)

        content_loss = 0
        style_loss = 0

        for c in net.content_losses: content_loss += c.loss
        for s in net.style_losses: style_loss += s.loss

        if(torch.isnan(style_loss)):
            return np.inf
            
        style_loss /= len(net.style_losses)
        
        b = (1/loss_ratio)
        #a = loss_ratio
        #b = b / (a+b)
        #a = a / (a+b)
        style_loss   *= b
        #content_loss *= a
        loss = (content_loss + style_loss)
        
        #print("Content loss: {}, style loss: {}".format(content_loss, style_loss))
        loss.backward()
                
        return loss
        
    t_start = time.time()
    for i in range(epochs):
        l = optimizer.step(closure)
        #print('Epoch: %d \t Loss: %.3f' % (i+1, l))
        
        if((i+1) % 10 == 0):
            p = ((i+1)/epochs)
            c_t = (time.time() - t_start)/60
            print('Progress = %d / %d (%0.2f%%) \t Time = %.2f m \t Est Time %.2f m \t lr = %.4f m \t Loss: %.10f'%(\
                      i+1, epochs, p*100, c_t, c_t/p, lr, l))
        if(l == np.inf):
            break
        
        #if(i+1 in steps):
        #    lr *= lr_decay
        #    optimizer.param_groups[-1]['lr'] = lr

        
    x.data.clamp_(0, 1)
    return x