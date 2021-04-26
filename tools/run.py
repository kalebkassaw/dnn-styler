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

def run_styler(net, epochs, loss_ratio):
    print('Starting styler optimization.')
    x = torch.rand(net.input_shape)
    optimizer = optim.Adam(x, lr=1e-4)

    for i in tqdm(range(epochs), desc='Optimization progress'):
        x.data.clamp(0, 1)
        x = net(x)
        for c in net.content_losses:
            content_loss += c
        for s in net.style_losses:
            style_loss   += s

        style_loss /= loss_ratio
        loss = content_loss + style_loss
        loss.backward()
        optimizer.step()
        x.data.clamp(0, 1)

    return x