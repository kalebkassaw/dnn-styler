import torch
import torch.nn as nn
import torch.nn.functional as F

def gram(x):
    a, b, c, d = x.size()
    features = x.view(a*b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, t_feat):
        super().__init__()
        self.t = gram(t_feat).detach()

    def forward(self, x):
        self.loss = F.mse_loss(gram(x), self.t)
        return x

class ContentLoss(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.t)
        return x