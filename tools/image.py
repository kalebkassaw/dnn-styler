import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load(path, size=512):
    # Load in images as a specified size in a PyTorch Tensor.
    img = Image.open(path)
    torch_load = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()])
    img = torch_load(img).unsqueeze(0)
    return img.type(torch.float).to(device)

class Normalize(nn.Module):
    # Normalize images, enabling autograd in PyTorch through use of nn.Module.
    def __init__(self, m, s):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = torch.tensor(m).view(-1, 1, 1).to(device)
        self.std  = torch.tensor(s).view(-1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

def unload(i, ax, title=None):
    # Pass in a PyTorch Tensor and a matplotlib Axes object, let "unload" do the rest.
    torch_unload = transforms.ToPILImage()
    im = i.cpu().clone()
    im = im.squeeze(0)
    im = torch_unload(im)
    ax.imshow(im)
    ax.axis('off')
    if title is not None: ax.set_title(title)

def save(i, filename):
    torch_unload = transforms.ToPILImage()
    im = i.cpu().clone()
    im = im.squeeze(0)
    im = torch_unload(im)
    im.save("outputs/{}.jpg".format(filename))

def clipImage(image):
    im = Image.open(image)
    im = im.resize(512, 512)
    im.save("images/crop/{}".format(filename))