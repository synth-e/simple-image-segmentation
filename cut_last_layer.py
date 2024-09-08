import torch
import torch.nn as nn
from torchvision import models
import sys

if __name__ == '__main__':
    i, o = sys.args[1:3]

    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False)
    model.load_state_dict(torch.load(i, map_location="cpu"), strict=False)
    torch.save(model.classifier[4].state_dict(), o)
