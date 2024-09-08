import torch
import torch.nn as nn
from torchvision import models
import sys

if __name__ == '__main__':
    i, o, k = sys.argv[1:4]

    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False)
    model.classifier[4] = nn.Conv2d(
        256, int(k), kernel_size=(1, 1), stride=(1, 1)
    )
    model.load_state_dict(torch.load(i, map_location="cpu"), strict=False)
    torch.save(model.classifier[4].state_dict(), o)
