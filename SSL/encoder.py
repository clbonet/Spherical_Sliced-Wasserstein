r"""
Adapted from https://github.com/HobbitLong/CMC/blob/f25c37e49196a1fe7dc5f7b559ed43c6fce55f70/models/alexnet.py
"""

import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class ResNet(nn.Module):

    def __init__(self, in_channel: int = 3, feat_dim: int = 128):
        super().__init__()
        feat_before_dim = 32 * 32
        self.rn = resnet18(num_classes=feat_before_dim)

        self.rn.maxpool = nn.Identity()
        self.rn.conv1 = nn.Conv2d(in_channel, 64,
                kernel_size=3, stride=1, padding=2, bias=False)

        self.predictor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(32 * 32, feat_dim, bias=False),
            L2Norm(),
        )

    def forward(self, x, layer_index:int = -1):
        if layer_index == -1:
            return self.predictor(self.rn(x))

        if layer_index == -2:
            # try adding the relu as part of the -2th layer
            return F.relu(self.rn(x), inplace=True)

        raise NotImplementedError(layer_index)
