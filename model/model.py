import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck


class AdniModel18(ResNet):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock,  [2, 2, 2, 2], **kwargs)
        self.fc = nn.Linear(512 * BasicBlock.expansion, 2)


class AdniModel34(ResNet):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], **kwargs)
        self.fc = nn.Linear(512 * BasicBlock.expansion, 2)


class AdniModel50(ResNet):
    def __init__(self, **kwargs):
        super().__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.fc = nn.Linear(512 * Bottleneck.expansion, 2)


class AdniModel101(ResNet):
    def __init__(self, **kwargs):
        super().__init__(Bottleneck,[3, 4, 23, 3], **kwargs)
        self.fc = nn.Linear(512 * Bottleneck.expansion, 2)
