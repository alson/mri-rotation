import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class AdniModel(ResNet):
    def __init__(self, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1)
