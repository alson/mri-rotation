import torch.nn.functional as F


def l1_loss(output, target):
    return F.l1_loss(output[:, 0], target)
