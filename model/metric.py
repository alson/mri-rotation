from math import pi

import torch


def mean_absolute_error(output, target):
    with torch.no_grad():
        pred = output
        assert pred.shape == target.shape
        pred_angles = torch.atan(pred[:, 0] / pred[:, 1]) * 180 / pi
        target_angles = torch.atan(target[:, 0] / target[:, 1]) * 180 / pi
        pred_angles3 = pred_angles.repeat(3, 1).transpose(0, 1)
        target_angles3 = target_angles.repeat(3, 1).transpose(0, 1)
        offset3 = torch.tensor([0, 360, -360]).repeat(pred_angles.shape[0], 1)
        sum_abs_err = torch.sum(torch.min(torch.abs(pred_angles3 - target_angles3 + offset3), dim=1)[0])
    return sum_abs_err / len(target)


def root_mean_squared_error(output, target):
    with torch.no_grad():
        pred = output
        assert pred.shape == target.shape
        pred_angles = torch.atan(pred[:, 0] / pred[:, 1]) * 180 / pi
        target_angles = torch.atan(target[:, 0] / target[:, 1]) * 180 / pi
        pred_angles3 = pred_angles.repeat(3, 1).transpose(0, 1)
        target_angles3 = target_angles.repeat(3, 1).transpose(0, 1)
        offset3 = torch.tensor([0, 360, -360]).repeat(pred_angles.shape[0], 1)
        sum_sq_err = torch.sum(torch.min(torch.square(pred_angles3 - target_angles3 + offset3), dim=1)[0])
    return torch.sqrt(sum_sq_err / len(target))
