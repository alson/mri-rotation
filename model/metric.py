from math import sqrt

import torch


def mean_absolute_error(output, target):
    with torch.no_grad():
        pred = output[:, 0]
        assert pred.shape[0] == len(target)
        sum_abs_err = torch.sum(torch.abs(pred - target))
    return sum_abs_err / len(target)


def mean_squared_error(output, target):
    with torch.no_grad():
        pred = output[:, 0]
        assert pred.shape[0] == len(target)
        sum_sq_err = torch.sum(torch.square(pred - target))
    return sum_sq_err / len(target)


def root_mean_squared_error(output, target):
    with torch.no_grad():
        pred = output[:, 0]
        assert pred.shape[0] == len(target)
        sum_sq_err = torch.sum(torch.square(pred - target))
    return sqrt(sum_sq_err / len(target))
