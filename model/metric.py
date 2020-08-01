import torch

def mean_absolute_error(output, target):
    with torch.no_grad():
        pred = output[:, 0]
        assert pred.shape[0] == len(target)
        sum_abs_err = torch.sum(torch.abs(pred - target))
    return sum_abs_err / len(target)
