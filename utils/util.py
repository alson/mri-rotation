import json
import pickle

import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def load_checkpoint(path, logger):
    '''
    Load a checkpoint onto the CPU. Checkpoints are loaded weights-only (torch's safe default) when possible;
    checkpoints saved before the config was stored as a plain dict contain a pickled ConfigParser and need a
    full unpickle, so only load those from a trusted source.
    '''
    try:
        return torch.load(path, map_location='cpu')
    except pickle.UnpicklingError:
        logger.warning("Checkpoint {} is in the legacy format (pickled ConfigParser); loading it with "
                       "weights_only=False.".format(path))
        return torch.load(path, map_location='cpu', weights_only=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._keys = keys
        self.reset()

    def reset(self):
        self._total = dict.fromkeys(self._keys, 0.0)
        self._counts = dict.fromkeys(self._keys, 0)

    def update(self, key, value, n=1):
        value = float(value)
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._total[key] += value * n
        self._counts[key] += n

    def avg(self, key):
        return self._total[key] / self._counts[key]

    def result(self):
        return {key: self.avg(key) for key in self._keys}
