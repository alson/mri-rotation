import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from base import BaseDataLoader
from torchvision.datasets import DatasetFolder


class AdniDataset(DatasetFolder):
    def __init__(self, root, train=True, transforms=None, transform=None, target_transform=None):
        data_folder = os.path.join(root, ('train/' if train else 'test'))
        super().__init__(data_folder, AdniDataset.loader, extensions=('npy',), transform=transform, target_transform=target_transform)

    @staticmethod
    def loader(path):
        return np.load(path).astype(np.float32)


class AdniDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.data_dir = data_dir
        self.dataset = AdniDataset(self.data_dir, train=training, transform=trsfm, target_transform=float)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        sample_groups = {s[0].split('/')[-1] for s in self.dataset.samples}
        valid_grps, train_grps = map(set, train_test_split(sample_groups, test_size=split))

        train_idx = [idx for idx, samp in self.dataset.samples if samp[0].split('/')[-1] in train_grps]
        valid_idx = [idx for idx, samp in self.dataset.samples if samp[0].split('/')[-1] in valid_grps]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler