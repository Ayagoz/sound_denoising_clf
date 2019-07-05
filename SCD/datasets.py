import os
import numpy as np
from .utils import load_ids, load_npy, load_pairs, load_pair_ids


class Dataset:
    def __init__(self, path, split='train', download=True):
        self.path = os.path.join(path, split)
        if download:
            self.data, self.target = load_npy(self.path)
            self.ids = list(range(len(self.data)))
        else:
            self.ids = load_ids(self.path)

    def load_sound(self, i):
        if hasattr(self, 'data'):
            return self.data[i]
        else:
            return np.load(i).astype(np.float32).T[None]

    def load_label(self, i):
        if hasattr(self, 'target'):
            return self.target[i]
        else:
            if 'clean' in i:
                return 1
            if 'noisy' in i:
                return 0


class PairDataset:
    def __init__(self, path, split='train', download=True):
        self.path = os.path.join(path, split)
        if download:
            self.data, self.target = load_pairs(self.path)
            self.ids = list(range(len(self.data)))

        else:
            self.ids = load_pair_ids(self.path)

    def get_len(self, i):
        return self.load_sound(i).shape[-1]

    def load_sound(self, i):
        if hasattr(self, 'data'):
            return self.data[i]
        else:
            return np.load(i).astype(np.float32).T[None]

    def load_target(self, i):
        if hasattr(self, 'target'):
            return self.target[i]
        else:
            return np.load(i.replace('noisy', 'clean')).astype(np.float32).T[None]
