import os
import numpy as np
from .utils import load_ids, load_npy


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
            name = i[:]
            _, s = name.split('_')[:2]

            if _ == '0':
                subj_path = os.path.join(os.path.join(self.path, 'noisy'), s)

                return np.load(os.path.join(subj_path, name[2:])).astype(np.float32).T[None]
            elif _ == '1':
                subj_path = os.path.join(os.path.join(self.path, 'noisy'), s)

                return np.load(os.path.join(subj_path, name[2:])).astype(np.float32).T[None]
            else:
                raise ValueError('Not correct id type target{0/1}_id{x}_[unique id].npy')

    def load_label(self, i):
        if hasattr(self, 'target'):
            return self.target[i]
        else:
            return int(i.split('_')[0])
