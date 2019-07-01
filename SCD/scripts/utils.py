import numpy as np
import os


def load_npy(path):
    data, target = [], []
    path_clean = os.path.join(path, 'clean')
    for s in sorted(os.listdir(path_clean)):
        subject_path = os.path.join(path_clean, s)
        for n in sorted(os.listdir((subject_path))):
            file_path = os.path.join(subject_path, n)
            data.append(np.load(file_path).astype(np.float32).T[None])
            target.append(1)

    path_noisy = os.path.join(path, 'noisy')
    for s in sorted(os.listdir(path_noisy)):
        subject_path = os.path.join(path_noisy, s)
        for n in sorted(os.listdir((subject_path))):
            file_path = os.path.join(subject_path, n)
            data.append(np.load(file_path).astype(np.float32).T[None])
            target.append(0)
    return data, target


def load_ids(path):
    ids = []
    path_clean = os.path.join(path, 'clean')
    for s in sorted(os.listdir(path_clean)):
        subject_path = os.path.join(path_clean, s)
        for n in sorted(os.listdir(subject_path)):
            ids.append('1_' + n)

    path_noisy = os.path.join(path, 'noisy')
    for s in sorted(os.listdir(path_noisy)):
        subject_path = os.path.join(path_noisy, s)
        for n in sorted(os.listdir(subject_path)):
            ids.append('0_' + n)
    return ids

