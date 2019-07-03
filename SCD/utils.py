import numpy as np
import os
from .batch_iter import crop, cyclic_transform


def load_crop_data(dataset, length):
    idx = np.copy(dataset.ids)
    # np.random.shuffle(idx)
    x = np.stack([crop(dataset.load_sound(i), length) for i in idx], 0)
    y = np.stack([dataset.load_label(i) for i in idx], 0)
    return x, y


def load_cyclic_data(dataset, length):
    idx = np.copy(dataset.ids)
    # np.random.shuffle(idx)
    x, y, shapes = [], [], []
    for i in idx:
        sound = dataset.load_sound(i)
        data = cyclic_transform(sound, length)
        x.extend(data)
        y.extend(np.repeat(dataset.load_label(i), len(data)))
        shapes.append(sound.shape[-1])

    return np.array(x).astype(np.float32), np.array(y), np.array(shapes)


def load_npy(path):
    data, target = [], []
    path_clean = os.path.join(path, 'clean')
    for s in sorted(os.listdir(path_clean)):
        subject_path = os.path.join(path_clean, s)
        for n in sorted(os.listdir(subject_path)):
            file_path = os.path.join(subject_path, n)
            data.append(np.load(file_path).astype(np.float32).T[None])
            target.append(1)

    path_noisy = os.path.join(path, 'noisy')
    for s in sorted(os.listdir(path_noisy)):
        subject_path = os.path.join(path_noisy, s)
        for n in sorted(os.listdir(subject_path)):
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
            ids.append(os.path.join(subject_path, n))

    path_noisy = os.path.join(path, 'noisy')
    for s in sorted(os.listdir(path_noisy)):
        subject_path = os.path.join(path_noisy, s)
        for n in sorted(os.listdir(subject_path)):
            ids.append(os.path.join(subject_path, n))
    return ids
