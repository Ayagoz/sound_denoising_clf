import numpy as np
import os
from .batch_iter import crop, cyclic_transform


def load_crop_data(dataset, length):
    '''
    :param dataset: Dataset object
    :param length: Length of crop
    :return: Loaded and croped data: X, y
    '''
    idx = np.copy(dataset.ids)
    # np.random.shuffle(idx)
    x = np.stack([crop(dataset.load_sound(i), length) for i in idx], 0)
    y = np.stack([dataset.load_label(i) for i in idx], 0)
    return x, y


def load_cyclic_data(dataset, length):
    '''
    :param dataset: Dataset object
    :param length: Length to cut object: if x shape is (1, 1, n, m) and x will be converted to
    (m//length + 1) parts and concatenate along axis=0 (batch)
    :return: Loaded data with cut and concatenated along batch axis
    '''
    idx = np.copy(dataset.ids)
    x, t, shapes = [], [], []
    for i in idx:
        x.extend(cyclic_transform(dataset.load_sound(i), length))
        t.extend(cyclic_transform(dataset.load_target(i), length))
        shapes.append(dataset.get_len(i))

    return np.array(x), np.array(t), np.array(shapes)


def load_npy(path):
    '''
    :param path: Data path, should contain folders 'clean' and 'noisy'
    :return: Loaded data and target depending on folder, (X,y)
    '''
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


def load_pairs(path):
    '''
    :param path: Data path, should contain folders 'clean' and 'noisy'
    :return: Loaded pairs clean/noisy
    '''
    source, target = [], []
    path_clean = os.path.join(path, 'clean')
    for s in sorted(os.listdir(path_clean)):
        subject_target = os.path.join(path_clean, s)
        subject_source = subject_target.replace('clean', 'noisy')
        for n in sorted(os.listdir(subject_target)):
            source.append(np.load(os.path.join(subject_source, n)).astype(np.float32).T[None])
            target.append(np.load(os.path.join(subject_target, n)).astype(np.float32).T[None])
    return source, target


def load_pair_ids(path):
    '''
    :param path: Data path, should contain folders 'clean' and 'noisy'
    :return: Full path to each entry from noisy folder
    '''
    ids = []
    path_noisy = os.path.join(path, 'noisy')
    for s in sorted(os.listdir(path_noisy)):
        subject_source = os.path.join(path_noisy, s)
        for n in sorted(os.listdir(subject_source)):
            ids.append(os.path.join(subject_source, n))
    return ids


def load_ids(path):
    '''
    :param path: Data path, should contain folders 'clean' and 'noisy'
    :return: Full path to each entry from clean and noisy folders
    '''
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
