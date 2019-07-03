import numpy as np


def pad(x, length, mode='reflect'):
    return np.pad(x[0], (0, length - x.shape[-1]), mode=mode)[:x.shape[1]][None]


def crop(x, length):
    if x.shape[-1] > length:
        i = np.random.randint(0, x.shape[-1] - length)
        return x[:, :, i:i + length]
    elif x.shape[-1] == length:
        return x
    else:
        return pad(x, length)


def batch_iter_crop(dataset, batch_size=200, length=500):
    idx = np.copy(dataset.ids)
    np.random.shuffle(idx)

    for i in range(0, len(idx), batch_size):
        x = np.stack([crop(dataset.load_sound(idx[j]), length) for j in range(i, i + batch_size)], 0)
        y = np.stack([dataset.load_label(idx[j]) for j in range(i, i + batch_size)], 0)
        yield np.array(x).astype(np.float32), np.array(y)


def cyclic_transform(x, length):
    if x.shape[-1] > length:
        k = x.shape[-1] // length
        xx = pad(x, (k + 1) * length)
        return xx.reshape(-1, 1, x.shape[-2], length)
    else:
        return pad(x, length)[None]


def batch_iter_cyclic(dataset, batch_size=200, length=500):
    idx = np.copy(dataset.ids)
    np.random.shuffle(idx)

    i = 0
    while i < len(idx):
        x = []
        y = []
        for j in range(i, min(i + batch_size, len(idx))):
            data = cyclic_transform(dataset.load_sound(idx[j]), length)
            x.extend(data)
            y.extend(np.repeat(dataset.load_label(idx[j]), len(data)))
            if len(x) >= batch_size:
                i = j + 1
                break
            else:
                i += batch_size + 1
        yield np.array(x).astype(np.float32), np.array(y)
