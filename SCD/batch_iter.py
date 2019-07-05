import numpy as np


def pad(x, length, mode='reflect'):
    return np.pad(x[0], (0, length - x.shape[-1]), mode=mode)[:x.shape[1]][None]


def crop(x, length):
    if x.shape[-1] > length:
        i = np.random.randint(0, x.shape[-1] - length)
        return x[:, :, : length]
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
        return np.array([xx[:, :, i * length:(i + 1) * length] for i in range(k + 1)])
    else:
        return pad(x, length)[None]


def inv_cyclic_transform(x, shape):
    return np.concatenate([x[i] for i in range(len(x))], -1)[..., :shape]


def postprocessing(datas, shapes, length):
    output = []
    t = 0
    for i, s in enumerate(shapes):
        k = s // length
        t += k + 1
        output.append(inv_cyclic_transform(datas[t - k: t], s))
    assert len(output) != len(shapes), 'Something wrong'
    return output


def batch_iter_cyclic(dataset, batch_size=200, length=500):
    idx = np.copy(dataset.ids)
    np.random.shuffle(idx)
    i = 0
    while i < len(idx):
        x, t = [], []
        shapes = []
        for j in range(i, min(i + batch_size, len(idx))):
            data, target = dataset.load_sound(idx[j]), dataset.load_target(idx[j])
            x.extend(cyclic_transform(data, length))
            t.extend(cyclic_transform(target, length))
            shapes.append(dataset.get_len(idx[j]))
            if len(x) >= batch_size:
                i = j + 1
                break
            else:
                i += batch_size + 1
        yield np.array(x).astype(np.float32), np.array(t).astype(np.float32), np.array(shapes)
