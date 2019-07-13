import functools
from dpipe.dataset import Dataset
from dpipe.dataset.wrappers import apply


def normalized(dataset: Dataset, mean: bool = True, std: bool = True, axis: int = 1) -> Dataset:
    '''
    :param dataset: Dataset object
    :param mean: If True, center the data before scaling
    :param std: If True, scale the data before scaling
    :param axis: The which axis along normalize
    :return: wrapped Dataset object
    '''
    return apply(dataset, load_sound=functools.partial(freq_normalize,
                                                       mean=mean, std=std, axis=axis))


def freq_normalize(sound, mean=True, std=True, axis=1, eps=1e-8):
    if mean and std:
        mean = sound.mean(axis, keepdims=True)
        std = sound.std(axis, keepdims=True)
        return (sound - mean) / (std + eps)

    if mean and not std:
        mean = sound.mean(axis, keepdims=True)
        return sound - mean

    if std and not mean:
        std = sound.std(axis, keepdims=True)
        return sound / (std + eps)
