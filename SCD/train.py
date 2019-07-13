from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
from torch import nn
import numpy as np

from .batch_iter import postprocessing

from dpipe.torch import to_np, to_var, is_on_cuda, sequence_to_var, sequence_to_np, save_model_state, load_model_state
from dpipe.train.logging import NamedTBLogger
from dpipe.medim.io import dump_json


def evaluate(model, data, targets):
    '''
    :param model: PyTorch Neural Network model
    :param data:  Loaded numpy data to evaluate model
    :param targets:  Numpy categorical target
    :return: Accuracy score of the model
    '''
    model.eval()

    preds = [to_np(
        model(to_var(d, is_on_cuda(model))[None])
    ).argmax(1) for d in data]
    return accuracy_score(targets, preds)


def train(model, optimizer, criterion, batch_iter, n_epochs,
          train_dataset, val_data, val_labels, path, batch_size=200, length=500):
    '''
    :param model: PyTorch Neural Network model
    :param optimizer: Torch optimization strategy: SGD, Adam, AdaDelta, ...
    :param criterion: Loss function
    :param batch_iter: Batch iterator function
    :param n_epochs:  Number of epochs to train model
    :param train_dataset:  Dataset object to train model
    :param val_data: Loaded numpy data to evaluate model
    :param val_labels: Loaded numpy target to evaluate model
    :param path: Experiment output path
    :param batch_size: Size of batches
    :param length: Length of crop to load train data
    :return: None
    '''
    # initial setup
    path = Path(path)
    logger = NamedTBLogger(path / 'logs', ['loss'])
    model.eval()

    best_score = None

    for step in tqdm(range(n_epochs)):

        model.train()
        losses = []
        for inputs in batch_iter(train_dataset, batch_size, length):
            *inputs, target = sequence_to_var(*tuple(inputs), cuda=is_on_cuda(model))

            logits = model(*inputs)

            if isinstance(criterion, nn.BCELoss):
                target = target.float()

            total = criterion(logits, target)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            losses.append(sequence_to_np(total))

        logger.train(losses, step)

        # validation
        model.eval()

        # metrics
        score = evaluate(model, val_data, val_labels)
        dump_json(score, path / 'val_accuracy.json')
        print(f'Val score {score}')
        logger.metrics({'accuracy': score}, step)

        # best model
        if best_score is None or best_score < score:
            best_score = score
            save_model_state(model, path / 'best_model.pth')

    save_model_state(model, path / 'model.pth')


def evaluate_on_test(model, data, labels, path, result_path):
    '''
    :param model: PyTorch Neural Network model
    :param data: Loaded numpy data
    :param labels: Loaded categorical target
    :param path: Path to model weights
    :param result_path: Experiment path to save accuracy on test
    :return: Accuracy score on given data
    '''
    # load_best_model
    model = load_model_state(model, path)
    score = evaluate(model, tqdm(data), labels)
    dump_json(score, Path(result_path) / 'test_accuracy.json')
    return score


def val_loss(model, val_data, val_labels, criterion):
    '''
    :param model: PyTorch Neural Network model
    :param val_data: Loaded numpy data
    :param val_labels: Loaded numpy target
    :param criterion: Loss function
    :return: Mean value of Loss function
    '''
    losses = [to_np(criterion(model(to_var(val_data[i])[None]),
                              to_var(val_labels[i])[None]))
              for i in range(len(val_data))]
    return np.mean(losses)


def train_autoencoder(model, optimizer, criterion, batch_iter, n_epochs,
                      train_dataset, path, batch_size=200, length=500):
    '''
    :param model: PyTorch Neural Network model
    :param optimizer: Torch optimization strategy: SGD, Adam, AdaDelta, ...
    :param criterion: Loss function
    :param batch_iter: Batch iterator function
    :param n_epochs:  Number of epochs to train model
    :param train_dataset:  Dataset object to train model
    :param path: Experiment output path
    :param batch_size: Size of batches
    :param length: Length of crop to load train data
    :return: None
    '''
    # initial setup
    path = Path(path)
    logger = NamedTBLogger(path / 'logs', ['loss'])
    model.eval()

    for step in tqdm(range(n_epochs)):

        model.train()
        losses = []
        for inputs in batch_iter(train_dataset, batch_size, length):

            *inputs, target, = sequence_to_var(*tuple(inputs[:2]), cuda=is_on_cuda(model))

            logits = model(*inputs)

            if isinstance(criterion, nn.BCELoss):
                target = target.float()

            total = criterion(logits, target)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            losses.append(sequence_to_np(total))

        logger.train(losses, step)
        print(f'Loss: {losses}')

    save_model_state(model, path / 'model.pth')


def denoise_on_test(model, data, shapes, length, path, result_path, names=None):
    '''
    :param model: PyTorch Neural Network model
    :param data: Loaded numpy data
    :param shapes: Shapes to convert back cyclic data
    :param length: The length to convert back cyclic data
    :param path: Model's weights path
    :param result_path: Experiment path to save denoised input
    :param names: If None each entry save with name 'i.npy', in other way name for each entry can be provided
    :return: None
    '''
    # load_best_model
    model = load_model_state(model, path)

    result = [to_np(model(to_var(d, is_on_cuda(model))[None])) for d in data]
    output = postprocessing(result, shapes, length)

    result_path = Path(result_path)

    if names is not None:
        for i, n in enumerate(names):
            s = (result_path / n)
            s.mkdir(exist_ok=True)
            np.save(s, np.squeeze(output[i]).T)
    else:
        for i, o in enumerate(output):
            name = str(i) + '.npy'
            np.save(result_path / name, np.squeeze(o).T)
