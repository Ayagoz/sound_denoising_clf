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
    model.eval()

    preds = [to_np(
        model(to_var(d, is_on_cuda(model))[None])
    ).argmax(1) for d in data]
    return accuracy_score(targets, preds)


def train(model, optimizer, criterion, batch_iter, n_epochs,
          train_dataset, val_data, val_labels, path, batch_size=200, length=500):
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
    # load_best_model
    model = load_model_state(model, path)
    score = evaluate(model, tqdm(data), labels)
    dump_json(score, Path(result_path) / 'test_accuracy.json')
    return score


def val_loss(model, val_data, val_labels, criterion):
    losses = [to_np(criterion(model(to_var(val_data[i])[None]),
                              to_var(val_labels[i])[None]))
              for i in range(len(val_data))]
    return np.mean(losses)


def train_autoencoder(model, optimizer, criterion, batch_iter, n_epochs,
                      train_dataset, path, batch_size=200, length=500):
    # initial setup
    path = Path(path)
    logger = NamedTBLogger(path / 'logs', ['loss'])
    model.eval()

    # best_score = None

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

        # validation
        # model.eval()

        # metrics
    #         score = val_loss(model, val_data, val_labels, criterion)
    #         dump_json(score, path / 'val_loss.json')
    #         print(f'Val score {score}')
    #         logger.metrics({'loss': score}, step)

    #         # best model
    #         if best_score is None or best_score < score:
    #             best_score = score
    #             save_model_state(model, path / 'best_model.pth')

    save_model_state(model, path / 'model.pth')


def denoise_on_test(model, data, shapes, path, result_path, names=None, length=200):
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
