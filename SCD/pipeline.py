from .train import train, train_autoencoder, denoise_on_test, val_loss, evaluate_on_test
from .datasets import Dataset, PairDataset
from .batch_iter import batch_iter_crop, batch_iter_cyclic
from .utils import load_crop_data, load_cyclic_data
from .classifier import clf
from .denosing import autoencoder

from torch.optim import Adam
from torch import nn
from pathlib import Path
import os


def classification_training(path, exp_path, length=80, lr=1e-4, wd=0, n_epochs=20, batch_size=500):
    '''
    :param path: Data path, with folders train/val consist of clean/noisy parts
    :param exp_path: Path to experiment outputs
    :param length: Length of cropping data
    :param lr: Learning rate for optimization
    :param wd: Weight decay for optimization
    :param n_epochs: Number of epochs to train
    :param batch_size: Size of batches
    :return: None
    '''
    dataset = Dataset(path, 'train', False)
    val_dataset = Dataset(path, 'val', False)

    val_data, val_labels = load_crop_data(val_dataset, length)

    model = clf.cuda()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.NLLLoss()

    train(model=model, optimizer=optimizer, criterion=criterion, batch_iter=batch_iter_crop,
          n_epochs=n_epochs, train_dataset=dataset, val_data=val_data, val_labels=val_labels,
          path=exp_path, batch_size=batch_size, length=length)


def denosing_training(path, exp_path, length=80, lr=1e-4, wd=1e-6, n_epochs=40, batch_size=100):
    '''
    :param path: Data path, with folders train/val consist of clean/noisy parts
    :param exp_path: Path to experiment outputs
    :param length: Length of cyclic transform data
    :param lr: Learning rate for optimization
    :param wd: Weight decay for optimization
    :param n_epochs: Number of epochs to train
    :param batch_size: Size of batches
    :return: None
    '''
    dataset = PairDataset(path, 'train', False)
    val_dataset = PairDataset(path, 'val', False)

    val_data, val_labels, _ = load_cyclic_data(val_dataset, length)

    model = autoencoder.cuda()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    train_autoencoder(model=model, optimizer=optimizer, criterion=criterion, batch_iter=batch_iter_cyclic,
                      n_epochs=n_epochs, train_dataset=dataset, path=exp_path, batch_size=batch_size, length=length)

    score = val_loss(model, val_data, val_labels, criterion)
    print(f'Validation MSE loss: {score}')


def evaluate_on_test_all(path_data, path_model, exp_path, cuda):
    '''
    :param path_data: Data path, with folders train/val consist of clean/noisy parts
    :param path_model: Full path to model weights: should contain files `clf.pth` and `denoise.pth`
    :param exp_path: Path to experiment, where to save outputs
    :param cuda: If True will move model and data to GPU
    :return: None
    '''
    print('Evaluate classification accuracy on test')

    test_dataset = Dataset(path_data, 'test', False)
    test_data, test_labels = load_crop_data(test_dataset, 80)
    print(f'Test data shape {test_data.shape}')

    exp_path = Path(exp_path)
    exp_path.mkdir(exist_ok=True)

    clf_path = Path(exp_path) / "classification"
    clf_path.mkdir(exist_ok=True)

    if cuda:
        model = clf.cuda()
    else:
        model = clf

    score = evaluate_on_test(model, test_data, test_labels, Path(os.path.join(path_model, 'clf.pth')), clf_path)
    print(f'Test score {score}')

    print('Evaluate denoising on test')

    test_dataset = PairDataset(path_data, 'test', False)

    test_data, test_labels, test_shapes = load_cyclic_data(test_dataset, 80)

    denoise_path = Path(exp_path) / "denoise"
    denoise_path.mkdir(exist_ok=True)

    if cuda:
        model = autoencoder.cuda()
    else:
        model = autoencoder

    denoise_on_test(model, test_data, test_shapes, Path(os.path.join(path_model, 'denoise.pth')), denoise_path,
                    length=80)

    print('Result of denoising in dir: ', denoise_path)
