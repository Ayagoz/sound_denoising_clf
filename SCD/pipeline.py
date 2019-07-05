from .train import train, train_autoencoder, denoise_on_test, val_loss
from .datasets import Dataset, PairDataset
from .batch_iter import batch_iter_crop, batch_iter_cyclic
from .utils import load_crop_data, load_cyclic_data
from .classifier import classifier
from .denosing import autoencoder

from torch.optim import Adam
from torch import nn


def classification_training(path, exp_path, length=80, lr=1e-4, wd=0, n_epochs=20, batch_size=500):
    dataset = Dataset(path, 'train', False)
    val_dataset = Dataset(path, 'val', False)

    val_data, val_labels = load_crop_data(val_dataset, length)

    model = classifier.cuda()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCELoss()

    train(model=model, optimizer=optimizer, criterion=criterion, batch_iter=batch_iter_crop,
          n_epochs=n_epochs, train_dataset=dataset, val_data=val_data, val_labels=val_labels,
          path=exp_path, batch_size=batch_size, length=length)


def denosing_training(path, exp_path, length=80, lr=1e-4, wd=1e-6, n_epochs=20, batch_size=500):
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