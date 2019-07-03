from torch import nn
from dpipe import layers


def build_autoencoder(channels, n_features, n_narrow, kernel_size, padding, stride):
    encoder = nn.Sequential(
        layers.make_consistent_seq(
            layers.ResBlock2d, [1, *channels], kernel_size=kernel_size, padding=padding, stride=stride
        ),
        layers.Reshape('0', -1),
        nn.Linear(n_features, n_narrow)
    )

    decoder = nn.Sequential(
        nn.Linear(n_narrow, n_features, bias=False),
        layers.Reshape('0', 64, ),
        layers.make_consistent_seq(
            layers.ResBlock2d, [*channels[::-1], 4], kernel_size=kernel_size, padding=padding, stride=stride,
            conv_module=nn.ConvTranspose2d
        ),
        layers.PreActivation2d(2, 2, kernel_size=4),
        layers.PreActivation2d(2, 2, kernel_size=3),
    )

    autoencoder = nn.Sequential(encoder, decoder)

    return autoencoder


channels = [8, 16, 32, 64, 128]
n_features = 64 * 2 * 2
n_narrow = 10

autoencoder = build_autoencoder(channels=channels, n_features=n_features, n_narrow=n_narrow,
                                kernel_size=3, padding=0, stride=1)