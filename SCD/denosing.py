from torch import nn
from dpipe import layers
from SCD.classifier import convs


def build_encoder(nums_conv, channels, n_features, n_narrow, kernel_sizes, padding, stride, poolings):
    encoder = nn.Sequential(
        convs(nums_conv, channels, kernel_sizes, [padding] * nums_conv,
              [stride] * nums_conv, poolings, nn.Sequential(),
              conv_module=nn.Conv2d),
        layers.Reshape('0', -1),
        nn.Linear(n_features, n_narrow)
    )

    return encoder


def build_decoder(nums_conv, channels, n_features, n_narrow, kernel_sizes, padding, stride, poolings):
    decoder = nn.Sequential(
        nn.Linear(n_narrow, n_features, bias=False),
        layers.Reshape('0', 64, 2, 2),
        convs(nums_conv, channels, kernel_sizes, [padding] * nums_conv,
              [stride] * nums_conv, poolings, nn.Sequential(), conv_module=nn.ConvTranspose2d),

    )
    return decoder


def build_autoencoder(nums_conv, channels, n_features, n_narrow, kernel_sizes, padding, stride, poolings):
    encoder = build_encoder(nums_conv['encoder'], channels['encoder'], n_features, n_narrow, kernel_sizes['encoder'],
                            padding, stride, poolings['encoder'])

    decoder = build_decoder(nums_conv['decoder'], channels['decoder'], n_features, n_narrow, kernel_sizes['decoder'],
                            padding, stride, poolings['decoder'])

    autoencoder = nn.Sequential(encoder, decoder)

    return autoencoder


channels = {'encoder': [1, 8, 8, 16, 16, 32, 32, 64],
            'decoder': [64, 32, 32, 16, 16, 8, 8, 1]}

kernel_sizes = {'encoder': [3, 3, 3, 3],
                'decoder': [3, 3, 3, 3]}

n_features = 64 * 2 * 2
n_narrow = 512

poolings = {'encoder': [nn.Sequential(), nn.AvgPool2d(2), nn.AvgPool2d(3), nn.MaxPool2d(3)],
            'decoder': [nn.Sequential(), nn.UpsamplingBilinear2d(scale_factor=1.5),
                        nn.UpsamplingBilinear2d(scale_factor=1.9),
                        nn.UpsamplingBilinear2d(scale_factor=2)]}
nums_conv = {'encoder': 4,
             'decoder': 4}

autoencoder = build_autoencoder(nums_conv=nums_conv, channels=channels, n_features=n_features, n_narrow=n_narrow,
                                kernel_sizes=kernel_sizes, padding=0, stride=1, poolings=poolings)
