from torch import nn
from torch.optim import Adam
from dpipe.layers import Reshape


def conv2block(channels, kernel_size, padding, stride, pooling):
    block = nn.Sequential(
        nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(channels[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels[1], channels[1], kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(channels[1]),
        nn.ReLU(inplace=True),
        pooling
    )
    return block


def fc2block(shape, p):
    block = nn.Sequential(
        nn.Linear(shape[0], shape[1]),
        nn.PReLU(),
        nn.BatchNorm1d(shape[1]),
        nn.Dropout(p),
    )
    return block


def fc(nums_block, shapes, probas, output_layers):
    clf = nn.Sequential(
        nn.Dropout(0.5),
        *[fc2block(shapes[i * 2:(i + 1) * 2], probas[i]) for i in range(nums_block)],
        output_layers
    )
    return clf


def convs(num_block, channels, kernel_sizes, paddings, strides, poolings, fc):
    model = nn.Sequential(
        *[conv2block(channels[i * 2:(i + 1) * 2], kernel_sizes[i], paddings[i], strides[i], poolings[i])
          for i in range(num_block)],
        Reshape('0', -1),
        fc
    )
    return model


def build_model(fc_blocks, shapes, probas, conv_blocks, channels, kernel_sizes, paddings, strides, poolings):
    output_layer = nn.Sequential(
        nn.Linear(128, 2),
        nn.LogSoftmax()
    )

    fc_part = fc(fc_blocks, shapes, probas, output_layer)

    model = convs(conv_blocks, channels, kernel_sizes, paddings, strides, poolings, fc_part)

    return model


model = build_model(fc_blocks=1, shapes=[8 * 8 * 128, 128], probas=[0.1],
                   conv_blocks=2, channels=[1, 64, 64, 128], kernel_sizes=[3, 3],
                   paddings=[1, 1], strides=[1, 1], poolings=[nn.AvgPool2d(3), nn.MaxPool2d(3)]).cuda()

lr = 1e-4
wd = 0
optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.NLLLoss()
