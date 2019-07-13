from torch import nn
from dpipe.layers import Reshape


def conv2block(channels, kernel_size, padding, stride, pooling, conv_module):

    '''
    :param channels: List of two ints, in and out channels
    :param kernel_size: Size of kernel in convolutions
    :param padding: Padding size in convolutions
    :param stride: Stride size in convolutions
    :param pooling: Pooling module
    :param conv_module: Convolution module
    :return:
    '''
    block = nn.Sequential(
        conv_module(channels[0], channels[1], kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(channels[1]),
        nn.ReLU(inplace=True),
        conv_module(channels[1], channels[1], kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(channels[1]),
        nn.ReLU(inplace=True),
        pooling
    )
    return block


def fc2block(shape, p):
    '''
    :param shape: List of two ints, input and output shape in fully connected layer
    :param p: Dropout probability
    :return: Fully connected block: [Linear, PRelu, BatchNorm, Dropout]
    '''
    block = nn.Sequential(
        nn.Linear(shape[0], shape[1]),
        nn.PReLU(),
        nn.BatchNorm1d(shape[1]),
        nn.Dropout(p),
    )
    return block


def fc(nums_block, shapes, probas, output_layers):
    '''
    :param nums_block: Number of fully connected blocks
    :param shapes: List of input and output shapes for fully connected blocks
    :param probas: List of probabilities for DropOut
    :param output_layers: Specified Output Layer: SoftMax, Sigmoid, LogSoftMax, ...
    :return:
    '''
    clf = nn.Sequential(
        Reshape('0', -1),
        nn.Dropout(0.5),
        *[fc2block(shapes[i * 2:(i + 1) * 2], probas[i]) for i in range(nums_block)],
        output_layers,
    )
    return clf


def convs(num_block, channels, kernel_sizes, paddings, strides, poolings, fc, conv_module=nn.Conv2d):
    '''
    :param num_block: int, number of convolutional blocks
    :param channels: List of ints, the length should be divided by 2. Provides number of channels for each convolution.
    :param kernel_sizes: Size of kernel in convolutions
    :param paddings: List of padding size in convolutions
    :param strides: List of stride size in convolutions
    :param poolings: The list of poolings after conv blocks
    :param fc: Fully connected blocks
    :param conv_module: Convolution type: Conv2D, DeConv2D, ...
    '''
    model = nn.Sequential(
        *[conv2block(channels[i * 2:(i + 1) * 2], kernel_sizes[i], paddings[i], strides[i], poolings[i],
                     conv_module)
          for i in range(num_block)],

        fc
    )
    return model


def build_model(fc_blocks, shapes, probas, conv_blocks, channels, kernel_sizes, paddings, strides, poolings):
    '''
    :param fc_blocks: int, Number of fully connected blocks
    :param shapes: List of input and output shapes for fully connected blocks
    :param probas: List of probabilities for DropOut
    :param conv_blocks:   int, number of convolutional blocks
    :param channels: List of ints, the length should be divided by 2. Provides number of channels for each convolution.
    :param kernel_sizes: Size of kernel in convolutions
    :param paddings: List of padding size in convolutions
    :param strides:  List of stride size in convolutions
    :param poolings:  The list of poolings after conv blocks
    :return:
    '''
    output_layer = nn.Sequential(
        nn.Linear(128, 2),
        nn.LogSoftmax()
    )

    fc_part = fc(fc_blocks, shapes, probas, output_layer)

    model = convs(conv_blocks, channels, kernel_sizes, paddings, strides, poolings, fc_part)

    return model


clf = build_model(fc_blocks=1, shapes=[8 * 8 * 128, 128], probas=[0.1],
                    conv_blocks=2, channels=[1, 64, 64, 128], kernel_sizes=[3, 3],
                    paddings=[1, 1], strides=[1, 1], poolings=[nn.AvgPool2d(3), nn.MaxPool2d(3)]).cuda()


