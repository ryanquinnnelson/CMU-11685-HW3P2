import octopus.models.CNN as cn
import torch.nn as nn


def test__calc_output_size_no_padding():
    input_size = 6
    padding = 0
    dilation = 1
    kernel_size = 3
    stride = 1

    expected = 4
    actual = cn._calc_output_size(input_size, padding, dilation, kernel_size, stride)
    assert actual == expected


def test__calc_output_size_padding():
    input_size = 6
    padding = 1
    dilation = 1
    kernel_size = 3
    stride = 1

    expected = 6
    actual = cn._calc_output_size(input_size, padding, dilation, kernel_size, stride)
    assert actual == expected


def test__calc_output_size_from_dict():
    input_size = 6
    d = {
        'padding': 1,
        'dilation': 1,
        'kernel_size': 3,
        'stride': 1}
    expected = 6
    actual = cn._calc_output_size_from_dict(input_size, d)
    assert actual == expected


def test__build_cnn2d_sequence_1():  # one conv, no batch norm, no pooling

    conv_dicts = []
    conv_dict = {
        'in_channels': 3,
        'out_channels': 4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1

    }
    conv_dicts.append(conv_dict)

    pool_dicts = []
    pool_class = None
    activation_func = 'ReLU'
    batch_norm = False

    input_size = 64
    expected_output_size = 62
    expected_out_channels = 4

    actual_sequence, actual_output_size, actual_out_channels = cn._build_cnn2d_sequence(input_size, activation_func, batch_norm, conv_dicts,
                                                                   pool_class, pool_dicts)
    assert actual_output_size == expected_output_size
    assert actual_out_channels == expected_out_channels

    # check actual sequence
    expected_name_1 = 'conv1'
    expected_str_1 = str(nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=0, dilation=1))
    print(actual_sequence)
    expected_name_2 = 'ReLU1'
    expected_str_2 = str(nn.ReLU(inplace=True))

    actual_name_1 = actual_sequence[0][0]
    actual_str_1 = str(actual_sequence[0][1])

    assert actual_name_1 == expected_name_1
    assert actual_str_1 == expected_str_1

    actual_name_2 = actual_sequence[1][0]
    actual_str_2 = str(actual_sequence[1][1])

    assert actual_name_2 == expected_name_2
    assert actual_str_2 == expected_str_2


def test__build_cnn2d_sequence_2():  # one conv, batch norm, pooling
    conv_dicts = []
    conv_dict = {
        'in_channels': 3,
        'out_channels': 4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1

    }
    conv_dicts.append(conv_dict)

    pool_dicts = []
    pool_dict = {
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
        'dilation': 1
    }
    pool_dicts.append(pool_dict)

    pool_class = 'MaxPool2d'
    activation_func = 'ReLU'
    batch_norm = True

    input_size = 64
    expected_output_size = 31
    expected_out_channels = 4

    actual_sequence, actual_output_size, actual_out_channels = cn._build_cnn2d_sequence(input_size, activation_func, batch_norm, conv_dicts,
                                                                   pool_class, pool_dicts)

    assert actual_output_size == expected_output_size
    assert  actual_out_channels == expected_out_channels
    print(actual_sequence)

    # check actual sequence
    expected_name_1 = 'conv1'
    expected_str_1 = str(nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=0, dilation=1))
    actual_name_1 = actual_sequence[0][0]
    actual_str_1 = str(actual_sequence[0][1])
    assert actual_name_1 == expected_name_1
    assert actual_str_1 == expected_str_1

    expected_name_2 = 'bn1'
    expected_str_2 = str(nn.BatchNorm2d(4))
    actual_name_2 = actual_sequence[1][0]
    actual_str_2 = str(actual_sequence[1][1])
    assert actual_name_2 == expected_name_2
    assert actual_str_2 == expected_str_2

    expected_name_3 = 'ReLU1'
    expected_str_3 = str(nn.ReLU(inplace=True))
    actual_name_3 = actual_sequence[2][0]
    actual_str_3 = str(actual_sequence[2][1])
    assert actual_name_3 == expected_name_3
    assert actual_str_3 == expected_str_3

    expected_name_4 = 'pool1'
    expected_str_4 = str(nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, padding=0))
    actual_name_4 = actual_sequence[3][0]
    actual_str_4 = str(actual_sequence[3][1])
    assert actual_name_4 == expected_name_4
    assert actual_str_4 == expected_str_4


def test_CNN__init___one_cnn_layer():
    conv_dicts = []
    conv_dict = {
        'in_channels': 3,
        'out_channels': 4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1

    }
    conv_dicts.append(conv_dict)

    pool_dicts = []
    pool_dict = {
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
        'dilation': 1
    }
    pool_dicts.append(pool_dict)

    pool_class = 'MaxPool2d'
    activation_func = 'ReLU'
    batch_norm = True

    input_size = 64
    output_size = 4000

    cnn = cn.CNN2d(input_size, output_size, activation_func, batch_norm, conv_dicts, pool_class, pool_dicts)
    print(cnn)


def test_CNN__init___two_cnn_layer():
    conv_dicts = []
    conv_dict = {
        'in_channels': 3,
        'out_channels': 7,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1

    }
    conv_dicts.append(conv_dict)
    conv_dict2 = {
        'in_channels': 7,
        'out_channels': 4,
        'kernel_size': 3,
        'stride': 1,
        'padding': 0,
        'dilation': 1

    }
    conv_dicts.append(conv_dict2)

    pool_dicts = []
    pool_dict = {
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
        'dilation': 1
    }
    pool_dicts.append(pool_dict)
    pool_dicts.append(pool_dict)

    pool_class = 'MaxPool2d'
    activation_func = 'ReLU'
    batch_norm = True

    input_size = 64
    output_size = 4000

    cnn = cn.CNN2d(input_size, output_size, activation_func, batch_norm, conv_dicts, pool_class, pool_dicts)
    print(cnn)

