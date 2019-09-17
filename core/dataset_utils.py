import copy
import datetime
import os

import numpy as np
import pandas


def data_sliding(data, y=None, time_step=60, data_format='time_first', time_stride=1):
    nb_features = data.shape[-1]
    nb_sample = data.shape[0] - time_step * time_stride + 1
    strides = np.array([data.strides[0], data.strides[0] * time_stride, data.strides[1]])

    #
    output_shape = (nb_sample, time_step, nb_features)
    _data = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=strides)
    if data_format == 'time_last':
        _data = np.transpose(_data, [0, 2, 1])

    if y is not None:
        return _data, y[time_step * time_stride - 1:]
    else:
        return _data


def split_data(datas, split=[0.6, 0.1, 0.3], sampling=5, shuffle=False, show_status=False):
    # parse inputs
    if isinstance(datas, np.ndarray):
        datas = [datas]
    elif not isinstance(datas, list):
        raise TypeError('datas should be a numpy.ndarray or list instance')
    else:
        for d in datas:
            if not isinstance(d, np.ndarray):
                raise TypeError('member in datas should be ndarray')

    # sampling
    for k in range(len(datas)):
        datas[k] = datas[k][::sampling]
        if shuffle:
            # set random state
            np.random.seed(20190717)
            datas[k] = np.random.permutation(datas[k])

    # split data
    assert len(split) == 3
    split = np.cumsum(np.array(split) / sum(split))
    nb_sample = datas[0].shape[0]
    at_train = int(nb_sample * split[0])
    at_val = int(nb_sample * split[1])

    trains, vals, tests = [], [], []
    for d in datas:
        trains += [d[:at_train]]
        vals += [d[at_train:at_val]]
        tests += [d[at_val:]]

    # show
    if show_status:
        print("train:{}, val:{}, test:{}".format(at_train, at_val - at_train, nb_sample - at_val))

    return trains, vals, tests


def split_data_by_interval(datas, split=[0.6, 0.1, 0.3], nb_int=4, sampling=5):
    # get total
    nb_sample = datas[0].shape[0]
    nb_each_int = nb_sample // nb_int
    trains, vals, tests = [], [], []

    def split_and_concatenate(start, stop):
        _datas = [data[start:stop] for data in datas]
        _trains, _vals, _tests = split_data(_datas, split, sampling, show_status=False)
        if len(trains) == 0:
            trains.extend(_trains)
            vals.extend(_vals)
            tests.extend(_tests)
        else:
            for k in range(len(_trains)):
                trains[k] = np.concatenate([trains[k], _trains[k]], axis=0)
                vals[k] = np.concatenate([vals[k], _vals[k]], axis=0)
                tests[k] = np.concatenate([tests[k], _tests[k]], axis=0)

    for i in range(nb_int - 1):
        split_and_concatenate(i * nb_each_int, (i + 1) * nb_each_int)
    split_and_concatenate((nb_int - 1) * nb_each_int, nb_sample)

    # show
    print("train: {}, val: {}, test: {}".format(trains[0].shape[0], vals[0].shape[0], tests[0].shape[0]))

    return trains, vals, tests


def ema(x, n):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = (n - 1) / (n + 1) * y[i - 1] + 2 / (n + 1) * x[i]
    return y


def ma(x, n):
    y = np.lib.stride_tricks.as_strided(x, (x.shape[0] - n + 1, n), (x.strides[0], x.strides[0]))
    return np.mean(y, axis=1)


def data_func(datas, time_steps=15, norm_window_size=10, levels=15, delay=87):
    mid_price = datas.mid_rate
    y = mid_price[delay:] - mid_price[:-delay]
    x = datas.features[:-delay]

    x = np.concatenate([x[..., i * 15:i * 15 + levels] for i in range(4)], axis=-1)

    x, y = data_sliding(x, y, time_step=time_steps + norm_window_size - 1)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data([x, y], split=[0.6, 0.1, 0.3], sampling=5)

    train_valid = train_x.shape[0] // 256 * 256
    val_valid = val_x.shape[0] // 256 * 256

    return (train_x[:train_valid], train_y[:train_valid]), (val_x[:val_valid], val_y[:val_valid]), (test_x, test_y)


def volume_data(datas, time_steps, norm_window_size=1, levels=15, delay=87, split=True):
    #
    features = copy.deepcopy(datas.features[:-delay])
    label_y = datas.mid_rate[delay:] - datas.mid_rate[:-delay]

    ask_rate = np.log10(features[..., 15:15 + levels])
    bid_rate = np.log10(features[..., 45:45 + levels])
    x = np.concatenate([ask_rate, bid_rate], axis=-1)
    x, y = data_sliding(x, label_y, time_step=time_steps + norm_window_size - 1)

    if split:
        (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data([x, y], split=[0.6, 0.1, 0.3], sampling=5)

        train_valid = train_x.shape[0] // 256 * 256
        val_valid = val_x.shape[0] // 256 * 256

        return (train_x[:train_valid], train_y[:train_valid]), (val_x[:val_valid], val_y[:val_valid]), (test_x, test_y)
    else:
        return x, y


def volume_date_stand_label(datas, time_steps, norm_window_size=1, levels=15, delay=87):
    features = copy.deepcopy(datas.features[:-delay])
    label_y = datas.mid_rate[delay:] - datas.mid_rate[:-delay]

    ask_rate = np.log10(features[..., 15:15 + levels])
    bid_rate = np.log10(features[..., 45:45 + levels])
    x = np.concatenate([ask_rate, bid_rate], axis=-1)
    x, y = data_sliding(x, label_y, time_step=time_steps + norm_window_size - 1)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data([x, y], split=[0.6, 0.1, 0.3], sampling=5)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def volume_data_norm(datas, time_steps, norm_window_size=1, levels=5):
    features = copy.deepcopy(datas.features)
    label_y = copy.deepcopy(datas.label_y)

    ask_rate = features[..., 15:15 + levels]
    bid_rate = features[..., 45:45 + levels]
    x = np.concatenate([ask_rate, bid_rate], axis=-1)
    x, y = data_sliding(x, label_y, time_step=time_steps + norm_window_size - 1)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data([x, y], split=[0.6, 0.1, 0.3], sampling=5)

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


class CSVRecord():
    def __init__(self, heads, dir="datas/records", filename=None):
        if filename is None:
            filename = "records_" + datetime.datetime.now().strftime("%m%d_%H%M") + ".csv"
        self._filename = os.path.join(dir, filename)
        # write head
        s = ""
        for head in heads:
            s += head + ","
        s = s + "min_train_loss,min_val_loss,val_score,test_score" + '\n'
        with open(self._filename, 'w') as f:
            f.write(s)

    def add_record(self, params, min_train_loss, min_val_loss, val_score, test_score):
        s = ""
        for i in range(len(params)):
            s += str(params[i]) + ","
        s = s + str(min_train_loss) + "," + str(min_val_loss) + "," + str(val_score) + "," + str(test_score) + '\n'
        with open(self._filename, 'a') as f:
            f.write(s)

    def print_record(self):
        csv = pandas.read_csv(self._filename)
        print(csv)

