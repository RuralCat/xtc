
import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
import pickle
import seaborn as sb
import matplotlib.pyplot as plt


def train_test_plot(train_data, test_data=None, kde=True, cumulative=True, title=None):
    axe = plt.axes() 
    if kde:
        sb.kdeplot(train_data, cumulative=cumulative, legend=True, ax=axe, label='train')
        if test_data is not None:
            sb.kdeplot(test_data, cumulative=cumulative, legend=True, ax=axe, label='test')
    else:
        sb.distplot(train_data, hist=False, kde=True, label='train')
        if test_data is not None:
            sb.distplot(test_data, hist=False, kde=True, label='test')
    axe.legend()
    if title and isinstance(title, str):
        plt.title(title)
    plt.show()


def data_statistic(data, window_size):
    pass


def load_raw_data(save_path='raw_data.pic'):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = np.nan_to_num(np.array(pd.read_csv('data-training.csv'), dtype=np.float32))
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    return data


def data_sliding(data, y=None, time_step=60, data_format='time_first'):
    nb_features = data.shape[-1]
    nb_sample = data.shape[0] - time_step + 1
    strides = np.array([data.strides[0], data.strides[0], data.strides[1]])

    #
    output_shape = (nb_sample, time_step, nb_features)
    _data = np.lib.stride_tricks.as_strided(data, shape=output_shape, strides=strides)
    if data_format == 'time_last':
        _data = np.transpose(_data, [0, 2, 1])

    if y is not None:
        return _data, y[time_step - 1:]
    else:
        return _data


def split_data(datas,split=[0.6, 0.1, 0.3], sampling=5, shuffle=False):
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
    print("train:{}, val:{}, test:{}".format(at_train, at_val-at_train, nb_sample-at_val))

    return trains, vals, tests


class XTXDataset:
    def __init__(self, filename, norm=True):
        # read data
        if isinstance(filename, np.ndarray):
            self._data = filename
        else:
            self._data = self._read_xtx_data(filename)
            # normlization
            if norm:
                self._data = self._normlization()

    def _read_xtx_data(self, filename):
        return np.nan_to_num(np.array(pd.read_csv(filename), dtype=np.float32))

    def _normlization(self):
        # normed_data = np.zeros_like(self._data)
        normed_data = np.zeros(self._data.shape, dtype=self._data.dtype)

        # ask & bid rate normlization
        normed_data[:, :15] = self._data[:, :15] - np.expand_dims(self._data[:, 0], axis=1)
        normed_data[:, 30:45] = self._data[:, 30:45] - np.expand_dims(self._data[:, 0], axis=1)

        # ask & bid size normlization
        asksize_book = {x: 0 for x in np.arange(1500, 1800, 0.5)}
        bidsize_book = {x: 0 for x in np.arange(1500, 1800, 0.5)}

        temp_asksize_book = {}
        temp_bidsize_book = {}
        for i in range(self._data.shape[0]):
            for j in range(15):
                # for ask
                if self._data[i, j] not in temp_asksize_book and j < 12:
                    normed_data[i, j + 15] = self._data[i, j + 15]
                else:
                    normed_data[i, j + 15] = self._data[i, j + 15] - asksize_book[self._data[i, j]]
                # update ask size book
                asksize_book[self._data[i, j]] = self._data[i, j + 15]

            for j in range(30, 45):
                # for bid
                if self._data[i, j] not in temp_bidsize_book and j < 42:
                    normed_data[i, j + 15] = self._data[i, j + 15]
                else:
                    normed_data[i, j + 15] = self._data[i, j + 15] - bidsize_book[self._data[i, j]]
                # update bid size book
                bidsize_book[self._data[i, j]] = self._data[i, j + 15]

            # update temp book
            temp_asksize_book = {self._data[i, k]: self._data[i, k + 15] for k in range(15)}
            temp_bidsize_book = {self._data[i, k + 30]: self._data[i, k + 45] for k in range(15)}

        # y normlization
        normed_data[:, 60] = self._data[:, 60]

        #
        normed_data[normed_data < -1000] = 0

        return normed_data

    def _tile(self, data, n):
        x = np.zeros((data.shape[0]-n+1, data.shape[1]-1, n), dtype=np.float32)
        for i in range(n):
            if i > 0:
                x[..., i] = data[n-i-1:-i, :-1]
            else:
                x[..., 0] = data[n-1:, :-1]
        x = np.transpose(x, [0, 2, 1])
        y = data[n-1:, -1]
        return x, y

    @staticmethod
    def label_quantification(y, reversed=False):
        y = np.copy(y)
        if reversed:
            return y * 0.25 - 1.25
        else:
            y[y > 1.25] = 1.25
            y[y < -1.25] = -1.25
            return to_categorical((y + 1.25) / 0.25)

    def train_data(self, split=0.1, use_tile=False, tile_n=0, sampling=1, label_quant=True):
        #
        data, y = self._tile(self._data, tile_n) if use_tile else (self._data[:, :-1], self._data[:,-1])
        if label_quant:
            y = XTXDataset.label_quantification(y)
        nb_train = np.int(data.shape[0] * (1 - split))
        # train data
        train_x = data[:nb_train:sampling, :60]
        train_y = y[:nb_train:sampling]
        # test data
        test_x = data[nb_train::sampling, :60]
        test_y = y[nb_train::sampling]
        return train_x, train_y, test_x, test_y


    def show(self):
        pass


def dataset_whiten_norm():
    from normlization import whiten_norm
    raw_data = load_raw_data()
    x = whiten_norm(raw_data[:, :60])
    y = raw_data[:, 60:]
    x, y = data_sliding(np.concatenate([x, y], axis=1))
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(x, y)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def dataset_slide_whiten_norm():
    raw_data = load_raw_data()
    x = raw_data[:, :60]
    mean = np.reshape(x, (-1, 1000))


def dataset_slide_norm(time_step=60, norm_window_size=60, label_smoothing=False):
    from normlization import slide_norm
    raw_data = load_raw_data()
    normed_data = slide_norm(raw_data[:, :60], window_size=norm_window_size)
    if label_smoothing:
        y = ema(raw_data[:, 60], 5)
    else:
        y = raw_data[:, 60]
    x, y = data_sliding(normed_data, y, time_step=time_step)
    return split_data(x, y, split=[0.6, 0.1, 0.3], sampling=5)


def ema(x, n):
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, x.shape[0]):
            y[i] = (n - 1) / (n + 1) * y[i - 1] + 2 / (n + 1) * x[i]
        return y

def ma(x, n):
    y = np.lib.stride_tricks.as_strided(x, (x.shape[0]-n+1, n), (x.strides[0], x.strides[0]))
    return np.mean(y, axis=1)

def ema_rate_cum_y():
    # load data
    raw_data = load_raw_data()
    askrate0 = raw_data[:, 0]
    y = raw_data[:, 60]
    # processing
    ema_askrate0 = ema(askrate0, 66)
    cum_y = np.cumsum(y)
    normed_askrate0 = (ema_askrate0 - np.mean(ema_askrate0)) * np.std(cum_y) / np.std(ema_askrate0) + np.mean(cum_y)
    # show
    plt.plot(normed_askrate0[110:], label='askrate')
    plt.plot(cum_y, label='y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import pickle
    with open('normed_data.pickle', 'rb') as f:
        normed_data = pickle.load(f)
    dataset = XTXDataset(normed_data)

    _, _, _, _ = dataset.train_data(split=0.3, use_tile=True, tile_n=60, sampling=20, label_quant=False)