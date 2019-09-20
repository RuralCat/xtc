import os

import pickle
import numpy as np
import pandas as pd
# from keras.utils import Sequence
from tensorflow.keras.utils import Sequence

from core.dataset_utils import data_sliding
from core.dataset_utils import ema
from core.dataset_utils import split_data
from normlization import slide_norm


def load_raw_data(save_path='raw_data.pic'):
    # read data
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = np.nan_to_num(np.array(pd.read_csv('data-training.csv'), dtype=np.float32))
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    # parse data
    # TODO

    return data


days_index = [
    0, 46013, 94711, 144351, 256399, 359111, 464191, 573884,
    644312, 703509, 759651, 899324, 953594, 1055570, 1105677,
    1153404, 1209507, 1342361, 1390259, 1437723, 1486955, 1545235,
    1603361, 1658726, 1718263, 1777602, 1877675, 1912710, 2022189,
    2081213, 2198907, 2271018, 2341003, 2467227, 2515866, 2573037,
    2677551, 2732082, 2788996, 2856131, 2971348, 3000000
]

days_index_nonzero = [
    11, 46018, 94728, 144366, 256401, 359115, 464198,
    573890, 644316, 703515, 759658, 899331, 953601, 1055586,
    1105689, 1153414, 1209523, 1342373, 1390261, 1437724, 1486959,
    1545254, 1603374, 1658744, 1718275, 1777610, 1877708, 1912718,
    2022200, 2081225, 2198924, 2271031, 2341017, 2467236, 2515868,
    2573070, 2677555, 2732084, 2789011, 2856151, 2971364
]
# index_nonzero should + 1


class XTXDataset:
    def __init__(self, filename='datas/raw_data.pic'):
        # read data
        self._filename = filename
        self.reset_raw_data()

    def reset_raw_data(self):
        self.raw_data = load_raw_data(self._filename)
        print("Time series length: {}".format(self.raw_data.shape[0]))

    def delete_invalid_rows(self):
        index = np.transpose(np.nonzero(self.raw_data[:, 14] == 0))[:, 0]
        self.raw_data = np.delete(self.raw_data, list(index), axis=0)

    def split_rate_size(self, data=None):
        # return ask_rate, ask_size, bid_rate, bid_size
        if data is None:
            return self.raw_data[:, :15], self.raw_data[:, 15:30], self.raw_data[:, 30:45], self.raw_data[:, 45:60]
        else:
            assert data.ndims == 2 and data.shape[1] == 60
            return data[:, :15], data[:, 15:30], data[:, 30:45], data[:, 45:60]

    def split_to_days(self):
        _diff = np.diff(np.mean(np.diff(self.raw_data[:, :15], axis=1), axis=1))
        days_index = np.transpose(np.nonzero(_diff < -1))[:, 0]
        days_data = [self.raw_data[:days_index[0]]]
        for i in range(len(days_index) - 1):
            days_data += [self.raw_data[days_index[i]:days_index[i + 1]]]
        days_data += [self.raw_data[days_index[-1]:]]

        return days_data

    def days_start_index(self, delete_invalid=True):
        if delete_invalid:
            invalid_num = np.asarray(days_index_nonzero) - np.asarray(days_index[:-1])
            invalid_num[0] += 1
            return np.asarray(days_index_nonzero) + 1 - np.cumsum(invalid_num)
        else:
            start_index = np.asarray(days_index)
            start_index[1:] += 1
            return start_index

    @property
    def features(self):
        return self.raw_data[:, :60]

    @property
    def avbv_features(self):
        # [ask_rate_i, ask_size_i, bid_rate_i, bid_size_i] * 15
        return np.concatenate([self.raw_data[:, i:60:15] for i in range(15)], axis=1)

    @property
    def rate(self):
        return np.concatenate([self.raw_data[:, :15], self.raw_data[:, 30:45]], axis=1)

    @property
    def mid_rate(self):
        return (self.features[:, 0] + self.features[:, 30]) / 2

    @property
    def size(self):
        return np.concatenate([self.raw_data[:, 15:30], self.raw_data[:, 45:60]], axis=1)

    @property
    def label_y(self):
        return self.raw_data[:, 60]

    def redefine_label_y(self, n=87):
        return self.mid_rate[n:] - self.mid_rate[:-n]

    def label_y_sign(self):
        return np.sign(self.label_y)

    def label_y_value(self):
        return np.abs(self.label_y)

    def data_generator(self, data_func, valid_data_index, batch_size=256, shuffle=True, **kwargs):
        #
        batch_id = 0
        batch_inds = valid_data_index
        #
        while True:
            print('batch_id:', batch_id)
            # shuffle
            if batch_id == 0 or batch_id >= len(batch_inds):
                batch_id == 0
                if shuffle: batch_inds = np.random.permutation(batch_inds)
            #
            inputs = []
            label_y = []
            for i in range(batch_size):
                _data, _y = data_func(self.raw_data, batch_inds[batch_id + i], **kwargs)
                inputs.append(_data)
                label_y.append(_y)
            yield (np.array(inputs), np.array(label_y))
            batch_id = batch_id + batch_size


class DataGenerator(Sequence):
    def __init__(self, data, indexs, batch_size=256, shuffle=True):
        self._data = data
        self._indexs = indexs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self._indexs) // self.batch_size

    def __getitem__(self, index):
        inputs = []
        label_y = []
        for i in range(self.batch_size):
            _data, _y = slide_norm_data_func(self._data, self._indexs[index * self.batch_size + i])
            inputs.append(_data)
            label_y.append(_y)
        return np.array(inputs), np.array(label_y)

    def on_epoch_end(self):
        if self.shuffle:
            self._indexs = np.random.permutation(self._indexs)


def slide_norm_data_func(data, index, time_steps=15, norm_window_size=10):
    # data should be raw data
    if data.shape[1] != 61:
        raise
    if index < time_steps + norm_window_size - 2:
        raise IndexError("index out of bound")
    _data = slide_norm(data[index - time_steps - norm_window_size + 2: index + 1, :60],
                       window_size=norm_window_size,
                       padding=False)
    _y = data[index, 60]

    return _data, _y


def dataset_whiten_norm():
    from normlization import whiten_norm
    raw_data = load_raw_data()
    x = whiten_norm(raw_data[:, :60])
    y = raw_data[:, 60:]
    x, y = data_sliding(np.concatenate([x, y], axis=1))
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(x, y)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def dataset_slide_norm(time_step=60, norm_window_size=60, label_smoothing=False):
    from normlization import slide_norm
    raw_data = load_raw_data()
    normed_data = slide_norm(raw_data[:, :60], window_size=norm_window_size)
    if label_smoothing:
        y = ema(raw_data[:, 60], 5)
    else:
        y = raw_data[:, 60]
    x, y = data_sliding(normed_data, y, time_step=time_step)
    return split_data([x, y], split=[0.6, 0.1, 0.3], sampling=1)


if __name__ == '__main__':
    import pickle
    from main import simple_model, tf_score
    from keras.optimizers import Adam

    # create dataset
    dt = XTXDataset()
    dt.delete_invalid_rows()
    # create model
    model = simple_model(time_step=15, regression=True)
    model.compile(Adam(2e-4), loss=tf_score, metrics=['accuracy'])
    # setting
    time_step = 15
    norm_wsz = 10
    batch_size = 256
    total_sample = dt.raw_data.shape[0]
    train_index = np.arange(time_step + norm_wsz - 2, np.int(total_sample * 0.7), 5)
    val_index = np.arange(train_index[-1], np.int(total_sample * 0.8), 5)
    train_gen = DataGenerator(dt.raw_data, train_index)
    val_gen = DataGenerator(dt.raw_data, val_index)

    train_steps = len(train_index) // batch_size
    val_steps = len(val_index) // batch_size
    # train
    model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=2,
                        validation_data=val_gen, validation_steps=val_steps, max_queue_size=16,
                        workers=16, use_multiprocessing=True)
