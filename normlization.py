
import numpy as np
import pickle

def norm_by_step(data):
    normed_data = np.zeros(data.shape, dtype=data.dtype)

    # ask & bid rate normlization
    normed_data[:, :15] = data[:, :15] - np.expand_dims(data[:, 0], axis=1)
    normed_data[:, 30:45] = data[:, 30:45] - np.expand_dims(data[:, 0], axis=1)

    # ask & bid size normlization
    asksize_book = {x: 0 for x in np.arange(1500, 1800, 0.5)}
    bidsize_book = {x: 0 for x in np.arange(1500, 1800, 0.5)}

    temp_asksize_book = {}
    temp_bidsize_book = {}
    for i in range(data.shape[0]):
        for j in range(15):
            # for ask
            if data[i, j] not in temp_asksize_book and j < 12:
                normed_data[i, j + 15] = data[i, j + 15]
            else:
                normed_data[i, j + 15] = data[i, j + 15] - asksize_book[data[i, j]]
            # update ask size book
            asksize_book[data[i, j]] = data[i, j + 15]

        for j in range(30, 45):
            # for bid
            if data[i, j] not in temp_bidsize_book and j < 42:
                normed_data[i, j + 15] = data[i, j + 15]
            else:
                normed_data[i, j + 15] = data[i, j + 15] - bidsize_book[data[i, j]]
            # update bid size book
            bidsize_book[data[i, j]] = data[i, j + 15]

        # update temp book
        temp_asksize_book = {data[i, k]: data[i, k + 15] for k in range(15)}
        temp_bidsize_book = {data[i, k + 30]: data[i, k + 45] for k in range(15)}

        # y normlization
        normed_data[:, 60] = data[:, 60]

        #
        normed_data[normed_data < -1000] = 0
    return normed_data


def whiten_norm(data, use_std=True):
    mean_map = np.mean(data, axis=0, keepdims=True)
    if use_std:
        std_map = np.std(data, axis=0, keepdims=True)
        normed_data = (data - mean_map) / std_map
    else:
        normed_data = (data - mean_map) / mean_map
    return normed_data


def local_norm(data, use_std=False):
    #
    sample_sum = lambda x: np.sum(np.sum(x, axis=1, keepdims=True), axis=2, keepdims=True)

    # data (sample, time_step, features)
    # assert data.ndim == 2 and data.shape[2] == 60
    # norm = np.zeros_like(data)
    norm = np.zeros(data.shape, dtype=data.dtype)
    
    for k in [0, 1]:
        mean = (sample_sum(data[..., 15*k : 15*k+15]) + sample_sum(data[..., 15*k+30 : 15*k+45])) / (30 * 60)
        if use_std:
            _diff1 = data[..., 15*k : 15*k+15] - mean
            _diff2 = data[..., 15*k+30 : 15*k+45] - mean
            std = np.sqrt((sample_sum(_diff1**2 + _diff2**2)) / 1800)
            norm[..., 15*k : 15*k+15] =  _diff1 / std 
            norm[..., 15*k+30 : 15*k+45] = _diff2 / std
        else:
            norm[..., 15*k : 15*k+15] = (data[..., 15*k : 15*k+15] - mean) / mean
            norm[..., 15*k+30 : 15*k+45] = (data[..., 15*k+30 : 15*k+45] - mean) / mean

    return norm


def slide_norm(data, window_size=60, padding=True):
    from dataset import data_sliding
    if padding:
        data = np.pad(data, ((window_size-1, 0), (0,0)), 'edge')
    x = data_sliding(data, time_step=window_size)
    x = (x[:, -1] - np.mean(x, axis=1)) / (np.std(x, axis=1) + 1e-4)
    return x


