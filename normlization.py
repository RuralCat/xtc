
import numpy as np
import pickle

def norm_by_step(data):
    with open('./normed_data.pickle', 'rb') as f:
        return pickle.load(f)


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


def slide_norm(data, window_size=60):
    from dataset import data_sliding
    data = np.pad(data, ((window_size-1, 0), (0,0)), 'edge')
    x = data_sliding(data, time_step=window_size)
    x = (x[:, -1] - np.mean(x, axis=1)) / (np.std(x, axis=1) + 1e-4)
    return x


def norm_use_train(train_x, val_x, data_x):
    mean = np.mean(train_x, axis=0, keepdims=True)
    