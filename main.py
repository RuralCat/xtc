
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import keras.backend as K
from keras.losses import categorical_crossentropy
from dataset import XTXDataset
from xtx_model import simple_model
from keras.layers import Conv2D


def score(y, y_pred):
    y_pred = np.squeeze(y_pred)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
        y = XTXDataset.label_quantification(y, reversed=True)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = XTXDataset.label_quantification(y_pred, reversed=True)
    return 1 - np.sum(np.power(y-y_pred, 2)) / np.sum(np.power(y, 2))


def tf_score(y, y_pred):
    return K.sum(K.pow(y - y_pred, 2)) / K.sum(K.pow(y, 2))


def model_score(model, train_x, train_y, test_x, test_y):
    train_y_pred = model.predict(train_x, batch_size=128, verbose=1)
    train_score = score(train_y, train_y_pred)
    print('train score: {:.5f}'.format(train_score))

    test_y_pred = model.predict(test_x, batch_size=128, verbose=1)
    test_score = score(test_y, test_y_pred)
    print('test score: {:.5f}'.format(test_score))

    return train_score, test_score


if __name__ == '__main__':
    # read data
    dataset = XTXDataset('data-training.csv')
    train_x, train_y, test_x, test_y = \
        dataset.train_data(
            split=0.3,
            use_tile=True,
            tile_n=12,
            label_quant=False,
        )

    # create model
    model = simple_model()
    model.compile(Adam(3e-4), loss=tf_score, metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=64, epochs=1)
