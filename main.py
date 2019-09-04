
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

import keras.backend as K
from dataset import XTXDataset
from xtx_model import simple_model





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
