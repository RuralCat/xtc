
import keras.layers as KL
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


def tf_score(y, y_pred):
    return K.sum(K.pow(y - y_pred, 2)) / K.sum(K.pow(y, 2))

def data_norm(data, ts, ws):
    def sliding(x, step):
        strides = np.array([x.strides[0], x.strides[0], x.strides[1]])
        output_shape = (x.shape[0] - step + 1, step, x.shape[-1])
        return np.lib.stride_tricks.as_strided(x, shape=output_shape, strides=strides)
    # norm
    _data = sliding(data, ws)
    _data = (_data[:, -1] - np.mean(_data, axis=1)) / (np.std(_data, axis=1) + 1e-4)
    return sliding(_data, ts)

class MLPModel:
    def __init__(self, time_step, window_size, weights=None):
        self._time_step = time_step
        self._window_size = window_size
        # create model
        self.model = self._create_model(time_step)
        # compile model
        self._compile(weights)
        # define data
        self._raw_data = None


    def _create_model(self, time_step):
        # inputs: [time_step, features]
        inputs = KL.Input(shape=(time_step, 60))

        def dense_unit(x, units, dropout=0.45):
            kernel_initializer = 'glorot_uniform'
            x = KL.Dense(units,
                        activation='relu',
                        use_bias=False,
                        kernel_initializer=kernel_initializer)(x)
            if dropout > 0:
                x = KL.Dropout(dropout)(x)

            return x

        x = inputs
        x = KL.Flatten()(x)

        scale = 1
        nb_filters = [256, 256, 1024]
        for nb_filter in nb_filters:
            x = dense_unit(x, nb_filter * scale)
        x = dense_unit(x, 2048, dropout=0.00)

        x = KL.Dense(1, activation='linear')(x)

        return Model(inputs=inputs, outputs=x)

    def _compile(self, weights):
        # compile
        self.model.compile(Adam(2e-4), tf_score)

        if weights is not None:
            self.model.load_weights(weights)

    def predict(self, x):
        if np.isnan(x).any():
            x = np.nan_to_num(x)
        x = np.expand_dims(x, axis=0)
        if self._raw_data is None:
            self._raw_data = x
            return 0.0025 * (x[0, 45] - x[0, 15])
        elif self._raw_data.shape[0] < self._time_step + self._window_size - 1:
            self._raw_data = np.concatenate((self._raw_data, x), axis=0)
            return 0.0025 * (x[0, 45] - x[0, 15])
        else:
            self._raw_data = np.concatenate((self._raw_data[1:], x), axis=0)
            norm_data = data_norm(self._raw_data, self._time_step, self._window_size)
            prediction= self.model.predict(norm_data, batch_size=1)
            return prediction[0, 0]


