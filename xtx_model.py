
import numpy as np
import keras.layers as KL
from keras.models import Model
import keras.backend as K
from keras.initializers import *
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# a classification model
def simple_model(regression=False):
    # inputs: [time_step, features]
    inputs = KL.Input(shape=(60, 60))
    x = inputs

    #
    def dense_unit(x, units, dropout=0.5):
        regularizer = KL.regularizers.l2(0.001)
        x = KL.TimeDistributed(KL.Dense(units, activation='relu', kernel_regularizer=regularizer))(x)
        if dropout > 0:
            x = KL.TimeDistributed(KL.Dropout(dropout))(x)

        return x
    x = dense_unit(x, 128)
    x = dense_unit(x, 256)
    x = dense_unit(x, 512)

    # x = KL.LSTM(128, return_sequences=True)(x)
    x = KL.LSTM(256)(x)

    if regression:
        x = KL.Dense(1, activation='linear')(x)
    else:
        x = KL.Dense(11, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def simple_cnn():
    # inputs : [features, timestep]
    inputs = KL.Input(shape=(62, 60))

    scale = 1
    def conv_unit(x, units, with_pooling=True):
        x = KL.Conv1D(units*scale, 3)(x)
        x = KL.ReLU()(x)
        if with_pooling:
            x = KL.MaxPooling1D()(x)
        return x
    x = inputs
    x = conv_unit(x, 64)
    x = conv_unit(x, 64)
    x = conv_unit(x, 32)

    x = KL.Flatten()(x)

    x = KL.Dense(128, activation='relu')(x)
    x = KL.Dense(11, activation='softmax')(x)

    outputs = x

    return Model(inputs=inputs, outputs=outputs)


def simple_mlp(time_step=60, regression=False, dropout=0.45, batch_norm=False):
    # inputs: [time_step, features]
    inputs = KL.Input(shape=(time_step, 60))

    def dense_unit(x, units, dropout=dropout):
        regularizer = KL.regularizers.l2(0.000)
        kernel_initializer = {0:'glorot_uniform', 1:'he_normal', 2:'he_uniform'}[0]
        x = KL.Dense(units,
                     activation='relu',
                     use_bias=False,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=regularizer)(x)
        if dropout > 0:
            x = KL.Dropout(dropout)(x)
        if batch_norm:
            x = KL.BatchNormalization(momentum=0.99)(x)

        return x

    x = inputs

    x = KL.Flatten()(x)

    scale = 1
    nb_filters = [256, 256, 1024]
    for nb_filter in nb_filters:
        x = dense_unit(x, nb_filter * scale)
    x = dense_unit(x, 2048, dropout=0.00)

    if regression:
        x = KL.Dense(1, activation='linear')(x)
    else:
        x = KL.Dense(11, activation='softmax')(x)

    outputs = x

    return Model(inputs=inputs, outputs=outputs)


def ask_bid_fusion(time_step=60, dropout=0.45):
    ask_input = 1
    pass


def get_weight_grad(model, data, label):
    assert isinstance(model, Model)
    means = []
    stds = []

    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)

    f = K.function(symb_inputs, grads)
    x, y, sample_weights = model._standardize_user_data(data, label)

    output_grad = f(x + y + sample_weights)

    for layer in range(len(model.layers)):
        if model.layers[layer].__class__.__name__ == '':
            means.append(output_grad[layer].mean())
            stds.append((output_grad[layer].std()))

    return means, stds


def gpu_setting():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))


if __name__ == '__main__':
    model = simple_model()
    print(model.summary())

