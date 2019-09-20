
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from core.layers import PreprocessingLayer
from core.layers import ConditionPass
from core.losses import score
from core.dataset_utils import CSVRecord


import tensorflow.keras.layers as KL
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model as TFModel


def simple_model(time_step, regression=False):
    # inputs: [time_step, features]
    inputs = KL.Input(shape=(time_step, 60))
    x = inputs

    #
    def dense_unit(x, units, dropout=0.45):
        regularizer = KL.regularizers.l2(0.001)
        x = KL.TimeDistributed(KL.Dense(units, activation='relu', kernel_regularizer=regularizer))(x)
        if dropout > 0:
            x = KL.TimeDistributed(KL.Dropout(dropout))(x)

        return x

    x = dense_unit(x, 256)
    x = dense_unit(x, 512)
    x = dense_unit(x, 1024, dropout=0)

    # x = KL.LSTM(128, return_sequences=True)(x)
    x = KL.LSTM(256)(x)

    if regression:
        x = KL.Dense(1, activation='linear')(x)
    else:
        x = KL.Dense(2, activation='softmax')(x)

    model = TFModel(inputs=inputs, outputs=x)

    return model


def simple_cnn():
    # inputs : [features, timestep]
    inputs = KL.Input(shape=(62, 60))

    scale = 1

    def conv_unit(x, units, with_pooling=True):
        x = KL.Conv1D(units * scale, 3)(x)
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

    return TFModel(inputs=inputs, outputs=outputs)


def dense_unit(x, units, dropout=0.45, batch_norm=False, leak_relu=True, relu_alpha=0.1, weight_decay=0.0):
    regularizer = regularizers.l2(weight_decay)
    kernel_initializer = {0: 'glorot_uniform', 1: 'he_normal', 2: 'he_uniform'}[0]
    x = KL.Dense(units,
                 use_bias=False,
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=regularizer)(x)
    if batch_norm:
        x = KL.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    if leak_relu:
        x = KL.LeakyReLU(alpha=relu_alpha)(x)
    else:
        x = KL.ReLU(max_value=1, negative_slope=relu_alpha, threshold=0.)(x)
    if dropout > 0:
        x = KL.Dropout(dropout)(x)

    return x


def residual_dense_unit(x, units, dropout=0.45):
    _x = dense_unit(x, units, dropout=dropout)
    x = KL.Add()([x, _x])
    # if dropout > 0:
    #     x = KL.Dropout(dropout)(x)
    return x


def simple_mlp(time_step=15, regression=False, dropout=0.45, batch_norm=False, leaky_relu_alpha=0.1):
    # inputs: [time_step, features]
    nb_features = 60
    norm_window_size = 10
    inputs = KL.Input(batch_shape=(256, time_step + norm_window_size - 1, nb_features))
    x = PreprocessingLayer(batch_size=256)(inputs)
    x = KL.Flatten()(x)

    scale = 1
    nb_filters = [256, 512, 2048]
    residuals = [False, False, False]
    for nb_filter, residual in zip(nb_filters, residuals):
        if residual:
            x = residual_dense_unit(x, nb_filter * scale, dropout)
        else:
            x = dense_unit(x,
                           nb_filter * scale,
                           dropout=dropout,
                           batch_norm=batch_norm,
                           leak_relu=True,
                           relu_alpha=leaky_relu_alpha
                           )
    x = dense_unit(x, 2048, dropout=0.00, relu_alpha=leaky_relu_alpha)

    if regression:
        x = KL.Dense(1, activation='linear')(x)
    else:
        x = KL.Dense(2, activation='softmax')(x)

    outputs = x

    return TFModel(inputs=inputs, outputs=outputs)


def mid_price_move_mlp(time_step, nb_features, regression=True, dropout=0.45, batch_norm=False):
    #
    inputs = KL.Input(shape=(time_step, nb_features))
    x = KL.Flatten()(inputs)

    nb_filters = [256, 512, 2048]
    for nb_filter in nb_filters:
        x = dense_unit(x, nb_filter, dropout, batch_norm, leak_relu=False, relu_alpha=0.1)


def deep_lob(time_steps=100, nb_features=40, regression=False):
    # inputs : [time_steps, nb_features]
    inputs = KL.Input(shape=(time_steps, nb_features, 1))

    def conv_1d(x, nb_filter, kernel_size, stride=1, by_time=True):
        by_time = int(by_time)
        padding = ['valid', 'same'][by_time]
        kernel_size = [(1, kernel_size), (kernel_size, 1)][by_time]
        stride = [(1, stride), (stride, 1)][by_time]

        x = KL.Conv2D(nb_filter, kernel_size, strides=stride, padding=padding)(x)
        x = KL.LeakyReLU(alpha=0.01)(x)

        return x

    def conv_block(x, feature_size, stride=2):
        x = conv_1d(x, 16, feature_size, stride=stride, by_time=False)
        x = conv_1d(x, 16, 4)
        x = conv_1d(x, 16, 4)
        return x

    def inception_module(x):
        x0 = conv_1d(x, 32, 1)
        x0 = conv_1d(x0, 32, 3)

        x1 = conv_1d(x, 32, 1)
        x1 = conv_1d(x1, 32, 5)

        x2 = KL.MaxPooling2D(pool_size=(3, 1), strides=1, padding='same')(x)
        x2 = conv_1d(x2, 32, 1)

        return KL.Concatenate()([x0, x1, x2])

    def keras_squeeze(x):
        return K.squeeze(x, axis=2)

    def keras_squeeze_output_shape(input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

    #
    x = inputs
    features = [2, 2, 15]
    strides = [2, 2, 1]
    for feature, stride in zip(features, strides):
        x = conv_block(x, feature, stride)

    x = inception_module(x)
    x = KL.Lambda(keras_squeeze, output_shape=keras_squeeze_output_shape)(x)
    x = KL.LSTM(64)(x)

    if regression:
        x = KL.Dense(1, activation='linear')(x)
    else:
        x = KL.Dense(3, activation='softmax')(x)

    return TFModel(inputs=inputs, outputs=x)


def lstm_fcn(time_step=60, regression=False, dropout=0.8, batch_norm=True):
    # inputs
    nb_feature = 60
    input = KL.Input(shape=(time_step, nb_feature))

    # lstm branch
    # dimension shuffle
    lstm_x = input
    lstm_x = KL.Reshape((1, time_step * nb_feature))(lstm_x)
    # lstm
    lstm_x = KL.LSTM(units=128)(lstm_x)
    lstm_x = KL.Dropout(dropout)(lstm_x)

    # fcn branch
    def conv_unit(x, units, kernel_size=3, batch_norm=batch_norm):
        x = KL.Conv1D(units, kernel_size, padding='valid', kernel_initializer='he_uniform')(x)
        if batch_norm:
            x = KL.BatchNormalization()(x)
        x = KL.ReLU()(x)
        return x

    fcn_x = input
    fcn_x = conv_unit(fcn_x, 128, kernel_size=8)
    fcn_x = conv_unit(fcn_x, 256, kernel_size=5)
    fcn_x = conv_unit(fcn_x, 128, kernel_size=3)
    fcn_x = KL.GlobalAveragePooling1D()(fcn_x)

    # concat
    x = KL.concatenate([lstm_x, fcn_x])
    if regression:
        x = KL.Dense(1, activation='linear')(x)
    else:
        x = KL.Dense(2, activation='softmax')(x)

    return TFModel(inputs=input, outputs=x)


def rate_size_fusion(time_step=60, dropout=0.45):
    # two branches
    def mlp_branch(x):
        nb_filters = [256, 256]
        for nb_filter in nb_filters:
            x = dense_unit(x, nb_filter, dropout)
        return x

    # rate branch
    rates = KL.Input(shape=(time_step, 30))
    rates_x = KL.Flatten()(rates)
    rates_x = mlp_branch(rates_x)

    # size branch
    sizes = KL.Input(shape=(time_step, 30))
    sizes_x = KL.Flatten()(sizes)
    sizes_x = mlp_branch(sizes_x)

    # fusion
    x = KL.concatenate(([rates_x, sizes_x]))

    # outputs
    x = dense_unit(x, 1024, dropout=dropout)
    x = dense_unit(x, 2048, dropout=0)
    x = KL.Dense(1, activation='linear')(x)

    return TFModel(inputs=[rates, sizes], outputs=x)


def get_weight_grad(model, data, label):
    assert isinstance(model, TFModel)
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
    from tensorflow.keras.backend import set_session
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))


class Model(object):
    def __init__(self, config, **kwargs):
        # initial status
        self.built = False
        self.compiled = False
        self.data_settled = False
        # set params
        self._config = config
        self._model = self.build_model(**kwargs)
        # assert isinstance(self._config, XTXConfig)
        assert isinstance(self._model, TFModel), "self.build_model should return a keras Model"

    def build_model(self):
        # clear session & release gpu memory before build a new model
        # clear session
        # K.clear_session()
        # release gpu memory
        # todo
        raise NotImplementedError

    def _parse_args(self, arg_names, **kwargs):
        if isinstance(arg_names, str): arg_names = [arg_names]
        args = []
        for arg_name in arg_names:
            if arg_name in kwargs:
                args.append(kwargs[arg_name])
            elif arg_name in self._config:
                args.append(self._config[arg_name])
            else:
                raise IndexError(arg_name + " is not exist in self._config dict and arguments!")
        return args if len(args) > 1 else args[0]

    def _parse_datas(self, **kwargs):
        # get args
        time_steps, norm_window_size, levels = \
            self._parse_args(["time_steps", "norm_window_size", "levels"], **kwargs)
        args = dict(time_steps=time_steps, norm_window_size=norm_window_size, levels=levels)
        return self.data_func(self.datas, **args)

    def set_data(self, datas, data_func=None):
        self.datas = datas
        self.data_func = data_func
        self.data_settled = True

    def compile(self, **kwargs):
        if not self.built:
            raise NotImplementedError("should build model before compile")
        # get compile params from config
        opt, opt_args, loss, metrics = self._parse_args(["optimizer", "optimizer_args", "loss", "metrics"], **kwargs)
        # compile and set status
        self._model.compile(optimizer=opt(**opt_args), loss=loss, metrics=metrics)
        self.compiled = True

    def train(self, **kwargs):
        # check compile status
        if not (self.compiled or self.built or self.data_settled):
            raise ValueError("Please check model building, model compile or dataset is all ready!")
        # get train params
        batch_size, epochs, callbacks = self._parse_args(["batch_size", "epochs", "callbacks"], **kwargs)
        # get dataset
        (x, y), (val_x, val_y), (_, _) = self._parse_datas(**kwargs)
        # train
        history = self._model.fit(x, y,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=(val_x, val_y))
        return history

    def cross_train(self, **kwargs):
        pass

    def retrain(self, **kwargs):
        self._model = self.build_model(**kwargs)
        self.compile(**kwargs)
        return self.train(**kwargs)

    def ensemble_train(self, nb_models, **kwargs):
        self.ensemble_models = []
        for _ in range(nb_models):
            self.retrain(**kwargs)
            self.ensemble_models.append(self._model.weights)
        # todo

    def parameter_search(self, search_dict):
        """
        train model over different parameter combination to find best hyperparamters

        format: search_dict = {"para_name1": [value_1_1, value_1_2, ...], "para_name2": [value_2_1, ...], ...}
        example:
            search_dict = {"batch_size": [32, 64, 256], "lr", [np.power(10, l) for l in range(-6, -1)]}
            model.parameter_search(search_dict)
        """
        assert isinstance(search_dict, dict)
        params_name = list(search_dict.keys())
        # create parameter combinations
        p = itertools.product(*search_dict.values())
        # create empty recorder
        val_scores, test_scores = {}, {}
        recorder = CSVRecord(params_name)
        # searching
        for params in p:
            s  = ''
            for i, param in enumerate(params):
                s = s + params_name[i] + ': ' + str(param) + ' '
            print('Train on ' + s)
            # train
            params_ = {params_name[i]: params[i] for i in range(len(params))}
            history = self.retrain(**params_)
            # validation score & test score
            (_, _), (val_x, val_y), (test_x, test_y) = self._parse_datas(**params_)
            val_scores[s] = score(val_y, self.predict(val_x))
            test_scores[s] = score(test_y, self.predict(test_x))
            # write recorder
            min_train_loss = min(history.history["loss"])
            min_val_loss = min(history.history["val_loss"])
            recorder.add_record(params, min_train_loss, min_val_loss, val_scores[s], test_scores[s])
        recorder.print_record()

        return val_scores, test_scores

    def predict(self, x):
        batch_size = self._parse_args(["batch_size"])
        return self._model.predict(x, batch_size=batch_size, verbose=0)

    def evaluate(self, x):
        pass

    def ensemble_evaluate(self):
        pass


class BaseMLP(Model):
    def build_model(self, **kwargs):
        # set status
        # K.clear_session()
        self.built = True
        # parse args
        batch_size, dropout, time_steps, model_type, norm_window_size, relu_alpha = \
            self._parse_args([
            "batch_size", "dropout", "time_steps", "model_type", "norm_window_size", "relu_alpha"
        ], **kwargs)
        levels = self._parse_args("levels", **kwargs)
        scale = self._parse_args(["mlp_scale"], **kwargs) 
        weight_decay  =self._parse_args(["weight_decay"], **kwargs)

        # create model
        nb_features = levels * 2
        input_shape = (time_steps+norm_window_size-1, nb_features)
        inputs = KL.Input(shape=input_shape)
        x = inputs

        preprocessing = self._parse_args("preprocessing")
        if preprocessing is not None:
            x = PreprocessingLayer(time_steps=time_steps,
                                   norm_window_size=norm_window_size,
                                   nb_features=nb_features,
                                   batch_size=batch_size)(x)

        x = KL.Flatten()(x)
        nb_filters = [32, 64, 256]
        for nb_filter in nb_filters:
            x = dense_unit(x,
                           nb_filter * scale,
                           dropout=dropout,
                           relu_alpha=relu_alpha,
                           weight_decay=weight_decay)
        x = dense_unit(x, 512*scale, dropout=0, relu_alpha=relu_alpha, weight_decay=weight_decay)

        if model_type == 'regression':
            x = KL.Dense(1, activation='tanh', use_bias=True)(x)
            # x = KL.Lambda(lambda x: 5*x)(x)
        else:
            x = KL.Dense(2, activation='softmax')(x)

        outputs = x

        return TFModel(inputs=inputs, outputs=outputs)


class DiffMLP(Model):
    def build_model(self, **kwargs):
        # set status
        # K.clear_session()
        self.built = True
        # parse args
        batch_size, dropout, time_steps, model_type, norm_window_size, relu_alpha = \
            self._parse_args([
                "batch_size", "dropout", "time_steps", "model_type", "norm_window_size", "relu_alpha"
            ], **kwargs)
        levels = self._parse_args("levels", **kwargs)
        scale = self._parse_args(["mlp_scale"], **kwargs)
        weight_decay = self._parse_args(["weight_decay"], **kwargs)
        channels = 2

        # create model
        nb_features = levels * 2
        input_shape = (channels, time_steps + norm_window_size - 1, nb_features)
        inputs = KL.Input(shape=input_shape)
        x = inputs

        preprocessing = self._parse_args("preprocessing")
        if preprocessing is not None:
            x = PreprocessingLayer(time_steps=time_steps,
                                   norm_window_size=norm_window_size,
                                   nb_features=nb_features,
                                   batch_size=batch_size)(x)

        x = KL.TimeDistributed(KL.Flatten())(x)

        def timedistributed_dense_unit(_x, units, dropout=0.45, relu_alpha=0.1, weight_decay=0.0):
            regularizer = regularizers.l2(weight_decay)
            kernel_initializer = {0: 'glorot_uniform', 1: 'he_normal', 2: 'he_uniform'}[0]
            _x = KL.TimeDistributed(KL.Dense(units,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=regularizer))(_x)
            _x = KL.LeakyReLU(alpha=relu_alpha)(_x)
            _x = KL.TimeDistributed(KL.Dropout(dropout))(_x)
            return _x

        nb_filters = [32, 64, 256]
        for nb_filter in nb_filters:
            x = timedistributed_dense_unit(x,
                           nb_filter * scale,
                           dropout=dropout,
                           relu_alpha=relu_alpha,
                           weight_decay=weight_decay)
        x = timedistributed_dense_unit(x,
                                       512 * scale,
                                       dropout=0, relu_alpha=relu_alpha, weight_decay=weight_decay)

        x = KL.TimeDistributed(KL.Dense(1, activation='tanh'))(x)
        x = KL.TimeDistributed(KL.Lambda(lambda x: 5 * x))(x)

        outputs = x

        return TFModel(inputs=inputs, outputs=outputs)



class ConditionMLP(Model):
    def _build_model(self, **kwargs):
        # set status
        K.clear_session()
        self.built = True
        # parse args
        batch_size, dropout, time_steps, model_type, norm_window_size, relu_alpha = \
            self._parse_args([
            "batch_size", "dropout", "time_steps", "model_type", "norm_window_size", "relu_alpha"
        ], **kwargs)
        levels = self._parse_args("levels", **kwargs)
        scale = self._parse_args(["mlp_scale"], **kwargs) # default 8
        weight_decay = self._parse_args(["weight_decay"], **kwargs)
        pass_alpha = self._parse_args("pass_alpha", **kwargs)

        # create model
        nb_features = levels * 2
        input_shape = (time_steps+norm_window_size-1, nb_features)
        # inputs: [time_step, features]
        inputs = KL.Input(batch_shape=(batch_size,) + input_shape)
        x = inputs
        x = KL.Flatten()(x)

        nb_filters = [32, 64, 256]
        for nb_filter in nb_filters:
            x = dense_unit(x,
                           nb_filter * scale,
                           dropout=dropout,
                           relu_alpha=relu_alpha,
                           weight_decay=weight_decay)
        x0 = dense_unit(x, 256*scale, dropout=0, relu_alpha=relu_alpha, weight_decay=weight_decay)
        x1 = dense_unit(x, 256*scale, dropout=0, relu_alpha=relu_alpha, weight_decay=weight_decay)

        x0 = KL.Dense(1, activation='linear')(x0)
        x1 = KL.Dense(1, activation='sigmoid')(x1)
        x = ConditionPass(pass_alpha)([x0, x1])

        outputs = x

        return TFModel(inputs=inputs, outputs=outputs)

class LSTMFCN(Model):
    def _build_model(self, **kwargs):
        self.built = True
        # arguments
        batch_size, input_shape, dropout, time_steps, model_type, norm_window_size = \
            self._parse_args([
            "batch_size", "input_shape", "dropout", "time_steps", "model_type", "norm_window_size"
        ], **kwargs)
        levels = self._parse_args("levels", **kwargs)
        nb_features = levels * 2
        preprocessing = self._parse_args("preprocessing")

        # inputs
        input_shape = (time_steps+norm_window_size-1, nb_features)
        # inputs: [time_step, features]
        inputs = KL.Input(batch_shape=(batch_size,) + input_shape)

        # lstm branch
        # dimension shuffle
        lstm_x = inputs
        if preprocessing is not None:
            lstm_x = PreprocessingLayer(time_steps=time_steps,
                                        norm_window_size=norm_window_size,
                                        nb_features=nb_features,
                                        batch_size=batch_size)(lstm_x)

        lstm_x = KL.Reshape((1, time_steps * input_shape[1]))(lstm_x)
        # lstm
        lstm_x = KL.LSTM(units=128)(lstm_x)
        lstm_x = KL.Dropout(dropout)(lstm_x)

        # fcn branch
        def conv_unit(x, units, kernel_size=3, batch_norm=True):
            x = KL.Conv1D(units, kernel_size, padding='valid', kernel_initializer='he_uniform')(x)
            if batch_norm:
                x = KL.BatchNormalization()(x)
            x = KL.ReLU()(x)
            return x

        fcn_x = input
        fcn_x = conv_unit(fcn_x, 128, kernel_size=8)
        fcn_x = conv_unit(fcn_x, 256, kernel_size=5)
        fcn_x = conv_unit(fcn_x, 128, kernel_size=3)
        fcn_x = KL.GlobalAveragePooling1D()(fcn_x)

        # concat
        x = KL.concatenate([lstm_x, fcn_x])
        if model_type == 'regression':
            x = KL.Dense(1, activation='linear')(x)
        else:
            x = KL.Dense(2, activation='softmax')(x)

        return TFModel(inputs=inputs, outputs=x)


class DeepLOB(Model):
    def _build_model(self):
        self.built = True



if __name__ == '__main__':
    from xtx_config import XTXConfig
    # model
    model = BaseMLP(XTXConfig)
    print(model.summary())
