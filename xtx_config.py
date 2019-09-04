from keras.callbacks import *
from keras.optimizers import Adam

from core.losses import tf_score

monitor = 'val_loss'
earlystopping = EarlyStopping(monitor=monitor, min_delta=1e-4, patience=10, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('', monitor=monitor, verbose=1, save_best_only=True)
reducelr = ReduceLROnPlateau(monitor=monitor, factor=0.6, patience=3, min_delta=1e-4, verbose=1)
callbacks = [earlystopping, reducelr]


class XTXConfig(object):
    def __init__(self):
        self._set_base_params()

    def _set_base_params(self):
        # data setting
        self.time_steps = 15
        self.norm_window_size = 10
        self.input_shape = (self.time_steps + self.norm_window_size - 1, 60)
        # model setting
        self.model_type = "regression"
        self.optimizer = Adam
        self.optimizer_args = {"learning_rate": 2e-4}
        self.loss = tf_score
        # train setting
        self.batch_size = 256
        self.epochs = 25
        self.callbacks = callbacks

    def set_param(self, param_name, param):
        pass


# callbacks
XTX_Config = {
    # data setting
    "time_steps"      : 15,
    "norm_window_size": 10,
    "levels"          : 15,
    "input_shape"     : (24, 60),
    # model setting
    "model_type"      : "regression",
    "optimizer"       : Adam,
    "optimizer_args"  : {"lr": 2e-4},
    "loss"            : tf_score,
    "metrics"         : ['accuracy'],
    "dropout"         : 0.5,
    "relu_alpha"      : 0.1,
    "mlp_scale"       : 6,
    "weight_decay"    : 0.0,
    # train setting
    "batch_size"      : 256,
    "epochs"          : 25,
    "callbacks"       : callbacks,
}
XTX_Config["input_shape"] = (XTX_Config["time_steps"] + XTX_Config["norm_window_size"] - 1,
                             XTX_Config["levels"] * 4)
