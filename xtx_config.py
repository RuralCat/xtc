
# from keras.callbacks import *
# from keras.optimizers import Adam

from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from keras.models import Model


from core.losses import tf_score_loss
from core.losses import tf_score_metric
from core.losses import tf_score_no_bias
from core.clr import OneCycleLR

monitor = 'val_loss'
earlystopping = EarlyStopping(monitor=monitor, min_delta=1e-4,
                              patience=25, verbose=1,
                              restore_best_weights=True, mode='min')
checkpoint = ModelCheckpoint('', monitor=monitor,
                             verbose=1, save_best_only=True)
reducelr = ReduceLROnPlateau(monitor=monitor, factor=0.6,
                             patience=10, min_delta=1e-4, verbose=1)
lr_manager = OneCycleLR(max_lr=1.6e-3,
                        maximum_momentum=0.99, minimum_momentum=0.99)
callbacks = [earlystopping, reducelr]


class XTXConfig(object):
    def __init__(self):
        self._set_base_params()

    def _set_base_params(self):
        # data setting
        self.time_steps = 15
        self.norm_window_size = 10
        self.levels = 5
        self.input_shape = (self.time_steps + self.norm_window_size - 1, 60)
        # model setting
        self.model_type = "regression"
        self.optimizer = Adam
        self.optimizer_args = {"learning_rate": 2e-4}
        self.loss = tf_score_loss
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
    "norm_window_size": 1,
    "levels"          : 5,
    "input_shape"     : (24, 60),
    # model setting
    "preprocessing"   : None,
    "model_type"      : "regression",
    "optimizer"       : Adam,
    "optimizer_args"  : {"lr": 2e-4},
    "loss"            : 'mse',
    "metrics"         : [tf_score_metric], #['accuracy'],
    "dropout"         : 0.5,
    "relu_alpha"      : 0.1,
    "mlp_scale"       : 2,
    "weight_decay"    : 0.0,
    "pass_alpha"      : 0.5,
    # train setting
    "batch_size"      : 256,
    "epochs"          : 320,
    "callbacks"       : callbacks,
}

