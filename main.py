
import tensorflow as tf

from dataset import XTXDataset
from xtx_model import BaseMLP
from xtx_config import XTX_Config
from core.dataset_utils import volume_data
from core.dataset_utils import volume_data_norm
from core.dataset_utils import data_func
from core.losses import score
from core.losses import score_metric
from core.losses import Score
import inspect
import json
"""
demo:
def foo():
    return 1
lines = inspect.getsource(foo)
"""
"""
read json
with open(file-path, 'r') as f:
    data = json.load(f)
write json
with open(file_path, 'w') as f:
    data_store = json.dump(f)

"""

if __name__ == '__main__':
    # load data
    dataset = XTXDataset()
    dataset.delete_invalid_rows()
    # set memory fraction
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # create mlp_model
    mlp_model = BaseMLP(XTX_Config)
    mlp_model.set_data(dataset, data_func=volume_data)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = mlp_model._parse_datas()

    score_callback = Score(train_x, train_y, val_x, val_y)
    # callbacks = [score_callback] + XTX_Config["callbacks"]
    callbacks = XTX_Config["callbacks"]
    metrics = None
    mlp_model.compile(metrics=metrics)
    # mlp_model._model.summary()
    _ = mlp_model.train(callbacks=callbacks)
    # predict
    # mlp_model._model.load_weights("datas/weights/scale_2_lastdense_512_val_202_test_199.hdf5")
    print("train_score:", score(train_y, mlp_model.predict(train_x)))
    print("val_score:", score(val_y, mlp_model.predict(val_x)))
    print("test_score:", score(test_y, mlp_model.predict(test_x)))

from keras.layers import GRU
