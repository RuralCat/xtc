
from xtx_config import XTX_Config
from xtx_model import BaseMLP
from core.clr import LRFinder
from dataset import XTXDataset
from core.dataset_utils import ema

from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

import os
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt


# find best learning_rate
class HyperFinder:
    def __init__(self, model_cls, model_args, data, data_func, root_dir='datas'):
        self.model_cls = model_cls
        # self.model_args = copy.deepcopy(model_args)
        self.model_args = model_args
        self.data = data
        self.data_func = data_func
        self.root_dir = os.path.join(root_dir, 'hypers_' + datetime.datetime.now().strftime("%m%d_%H%M"))
        self._best_lr = None
        self._best_momentum = None

    def set_best_lr(self, best_lr):
        self._best_lr = best_lr

    def set_best_momentum(self, best_momentum):
        self._best_momentum = best_momentum

    def _run(self, min_lr=1e-6, max_lr=1e-1, momentum=0.9, weight_decay=0.0, save_dir='', find_lr=True):
        # create model
        K.clear_session()
        self.model_args["weight_decay"] = weight_decay
        model = self.model_cls(XTX_Config, **self.model_args)
        # load data
        model.set_data(self.data, data_func=self.data_func)
        (x, y), (val_x, val_y), (_, _) = model._parse_datas(**self.model_args)
        # optimizer
        self.model_args["optimizer"] = SGD
        self.model_args["optimizer_args"] = {"lr": 3e-4, "momentum": momentum}
        # lr callback
        if find_lr:
            lr_scale = 'exp'
            val_data = None
        else:
            lr_scale = 'linear'
            val_data = (val_x, val_y)
        batch_size = model._parse_args("batch_size", **self.model_args)
        lr_callback = LRFinder(num_samples=x.shape[0], batch_size=batch_size,
                               minimum_lr=min_lr, maximum_lr=max_lr,
                               lr_scale=lr_scale,
                               validation_data=val_data, validation_sample_rate=10,
                               loss_smoothing_beta=0.99, save_dir=save_dir,
                               stopping_criterion_factor=None, verbose=0)
        self.model_args["callbacks"] = [lr_callback]
        # cyclic train
        self.model_args["epochs"] = 1
        model.compile(**self.model_args)
        model.train(**self.model_args)


    def find_lr(self, min_lr=1e-6, max_lr=1e-1):
        save_dir = os.path.join(self.root_dir, 'learning_rate')
        self._run(min_lr=min_lr, max_lr=max_lr, save_dir=save_dir, find_lr=True)

    def find_momentum(self, momentums):
        if self._best_lr is None:
            raise ValueError("Before find momentum, you should find learning rate firstly.")
        for momentum in momentums:
            save_dir = os.path.join(self.root_dir, 'momentum-%s'%str(momentum))
            self._run(min_lr=self._best_lr/10, max_lr=self._best_lr,
                      momentum=momentum, save_dir=save_dir, find_lr=False)

    def find_weight_decay(self, weight_decays):
        if self._best_lr is None or self._best_momentum is None:
            raise ValueError("Before find momentum, you should find learning rate and momentum firstly.")
        for weight_decay in weight_decays:
            save_dir = os.path.join(self.root_dir, 'weight_decay-%s'%str(weight_decay))
            self._run(min_lr=self._best_lr/10, max_lr=self._best_lr, momentum=self._best_momentum,
                      weight_decay=weight_decay, save_dir=save_dir, find_lr=False)

    def _vis(self, title, labels, dirs, begin=0, end=-1):
        for label, dir in zip(labels, dirs):
            losses, lrs = LRFinder.restore_schedule_from_dir(dir)
            losses = ema(losses, 25)
            plt.plot(lrs[begin:end], losses[begin:end], label=label)
        plt.title(title)
        plt.xlabel("Learning rate")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.show()

    def vis_lr(self, begin=0, end=-1):
        labels = ['learning_rate']
        dirs = [os.path.join(self.root_dir, label) for label in labels]
        self._vis('Learning rate vs. Loss', labels, dirs, begin, end)

    def vis_momentum(self, momentums, begin=0, end=-1):
        labels = ['momentum-%s'%str(momentum) for momentum in momentums]
        dirs = [os.path.join(self.root_dir, label) for label in labels]
        self._vis('Momentum', labels, dirs, begin, end)

    def vis_weight_decay(self, decays, begin, end):
        labels = ['weight_decay-%s'%str(decay) for decay in decays]
        dirs = [os.path.join(self.root_dir, label) for label in labels]
        self._vis('Weight Decay', labels, dirs, begin, end)


def finder_demo():
    from core.dataset_utils import data_func

    # read data
    dataset = XTXDataset()
    dataset.delete_invalid_rows()

    # create finder
    finder = HyperFinder(BaseMLP, XTX_Config, dataset, data_func=data_func)

    # find best learning rate
    finder.find_lr(min_lr=1e-6, max_lr=1)
    finder.vis_lr(begin=500, end=-200)
    # set learning rate
    lr = float(input("set learning rate:"))
    finder.set_best_lr(lr)

    # find best momentum
    momentums = [0.85, 0.9, 0.95, 0.99, 0.999]
    finder.find_momentum(momentums)
    finder.vis_momentum(momentums, begin=500, end=-200)
    # set best momentum
    momentum = float(input("set best momentum:"))
    finder.set_best_momentum(momentum)

    # find besr weight decay
    weight_decays = [np.power(0.1, x) for x in np.arange(1, 3, 1)]
    weight_decays.append(0)
    finder.find_weight_decay(weight_decays)
    finder.vis_weight_decay(weight_decays, begin=900, end=-200)
    # set weight decay
    weight_decay = float(input("set weight decay:"))

    # show res
    print("The best hyperparameter for " + BaseMLP.__name__ + " : \n",
          "learning rate: " + str(lr) + '\n',
          "     momentum: " + str(momentum) + '\n',
          " weight_decay: " + str(weight_decay))


if __name__ == '__main__':
    pass
