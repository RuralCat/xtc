
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


def train_test_plot(train_data, test_data=None, kde=True, cumulative=True, title=None):
    axe = plt.axes()
    if kde:
        sb.kdeplot(train_data, cumulative=cumulative, legend=True, ax=axe, label='train')
        if test_data is not None:
            sb.kdeplot(test_data, cumulative=cumulative, legend=True, ax=axe, label='test')
    else:
        sb.distplot(train_data, hist=False, kde=True, label='train')
        if test_data is not None:
            sb.distplot(test_data, hist=False, kde=True, label='test')
    axe.legend()
    if title and isinstance(title, str):
        plt.title(title)
    plt.show()


def plot_askrate_and_y(askrate, y):
    fig = plt.figure()

    n = y.shape[0]
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(np.arange(n), askrate, '-', label='askrate')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(np.arange(n), y, '-', label='y', color='r')

    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    ax1.grid()
    ax1.set_ylim(min(askrate) - 2, max(askrate) + 2)
    ax2.set_ylim(min(y) - 10, max(y) + 10)


