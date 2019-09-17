
import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import Callback

def score(y, y_pred):
    y_pred = np.squeeze(y_pred)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return 1 - np.sum(np.power(y - y_pred, 2)) / np.sum(np.power(y, 2))


def tf_score_no_bias(y, y_pred):
    # return K.sum(K.pow(y - y_pred, 2)) / K.sum(K.pow(y-K.mean(y), 2))
    return K.sum(K.pow(y - y_pred, 2)) * K.mean(K.pow(y-K.mean(y), 2))


def tf_score_loss(y, y_pred):
    return K.sum(K.pow(y - y_pred, 2)) / K.sum(K.pow(y, 2))


def tf_score_metric(y, y_pred):
    return 1 - K.sum(K.pow(y - y_pred, 2)) / K.sum(K.pow(y, 2))


def score_metric(val_y, batch_size):
    total = np.sum(np.power(val_y, 2))
    steps = val_y.shape[0] // batch_size + 1
    def tf_score_metric(y, y_pred):
        return 1 - steps * K.sum(K.pow(y - y_pred, 2) / total)
    return tf_score_metric


def weights_mse(y, y_pred):
    pass


def model_score(model, train_x, train_y, val_x, val_y, test_x, test_y):
    scores = []
    for x, y, s in zip([train_x, val_x, test_x], [train_y, val_y, test_y], ['train', 'val', 'test']):
        pred = model.predict(x, batch_size=256, verbose=1)
        scores.append(score(y, pred))
        print('{} score: {:.5f}'.format(s, scores[-1]))

    return scores

class Score(Callback):
    def __init__(self, train_x, train_y, val_x, val_y):
        self.trains = (train_x, train_y)
        self.vals = (val_x, val_y)
        self.best = -np.Inf
        super(Score, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        train_score = score(self.trains[1],
                          self.model.predict(self.trains[0], batch_size=256))
        val_score = score(self.vals[1],
                          self.model.predict(self.vals[0], batch_size=256))
        if val_score > self.best:
            self.best = val_score
            flag = '[best]'
        else:
            flag = ''
        print("\nEpoch %s: train_ score: %.4f - val_score: %.4f%s" % (epoch+1, train_score, val_score, flag))

