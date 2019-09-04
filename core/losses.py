
import numpy as np
import keras.backend as K


def score(y, y_pred):
    y_pred = np.squeeze(y_pred)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return 1 - np.sum(np.power(y - y_pred, 2)) / np.sum(np.power(y, 2))


def tf_score(y, y_pred):
    return K.sum(K.pow(y - y_pred, 2)) / K.sum(K.pow(y, 2))


def model_score(model, train_x, train_y, val_x, val_y, test_x, test_y):
    scores = []
    for x, y, s in zip([train_x, val_x, test_x], [train_y, val_y, test_y], ['train', 'val', 'test']):
        pred = model.predict(x, batch_size=256, verbose=1)
        scores.append(score(y, pred))
        print('{} score: {:.5f}'.format(s, scores[-1]))

    return scores
