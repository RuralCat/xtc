
import keras.layers as KL
import keras.backend as K
import tensorflow as tf

class PreprocessingLayer(KL.Layer):
    def __init__(self, time_steps=15, norm_window_size=10, nb_features=60, **kwargs):
        self.time_steps = time_steps
        self.norm_window_size = norm_window_size
        self.nb_features = nb_features
        self.batch_size = kwargs.get('batch_size')
        super(PreprocessingLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pi = []
        for i in range(self.time_steps):
            # slice
            block = tf.strided_slice(inputs,
                                     begin=[0, i, 0],
                                     end=[self.batch_size, i+self.norm_window_size, self.nb_features],
                                     strides=[1, 1, 1])
            # compute mean & standard deviation
            mean = K.mean(block, axis=1, keepdims=False)
            std  = K.std(inputs[:, i:i+self.norm_window_size], axis=1, keepdims=False)
            # normlization
            input = tf.strided_slice(inputs,
                                     begin=[0, i+self.norm_window_size-1, 0],
                                     end=[self.batch_size, i+self.norm_window_size, self.nb_features],
                                     strides=[1, 1, 1])
            res = (K.squeeze(input, axis=1) - mean) / (std + K.epsilon())
            pi.append(res)
        return K.stack(pi, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.time_steps, input_shape[2])

