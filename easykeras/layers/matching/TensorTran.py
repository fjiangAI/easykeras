__author__ = 'jf'
from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
class TensorTran(Layer):
    def __init__(self, d2=80, r=10, W_regularizer=None, W_constraint=None, **kwargs):
        '''

        :param d2:
        :param r:
        :param W_regularizer:
        :param W_constraint:
        :param kwargs:
        :return:
        '''
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.d2 = d2
        self.r = r
        super(TensorTran, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.d1 = input_shape[-1]
        self.W = self.add_weight((self.d1, self.d2),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.P = self.add_weight((self.d1, self.r, self.d2),
                                 initializer=self.init,
                                 name='{}_P'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.Q = self.add_weight((self.r, self.d1, self.d2),
                                 initializer=self.init,
                                 name='{}_Q'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.b = self.add_weight((self.d2,),
                                 initializer=self.init,
                                 name='{}_b'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        t = K.permute_dimensions(K.dot(K.permute_dimensions(self.P, (2, 1, 0)), K.permute_dimensions(x, (1, 0))), (0, 2, 1))
        tv = K.batch_dot(K.permute_dimensions(K.batch_dot(t, K.permute_dimensions(self.Q, (2, 0, 1))), (1, 0, 2)), x)
        return K.relu(K.dot(x, self.W) + tv + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.d2
