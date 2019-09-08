__author__ = 'jf'
from keras.layers import Layer
from keras import backend as K

class CustomLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs
'''使用例子
class FeedForward(CustomLayer):
    """FeedForward层，其实就是两个Dense层的叠加
    """
    def __init__(self, units, activation='relu', **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
    #初始化好所需要用的层
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]
        self.dense_1 = Dense(self.units, activation=self.activation)
        self.dense_2 = Dense(output_dim)

    def call(self, inputs):
        x = self.reuse(self.dense_1, inputs)
        x = self.reuse(self.dense_2, x)
        return x
    #如果有维度变换，还应当重新计算维度
    def compute_output_shape(self, input_shape):
        pass
'''