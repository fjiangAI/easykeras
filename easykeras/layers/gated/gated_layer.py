__author__ = 'jf'
from keras.layers import Dense, multiply, Conv1D
import keras


class GluLayer(keras.Model):
    def __init__(self,
                 kernel_layer="CNN",
                 h_dim=None,
                 c_filters=None,
                 c_kernel_size=None):
        """
        门控线性单元
        :param kernel_layer: 使用的核心类型，可以使CNN也可以Dense
        :param h_dim: 使用Dense的隐藏层（输出维度）
        :param c_filters: 使用CNN的核心数（输出维度）
        :param c_kernel_size: 使用CNN的核的大小
        """
        super(GluLayer, self).__init__(name='glu_layer')
        if kernel_layer == "CNN":
            self.sig_layer = Conv1D(c_filters, c_kernel_size, padding='same', strides=1, activation="sigmoid")
            self.original_layer = Conv1D(c_filters, c_kernel_size, padding='same', strides=1)
        else:
            self.sig_layer = Dense(h_dim, activation="sigmoid")
            self.original_layer = Dense(h_dim)

    def call(self, inputs, mask=None):
        # GLU(X)=Kernal_layer(X) element-wise product sigmoid(Kernal_layer(X))
        sig_text = self.sig_layer(inputs)
        original_text = self.original_layer(inputs)
        xo = multiply([sig_text, original_text])
        return xo

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)  # 继承输入的形状
        output_shape[-1] = self.sig_layer.output_shape[-1]  # 改变最后一维为当前单元数
        return tuple(output_shape)
