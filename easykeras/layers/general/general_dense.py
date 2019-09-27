__author__ = 'jf'
from keras.layers import Dense,Dropout
import keras


class DoubleDenseLayer(keras.Model):
    def __init__(self, hidden_size=100,
                 hidden_activation="relu",
                 output_size=3,
                 output_activation="softmax",
                 loss=0.6):
        """
        双层dense，一般可用于最终的分类
        :param hidden_size: 隐藏dense单元数
        :param hidden_activation: 隐藏层激活函数
        :param output_size: 输出dense单元数
        :param output_activation:  输出层激活函数
        :param loss:  遗忘概率
        :return: 预测结果
        """
        super(DoubleDenseLayer, self).__init__(name='double_dense_layer')
        self.first_dense = Dense(hidden_size, activation=hidden_activation)
        self.dropout = Dropout(loss)
        self.second_dense = Dense(output_size, activation=output_activation)

    def call(self, inputs, mask=None):
        first_dense_out = self.first_dense(inputs)
        drop_out = self.dropout(first_dense_out)
        final_out = self.second_dense(drop_out)
        return final_out

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)  # 继承输入的形状
        output_shape[-1] = self.second_dense.output_shape[-1]  # 改变最后一维为当前单元数
        return tuple(output_shape)
