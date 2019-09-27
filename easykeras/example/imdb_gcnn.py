__author__ = 'jf'
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
import keras
from easykeras.layers.gated.gated_layer import gated_linear_unit
from easykeras.layers.general.general_dense import double_dense_layer
from easykeras.example.imdb_util import get_train_test


# epoch 10:89.02
def get_config():
    return {"input_shape": (400,),
            "max_features": 5000,
            "max_len": 400,
            "embedding_dims": 50,
            "filters": 250,
            "kernel_size": 3,
            "hidden_dims": 100,
            "class_num": 1,
            "activation": 'sigmoid'}


class GcnnModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(GcnnModel, self).__init__(name='gcnn_model')
        self.embedding_layer = Embedding(config.max_features, config.embedding_dims, input_length=config.max_len)
        self.gcnn_layer = gated_linear_unit(input_shape=self.embedding_layer.output_shape,
                                            c_kernel_size=config.kernel_size,
                                            c_filters=config.filters,
                                            h_dim=config.hidden_dims,
                                            kernel_layer="CNN")
        self.global_pooling_layer = GlobalMaxPooling1D()
        self.class_layer = double_dense_layer(input_shape=self.global_pooling_layer.output_shape, output_activation='sigmoid', output_size=1)


    def call(self, inputs):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        embedding_output = self.embedding_layer(inputs)
        gcnn_output = self.gcnn_layer(embedding_output)
        global_pooling_output = self.global_pooling_layer(gcnn_output)
        final_out = self.class_layer(global_pooling_output)
        return final_out


# 设定编译参数
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']


class Config:
    def __init__(self, config={}):
        for key, value in config.items():
            setattr(self, key, value)


# 主程序
x_train, y_train, x_test, y_test = get_train_test()
model_config = Config(get_config())
gcnn_model = GcnnModel(model_config)
gcnn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


# 设置训练超参数
batch_size = 32
epochs = 10
# 训练过程
gcnn_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
