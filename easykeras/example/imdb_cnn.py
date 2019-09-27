__author__ = 'jf'
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import keras
from easykeras.example.imdb_util import get_train_test


# epoch 10: 88.03
def get_config():
    return {"input_shape": (400,),
            "max_features": 5000,
            "max_len": 400,
            "embedding_dims": 50,
            "filters": 250,
            "kernel_size": 3,
            "hidden_dims": 250,
            "class_num": 1,
            "activation": 'sigmoid'}


class CnnModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(CnnModel, self).__init__(name='cnn_model')
        self.embedding_layer = Embedding(config.max_features, config.embedding_dims, input_length=config.max_len)
        self.dropout_layer = Dropout(0.2)
        self.con_layer = Conv1D(config.filters, config.kernel_size, padding='valid', activation='relu', strides=1)
        self.global_pooling_layer = GlobalMaxPooling1D()
        self.dense_layer = Dense(config.hidden_dims)
        self.dropout_layer2 = Dropout(0.2)
        self.activation_layer2 = Activation('relu')
        self.final_activation = config.activation
        self.class_layer = Dense(config.class_num, activation=self.final_activation)

    def call(self, inputs):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        embedding_output = self.embedding_layer(inputs)  # (?,400,50)
        dropout_output = self.dropout_layer(embedding_output)
        con_output = self.con_layer(dropout_output)  # (?,398,250)
        global_pooling_output = self.global_pooling_layer(con_output)  # （？,250）
        dense_output = self.dense_layer(global_pooling_output)  # (?,250)
        activation_output2 = self.activation_layer2(dense_output)
        final_out = self.class_layer(activation_output2)  # (?,1)
        return final_out


class Config:
    def __init__(self, config={}):
        for key, value in config.items():
            setattr(self, key, value)


if __name__ == "__main__":
    # 主程序
    x_train, y_train, x_test, y_test = get_train_test()
    model_config = Config(get_config())
    cnn_model = CnnModel(model_config)

    # 设定编译参数
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    cnn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # 设置训练超参数
    batch_size = 32
    epochs = 10
    # 训练过程
    cnn_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
