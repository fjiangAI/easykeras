__author__ = 'jf'

from easykeras.layers.capsule import Capsule
from keras.layers import *
from easykeras.example.imdb_util import get_train_test
import keras

# epoch 10: 86.29


class CapsuleModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(CapsuleModel, self).__init__(name='capsule_model')
        self.embedding_layer = Embedding(config.max_features, config.embedding_dims, input_length=config.max_len)
        self.dropout_layer = Dropout(0.2)
        self.conv_layer = Conv1D(config.filters, config.kernel_size, padding='valid', activation='relu', strides=1)

    def call(self, inputs, mask=None):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        embedding_output = self.embedding_layer(inputs)  # (?,400,50)
        dropout_output = self.dropout_layer(embedding_output)
        conv_output = self.conv_layer(dropout_output)  # (?,398,250)
        capsule = Capsule(2, 8, 3, True)(conv_output)  # (?,2,32)
        output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)  # (?,2)
        return output


def get_config():
    return {"max_features": 5000,
            "max_len": 400,
            "embedding_dims": 50,
            "filters": 250,
            "kernel_size": 3,
            "hidden_dims": 100,
            "class_num": 1,
            "activation": 'sigmoid',
            "num_classes": 2}


class Config:
    def __init__(self, config={}):
        for key, value in config.items():
            setattr(self, key, value)


if __name__ == "__main__":
    # 主程序
    model_config = Config(get_config())
    x_train, y_train, x_test, y_test = get_train_test(one_hot=True, num_classes=model_config.num_classes)
    capsule_model = CapsuleModel(model_config)

    # 设定编译参数
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    capsule_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # 设置训练超参数
    batch_size = 32
    epochs = 10
    # 训练过程
    capsule_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
