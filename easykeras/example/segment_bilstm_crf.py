import numpy as np
from keras.layers import Embedding, Input, LSTM, Dropout, Dense, Bidirectional
from keras.models import Model

__author__ = 'jf'
import keras
from easykeras.example.imdb_util import Config
from easykeras.layers.sequence.crf_layer import CRFLayer

# epoch 10:89.02
def get_config():
    return {"vocab_size": 10000,
            "word_embedding_size": 100,
            "num_word_lstm_units": 100,
            "dropout": 0.5,
            "ntags": 10,
            "maxlen": 2,
            "batch_size": 32,
            "n_classes": 10
            }

def get_train(config):

    # Random features.
    x = np.random.randint(1, config.vocab_size, size=(config.batch_size, config.maxlen))

    # Random tag indices representing the gold sequence.
    y = np.random.randint(config.n_classes, size=(config.batch_size, config.maxlen))
    y = np.eye(config.n_classes)[y]

    # All sequences in this example have the same length, but they can be variable in a real model.
    s = np.asarray([config.maxlen] * config.batch_size, dtype='int32')
    return [x, s], y


class BiLSTMCRFModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(BiLSTMCRFModel, self).__init__(name='bilstmcrf_model')
        self.embedding_layer = Embedding(input_dim=config.vocab_size, output_dim=config.word_embedding_size, mask_zero=True)
        self.bilstm_layer = Bidirectional(LSTM(units=config.num_word_lstm_units, return_sequences=True))
        self.dropout_layer= Dropout(config.dropout)
        self.dense_layer= Dense(config.ntags)
        self.crf = CRFLayer()

    def call(self, inputs):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        embedding_output = self.embedding_layer(inputs[0])
        bilstm_output = self.bilstm_layer(embedding_output)
        dropout_output= self.dropout_layer(bilstm_output)
        pred = self.crf([dropout_output, inputs[1]])
        return pred


if __name__ == "__main__":
    # 主程序
    model_config = Config(get_config())
    bilstm_crf_model = BiLSTMCRFModel(model_config)
    x_train, y_train = get_train(model_config)

    # 设定编译参数
    loss = bilstm_crf_model.crf.loss
    optimizer = 'sgd'
    metrics = ['accuracy']
    bilstm_crf_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # 设置训练超参数
    batch_size = 32
    epochs = 10
    # 训练过程
    bilstm_crf_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)