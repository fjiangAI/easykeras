__author__ = 'jf'
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
import sys
sys.path.append("../../")
from easykeras.layers.embedding.bert_embedding import bert_embedding_by_model, get_model_and_config
import numpy as np
import keras
from easykeras.example.imdb_util import get_train_test
from easykeras.example.imdb_util import get_new_x
from easykeras.example.imdb_util import Config


def BERT_embedding(seq_len, tokenizer, model, x):
    x_train_args = bert_embedding_by_model(seq_len, tokenizer, model, [x])
    x = np.asarray(x_train_args[0])
    return x


def get_config():
    return {"max_features": 5000,
            "max_len": 400,
            "filters": 250,
            "kernel_size": 3,
            "hidden_dims": 100,
            "class_num": 1,
            "activation": 'sigmoid',
            "model_path": "../pretrainmodel/cased_L-12_H-768_A-12"}


class CnnBERTModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(CnnBERTModel, self).__init__(name='cnnbert_model')
        self.seq_len, self.tokenizer, self.bert_model = get_model_and_config(config.model_path, cased=False)
        self.dropout_layer = Dropout(0.2)
        self.conv_layer = Conv1D(self.filters, self.kernel_size,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)
        self.global_pooling_layer = GlobalMaxPooling1D()
        self.dense_layer = Dense(config.hidden_dims)
        self.dropout_layer2 = Dropout(0.2)
        self.activation_layer = Activation('relu')
        self.final_dense_layer = Dense(config.class_num)
        self.final_activation_layer = Activation(config.activation)

    def call(self, inputs):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        embedding_output = BERT_embedding(self.seq_len, self.tokenizer, self.bert_model, inputs)
        dropout_output = self.dropout_layer(embedding_output)
        conv_output = self.conv_layer(dropout_output)  # (?,510,250)
        global_pooling_output = self.globalpooling_layer(conv_output)  # （？,250）
        dense_layer_output = self.dense_layer(global_pooling_output)  # (?,250)
        dropout_layer2_output = self.dropout_layer2(dense_layer_output)
        activation_output = self.activation_layer(dropout_layer2_output)
        final_dense_out = self.final_dense_layer(activation_output)  # (?,1)
        final_activation_out = self.final_activation_layer(final_dense_out)
        return final_activation_out


if __name__ == "__main__":
    # 主程序
    x_train, y_train, x_test, y_test = get_train_test()
    x_train = get_new_x(x_train)
    x_test = get_new_x(x_test)
    model_config = Config(get_config())
    cnnbert_model = CnnBERTModel(model_config)

    # 设定编译参数
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    cnnbert_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # 设置训练超参数
    batch_size = 32
    epochs = 10
    # 训练过程
    cnnbert_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))
