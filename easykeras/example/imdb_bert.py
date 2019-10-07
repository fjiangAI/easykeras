__author__ = 'jf'
from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数
from keras.layers import *
import numpy as np
import sys
sys.path.append("../../")
from easykeras.keras_bert.util import convert_text_to_index
from keras.optimizers import Adam
from easykeras.layers.embedding.bert_embedding import get_model_and_config
from easykeras.example.imdb_util import get_train_test
from easykeras.example.imdb_util import get_new_x
from easykeras.example.imdb_util import Config
import keras
# epoch 2: 87.87
# set parameters:
max_features = 5000


def get_config():
    return {"max_features": 5000,
            "class_num": 2,
            "activation": 'softmax',
            "model_path": "../pretrainmodel/cased_L-12_H-768_A-12"}


def convert_inputs(x, seq, tokenizer):
    """
    根据BERT的配置进行输入的转换
    :param x: 输入的序列
    :param seq: 序列长度（默认512）
    :param tokenizer: BERT的词和ID对应
    :return:
    """
    new_x = np.asarray(convert_text_to_index(seq, tokenizer, x)[0])
    return [np.asarray(new_x[0]), np.asarray(new_x[1])]


class BERTModel(keras.Model):
    def __init__(self, config):
        """
        初始化模型，准备需要的类
        :param config: 参数集合
        """
        super(BERTModel, self).__init__(name='bert_model')
        self.seq_len, self.tokenizer, self.bert_model = get_model_and_config(config.model_path, cased=False)
        # 设置BERT可训练
        for l in self.bert_model.layers:
            l.trainable = True
        self.dense_layer = Dense(config.class_num, activation=config.activation)

    def call(self, inputs):
        """
        构建模型过程,使用函数式模型
        :return:
        """
        inputs = convert_inputs(inputs, self.seq_len, self.tokenizer)
        bert_out = self.bert_model(inputs)
        bert2_out = Lambda(lambda x: x[:, 0])(bert_out)
        final_out = self.dense_layer(bert2_out)
        return final_out


if __name__ == "__main__":
    # 主程序
    model_config = Config(get_config())
    x_train, y_train, x_test, y_test = get_train_test(one_hot=True, num_classes=model_config.class_num)
    x_train = get_new_x(x_train)
    x_test = get_new_x(x_test)
    bert_model = BERTModel(model_config)

    # 设定编译参数
    loss = 'binary_crossentropy'
    optimizer = Adam(1e-5),
    metrics = ['accuracy']
    bert_model.summary()
    parallel_model = multi_gpu_model(bert_model, gpus=2)  # 设置使用2个gpu，该句放在模型compile之前
    parallel_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # 设置训练超参数
    batch_size = 32
    epochs = 10
    # 并行训练过程
    parallel_model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       shuffle=True,
                       validation_data=(x_test, y_test))
