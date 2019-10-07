from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import utils
import numpy as np


def get_train_test(max_features=5000, max_len=400, one_hot=False, num_classes=None):
    """
    获取标准IMDB数据集的训练和测试集（与Keras样例一致）
    :param max_features: 默认5000
    :param max_len: 默认400
    :param one_hot: 标签是否离散化(softmax分类都需要)
    :param num_classes: 如果标签离散化，则需要指明标签类别数
    :return: 训练样本，训练集标签，测试样本，测试标签
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    if one_hot is True:
        y_train = utils.to_categorical(y_train, num_classes)
        y_test = utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def get_new_x(x):
    """
    复原IMDB语料库的语料内容
    :param x: 原有的词ID
    :return:
    """
    new_dict = {v: k for k, v in imdb.get_word_index()}
    new_x = []
    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            x[i][j] = new_dict[x[i][j]]
        new_x.append(" ".join(x[i]))
    new_x = np.asarray(new_x)
    return new_x


class Config:
    def __init__(self, config={}):
        for key, value in config.items():
            setattr(self, key, value)
