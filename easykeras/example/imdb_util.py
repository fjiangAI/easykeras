from keras.datasets import imdb
from keras.preprocessing import sequence


def get_train_test(max_features=5000, max_len=400):
    """
    获取标准IMDB数据集的训练和测试集（与Keras样例一致）
    :param max_features: 默认5000
    :param max_len: 默认400
    :return: 训练样本，训练集标签，测试样本，测试标签
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    return x_train, y_train, x_test, y_test
