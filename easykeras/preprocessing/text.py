#coding:utf-8

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class TextProcessor():
    """文本预处理

    将文本（或词语序列）转换为编号序列

    Args:
        max_word_num: 保留出现频率最高前 `max_word_num` 个词
                      为None，则全部保留
    
    """

    def __init__(self, max_word_num=None):
        """初始化文本预处理类

        Attributes:
            tokenizer: Keras自带的文本处理工具类
            has_dic: 是否已经生成词表
        """
        self.tokenizer = Tokenizer(num_words=max_word_num)
        self.has_dic = False

    def get_vocab(self):
        """获取词表

        Returns:
            词表字典，将词语映射到词语编号。例如:

            {'I': 54,
            'love': 23,
            'Keras': 6}

            词语编号从1开始
        """
        if (not self.has_dic):
            print('请先调用 read_all_texts() 函数生成词表!')
            return None
        return self.tokenizer.word_index

    def get_vocab_size(self):
        """获取词表大小

        Returns:
            词表中的词语数量
        """
        if (not self.has_dic):
            print('请先调用 read_all_texts() 函数生成词表!')
            return 0
        return len(self.tokenizer.word_index)

    def read_all_texts(self, *texts):
        """浏览所有的文本

        对所有样例文本中的词语进行编号，建立词表

        Args:
            texts: 一个或多个文本列表，
                   每个元素可以是用空格分开的原始文本，
                   也可以是词语列表

        Returns:
            词表字典，将词语映射到词语编号。例如:

            {'I': 54,
            'love': 23,
            'Keras': 6}

            词语编号从1开始
        """
        list = []
        for text in texts:
            list.extend(text)
        if (len(list) == 0):
            print('传入文本为空!')
            return None
        self.tokenizer.fit_on_texts(list)
        self.has_dic = True
        return self.tokenizer.word_index
    
    def _process_text(self, texts, length):
        # 词语字符串->词语编号列表
        sequences = self.tokenizer.texts_to_sequences(texts)
        # 统一长度
        return pad_sequences(sequences, maxlen=length)

    def _one_hot(self, samples, length):
        result = []
        for sample in samples:
            if (isinstance(sample, str)):
                sample = sample.split()
            # 固定序列长度，不足在前面补空串
            new_sample = ['' for i in range(length)]
            for i, word in enumerate(sample[:length]):
                new_sample[length-i-1] = word
            word_mat = [[word] for word in new_sample]
            result.append(self.tokenizer.texts_to_matrix(word_mat, mode='binary'))
        return np.asarray(result)

    def texts_to_num(self, length, *texts):
        """文本转换为数字编号序列

        将文本中的词语映射为对应的词语编号

        Args:
            length: 转换后的序列长度(词语数量)
                    超过截断，不足补零
            texts: 一个或多个文本列表，
                   每个元素可以是用空格分开的原始文本，
                   也可以是词语列表

        Returns:
            一个或多个转换后的 `(sample_num, length)` 二维矩阵。
            例如：

            ['a b c', 'a b', 'd'] -> [[2 1 3]
                                      [0 2 1]
                                      [0 0 4]]

            数字编号从1开始，不足在开头补0.
        """
        if (not self.has_dic):
            print('请先调用 read_all_texts() 函数生成词表!')
            if (len(texts) == 1):
                return None
            return tuple([None for x in range(len(texts))])
        if (len(texts) == 1):
            return self._process_text(texts[0], length)
        return tuple([self._process_text(x, length) for x in texts])
    
    def texts_to_bow(self, *texts):
        """文本转换为词袋编号序列

        将文本转换为词袋向量，长度为词表
        向量每一维对应一个词语，文本中出现为1，未出现为0

        Args:
            texts: 一个或多个文本列表，
                   每个元素可以是用空格分开的原始文本，
                   也可以是词语列表

        Returns:
            一个或多个转换后的 `(sample_num, vocab_size)` 二维矩阵。
            例如：

            ['a b c', 'a b', 'd'] -> [[0. 1. 1. 1. 0. 0.]
                                      [0. 1. 1. 0. 0. 0.]
                                      [0. 0. 0. 0. 1. 0.]]

            每一维对应一个词语，文本中出现为1，未出现为0
        """
        if (not self.has_dic):
            print('请先调用 read_all_texts() 函数生成词表!')
            if (len(texts) == 1):
                return None
            return tuple([None for x in range(len(texts))])
        if (len(texts) == 1):
            return self.tokenizer.texts_to_matrix(texts[0], mode='binary')
        return tuple([self.tokenizer.texts_to_matrix(x, mode='binary') for x in texts])
    
    def texts_to_one_hot(self, length, *texts):
        """文本转换为one-hot编号序列

        将文本转换为one-hot序列

        Args:
            length: 序列长度
            texts: 一个或多个文本列表，
                   每个元素可以是用空格分开的原始文本，
                   也可以是词语列表

        Returns:
            一个或多个转换后的 `(sample_num, length, vocab_size)` 三维矩阵。
            例如：

            ['a b c', 'a b', 'd'] -> [[[0. 0. 0. 1. 0.]
                                       [0. 0. 1. 0. 0.]
                                       [0. 1. 0. 0. 0.]]

                                      [[0. 0. 0. 0. 0.]
                                       [0. 0. 1. 0. 0.]
                                       [0. 1. 0. 0. 0.]]

                                      [[0. 0. 0. 0. 0.]
                                       [0. 0. 0. 0. 0.]
                                       [0. 0. 0. 0. 1.]]]
        """
        if (not self.has_dic):
            print('请先调用 read_all_texts() 函数生成词表!')
            if (len(texts) == 1):
                return None
            return tuple([None for x in range(len(texts))])
        if (len(texts) == 1):
            return self._one_hot(texts[0], length)
        return tuple([self._one_hot(x, length) for x in texts])

if __name__ == "__main__":
    texts_1 = ['中国 的 首都 是 北京', '北京 天安门', '中国']
    texts_2 = ['我 在 中国', '北京 是 中国 的 首都']
    # texts_1 = [['中国', '的', '首都', '是', '北京'], ['北京', '天安门'], ['中国']]
    # texts_2 = [['我', '在', '中国'], ['北京', '是', '中国', '的', '首都']]
    print('texts1:', texts_1)
    print('texts2:', texts_2)

    processor = TextProcessor() # 文本预处理器
    # 读取文本，生成词表
    processor.read_all_texts(texts_1, texts_2)
    print('词表大小:', processor.get_vocab_size())
    print('词表:', processor.get_vocab())
    # 文本转换为数字编号序列
    print('转换为数字序列(长度4)：')
    texts_1_num, texts_2_num = processor.texts_to_num(4, texts_1, texts_2)
    print('texts1:\n', texts_1_num)
    print('texts2:\n', texts_2_num)
    # 转换为词袋编号序列
    print('转换为词袋编号序列：')
    texts_1_bow, texts_2_bow = processor.texts_to_bow(texts_1, texts_2)
    print('texts1:\n', texts_1_bow)
    print('texts2:\n', texts_2_bow)
    # 转换为one-hot序列
    print('转换为one-hot序列(长度4)：')
    texts_1_one_hot, texts_2_one_hot = processor.texts_to_one_hot(4, texts_1, texts_2)
    print('texts1:\n', texts_1_one_hot)
    print('texts2:\n', texts_2_one_hot)
    