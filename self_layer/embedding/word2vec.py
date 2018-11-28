__author__ = 'jf'
from gensim.models import KeyedVectors
from keras.layers import Embedding
import numpy as np
def w2v_encoding_layer(tokenizer,embeddings_dic,embedding_dim):
    '''
    使用已有的word2vec模型来获得embedding层
    :param tokenizer: 所有词的tokenizer
    :param embeddings_dic: word2vec模型加载后词典
    :param embedding_dim: 词向量的维度
    :return: embedding层
    '''
    tokenizer = tokenizer
    #word_index形式为"词：序号";
    word_index = tokenizer.word_index
    #样本中词的数目+1
    num_words = len(word_index)+1
    #创建映射矩阵，行为词的个数，列为词向量长度。
    embedding_matrix = np.zeros((num_words, embedding_dim))
    #对于样本中的每一个词及编号
    for word, i in word_index.items():
        #如果有词向量
        if word in embeddings_dic:
            embedding_matrix[i] = embeddings_dic[word]
    ## word embedding
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False)
    return embedding_layer