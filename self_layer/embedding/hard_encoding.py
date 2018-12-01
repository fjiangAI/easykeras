__author__ = 'jf'
from keras.layers import Embedding
def hard_encoding_layer(tokenizer,embedding_dim,trainable):
    '''
    硬编码层，可以应用于词性、结构等硬编码
    :param tokenizer:
    :param embedding_dim:
    :return:
    '''
    index = tokenizer.word_index
    num= len(index)+1
    embedding_layer = Embedding(num, embedding_dim,trainable=trainable)
    return embedding_layer