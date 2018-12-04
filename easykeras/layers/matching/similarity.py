__author__ = 'jf'
from keras.layers import dot
def cos_similarity(vector1,vector2):
    '''
    余弦相似度计算
    :param vector1: 向量1
    :param vector2: 向量2
    :return: 余弦相似度（数值）
    '''
    cos = dot([vector1, vector2], axes=-1, normalize=True)
    return cos