__author__ = 'jf'
from keras.layers import Dense,multiply
def gated_layer(gated_dim,encoded_controller,encoded_text):
    '''
    普通门控单元
    :param gated_dim: 被控制的输出维度
    :param encoded_controller: 门控控制者输入
    :param encoded_text: 被控制的语义输入
    :return: 门控后的语义输入
    '''
    gated=Dense(gated_dim, activation='sigmoid')(encoded_controller)
    multi=multiply([encoded_text,gated])
    return multi