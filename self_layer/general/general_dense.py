__author__ = 'jf'
from keras.layers import Dense,Dropout
def double_dense(input,hidden_size,hidden_activation="relu",output_size=3,output_activation="softmax",loss=0.6):
    '''
    双层dense，一般可用于最终的分类
    :param input: 输入向量
    :param hidden_size: 隐藏dense单元数
    :param hidden_activation: 隐藏层激活函数
    :param output_size: 输出dense单元数
    :param output_activation:  输出层激活函数
    :param loss:  遗忘概率
    :return: 预测结果
    '''
    ##全连接
    dense = Dense(hidden_size, activation=hidden_activation)(input)
    ##随机损失
    output = Dropout(loss)(dense)
    #最终预测结果，3分类
    preds = Dense(output_size, activation=output_activation)(output)
    return preds