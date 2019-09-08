__author__ = 'jf'
from keras.layers import Dense,Dropout,Input
from keras.models import Model
def double_dense_layer(input_shape=None,hidden_size=100,hidden_activation="relu",output_size=3,output_activation="softmax",loss=0.6):
    '''
    双层dense，一般可用于最终的分类
    :param input_shape: 输入向量的维度
    :param hidden_size: 隐藏dense单元数
    :param hidden_activation: 隐藏层激活函数
    :param output_size: 输出dense单元数
    :param output_activation:  输出层激活函数
    :param loss:  遗忘概率
    :return: 预测结果
    '''
    text=Input(shape=(int(input_shape[1]),),)
    ##全连接
    dense = Dense(hidden_size, activation=hidden_activation)(text)
    ##随机损失
    output = Dropout(loss)(dense)
    #最终预测结果
    preds = Dense(output_size, activation=output_activation)(output)
    model=Model(inputs=[text], outputs=preds)
    return model