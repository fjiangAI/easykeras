__author__ = 'jf'
from keras.layers import Dense,multiply,Conv1D
from keras.layers import Input
from keras.models import Model
def gated_linear_unit(input_shape=None,kernel_layer="CNN",h_dim=None,c_filters=None,c_kernel_size=None, **kwargs):
    '''
    门控线性单元
    :param input_shape: 输入的维度，使用输入对象.shape即可
    :param kernel_layer: 使用的核心类型，可以使CNN也可以Dense
    :param h_dim: 使用Dense的隐藏层（输出维度）
    :param c_filters: 使用CNN的核心数（输出维度）
    :param c_kernel_size: 使用CNN的核的大小
    :param kwargs:
    :return: 返回子模型
    '''
    if kernel_layer=="CNN":
        sig_layer = Conv1D(c_filters,c_kernel_size,padding='same',strides=1,activation="sigmoid")
        original_layer=Conv1D(c_filters,c_kernel_size,padding='same',strides=1)
    else:
        sig_layer =Dense(h_dim,activation="sigmoid")
        original_layer=Dense(h_dim)
    text=Input(shape=(int(input_shape[1]),int(input_shape[2])),)
    #GLU(X)=Kernal_layer(X) element-wise product sigmoid(Kernal_layer(X))
    sig_text = sig_layer(text)
    original_text = original_layer(text)
    xo = multiply([sig_text,original_text])
    model=Model(inputs=[text], outputs=xo)
    return model