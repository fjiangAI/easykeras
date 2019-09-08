__author__ = 'jf'
from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数
from keras.layers import *
from keras.datasets import imdb
from keras.models import Model
import numpy as np
import sys
sys.path.append("../../")
from easykeras.keras_bert.util import convert_text_to_index
from keras.optimizers import Adam
from keras import utils
from easykeras.layers.embedding.bert_embedding import getmodelandconfig
#epoch 2: 87.87
#set parameters:
max_features = 5000
batch_size = 8
epochs = 2
num_classes=2

def converntinputs(x,seq,tokenizer):
    '''
    根据BERT的配置进行输入的转换
    :param x: 输入的序列
    :param seq: 序列长度（默认512）
    :param tokenizer: BERT的词和ID对应
    :return:
    '''
    newx=np.asarray(convert_text_to_index(seq,tokenizer,x)[0])
    return [np.asarray(newx[0]),np.asarray(newx[1])]
def getnewtrain(x_train):
    '''
    复原IDMB语料库的语料内容
    :param x_train: 原有的词ID
    :return:
    '''
    wordIndex=imdb.get_word_index()
    new_dict = {v : k for k, v in wordIndex.items()}
    print(len(x_train))
    new_x_train=[]
    for i in range(0,len(x_train)):
        for j in range(0,len(x_train[i])):
            x_train[i][j]=new_dict[x_train[i][j]]
        new_x_train.append(" ".join(x_train[i]))
    new_x_train=np.asarray(new_x_train)
    return new_x_train
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train=getnewtrain(x_train)
x_test=getnewtrain(x_test)

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

print('Build model...')
modelpath="../pretrainmodel/cased_L-12_H-768_A-12"
seq_len,tokenizer,bert_model = getmodelandconfig(modelpath,cased=False)
x_train=converntinputs(x_train,seq_len,tokenizer)
x_test=converntinputs(x_test,seq_len,tokenizer)
#设置BERT可训练
for l in bert_model.layers:
    l.trainable = True
#模型的主体部分
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in,x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(num_classes, activation='softmax')(x)
model = Model([x1_in,x2_in], p)
parallel_model = multi_gpu_model(model, gpus=2) # 设置使用2个gpu，该句放在模型compile之前
parallel_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5), # 用足够小的学习率
        metrics=['accuracy']
    )
model.summary()
history=parallel_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              validation_data=(x_test, y_test))
