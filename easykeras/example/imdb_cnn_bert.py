__author__ = 'jf'
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Input
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.models import Model
import sys
sys.path.append("../../")
from easykeras.layers.embedding.bert_embedding import bert_embedding_bymodel,getmodelandconfig
import numpy as np

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
def BERT_embedding(seq_len,tokenizer,model,x):
    x_train_args=bert_embedding_bymodel(seq_len,tokenizer,model,[x])
    x=np.asarray(x_train_args[0])
    return x
#epoch 10: 81.54
#set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Reconstruct data...')
x_train=getnewtrain(x_train)
x_test=getnewtrain(x_test)
print('Bert model load...')
modelpath="../pretrainmodel/cased_L-12_H-768_A-12"
seq_len,tokenizer,bert_model = getmodelandconfig(modelpath,cased=False)
x_train=BERT_embedding(seq_len,tokenizer,bert_model,x_train)
x_test=BERT_embedding(seq_len,tokenizer,bert_model,x_test)
print('Build model...')
inputs=Input(shape=(512,768))
dropout_layer=Dropout(0.2)
dropout_output=dropout_layer(inputs)
conv_layer=Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)
conv_output=conv_layer(dropout_output) #(?,510,250)
globalpooling_layer=GlobalMaxPooling1D()
globalpooling_output=globalpooling_layer(conv_output)#（？,250）
dense_layer2=Dense(hidden_dims)
dense_layer2_output=dense_layer2(globalpooling_output)# (?,250)
dropout_layer2_layer=Dropout(0.2)
dropout_layer2_output=dropout_layer2_layer(dense_layer2_output)
activation2_layer=Activation('relu')
activation2_output=activation2_layer(dropout_layer2_output)
finaldense_layer=Dense(1)
finaldense_out=finaldense_layer(activation2_output)#(?,1)
finalactivation_layer=Activation('sigmoid')
finalactivation_out=finalactivation_layer(finaldense_out)#(?,1)
model= Model(inputs=[inputs], outputs=finalactivation_out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))