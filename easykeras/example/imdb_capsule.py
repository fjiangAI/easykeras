__author__ = 'jf'
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Model
import sys
sys.path.append("../../")
from easykeras.layers.capsule import Capsule
from keras.layers import *
#epoch 10: 86.29
# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
num_classes=2
from keras import utils
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


#Capsule分类是类似Softmax一样，需要先离散化。
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
inputs=Input(shape=(400,))
embedding_layer=Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)
embedding_output=embedding_layer(inputs) #(?,400,50)
dropout_layer=Dropout(0.2)
dropout_output=dropout_layer(embedding_output)
conv_layer=Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)
conv_output=conv_layer(dropout_output) #(?,398,250)
print("conv:")
print(conv_output.shape)
capsule = Capsule(2, 8, 3, True)(conv_output) # (?,2,32)
print("capsule:")
print(capsule.shape)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule) #(?,2)
print("output:")
print(output.shape)
model= Model(inputs=[inputs], outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
