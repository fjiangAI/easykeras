__author__ = 'jf'
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Input
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.models import Model
from easykeras.layers.gated.gated_layer import GatedLinearUnit
# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
inputs= Input(shape=(400,))
embedding_layer=Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen)
embedding_output=embedding_layer(inputs)
gcnn_layer=GatedLinearUnit(input_shape=embedding_output.shape,c_kernel_size=3,c_filters=250,h_dim=100,kernel_layer="CNN")
gcnn_output=gcnn_layer(embedding_output)
globalpooling_layer=GlobalMaxPooling1D()
globalpooling_output=globalpooling_layer(gcnn_output)
dense_layer2=Dense(hidden_dims)
dense_layer2_output=dense_layer2(globalpooling_output)
dropout_layer2_layer=Dropout(0.2)
dropout_layer2_output=dropout_layer2_layer(dense_layer2_output)
activation2_layer=Activation('relu')
activation2_output=activation2_layer(dropout_layer2_output)
finaldense_layer=Dense(1)
finaldense_out=finaldense_layer(activation2_output)
finalactivation_layer=Activation('sigmoid')
finalactivation_out=finalactivation_layer(finaldense_out)
model= Model(inputs=[inputs], outputs=finalactivation_out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
