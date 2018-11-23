__author__ = 'jf'
from keras.engine import Layer
from keras import backend as K
from keras import initializers,regularizers,constraints

class Attention(Layer):
    """
    Soft Attention

    Input:
    embedding后的文本序列

    Output:
    如果选择返回alpha，则返回权重值(samples,word_num,1)
    否则返回求和值(samples,wordvec_dim)

    Example:
    attention_encoder=Attention()
    attention=attention_encoder(text1_seq)
    """
    def __init__(self, return_sequence=False, return_alpha=False,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=False, **kwargs):
        self.return_sequences = False
        self.return_alpha = return_alpha
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[-1]

    def call(self, x, mask=None):
        M = K.tanh(x)
        alpha = K.softmax (K.squeeze(K.dot(M, K.expand_dims(self.W)),axis=-1))

        if self.return_alpha:
            return alpha
        else:
            #return K.tanh(K.dot(alpha,x))[0]
            return K.dot(alpha,x)[0]

    def compute_mask(self, inputs, masks=None):
        return None