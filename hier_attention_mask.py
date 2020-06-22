from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec
import keras
import tensorflow as tf
class Attention_mask(Layer):

    def __init__(self,hidden, da, r, init='glorot_uniform', activation='tanh', W1_regularizer=None,
                 W2_regularizer=None, W1_constraint=None, W2_constraint=None, return_attention=False,
                 attention_regularizer_weight=0.0, normalize=False,attmod='softmax',sharp_beta=1,**kwargs):
        self.W_initializer = initializers.get(init)
        self.activation = activations.get(activation)
        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.W2_regularizer = regularizers.get(W2_regularizer)
        self.W1_constraint = constraints.get(W1_constraint)
        self.W2_constraint = constraints.get(W2_constraint)
        self.hidden = hidden
        self.da = da
        self.r = r
        self.return_attention = return_attention
        self.attention_regularizer_weight = attention_regularizer_weight
        self.normalize = normalize
        self.attmod = attmod
        self.sharp_beta = sharp_beta
        super(Attention_mask, self).__init__(**kwargs)

    def build(self, input_shape):
        #input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W1 = self.add_weight(shape=(self.hidden, self.da), name='W1', initializer=self.W_initializer,
                                  regularizer=self.W1_regularizer, constraint=self.W1_constraint)
        self.W2 = self.add_weight(shape=(self.da, self.r), name='W2', initializer=self.W_initializer,
                                  regularizer=self.W2_regularizer, constraint=self.W2_constraint)
        self.built = True

    def call(self, H, mask=None):
        # energy = self.activation(K.dot(x, self.W0)+self.b0)
        # energy=K.dot(energy, self.W) + self.b
        # energy = K.reshape(energy, (-1, self.input_length))
        # energy = K.softmax(energy)
        # xx = K.batch_dot(energy,x, axes=(1, 1))
        # all=K.concatenate([xx,energy])
        # return all
        #      H_t=K.permute_dimensions(H,(0,2,1))        #H is [none, n, hidden] ; H_t is [none, hidden, n]
        #      temp=self.activation(K.permute_dimensions(K.dot(self.W1,H_t),(1,0,2)))   #tanh(W1 . Ht) was [da, none, n],  transpose to [none, da, n]
        #      temp=K.permute_dimensions(K.dot(self.W2,temp),(1,0,2))          #W2 . tanh(W1 . Ht) was [r, none, n], transpose to [none, r, n]
        H1=H[:,:,:-1] #(none,2666,32)
        attention_mask=H[:,:,-1] #(none,2666,1) (none,n,1)
        #print("success in hier_attention call")
        H_t = self.activation(K.dot(H1, self.W1)) #same with unmasked
        temp = K.permute_dimensions(K.dot(H_t, self.W2), (0, 2, 1))  # [?,r,n] #same with unmasked
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0 #[none,n,1]
        #adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10
        #adder = (1.0 - tf.cast(attention_mask, tf.float32)) * tf.math.reduce_min(temp,axis=) #use the minimum value of unmasked places. different minimum value not good!
        temp +=K.repeat(adder,self.r) #[none,r,n]
        if 'softmax' in self.attmod:
             A = K.softmax(temp*self.sharp_beta)  # A    [none, r, n]
        elif 'smooth' in self.attmod:
            _at = keras.activations.sigmoid(temp*self.sharp_beta)
            print("_at shape is:")
            print(_at.get_shape())
            s = K.sum(_at, axis=2, keepdims=True)
            A = _at/s
        
        if self.normalize:
            length = K.sum(tf.cast(attention_mask, tf.float32),axis=1,keepdims=True)/attention_mask.get_shape()[1].value#[none,1]
            print("length in attention")
            print(length.get_shape())
            lengthr = K.repeat(length,self.r)
            print(lengthr.get_shape())
            A = A * lengthr
        
        M = K.batch_dot(A, H1, axes=(2, 1))  # [none, r, hidden]

        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(A))

        if self.return_attention:
            return [M, A]

        # all=K.concatenate([M,A])  #[none, r, hidden+n]
        return M

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.r, self.hidden)
        if self.return_attention:
            attention_shape = (input_shape[0], self.r, input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'W_initializer': initializers.serialize(self.W_initializer),
            'W1_regularizer': regularizers.serialize(self.W1_regularizer),
            'W2_regularizer': regularizers.serialize(self.W2_regularizer),
            'W1_constraint': constraints.serialize(self.W1_constraint),
            'W2_constraint': constraints.serialize(self.W2_constraint),
            'da':self.da,
            'r':self.r,
            'return_attention':self.return_attention,
            'attention_regularizer_weight':self.attention_regularizer_weight,
            'hidden':self.hidden
        }
        base_config = super(Attention_mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # AAT - I
    def _attention_regularizer(self, attention):
        # batch_size = K.cast(K.shape(attention)[0], K.floatx())
        # input_len = K.shape(attention)[-1]
        # indices = K.tile(K.expand_dims(K.arange(input_len), axis=0), [input_len, 1])
        # diagonal = K.expand_dims(K.arange(input_len), axis=-1)
        # eye = K.cast(K.equal(indices, diagonal), K.floatx())
        # return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
        #    attention,
        #    K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size
        # batch_size=K.eval(K.shape(attention)[0])
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        identity = K.eye(self.r)  # [r,r]
        temp = K.batch_dot(attention, K.permute_dimensions(attention, (0, 2, 1))) - identity  # [none, r, r]
        penal = self.attention_regularizer_weight * K.sum(K.square(temp)) / batch_size
        return penal

