import tensorflow
from tensorflow.keras import activations
from tensorflow.keras.layers import *

class SimpleDense(Layer):
    def __init__(self, units, activation):
        super(SimpleDense, self).__init__()

        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='bias', shape=(self.units), initializer='zeros', trainable=True)

    def call(self, inputs):
        return self.activation(tensorflow.matmul(inputs, self.w) + self.b)

    def get_config(self):
        return super(SimpleDense, self).get_config()
