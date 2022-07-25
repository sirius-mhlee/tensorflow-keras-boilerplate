from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from Layer.SimpleDense import *

class SimpleModel():
    input_shape = (32, 32, 3)
    num_classes = 10

    def __getattr__(self, name):
        return getattr(self.model, name)
        
    def __init__(self):
        inputs = Input(shape=SimpleModel.input_shape)

        x = Conv2D(64, 3, activation='relu')(inputs)
        x = Conv2D(64, 3, activation='relu')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, 3, activation='relu')(x)
        x = MaxPooling2D()(x)
        
        x = Flatten()(x)

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = SimpleDense(512, activation='relu')(x)

        outputs = Dense(SimpleModel.num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
