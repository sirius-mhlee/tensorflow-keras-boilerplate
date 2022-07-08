import math
import numpy

from tensorflow.keras.utils import Sequence

class DataSequence(Sequence):
    def __init__(self, data_x, data_y, batch_size, shuffle=False, func_preprocess=None):
        self.data_x = data_x
        self.data_y = data_y

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.func_preprocess = func_preprocess

        self.indices = numpy.arange(len(self.data_x))

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.data_x) / self.batch_size)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.data_x[batch_indices]
        batch_y = self.data_y[batch_indices]

        if self.func_preprocess is not None:
            batch_x, batch_y = self.func_preprocess(batch_x, batch_y)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle == True:
            numpy.random.shuffle(self.indices)
