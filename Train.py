from datetime import datetime

from tensorflow import distribute
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from Util.DataSequence import *
from Util.Print import *
from Model.SimpleModel import *
from Metric.CategoricalAccuracy import *
from Loss.CategoricalCrossentropy import *
import Config

# Load Data
import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10
(train_x, train_y), (validation_x, validation_y) = cifar10.load_data()

def func_preprocess(x, y):
    return (x / 255.0), to_categorical(y, Config.num_classes)

# Prepare Generator
train_generator = DataSequence(train_x, train_y, batch_size=Config.train_batch_size, shuffle=True, func_preprocess=func_preprocess)
validation_generator = DataSequence(validation_x, validation_y, batch_size=Config.validation_batch_size, func_preprocess=func_preprocess)

# Define Model using Multi GPU
mirrored_strategy = distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = SimpleModel(input_shape=Config.input_shape, num_classes=Config.num_classes)
    model.compile(optimizer=SGD(learning_rate=Config.learning_rate), loss=CategoricalCrossentropy, metrics=[CategoricalAccuracy])

# Print Model Summary
print_model_summary(model)

# Define Callback
callbacks=[TensorBoard('./logs/{}'.format(datetime.now().astimezone().strftime('%Y%m%d_%H%M%S'))),
    ModelCheckpoint(filepath='./checkpoints/{epoch:02d}', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, save_freq='epoch')]

# Train Model
model.fit(epochs=Config.train_epochs,
    x=train_generator, steps_per_epoch=len(train_generator),
    validation_data=validation_generator, validation_steps=len(validation_generator),
    callbacks=callbacks,
    max_queue_size=Config.max_queue_size, workers=Config.workers)
