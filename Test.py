from datetime import datetime

from tensorflow import distribute
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import SGD

from Util.DataSequence import *
from Util.Print import *
from Model.SimpleModel import *
from Metric.CategoricalAccuracy import *
from Loss.CategoricalCrossentropy import *
import Config

# Load Data
import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10
(_, _), (test_x, test_y) = cifar10.load_data()

def func_preprocess(x, y):
    return (x / 255.0), to_categorical(y, Config.num_classes)

# Prepare Generator
test_generator = DataSequence(test_x, test_y, batch_size=Config.test_batch_size, func_preprocess=func_preprocess)

# Define Model using Multi GPU
mirrored_strategy = distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = SimpleModel(input_shape=Config.input_shape, num_classes=Config.num_classes)
    model.compile(optimizer=SGD(learning_rate=Config.learning_rate), loss=CategoricalCrossentropy, metrics=[CategoricalAccuracy])

# Print Model Summary
print_model_summary(model)

# Load Checkpoint Weight
model.load_weights('./checkpoints/{}'.format(Config.test_checkpoint_name))

# Test Model
model.evaluate(x=test_generator, steps=len(test_generator),
    max_queue_size=Config.max_queue_size, workers=Config.workers)

# Save Output Model Weight and Plot Image
# - need package install command : pip install pydot, apt install graphviz
output_name = './output_models/{}'.format(datetime.now().astimezone().strftime('%Y%m%d_%H%M%S'))
model.save_weights(output_name)
plot_model(model, to_file='{}.jpg'.format(output_name), show_shapes=True)
