import numpy

from tensorflow import distribute

from Util.Print import *
from Model.SimpleModel import *
import Config

# Define Model using Multi GPU
mirrored_strategy = distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = SimpleModel(input_shape=Config.input_shape, num_classes=Config.num_classes)

# Print Model Summary
print_model_summary(model)

# Load Output Model Weight
model.load_weights('./output_models/{}'.format(Config.predict_model_name))

# Predict Data
input = numpy.random.rand(1, 32, 32, 3)
output = model.predict(input)

print(output)
