from tensorflow.keras.metrics import categorical_crossentropy

def CategoricalCrossentropy(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)
