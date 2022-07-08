from tensorflow.keras.metrics import categorical_accuracy

def CategoricalAccuracy(y_true, y_pred):
    return categorical_accuracy(y_true, y_pred)
