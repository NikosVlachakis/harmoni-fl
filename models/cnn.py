import tensorflow as tf

def cnn():
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    model = tf.keras.applications.MobileNetV2((32, 32, 3), alpha=0.1, classes=10, weights=None)
    model.compile("adam", loss_function, metrics=["accuracy"])
    return model
