import tensorflow as tf

def cnn(learning_rate=0.01):
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    model = tf.keras.applications.MobileNetV2((32, 32, 3), alpha=0.1, classes=10, weights=None)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss_function, metrics=["accuracy"])
    return model
