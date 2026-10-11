from tensorflow.keras.datasets import mnist
import numpy as np
from tfutils.cnn import CNNModel
from tfutils.data import create_dataset
from tfutils.train import CustomTrainer

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

model = CNNModel(input_shape=(28, 28, 1), num_classes=10)

# Use built-in Keras training:
model.train(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Or custom training loop:
# ds = create_dataset(x_train, y_train, batch_size=64)
# trainer = CustomTrainer(model.model)
# trainer.train(ds, epochs=5)
