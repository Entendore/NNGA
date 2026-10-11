# tf_helper.py
import tensorflow as tf
from tensorflow.keras import layers, models

class TFModel:
    def __init__(self, input_shape, num_classes, hidden_units=[64, 64], activation='relu'):
        """
        Create a simple feedforward neural network model.
        Args:
            input_shape (tuple): Shape of input features, e.g. (784,)
            num_classes (int): Number of output classes
            hidden_units (list): List of units in each hidden layer
            activation (str): Activation function for hidden layers
        """
        self.model = models.Sequential()
        self.model.add(layers.InputLayer(input_shape=input_shape))

        for units in hidden_units:
            self.model.add(layers.Dense(units, activation=activation))

        if num_classes == 1:
            # Regression or binary classification with sigmoid
            self.model.add(layers.Dense(1, activation='sigmoid'))
        else:
            # Multi-class classification with softmax
            self.model.add(layers.Dense(num_classes, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        """
        Train the model.
        Args:
            x_train (np.array): Training features
            y_train (np.array): Training labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            validation_data (tuple): (x_val, y_val) for validation
        Returns:
            History object from model.fit()
        """
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=validation_data)
        return history

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on test data.
        Args:
            x_test (np.array): Test features
            y_test (np.array): Test labels
        Returns:
            Loss and metrics values
        """
        results = self.model.evaluate(x_test, y_test)
        return results

    def predict(self, x):
        """
        Predict using the trained model.
        Args:
            x (np.array): Input data
        Returns:
            Predictions as numpy array
        """
        preds = self.model.predict(x)
        return preds

    def save(self, filepath):
        """
        Save the model to a file.
        Args:
            filepath (str): File path to save model
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Load model from a file.
        Args:
            filepath (str): File path to load model
        """
        self.model = models.load_model(filepath)
