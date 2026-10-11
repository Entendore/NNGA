from tensorflow.keras import layers, models

class RNNModel:
    def __init__(self, input_shape, num_classes, rnn_units=64):
        """Simple RNN for sequence classification."""
        self.model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.SimpleRNN(rnn_units),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x, y, **kwargs): return self.model.fit(x, y, **kwargs)
    def evaluate(self, x, y): return self.model.evaluate(x, y)
    def predict(self, x): return self.model.predict(x)
