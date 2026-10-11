from tensorflow.keras import layers, models

class CNNModel:
    def __init__(self, input_shape, num_classes):
        """Simple CNN for image classification."""
        self.model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x, y, **kwargs): return self.model.fit(x, y, **kwargs)
    def evaluate(self, x, y): return self.model.evaluate(x, y)
    def predict(self, x): return self.model.predict(x)
