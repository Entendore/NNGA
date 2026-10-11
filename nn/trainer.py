import tensorflow as tf

class CustomTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None):
        self.model = model
        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        self.loss_fn = loss_fn or tf.keras.losses.SparseCategoricalCrossentropy()
        self.metrics = metrics or [tf.keras.metrics.SparseCategoricalAccuracy()]

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        for metric in self.metrics:
            metric.update_state(y, predictions)
        return loss

    def train(self, dataset, epochs=5):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            for metric in self.metrics:
                metric.reset_states()
            for step, (x_batch, y_batch) in enumerate(dataset):
                loss = self.train_step(x_batch, y_batch)
                if step % 100 == 0:
                    acc = self.metrics[0].result().numpy()
                    print(f"Step {step}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            print(f"End of Epoch {epoch+1} - Accuracy: {self.metrics[0].result().numpy():.4f}")
