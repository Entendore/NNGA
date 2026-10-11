from tensorflow.keras.models import load_model

def save_model(model, path):
    model.save(path)

def load_trained_model(path):
    return load_model(path)
