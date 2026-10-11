def create_tf_dataset(x, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
