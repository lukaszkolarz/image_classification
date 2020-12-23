import tensorflow as tf


def keras_loader(target_size, batch_size, source_train, source_val):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory("/net/people/plgkolarzl/dataset_clear",
                                                                   validation_split=0.1,
                                                                   color_mode='grayscale',
                                                                   label_mode='int',
                                                                   subset="training",
                                                                   seed=123,
                                                                   image_size=(target_size, target_size),
                                                                   batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory("/net/people/plgkolarzl/test",
                                                                 validation_split=0.9,
                                                                 color_mode='grayscale',
                                                                 label_mode='int',
                                                                 subset="validation",
                                                                 seed=123,
                                                                 image_size=(target_size, target_size),
                                                                 batch_size=batch_size)
    return train_ds, val_ds
