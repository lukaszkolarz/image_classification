import tensorflow as tf


def import_greyscale(target_size, batch_size, source_train, source_val, source_test):
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                           zoom_range=0.0,
                                                                           rotation_range=0,
                                                                           vertical_flip=False)
    train_ds = train_data_generator.flow_from_directory(source_train,
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        color_mode='grayscale',
                                                        target_size=(target_size, target_size),
                                                        shuffle=True)

    test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_ds = test_data_generator.flow_from_directory(source_val,
                                                     class_mode='categorical',
                                                     batch_size=batch_size,
                                                     color_mode='grayscale',
                                                     target_size=(target_size, target_size),
                                                     shuffle=True)
    test_ds = test_data_generator.flow_from_directory(source_test,
                                                      class_mode='categorical',
                                                      batch_size=batch_size,
                                                      color_mode='grayscale',
                                                      target_size=(target_size, target_size),
                                                      shuffle=True)

    input_shape = (target_size, target_size, 1)

    return train_ds, val_ds, test_ds, input_shape
