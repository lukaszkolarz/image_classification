import tensorflow as tf


def SEResNet50(input_shape, classes):
    input_shape = tf.keras.layers.Input(input_shape)

    # STAGE 1
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(input_shape)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # STAGE 2
    x = convolutional_block(x, kernel_size=(3, 3), filters=[64, 64, 256], strides=(1, 1))
    x = identity_block(x, filters=[64, 64, 256], kernel_size=(3, 3))
    x = identity_block(x, filters=[64, 64, 256], kernel_size=(3, 3))

    # STAGE 3
    x = convolutional_block(x, kernel_size=(3, 3), filters=[128, 128, 512], strides=(2, 2))
    x = identity_block(x, filters=[128, 128, 512], kernel_size=(3, 3))
    x = identity_block(x, filters=[128, 128, 512], kernel_size=(3, 3))
    x = identity_block(x, filters=[128, 128, 512], kernel_size=(3, 3))

    # STAGE 4
    x = convolutional_block(x, kernel_size=(3, 3), filters=[256, 256, 1024], strides=(2, 2))
    x = identity_block(x, filters=[256, 256, 1024], kernel_size=(3, 3))
    x = identity_block(x, filters=[256, 256, 1024], kernel_size=(3, 3))
    x = identity_block(x, filters=[256, 256, 1024], kernel_size=(3, 3))
    x = identity_block(x, filters=[256, 256, 1024], kernel_size=(3, 3))
    x = identity_block(x, filters=[256, 256, 1024], kernel_size=(3, 3))

    # STAGE 5
    x = convolutional_block(x, kernel_size=(3, 3), filters=[512, 512, 2048], strides=(2, 2))
    x = identity_block(x, filters=[512, 512, 2048], kernel_size=(3, 3))
    x = identity_block(x, filters=[512, 512, 2048], kernel_size=(3, 3))

    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=classes,
                              activation='softmax',
                              kernel_initializer=tf.initializers.glorot_uniform(seed=0))(x)
    return tf.keras.models.Model(inputs=input_shape, outputs=x, name='SEResNet50')


def identity_block(x, filters, kernel_size, reduction_ratio=16):
    f1, f2, f3 = filters

    shortcut = x

    x = tf.keras.layers.Conv2D(filters=f1,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=f2,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=f3,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = SE_bypass(x, f3, reduction_ratio)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def convolutional_block(x, filters, kernel_size, strides, reduction_ratio=16):

    f1, f2, f3 = filters

    shortcut_conv = x

    shortcut_conv = tf.keras.layers.Conv2D(filters=f3,
                                           kernel_size=(1, 1),
                                           strides=strides,
                                           padding='valid',
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(
        shortcut_conv)
    shortcut_conv = tf.keras.layers.BatchNormalization(axis=3)(shortcut_conv)

    x = tf.keras.layers.Conv2D(filters=f1,
                               kernel_size=(1, 1),
                               strides=strides,
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=f2,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=f3,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = SE_bypass(x, f3, reduction_ratio)
    x = tf.keras.layers.Add()([x, shortcut_conv])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def SE_bypass(x, filters, ratio):
    scale = x

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(filters // ratio)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.multiply([x, scale])

    return x
