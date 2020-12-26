import tensorflow as tf


def ResNet34(input_shape, classes):
    input_shape = tf.keras.layers.Input(input_shape)
    # stage1
    x = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(input_shape)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # stage2
    x = convolutional_block(x, filters=[64, 64], strides=1)
    x = identity_block(x, filters=[64, 64], kernel_size=3)
    x = identity_block(x, filters=[64, 64], kernel_size=3)

    # stage3
    x = convolutional_block(x, filters=[128, 128], strides=2)
    x = identity_block(x, filters=[128, 128], kernel_size=3)
    x = identity_block(x, filters=[128, 128], kernel_size=3)
    x = identity_block(x, filters=[128, 128], kernel_size=3)

    # stage4
    x = convolutional_block(x, filters=[256, 256], strides=2)
    x = identity_block(x, filters=[256, 256], kernel_size=3)
    x = identity_block(x, filters=[256, 256], kernel_size=3)
    x = identity_block(x, filters=[256, 256], kernel_size=3)
    x = identity_block(x, filters=[256, 256], kernel_size=3)
    x = identity_block(x, filters=[256, 256], kernel_size=3)

    # stage5
    x = convolutional_block(x, filters=[512, 512], strides=2)
    x = identity_block(x, filters=[512, 512], kernel_size=3)
    x = identity_block(x, filters=[512, 512], kernel_size=3)

    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes, activation='softmax',
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
                              kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(x)
    ResNet32Model = tf.keras.models.Model(inputs=input_shape, outputs=x, name='ResNet34')

    return ResNet32Model


def convolutional_block(x, filters, strides):
    f1, f2 = filters
    kernel_size = (1, 1)

    shortcut_conv = x

    shortcut_conv = tf.keras.layers.Conv2D(filters=f2,
                                           kernel_size=kernel_size,
                                           strides=(strides, strides),
                                           padding='valid',
                                           kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(
        shortcut_conv)
    shortcut_conv = tf.keras.layers.BatchNormalization(axis=3)(shortcut_conv)

    x = tf.keras.layers.Conv2D(filters=f1,
                               kernel_size=kernel_size,
                               strides=(strides, strides),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=f2,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, shortcut_conv])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def identity_block(x, filters, kernel_size):
    shortcut = x
    f1, f2 = filters

    x = tf.keras.layers.Conv2D(filters=f1,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=f2,
                               kernel_size=(kernel_size, kernel_size),
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x
