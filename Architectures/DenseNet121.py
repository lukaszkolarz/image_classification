import tensorflow as tf


def DenseNet121(input_shape, classes, init_filters=64, block_config=None, growth=4):

    if block_config is None:
        block_config = [6, 12, 24, 16]

    input_shape = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=init_filters,
                               kernel_size=(7, 7),
                               strides=2,
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(input_shape)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)

    num_filters = init_filters

    for i, layers in enumerate(block_config):
        for j in range(layers):
            x = dense_block(x, growth=growth)

        if i != len(block_config) - 1:
            num_filters = num_filters + layers * growth
            num_filters = num_filters // 2
            x = transition_block(x, filters=num_filters)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(classes,
                              activation='softmax',
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)

    DenseNet121Model = tf.keras.models.Model(inputs=input_shape, outputs=x, name='DenseNet121')

    return DenseNet121Model


def dense_block(x, growth=32, multi_factor=4):

    shortcut = x

    x = tf.keras.layers.Conv2D(filters=growth * multi_factor,
                               kernel_size=(1, 1),
                               strides=1,
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=growth,
                               kernel_size=(3, 3),
                               strides=1,
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Concatenate(axis=3)([shortcut, x])

    return x


def transition_block(x, filters):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=(1, 1),
                               strides=1,
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x
