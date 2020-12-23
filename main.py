import ResNet34
import ResNet50
import SEResNet50
import tensorflow as tf
import plot_result as plot

batch_size = 64
epochs = 2
learning_rate = 1e-3
classes = 4
target_size = 496


# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "/net/people/plgkolarzl/dataset_clear",
#     validation_split=0.1,
#     color_mode='grayscale',
#     label_mode='int',
#     subset="training",
#     seed=123,
#     image_size=(target_size, target_size),
#     batch_size=batch_size)
#
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#       "/net/people/plgkolarzl/test",
#       validation_split=0.9,
#       color_mode='grayscale',
#       label_mode='int',
#       subset="validation",
#       seed=123,
#       image_size=(target_size, target_size),
#       batch_size=batch_size)

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_ds = data_generator.flow_from_directory('/net/people/plgkolarzl/dataset_clear',
                                              class_mode='categorical',
                                              batch_size=batch_size,
                                              color_mode='grayscale',
                                              target_size=(target_size, target_size))
val_ds = data_generator.flow_from_directory('/net/people/plgkolarzl/test',
                                            class_mode='categorical',
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            target_size=(target_size, target_size))

input_shape = (target_size, target_size, 1)

strategy = tf.distribute.MirroredStrategy()
print('Number of GPUs: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():
    #model = ResNet34.ResNet34(input_shape, classes)
    #model = ResNet50.ResNet50(input_shape, classes)
    model = SEResNet50.SEResNet50(input_shape, classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])

model.summary()
#history = model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds)
history = model.fit(train_ds, steps_per_epoch=120, validation_steps=2, epochs=epochs, validation_data=val_ds)
print(history.history)

#test_loss, test_acc = model.evaluate(x_test, y_test)
#print(test_acc)

#tf.keras.utils.plot_model(model, to_file='SEResNet50_architecture.png', show_shapes=False, show_layer_names=False)
plot.plot_history(history, 'results.png')

