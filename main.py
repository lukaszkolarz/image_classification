from Architectures import ResNet34, ResNet50, SEResNet50, TestNet, DenseNet121
import tensorflow as tf
import plot_result as plot
from DataLoader import DataGenerator

batch_size = 8
epochs = 32
learning_rate = 1e-3
classes = 4
target_size = 800

train_ds, val_ds, test_ds, input_shape = DataGenerator.import_greyscale(target_size=target_size,
                                                                        batch_size=batch_size,
                                                                        source_train='/net/archive/groups/plggmlkt/dataset/train',
                                                                        source_val='/net/archive/groups/plggmlkt/dataset/validation',
                                                                        source_test='/net/archive/groups/plggmlkt/dataset/test')

strategy = tf.distribute.MirroredStrategy()
print('Number of GPUs: {}'.format(strategy.num_replicas_in_sync))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                             decay_steps=10000,
                                                             decay_rate=0.96,
                                                             staircase=True)

with strategy.scope():
    # model = ResNet34.ResNet34(input_shape, classes)
    # model = ResNet50.ResNet50(input_shape, classes)
    # model = SEResNet50.SEResNet50(input_shape, classes)
    # model = TestNet.testNet(input_shape, classes)
    # model = tf.keras.applications.DenseNet121(classes=4, weights=None)
    model = DenseNet121.DenseNet121(input_shape, classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])

model.summary()
# history = model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds)
history = model.fit(train_ds, steps_per_epoch=3500, epochs=epochs, validation_data=val_ds, validation_steps=100)
# print(history.history)

test_loss, test_acc = model.evaluate(test_ds, steps=14)
print('Test accuracy: ' + str(test_acc))
print('Test loss: ' + str(test_loss))

# tf.keras.utils.plot_model(model, to_file='SEResNet50_architecture.png', show_shapes=False, show_layer_names=False)
plot.plot_history(history, 'results.png')
