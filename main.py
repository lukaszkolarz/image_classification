from tensorflow.keras import datasets
import ResNet34
import ResNet50
import SEResNet50
import numpy as np
import tensorflow as tf
import plot_result as plot

batch_size = 128
epochs = 2
num_classes = 10
learning_rate = 1e-5
classes = 10

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(type(x_train))

x_train = x_train / 255
x_test = x_test / 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

#model = ResNet34.ResNet34(input_shape, classes)
#model = ResNet50.ResNet50(input_shape, classes)
model = SEResNet50.SEResNet50(input_shape, classes)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
#print(history.history)

test_loss, test_acc = model.evaluate(x_test, y_test)
#print(test_acc)

#tf.keras.utils.plot_model(model, to_file='SEResNet50_architecture.png', show_shapes=False, show_layer_names=False)
#plot.plot_history(history, 'results.png')

