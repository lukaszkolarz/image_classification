from tensorflow.keras import datasets
import ResNet34
import ResNet50
import SEResNet50
import numpy as np
import tensorflow as tf
import plot_result as plot
import ImageLoader

batch_size = 64
epochs = 2
num_classes = 10
learning_rate = 1e-5
classes = 2

#loader = ImageLoader.ImageLoader(train_path="/Users/lukaszkolarz/Downloads/OCT2017/proba/",
#                                 size=(496, 512))
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = loader.import_images()


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
              loss='binary_crossentopy',
              metrics=['acc'])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
#print(history.history)

#test_loss, test_acc = model.evaluate(x_test, y_test)
#print(test_acc)

#tf.keras.utils.plot_model(model, to_file='SEResNet50_architecture.png', show_shapes=False, show_layer_names=False)
#plot.plot_history(history, 'results.png')

