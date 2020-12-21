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
learning_rate = 1e-3
classes = 4

#loader = ImageLoader.ImageLoader(train_path="/Users/lukaszkolarz/Downloads/OCT2017/proba/",
#                                 size=(496, 512))
#(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = loader.import_images()


#x_train = x_train / 255
#x_test = x_test / 255


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/lukaszkolarz/Desktop/AGH/praca inżynierska/dataset.nosync/train_augmeted_resized",
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(496, 496),
    batch_size=128)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      "/Users/lukaszkolarz/Desktop/AGH/praca inżynierska/dataset.nosync/validation",
      validation_split=0.1,
      subset="validation",
      seed=123,
      image_size=(496, 496),
      batch_size=128)

print(train_ds, 'train samples')
#print(x_test.shape[0], 'test samples')
input_shape = (496, 496, 3)

#model = ResNet34.ResNet34(input_shape, classes)
#model = ResNet50.ResNet50(input_shape, classes)
model = SEResNet50.SEResNet50(input_shape, classes)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])

history = model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds)
print(history.history)

#test_loss, test_acc = model.evaluate(x_test, y_test)
#print(test_acc)

#tf.keras.utils.plot_model(model, to_file='SEResNet50_architecture.png', show_shapes=False, show_layer_names=False)
plot.plot_history(history, 'results.png')

