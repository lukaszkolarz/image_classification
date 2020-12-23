import os
import numpy as np
import matplotlib.image as plt
from tensorflow import image as tf


class ImageLoader:
    def __init__(self, train_path, test_path, size=(496, 512)):
        self.train_path = train_path
        self.test_path = test_path
        self.size = size
        self.train_image_names = os.listdir(self.train_path)
        self.test_image_names = os.listdir(self.test_path)
        self.imported_train_images = None
        self.imported_test_images = None
        self.imported_train_rates = None
        self.imported_test_rates = None

    def import_images(self, verbose=0):

        import_images_buffer = list()
        import_rates_buffer = list()

        for image in self.train_image_names:
            if 'CNV' in image:
                import_rates_buffer.append(np.array([1], dtype=int))
                if verbose >= 1:
                    print("New CNV rate for image: " + image)

            elif 'NORMAL' in image:
                import_rates_buffer.append(np.array([0], dtype=int))
                if verbose >= 1:
                    print("New NORMAL rate for image: " + image)
            else:
                print(image)
                raise Exception('Unknown class')

            img = plt.imread(self.train_path + image)
            img = np.asarray(img)
            img = np.expand_dims(img, axis=2)
            img = tf.resize_with_crop_or_pad(img, self.size[0], self.size[1])
            import_images_buffer.append(img)
            if verbose >= 1:
                print("New imported image: " + image)
        self.imported_train_images = np.stack(import_images_buffer, axis=0)
        self.imported_train_rates = np.stack(import_rates_buffer, axis=0)

        import_images_buffer = list()
        import_rates_buffer = list()

        for image in self.test_image_names:
            if 'CNV' in image:
                import_rates_buffer.append(np.array([1], dtype=int))
                if verbose >= 1:
                    print("New CNV rate for image: " + image)

            elif 'NORMAL' in image:
                import_rates_buffer.append(np.array([0], dtype=int))
                if verbose >= 1:
                    print("New NORMAL rate for image: " + image)
            else:
                print(image)
                raise Exception('Unknown class')

            img = plt.imread(self.test_path + image)
            img = np.asarray(img)
            img = np.expand_dims(img, axis=2)
            img = tf.resize_with_crop_or_pad(img, self.size[0], self.size[1])
            import_images_buffer.append(img)
            if verbose >= 1:
                print("New imported image: " + image)
        self.imported_test_images = np.stack(import_images_buffer, axis=0)
        self.imported_test_rates = np.stack(import_rates_buffer, axis=0)

        return (self.imported_train_images, self.imported_train_rates), (self.imported_test_images, self.imported_test_rates)
