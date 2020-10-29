import imageio
import os
import numpy as np
import matplotlib.image as plt
from tensorflow import image as tf


class ImageLoader:
    def __init__(self, disease_path, healthy_path, size=(496, 512)):
        self.disease_path = disease_path
        self.healthy_path = healthy_path
        self.size = size
        self.disease_names = os.listdir(self.disease_path)
        self.healthy_names = os.listdir(self.healthy_path)
        self.disease_names.sort()
        self.healthy_names.sort()
        self.imported_disease = None
        self.disease_rates = None
        self.imported_healthy = None
        self.healthy_rates = None
        self.imagesImported = False
        self.x = None
        self.y = None

    def import_images(self, verbose=0):

        import_disease_buffer = list()
        import_healthy_buffer = list()

        for image in self.disease_names:
            img = plt.imread(self.disease_path + image)
            img = np.asarray(img)
            img = np.expand_dims(img, axis=2)
            img = tf.resize_with_crop_or_pad(img, self.size[0], self.size[1])
            import_disease_buffer.append(img)
            if verbose >= 1:
                print("Imported 'disease' image:" + image)
        self.imported_disease = np.stack(import_disease_buffer, axis=0)

        for image in self.healthy_names:
            img = plt.imread(self.healthy_path + image)
            img = np.asarray(img)
            img = np.expand_dims(img, axis=2)
            img = tf.resize_with_crop_or_pad(img, self.size[0], self.size[1])
            import_healthy_buffer.append(img)
            if verbose == 1:
                print("Imported 'healthy' image:" + image)
        self.imported_healthy = np.stack(import_disease_buffer, axis=0)
        self.imagesImported = True
        print("Images imported!")

    def import_rates(self, verbose=0):
        if self.imagesImported:
            disease_images_count = self.imported_disease.shape[0]
            self.disease_rates = np.zeros(disease_images_count, dtype=np.uint8)
            if verbose >= 1:
                print("Disease rates imported!")
            healthy_images_count = self.imported_healthy.shape[0]
            self.healthy_rates = np.ones(healthy_images_count, dtype=np.uint8)
            if verbose >= 1:
                print("Healthy rates imported")
        else:
            print("[Error] Cannot import rates before images. Import images first")

    def build_dataset(self, verbose=0):
        self.import_images(verbose=verbose)
        
