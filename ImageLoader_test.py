import ImageLoader
from matplotlib import pyplot
from PIL import Image


def test_ImageLoader():
    loader = ImageLoader.ImageLoader(train_path="/Users/lukaszkolarz/Downloads/OCT2017/proba/",
                                     size=(496, 512))

    loader.import_train_images()
    #pyplot.imshow(loader.imported_disease[2])
    #pyplot.show()
    print(loader.imported_train_images.shape)
    #print(loader.imported_healthy)
    #loader.import_rates()
    print(loader.imported_train_rates)
if __name__ == '__main__':
    test_ImageLoader()
