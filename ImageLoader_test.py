import ImageLoader
from matplotlib import pyplot
from PIL import Image


def test_ImageLoader():
    loader = ImageLoader.ImageLoader(disease_path="/Users/lukaszkolarz/Downloads/OCT2017/proba/CNV/",
                                     healthy_path="/Users/lukaszkolarz/Downloads/OCT2017/proba/NORMAL/")

    loader.import_images(verbose=0)
    #pyplot.imshow(Image.fromarray(loader.imported_healthy[2]))
    #pyplot.show()
    #print(loader.imported_healthy.shape[0])
    #print(loader.imported_healthy)
    loader.import_rates()

if __name__ == '__main__':
    test_ImageLoader()
