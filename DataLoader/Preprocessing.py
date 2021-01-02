import Augmentor


def prepare_data(source, output):
    aug = Augmentor.Pipeline(source_directory=source, output_directory=output)
    aug.flip_top_bottom(0.3)
    aug.flip_left_right(0.25)
    aug.zoom(probability=0.2, min_factor=1.1, max_factor=1.7)
    aug.rotate(probability=0.6, max_left_rotation=15, max_right_rotation=15)
    aug.sample(40000, multi_threaded=True)


if __name__ == '__main__':
    prepare_data('/Users/lukaszkolarz/Desktop/AGH/praca inżynierska/dataset.nosync/train',
                 '/Users/lukaszkolarz/Desktop/AGH/praca inżynierska/dataset.nosync/train_augmeted')
