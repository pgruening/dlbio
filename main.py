from keras.utils.generic_utils import get_custom_objects


class IMain(object):
    def __init__(self):
        pass

    def train(self, model, generator):
        raise NotImplementedError

    def evaluate(self, model, generator):
        raise NotImplementedError

    def print_generator(self, model, generator, save_path):
        raise NotImplementedError

    def predict_single_sample(self, model, sample):
        raise NotImplementedError


class KerasMain(IMain):
    def init_costum_objects(self, costum_objects):
        get_custom_objects().update(
            costum_objects)
