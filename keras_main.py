from keras.utils.generic_utils import get_custom_objects
from main import IMain


class KerasMain(IMain):
    def init_costum_objects(self, costum_objects):
        get_custom_objects().update(
            costum_objects)
