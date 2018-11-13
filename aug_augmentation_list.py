import warnings


class AugmentationList(list):
    def __init__(self, augmentation_functions):
        self.aug_list = augmentation_functions
        self.is_setup = False

    def setup_list(self, model):
        for func in self.aug_list:
            if not hasattr(func, 'setup'):
                warnings.warn(
                    'No setup function found for {}'.format(func.__name__)
                )
            else:
                func.setup(model)

        self.is_setup = True

    def __len__(self):
        return len(self.aug_list)

    def __iter__(self):
        return iter(self.aug_list)

    def __getitem__(self, index):
        # when used in for loops the end of the loop is detected via index error
        if index >= len(self.aug_list):
            raise IndexError
        return self.aug_list[index]
