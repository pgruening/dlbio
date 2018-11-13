import warnings


class IGenerator(object):
    def __init__(self, data, augmentation_functions, IDs, **kwargs):
        # positional arguments
        self.data = data
        self.augmentation_functions = augmentation_functions

    def setup_augmentation_functions(self, model):
        for aug_function in self.augmentation_functions:
            if not hasattr(aug_function, 'setup'):
                warnings.warn('No setup method for: {}'.format(
                    aug_function.__name__))
            else:
                aug_function.setup(model)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
