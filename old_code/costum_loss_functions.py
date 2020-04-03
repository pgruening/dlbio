class INetworkLossFunction(object):

    def __call__(self, y_true, y_pred):
        raise NotImplementedError


class CostumMetric(INetworkLossFunction):
    def __init__(self, name, mode, func):
        self.mode = mode
        self.__name__ = name
        self.func = func

    # NOTE: This base code is meant for using keras functions
    def __call__(self, y_true, y_pred):
        return self.func(y_true, y_pred)
