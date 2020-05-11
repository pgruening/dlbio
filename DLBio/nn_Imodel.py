class IModel(object):
    def predict(self, input, do_pre_proc):
        raise NotImplementedError

    def do_task(self, input, do_pre_proc):
        raise NotImplementedError
