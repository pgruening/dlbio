class IEvaluation(object):
    def __init__(self, quality_measure):
        self.quality_measure = quality_measure
        raise NotImplementedError

    def evaluate_generator(self, model, generator):
        raise NotImplementedError

    def evaluate_sample(self, model, input, ground_truth, do_pre_proc=True):
        prediction = model.do_task(input, do_pre_proc)
        return self.quality_measure(prediction, ground_truth)
