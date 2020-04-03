class ITraining(object):
    def __call__(self,
                 model,
                 generator_train,
                 optimizer,
                 lr_policy,
                 loss_function,
                 number_of_epochs,
                 save_path,
                 generator_val=None,
                 **kwargs
                 ):
        raise NotImplementedError
