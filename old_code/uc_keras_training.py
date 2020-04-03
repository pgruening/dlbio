import os
import keras
from .uc_Itraining import ITraining

# NOTE: model is compiled during call


class KerasTraining(ITraining):

    def __call__(self,
                 keras_model,
                 generator_train,
                 optimizer,
                 loss_function,
                 lr_policy,
                 number_of_epochs,
                 save_path,
                 generator_val=None,
                 costum_metrics=None,
                 use_tensorboard=True,
                 reset_weights=True,
                 training_callbacks=[],
                 model_checkpoint_validation_period=1,
                 tensorboard_batch_size=32,
                 class_weight=None
                 ):
        """Method to train the keras keras_model.

        Parameters
        ----------
        save_path : str
        Folder where to save the keras_model.
        Model name is "best_model_on_[some_metric or normal].hpy5"
        generator_train : Data_Sequence
        Data_Sequence providing batches with samples and labels for training
        generator_val : Data_Sequence, optional
        Data_Sequence providing batches with samples and labels
        for validation
        during training
        (the default is None, which means no validation is used)
        number_of_epochs : int, optional
        how many epochs is the network trained
        (the default is config.get('number_of_epochs'),
        which is the value given in global_constants.py)
        use_tensorboard : bool, optional
        write data to tensorboard (the default is True)
        reset_weights : bool, optional
        For each new run the weights are set to the random initialization,
        needed when training cross-validation
        (the default is True,
        which means the weights are initialized randomly when this function is called).
        class_weights: list of float
        when using sparse cross entropy you can reweight the specific class loss
        Returns
        -------
        keras history object
        All information gathered during keras training,
        look at keras website for more information.
        """

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if reset_weights:
            keras_model.reset_model()

        # Before using the generators, they need to be setup, because some
        # of the functions need information from the keras_model.
        generator_train.setup_augmentation_functions(keras_model)
        if generator_val is not None:
            generator_val.setup_augmentation_functions(keras_model)

        lr_callback = keras.callbacks.LearningRateScheduler(lr_policy.schedule)
        training_callbacks.append(lr_callback)

        if use_tensorboard:
            parent_folder = "/".join(save_path.split("/")[0:-1])
            tb_save_path = os.path.join(
                parent_folder, keras_model.ID, "tb_logs")
            print('writing to tensorboard to {}'.format(tb_save_path))
            tb_callback = keras.callbacks.TensorBoard(log_dir=tb_save_path,
                                                      histogram_freq=0,
                                                      batch_size=tensorboard_batch_size,
                                                      write_graph=True,
                                                      write_grads=True,
                                                      write_images=False,
                                                      embeddings_freq=0,
                                                      embeddings_layer_names=[],
                                                      embeddings_metadata=None)
            training_callbacks.append(tb_callback)

        if costum_metrics is not None:
            early_stopping_metrics = [
                ('val_'+x.__name__, x.mode) for x in costum_metrics
            ]
        else:
            early_stopping_metrics = [('val_loss', 'min')]

        for (metric, mode) in early_stopping_metrics:
            model_name = os.path.join(
                save_path, "best_model_on_{}.h5".format(metric))
            checkpoint = keras.callbacks.ModelCheckpoint(
                model_name,
                monitor=metric,
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode=mode,
                period=model_checkpoint_validation_period
            )
            training_callbacks.append(checkpoint)

        # for full train, save keras_model if every 5 epochs the loss did decrease
        model_name = os.path.join(save_path, "best_model_on_normal.h5")
        checkpoint = keras.callbacks.ModelCheckpoint(model_name,
                                                     monitor='loss',
                                                     verbose=0,
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     mode='min',
                                                     period=model_checkpoint_validation_period)
        training_callbacks.append(checkpoint)

        #used_metrics = [x[0] for x in early_stopping_metrics]
        keras_model.cnn.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=costum_metrics
        )

        kwargs = {}
        if class_weight is not None:
            kwargs.update({'class_weight': class_weight})

        # *train keras_model*
        if generator_val:
            history = keras_model.cnn.fit_generator(generator_train,
                                                    validation_data=generator_val,
                                                    epochs=number_of_epochs,
                                                    callbacks=training_callbacks,
                                                    use_multiprocessing=True,
                                                    **kwargs
                                                    )
        else:
            history = keras_model.cnn.fit_generator(generator_train,
                                                    epochs=number_of_epochs,
                                                    callbacks=training_callbacks,
                                                    **kwargs
                                                    )
        # *save full run keras_model*
        model_name = os.path.join(save_path, "best_model_on_normal.h5")
        keras_model.cnn.save(model_name)

        return history
