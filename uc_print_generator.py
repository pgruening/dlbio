   ############# EVALUATION ############################################

   def print_generator(self, generator, save_path='./generator_output'):
        generator.setup_augmentation_functions(self)
        self._print_generator(generator, save_path)

    def _print_generator(self, generator, save_path, num_classes):
        for i in range(len(generator)):
            dir_name = os.path.join(
                save_path, 'batch_{}'.format(str(i).zfill(3)))
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

            batch = generator[i]
            num_images = batch[0].shape[0]
            if batch[0][1].ndim == 2:
                label_dim = 1
            else:
                label_dim = batch[0][1].shape[-1]

            if label_dim > 1:
                print(
                    "label dim is greater than 1 ({}). Printing only first dim".format(label_dim))

            for j in range(num_images):
                image = batch[0][j, ...]
                label = batch[1][j, ...]

                file_name = os.path.join(dir_name, str(j).zfill(3)+'.png')
                _, ax = plt.subplots(1, 2)
                image = helpers.to_uint8_image(image)
                ax[0].imshow(image)
                ax[1].imshow(label[..., 0], vmin=0,
                             vmax=num_classes)
                plt.savefig(file_name)
                plt.close()
