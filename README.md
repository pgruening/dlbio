# Deep Learning on Biomedial Data

This Repository aims to provide boiler plate code to quickly train, test, and evaluate deep learning architectures in Pytorch.

## Interesting modules:
 - `helpers.py`: utility functions for path operations, number conversions, detection rectangles, and so on. With `MyDataFrame`, a simple data structure based on dictionaries is provided to quickly setup Pandas DataFrames.
- `pt_training.py`: contains a class ‘training’ that comprises all necessary objects (optimizer, loss-functions, data-loader, …) and, upon calling, runs a typical pytorch training session. Furthermore, a default ArgumentParser is given that functions as a standard interface to select hyperparameters. Training results are saved to a log-file that is managed by the `Printer`.
- `pt_train_printer.py`: an object that prints intermediate training results in the terminal window and additionally, writes them to a `.json` file.
- `exe_log_gui.py`: opens a TKinter GUI in the current directory. Searches it and all subfolders for json files (search can be narrowed by with regular expressions). The found files are presented in a list. One can select a file that is assumed to be a dictionary with strings as keys and lists as values. The keys can be selected and plotted in a window.
- `pytorch_helpers.py`: contains some convenience functions specifically for the use of pytorch. For example, getting the current device, count the number of parameters in a model, or transform a torch image tensor to a numpy image array.
- `embedding_gui.py`: provides classes to visualize embeddings. It is possible to click on one of the datapoints in the embedding and display additional information about it in a new window.
- `process_image_patchwise.py`: meant for segmenting large images with a network. The whole image is processed in patches, inspired by the ‘seamless tiling’ strategy in the u-net paper.
- `ds_pt_dataset.py`: boiler plate code to implement custom datasets. For example, there is a SegmentationDataset class, that can be used on small sets that fit into RAM.
- `pt_run_parallel.py`: useful for running several training processes. ITrainingProcess is a good interface to quickly setup and run a training process.

## Naming Conventions:
### Module prefixes
- ‘pt’: pytorch, any module that imports, or at least heavily uses, pytorch packages.
- ‘ds’: dataset, should contain a class that inherits a pytorch Dataset and has a getter function for an appropriate pytorch DataLoader.
- ‘exe’: execute, modules that are meant to be called directly in the terminal. In general, exe modules should not be imported by other modules.
- ‘run’: a module that can be executed, but heavily relies on input parameters. Hence, it should be called by an exe module (e.g. with a class that implements `ITrainingProcess`).


### Class Prefixes
- ‘ISomeClass’: interface. Similar to interfaces in other languages, this class should not be instantiated, but rather, other classes should inherit it and implement the necessary functions.  

