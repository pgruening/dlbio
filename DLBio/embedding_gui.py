"""
GUI boilerplate code to visualize high dim vectors in a reduced space. To do 
this use IEmbedding function. 
You can visualize the embedding, when clicking on one of the data points,
the selected plot function is called.

-> EmbeddingGui._open_window
-> Generate ClickableEmbeddingGui
-> Embed plot_fcn.X
-> on_click -> plot_fcn(index)

#run e.g. like this:

data = load_data()

plot_fcns = [
    plot_fcn1(data),
    ...
    plot_fcn4(data)
]

embedding = [
    fcn1(),
    ...
    fcn7()
]


gui = EmbeddingGUI(embedding, plot_fcns)
tk.mainloop()

"""
import os
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

from .tk_multi_col_listbox import MultiColumnListbox

CHECKBOXES = None
CHECK_VARS = None


class EmbeddingGUI():
    """GUI to select an embedding and a plot function
    """

    def __init__(self, embed_fcns, data_fcns):
        self.master = tk.Tk()
        self.master.title('Embedding GUI')

        self.embed_fcns = {x.__name__: x for x in embed_fcns}
        self.data_fcns = {x.__name__: x for x in data_fcns}

        self.tk_param_list = MultiColumnListbox(
            ['Key', 'Value'], [('None', '-1'), ('None', '-1')]
        )

        self.add_embed_list()
        self.add_data_list()

        self.bt_cancel = tk.Button(self.master, text='cancel',
                                   width=25, command=self._end)
        self.bt_start = tk.Button(
            self.master, text='start', width=25, command=self._open_window
        )

        self.current_embedding = None
        self.current_data = None

        self._pack()

        self.open_guis = []

    def _pack(self):
        self.bt_start.pack()
        self.bt_cancel.pack()
        self.tk_data_list.pack()
        self.tk_embed_list.pack()
        self.tk_param_list.pack()

    def _end(self):
        self.master.destroy()
        plt.close('all')

    def _open_window(self):
        if self.current_embedding is None:
            print('No embedding chosen')
            return

        if self.current_data is None:
            print('No data chosen')
            return

        params = self.tk_param_list.get()
        params = {p[0]: p[1] for p in params}

        self.current_embedding.kwargs = params

        self.open_guis.append(
            ClickableEmbeddingGUI(
                self.current_embedding,
                self.current_data
            )
        )

        plt.show()

    def add_data_list(self):
        self.tk_data_list = self._setup_list_box(
            self.data_fcns,
            self._get_set_data_function
        )

    def add_embed_list(self):
        self.tk_embed_list = self._setup_list_box(
            self.embed_fcns,
            self.add_param_list
        )

    def _setup_list_box(self, items_, fcn):
        tk_list_box = tk.Listbox(self.master)

        for item in items_:
            tk_list_box.insert(tk.END, item)

        for bind_input in ["<Button-1>", "<Up>", "<Down>"]:
            tk_list_box.bind(bind_input, fcn)

        return tk_list_box

    def add_param_list(self, event):
        # somehow selection cannot be found on first click
        self.current_embedding = self._get_embedding_function(event)
        if self.current_embedding is None:
            return

        self.tk_param_list.pack_forget()
        self.tk_param_list = MultiColumnListbox(
            ['Key', 'Value'],
            self.current_embedding.get_params()
        )
        self.tk_param_list.pack()

    def _get_embedding_function(self, event):
        current_item = self._get_item(event)
        if current_item is None:
            return None

        return self.embed_fcns[current_item]

    def _get_set_data_function(self, event):
        current_item = self._get_item(event)
        if current_item is None:
            return None

        self.current_data = self.data_fcns[current_item]

    def _get_item(self, event):
        file_list = event.widget
        selection = file_list.curselection()
        if not selection:
            return None
        item = file_list.get(selection[0])
        return item

    def add_logfiles(self):
        full_paths = dict()
        for root, _, files_ in os.walk('.'):
            files_ = [x for x in files_ if os.path.splitext(x)[-1] == '.json']
            if not files_:
                continue
            last_folder = root.split('/')[-1]
            for file in files_:
                item = '.../' + os.path.join(last_folder, file)
                self.tk_embed_list.insert(tk.END, item)
                full_paths[item] = os.path.join(root, file)

        return full_paths

    def read_items(self, event, full_paths, master, lab_current_file):
        global CHECKBOXES, CHECK_VARS

        while CHECKBOXES:
            CHECK_VARS.pop()
            c = CHECKBOXES.pop()
            c.pack_forget()

        file_list = event.widget
        lab_current_file.config()


class ClickableEmbeddingGUI():
    """ A scatter plot that is computed via an embedding function. Furthermore,
    a plot function is activated when the user clicks on a datapoint in the 
    plot.
    """

    def __init__(self, embedding, plot_fcn, thres=np.inf, **kwargs):
        self.emb = embedding
        self.plot_fcn = plot_fcn

        self.Z = self.emb.fit_transform(plot_fcn.X)
        fig = self.emb.plot(self.Z, plot_fcn.__name__)

        # set clickable
        _ = fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.thres = thres

    def on_click(self, event):
        if not event.dblclick:
            return

        pos = np.array([event.xdata, event.ydata])
        dist = ((self.Z - pos)**2.).sum(1)

        index = np.argmin(dist)
        dist = dist[index]

        if dist > self.thres:
            return

        self.plot_fcn(index)
        plt.show()


class IEmbeddingFunction():
    """For a dataset x, compute the embedding and plot a scatter plot
    """

    def __init__(self, label=None):
        self.kwargs = dict()
        self.label = label
        self.no_emb_keys = []
        self.fcn_class = IPlotFunction('change to a real class')
        raise NotImplementedError()

    def fit_transform(self, x):
        emb_kwargs = {k: v for k, v in self.kwargs.items()
                      if k not in self.no_emb_keys
                      }
        fcn = self.fcn_class(**emb_kwargs)
        if x.ndim > 2:
            x = x[:, :, 0]
        return fcn.fit_transform(x)

    def get_params(self):
        return list(self.kwargs.items())

    def plot(self, Z, plot_fcn_name):
        fig, _ = plt.subplots()
        if self.label is not None:
            plt.scatter(
                Z[:, 0], Z[:, 1],
                c=self.label[:, self.kwargs['label_id']]
            )

        else:
            plt.scatter(Z[:, 0], Z[:, 1])

        plt.title(' '.join(
            [
                f'{k}:{v}' for k, v in self.kwargs.items()
            ] + [plot_fcn_name]))
        return fig


class IPlotFunction():
    """Given the data and an index. Plot the datapoint i
    """

    def __init__(self, x, **kwargs):
        self.X = x
        self.kwargs = kwargs
        self._set_name()

    def _set_name(self):
        """Needs to be implemented to be visible in the main gui.
        """
        self.__name__ = 'implement me'
        raise NotImplementedError()

    def __call__(self, index):
        # NOTE: plt.show() not necessary
        pass

##############################################################################
##############################################################################
##############################################################################
##############################################################################


class TSNEEmbedding(IEmbeddingFunction):
    def __init__(self, n_components=2, label=None):
        self.__name__ = 'TSNE'
        self.fcn_class = TSNE
        self.kwargs = {
            'n_components': n_components,
            'label_id': 0,
            'random_state': 0,
            'perplexity': 15,
        }
        self.no_emb_keys = ['label_id']
        self.label = label


class LLEEmbedding(IEmbeddingFunction):
    def __init__(self, n_components=2, n_neighbors=15, label=None):
        self.__name__ = 'LLE'
        self.fcn_class = LocallyLinearEmbedding
        self.kwargs = {
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'label_id': 0
        }
        self.no_emb_keys = ['label_id']
        self.label = label


class PCAEmbedding(IEmbeddingFunction):
    def __init__(self, label=None):
        self.__name__ = 'PCA'
        self.fcn_class = MyPCA
        self.kwargs = {
            'label_id': 0,
            'c0': 0,
            'c1': 1
        }
        self.no_emb_keys = ['label_id']
        self.label = label


class MyPCA():
    def __init__(self, c0=0, c1=1):
        self.pca = PCA()
        self.components = [c0, c1]

    def fit_transform(self, x):
        x = x - x.mean(0)
        self.pca.fit(x)
        return np.matmul(x, self.pca.components_[self.components, :].T)
