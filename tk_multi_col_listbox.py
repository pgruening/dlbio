'''
Here the TreeView widget is configured as a multi-column listbox
with adjustable column width and column-header-click sorting.
Taken from:
https://stackoverflow.com/questions/5286093/display-listbox-with-columns-using-tkinter
'''
try:
    import Tkinter as tk
    import tkFont
    import ttk
except ImportError:  # Python 3
    import tkinter as tk
    import tkinter.font as tkFont
    import tkinter.ttk as ttk


class MultiColumnListbox(object):
    """use a ttk.TreeView as a multicolumn ListBox"""

    def __init__(self, header, values):
        self.tree = None

        # creates a fixed window and pack cannot be used
        # self._setup_widgets(header)

        self.tree = ttk.Treeview(columns=header, show="headings")
        self.tree.bind("<Double-Button-1>", self.on_click)
        self._build_tree(header, values)

    def _build_tree(self, header, values):
        for col in header:
            self.tree.heading(col, text=col.title(),
                              command=lambda c=col: sortby(self.tree, c, 0))
            # adjust the column's width to the header string
            self.tree.column(col,
                             width=tkFont.Font().measure(col.title()))

        for item in values:
            self.tree.insert('', 'end', values=item)
            # adjust column's width if necessary to fit each value
            for ix, val in enumerate(item):
                col_w = tkFont.Font().measure(val)
                if self.tree.column(header[ix], width=None) < col_w:
                    self.tree.column(header[ix], width=col_w)

    def pack(self):
        self.tree.pack()

    def pack_forget(self):
        self.tree.pack_forget()

    def on_click(self, event):
        ''' Executed, when a row is double-clicked. Opens 
        read-only EntryPopup above the item's column, so it is possible
        to select text '''

        # close previous popups
        # self.destroyPopups()

        # what row and column was clicked on
        rowid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if column == "#1":
            return

        # get column position info
        x, y, width, height = self.tree.bbox(rowid, column)

        # place Entry popup properly
        var = tk.StringVar()
        var.set(self.tree.item(rowid, 'text'))
        self.entryPopup = EntryPopup(self.tree, rowid, var)
        self.entryPopup.place(x=x, y=y)

        self.tree.wait_variable(var)

        self.tree.set(rowid, column, var.get())

    def get(self):
        output = []
        for row_item in self.tree.get_children():
            output.append(self.tree.item(row_item)['values'])

        return output

    def _setup_widgets(self, header):
        s = """\click on header to sort by that column
to change width of column drag boundary
        """
        msg = ttk.Label(wraplength="4i", justify="left", anchor="n",
                        padding=(10, 2, 10, 6), text=s)
        msg.pack(fill='x')
        container = ttk.Frame()
        container.pack(fill='both', expand=True)
        # create a treeview with dual scrollbars
        self.tree = ttk.Treeview(columns=header, show="headings")
        vsb = ttk.Scrollbar(orient="vertical",
                            command=self.tree.yview)
        hsb = ttk.Scrollbar(orient="horizontal",
                            command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set,
                            xscrollcommand=hsb.set)
        self.tree.grid(column=0, row=0, sticky='nsew', in_=container)
        vsb.grid(column=1, row=0, sticky='ns', in_=container)
        hsb.grid(column=0, row=1, sticky='ew', in_=container)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)


def sortby(tree, col, descending):
    """sort tree contents when a column header is clicked on"""
    # grab values to sort
    data = [(tree.set(child, col), child)
            for child in tree.get_children('')]
    # if the data to be sorted is numeric change to float
    #data =  change_numeric(data)
    # now sort the data in place
    data.sort(reverse=descending)
    for ix, item in enumerate(data):
        tree.move(item[1], '', ix)
    # switch the heading so it will sort in the opposite direction
    tree.heading(col, command=lambda col=col: sortby(tree, col,
                                                     int(not descending)))


class EntryPopup(tk.Entry):

    def __init__(self, parent, iid, text, **kw):
        ''' If relwidth is set, then width is ignored '''
        super().__init__(parent, **kw)
        self.tv = parent
        self.iid = iid

        self.var = text

        self.insert(0, text)
        # self['state'] = 'readonly'
        # self['readonlybackground'] = 'white'
        # self['selectbackground'] = '#1BA1E2'
        self['exportselection'] = False

        self.focus_force()
        self.bind("<Return>", self.on_return)
        self.bind("<Control-a>", self.select_all)
        self.bind("<Escape>", lambda *ignore: self.destroy())

    def on_return(self, event):
        self.var.set(self.get())
        self.tv.item(self.iid, text=self.get())
        self.destroy()
        return

    def select_all(self, *ignore):
        ''' Set selection on the whole text '''
        self.selection_range(0, 'end')

        # returns 'break' to interrupt default key-bindings
        return 'break'

# the test data ...


CAR_HEADER = ['car', 'repair']
CAR_LIST = [
    ('Hyundai', 'brakes'),
    ('Honda', 'light'),
    ('Lexus', 'battery'),
    ('Benz', 'wiper'),
    ('Ford', 'tire'),
    ('Chevy', 'air'),
    ('Chrysler', 'piston'),
    ('Toyota', 'brake pedal'),
    ('BMW', 'seat')
]


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Multicolumn Treeview/Listbox")
    listbox = MultiColumnListbox(CAR_HEADER, CAR_LIST)
    root.mainloop()
