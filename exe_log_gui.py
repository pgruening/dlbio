import tkinter as tk
import os
import matplotlib.pyplot as plt
import json

CHECKBOXES = []
CHECK_VARS = []
CURRENT_FILE = ''


def run():
    master = tk.Tk()
    master.title('Logfile viewer')

    w = tk.Label(master, text="Choose logfile")
    lab_current_file = tk.Label(master, text='')

    file_list, full_paths = prepare_list(master, lab_current_file)

    def _end(): return end(master)
    bt_cancel = tk.Button(master, text='cancel',
                          width=25, command=_end)

    var_semilogy = tk.IntVar()
    var_semilogy.set(0)
    c_semilogy = tk.Checkbutton(master, text='semilogy', variable=var_semilogy)

    def ow(): return open_window(file_list, full_paths, var_semilogy)
    bt_start = tk.Button(master, text='start', width=25, command=ow)

    w.pack()
    file_list.pack()
    lab_current_file.pack()

    c_semilogy.pack()

    bt_start.pack()
    bt_cancel.pack()

    tk.mainloop()


def end(master):
    master.destroy()
    plt.close('all')


def prepare_list(master, lab_current_file):
    file_list = tk.Listbox(master)
    full_paths = add_logfiles(file_list)
    def fcn(x): return read_items(x, full_paths, master, lab_current_file)

    for bind_input in ["<Button-1>", "<Up>", "<Down>"]:
        file_list.bind(bind_input, fcn)

    return file_list, full_paths


def add_logfiles(file_list):
    full_paths = dict()
    for root, _, files_ in os.walk('.'):
        files_ = [x for x in files_ if os.path.splitext(x)[-1] == '.json']
        if not files_:
            continue
        last_folder = root.split('/')[-1]
        for file in files_:
            item = '.../' + os.path.join(last_folder, file)
            file_list.insert(tk.END, item)
            full_paths[item] = os.path.join(root, file)

    return full_paths


def read_items(event, full_paths, master, lab_current_file):
    global CHECKBOXES, CHECK_VARS

    while CHECKBOXES:
        CHECK_VARS.pop()
        c = CHECKBOXES.pop()
        c.pack_forget()

    file_list = event.widget
    file, item = load_current(file_list, full_paths)

    for key, value in file.items():
        name = f'{key}: {len(value)}'
        add_check_box(master, name)

    lab_current_file.config(text=item)


def add_check_box(master, name):
    global CHECKBOXES, CHECK_VARS
    var = tk.IntVar()
    c = tk.Checkbutton(master, text=name, variable=var)
    c.pack()
    CHECKBOXES.append(c)
    CHECK_VARS.append(var)


def open_window(file_list, full_paths, var_semilogy):
    is_semilogy = var_semilogy.get()
    plt.figure()
    ax = plt.gca()

    file, item = load_current(file_list, full_paths)

    counter = 0
    for var, c in zip(CHECK_VARS, CHECKBOXES):
        if var.get() == 1:
            counter += 1
            key = c.cget('text').split(':')[0]

            if is_semilogy:
                ax.semilogy(file[key])
            else:
                ax.plot(file[key], label=key)

    ax.set_title(item)

    ax.legend()
    ax.grid()
    plt.show()


def load_current(file_list, full_paths):
    selection = file_list.curselection()

    item = file_list.get(selection[0])
    path = full_paths[item]
    with open(path, 'r') as file:
        file = json.load(file)

    return file, item


if __name__ == "__main__":
    run()
