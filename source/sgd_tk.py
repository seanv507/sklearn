#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

#TODO
    # when press learn, take strings and check f numbers have changed.
   # if so mark  on timeline
#
from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("SGD Demo")

class Learn_Chart:
    def __init__(self, root):
        self.root=root

        # main frame
        self.fr_learn = Tk.Frame(root)

        # figure
        f = plt.Figure(figsize=(5,4), dpi=100)
        a = f.add_subplot(111)

    # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(f, master=self.fr_learn)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=Tk.RIGHT, expand=1, fill=Tk.BOTH )

        self.fr_buttons=Tk.Frame(self.fr_learn)
        self.labels={}
        self.vars={}
        var_list=[('alpha',0.01),
                  ('eta0',  0.1),
                  ('probe every', 1000),
                  ('learn for', 10000) ]
        for var, default in var_list:
            self.vars[var]=Tk.StringVar()
            self.vars[var].set(default)

            fr_row = Tk.Frame(self.fr_buttons)

            self.labels[var]= Tk.Label(fr_row, text=var)
            self.labels[var].pack(side=Tk.LEFT)
            self.labels[var].config(width=10)
            #label_width_max=max(map(lambda x: x.config('width'),labels))
            Tk.Entry(fr_row, textvariable=self.vars[var]).pack(side=Tk.RIGHT, expand=Tk.YES, fill=Tk.X)
            fr_row.pack(side=Tk.TOP)


        Tk.Button(self.fr_buttons,text='reset').pack(side=Tk.TOP)
        Tk.Button(self.fr_buttons,text='learn').pack(side=Tk.TOP)

        self.fr_buttons.pack(side=Tk.LEFT)
        self.fr_learn.pack(side=Tk.TOP,expand=1,fill=Tk.BOTH)

        self.toolbar = NavigationToolbar2TkAgg( self.canvas, self.root )
        self.toolbar.update()






def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

#button = Tk.Button(master=fr_learn, text='Quit', command=_quit)
#button.pack(side=Tk.BOTTOM)

a=Learn_Chart(root)

Tk.mainloop()
# If you put root.destroy() here, it will cause an error if
# the window is closed with the window manager.


