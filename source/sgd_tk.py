#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("SGD Demo")

# main frame
fr_learn = Tk.Frame(root)

# figure
f = plt.Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)

# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=fr_learn)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

variables=[]
labels=[]

fr_buttons=Tk.Frame(fr_learn)

var = Tk.StringVar()
var.set('0.01')
variables.append(var)

fr_row = Tk.Frame(fr_buttons)

labels.append( Tk.Label(fr_row, text='alpha'))
labels[-1].pack(side=Tk.LEFT)
Tk.Entry(fr_row, textvariable=var).pack(side=Tk.RIGHT, expand=Tk.YES, fill=Tk.X)
fr_row.pack(side=Tk.TOP)

var = Tk.StringVar()
var.set('0.02')
variables.append(var)
fr_row = Tk.Frame(fr_buttons)

labels.append(Tk.Label(fr_row, text='eta0'))

labels[-1].pack(side=Tk.LEFT)
Tk.Entry(fr_row, textvariable=var).pack(side=Tk.RIGHT, expand=Tk.YES, fill=Tk.X)
fr_row.pack(side=Tk.TOP)

Tk.Button(fr_buttons,text='reset').pack(side=Tk.TOP)

fr_row=Tk.Frame(fr_buttons)
labels.append(Tk.Label(fr_row, text='probe every'))
labels[-1].pack(side=Tk.LEFT)
Tk.Entry(fr_row).pack(side=Tk.RIGHT, expand=Tk.YES, fill=Tk.X)
fr_row.pack(side=Tk.TOP)

fr_row = Tk.Frame(fr_buttons)
labels.append(Tk.Label(fr_row, text='learn for'))
labels[-1].pack(side=Tk.LEFT)
Tk.Entry(fr_row).pack(side=Tk.RIGHT, expand=Tk.YES, fill=Tk.X)
fr_row.pack(side=Tk.TOP)

#label_width_max=max(map(lambda x: x.config('width'),labels))
for l in labels:
    l.config(width=10)

Tk.Button(fr_buttons,text='learn').pack(side=Tk.TOP)

fr_buttons.pack(side=Tk.LEFT)

toolbar = NavigationToolbar2TkAgg( canvas, root )
toolbar.update()

canvas._tkcanvas.pack(side=Tk.RIGHT, expand=1, fill=Tk.BOTH )

fr_learn.pack(side=Tk.TOP,expand=1,fill=Tk.BOTH)

def on_key_event(event):
    print('you pressed %s'%event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

#button = Tk.Button(master=fr_learn, text='Quit', command=_quit)
#button.pack(side=Tk.BOTTOM)

Tk.mainloop()
# If you put root.destroy() here, it will cause an error if
# the window is closed with the window manager.


