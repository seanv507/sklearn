# -*- coding: utf-8 -*-
"""
Created on Sun May 11 22:07:46 2014

@author: sean
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button, RadioButtons

#from sgdlearner import SGDLearner
# matplot lib has sliders, buttons and radio buttons

# eta0 alpha
# learning


class SGD_matplot:
    '''
    To do: have multiple windows,
    use widgets - edit boxes
    add timeframe for parameter changes, so add vline for each change
    '''

    def reset(self, event):
            self.sgd.reset()
            self.sgd.plot(self.ax_graph)
            # CLEAR SCREEN

    def learn(self, event):
            self.sgd.learn(self.s_learn_for.val, self.s_probe_every.val)
            self.sgd.plot(self.ax_graph)

    def __init__(self, sgd):
        fig, self.ax_graph = plt.subplots()
        fig.subplots_adjust(left=0.25, bottom=0.5)

        self.sgd = sgd

        axcolor = 'lightgoldenrodyellow'
        self.ax_learn_for = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        self.ax_probe_every  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        self.ax_alpha = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg=axcolor)
        self.ax_eta0  = plt.axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)
        self.s_learn_for = Slider(self.ax_learn_for, 'learnfor', 0, 10000, valinit=1000)
        self.s_probe_every = Slider(self.ax_probe_every, 'probe every', 0, 100, valinit=1000)
        self.s_alpha = Slider(self.ax_alpha, 'alpha', 0, 100, valinit=self.sgd.sgd.alpha)
        self.s_eta0 = Slider(self.ax_eta0, 'eta0', 0, 100, valinit=self.sgd.sgd.eta0)

        self.reset_ax = plt.axes([0.6, 0.025, 0.1, 0.04])
        self.learn_ax = plt.axes([0.8, 0.025, 0.1, 0.04])

        self.reset_but = plt.Button(self.reset_ax,'reset', color=axcolor, hovercolor='0.975')
        self.learn_but = plt.Button(self.learn_ax,'learn', color=axcolor, hovercolor='0.975')
        self.sgd.plot(self.ax_graph)



        self.reset_but.on_clicked(lambda event: self.reset(event))
        self.learn_but.on_clicked(lambda event: self.learn(event))


#def update(val):
#    amp = samp.val
#    freq = sfreq.val
#    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
#    fig.canvas.draw_idle()
#sfreq.on_changed(update)
#samp.on_changed(update)
#
#
#rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
#radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
#def colorfunc(label):
#    l.set_color(label)
#    fig.canvas.draw_idle()
#radio.on_clicked(colorfunc)
#
#plt.show()