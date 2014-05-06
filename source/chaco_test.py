# -*- coding: utf-8 -*-
"""
Created on Sun May  4 22:26:19 2014

@author: sean
"""

from chaco.api import ArrayPlotData, Plot
from enable.component_editor import ComponentEditor

from traits.api import HasTraits, Instance
from traitsui.api import View, Item
import numpy as np



class MyPlot(HasTraits):
    plotrange=Range(-14,14)
    plot = Instance(Plot)
    traits_view = View(Item('plot', editor = ComponentEditor(), show_label = False),
                       width = 500, height = 500,
                       resizable = True, title = "My line plot")
                       
    def __init__(self, x, y, *args, **kw):
            super(MyPlot, self).__init__(*args, **kw)
            plotdata = ArrayPlotData(x=x,y=y)
            plot = Plot(plotdata)
            plot.plot(("x","y"), type = "line", color = "blue")
            plot.title = "sin(x)*x**3"
            self.plot = plot


x = np.linspace(-14,14,100)
y = np.sin(x)*x**3
lineplot = MyPlot(x,y)
lineplot.configure_traits()