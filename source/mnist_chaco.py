# -*- coding: utf-8 -*-
"""
Created on Tue May 06 23:40:45 2014

@author: sv507
"""

"""
Visualization of simulated live data stream

Shows how Chaco and Traits can be used to easily build a data
acquisition and visualization system.

Two frames are opened: one has the plot and allows configuration of
various plot properties, and one which simulates controls for the hardware
device from which the data is being acquired; in this case, it is a mockup
random number generator whose mean and standard deviation can be controlled
by the user.
"""

# Major library imports
import numpy as np

# Enthought imports
from traits.api import (Array, Callable, Enum, Float, HasTraits, Instance, Int,
                        Trait)
from traitsui.api import Group, HGroup, Item, View, spring, Handler
import mnist

# Chaco imports
from chaco.chaco_plot_editor import ChacoPlotItem


class Viewer(HasTraits):
    """ This class just contains the two data arrays that will be updated
    by the Controller.  The visualization/editor for this class is a
    Chaco plot.
    """
    index = Array

    data = Array

    #plot_type = Enum("line", "scatter")

    view = View(ChacoPlotItem("index", "data",
                              type_trait="plot_type",
                              resizable=True,
                              x_label="Time",
                              y_label="Classification error",
                              color="blue",
                              bgcolor="white",
                              border_visible=True,
                              border_width=1,
                              padding_bg_color="lightgray",
                              width=800,
                              height=380,
                              marker_size=2,
                              show_label=False),
                HGroup(spring, Item("plot_type", style='custom'), spring),
                resizable = True,
                buttons = ["OK"],
                width=800, height=500)


class Controller(HasTraits):

    # A reference to the plot viewer object
    viewer = Instance(Viewer)

    # Some parameters controller the random signal that will be generated
    learning_rate = Enum('optimal', 'constant', 'invscaling')
    eta0 = Float(0.01)
    power_t = Float(0.5)
    alpha = Float(1.0)
    nsamples = Int(1000)

    # The max number of data points to accumulate and show in the plot
    #max_num_points = Int(100)

    # The number of data points we have received; we need to keep track of
    # this in order to generate the correct x axis data series.
    #num_ticks = Int(0)

    # private reference to the random number generator.  this syntax
    # just means that self._generator should be initialized to
    # random.normal, which is a random number function, and in the future
    # it can be set to any callable object.
    #_generator = Trait(SGDLearner, Callable)
    def __init__(self, sgd, *args, **kw):
        super(Controller, self).__init__(*args, **kw)
        self.sgd=sgd
        
    view = View(Group('learning_rate',
                      'eta0',
                      'power_t',
                      'alpha',
                      'nsamples',
                      orientation="vertical"),
                      buttons=["OK", "Cancel"])

    def learn_pressed(self, *args):
        """
        Callback function that should get called based on a timer tick.  This
        will generate a new random data point and set it on the `.data` array
        of our viewer object.
        """
        # Generate a new number and increment the tick count
        self.sgd.learn(self.nsamples)

        # grab the existing data, truncate it, and append the new point.
        # This isn't the most efficient thing in the world but it works.
        

        self.viewer.index = map(lambda x:x['timestep'],self.sgd.scores)
        self.viewer.data = map(lambda x:x['test'],self.sgd.scores)
        return

    def _learning_rate_changed(self):
        # This listens for a change in the learning rate.
        self.sgd.learning_rate = self.learning_rate
        
    def _eta0_changed(self):
        # This listens for a change in the learning rate.
        self.sgd.eta0 = self.eta0
    
    def _power_t_changed(self):
        # This listens for a change in the learning rate.
        self.sgd.power_t = self.power_t
    def _alpha_t_changed(self):
        # This listens for a change in the learning rate.
        self.sgd.alpha = self.alpha


class Demo(HasTraits):
    controller = Instance(Controller)
    viewer = Instance(Viewer, ())
    view = View(Item('controller', style='custom', show_label=False),
                Item('viewer', style='custom', show_label=False),
                handler=DemoHandler,
                resizable=True)

    def edit_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer=Timer(100, self.controller.timer_tick)
        return super(Demo, self).edit_traits(*args, **kws)

    def configure_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer=Timer(100, self.controller.timer_tick)
        return super(Demo, self).configure_traits(*args, **kws)

    def _controller_default(self):
        return Controller(viewer=self.viewer)


# NOTE: examples/demo/demo.py looks for a 'demo' or 'popup' or 'modal popup'
# keyword when it executes this file, and displays a view for it.
popup=Demo()


if __name__ == "__main__":
    popup.configure_traits()