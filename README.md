# Visual-Echo
Spatialisation of Reservoir computing networks for display and efficiency purposes.

## The objective
During this project, I will try to spatialize the reservoir of an Echo-State Network for display purposes, but also to take into account the layout of the human brain. I will additionally try to get interesting properties from this network.

## What was done:

* :ballot_box_with_check: Implementation of a basic ESN:
The first step was an implementation of an Echo State Network, in this case to predict Mackey-Glass' series.
It led to an ESN object, with some basic functions and methods.

* :ballot_box_with_check: A basic display:
The next part consisted in displaying this network, here simply by making it into a squared image, and showing the evolution of the activation.

* :ballot_box_with_check: Switch to a prediction of the sinus function.
Since it is easier to predict, it will be easier to see whether the network is completely broken or not. If it isn't, we can then switch back to Mackey-Glass.

* :ballot_box_with_check: The spatialized ESN:
This leads to the main part of the project: to give the neurons a position in the plan, before trying to use different properties and see how it learns.

* :clock1: Properly use the spatialization.
The plan is to try to check to what extent the properties of the ESN are kept, and how to adapt if there are differences.
This is currently undergoing.

## How to use it:
  This program uses an object "Spatial_ESN". To use it effectively, there are several steps required. if you only want to use the plot with a different input series, you can simply:

  * Initialize the "Spatial_ESN" object, given several hyper parameters (see doc of the function for more details): number_neurons, sparsity, number_input, number_output, spectral_radius, leak_rate
  * Import the data you want as an array.
  * Use the function compare_result with your data(see doc for the complete )

Everything is almost done by the simulation method. But if you want to do something more specific, here is how it is done:
  * Initialize the object
  * Use the begin_record method when you want to start recording the internal state.
  * Use the warmup method to initialize the reservoir, or manually with a while and the update method.
  * Train it using the train method (it is very important to use this, else the network can't work unless you do the regression manually)
  * Use the update method for as long as you want.
  * Use the end_record method to plot the internal state and eventually save it as a .mp4 file.


## What is left todo:
* :soon: Change the train function, so that only the connected neurons are updated.
* :soon: use set_aspect (from axis)
* :soon: use json format
* :soon: Use neurons modelisation.
* :soon: Fully check the working of the spatialized ESN
* :soon: Update the display function, and check whether it is working properly.
* :soon: Introduce different types of neurons (having different connecting behaviours)
* :soon: Test different properties on the connectivity of the neurons
* :soon: Try to use Voronoi Tesselation for display purpose.  
