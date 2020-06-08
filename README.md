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
TODO


## What is left todo:

* :soon: Fully check the working of the spatialized ESN
* :soon: Update the display function, and check wether it is working properly.
* :soon: Introduce different types of neurons (having different connecting behaviours)
* :soon: Test different properties on the connectivity of the neurons
* :soon: Try to use Voronoi Tesselation for display purpose.  
