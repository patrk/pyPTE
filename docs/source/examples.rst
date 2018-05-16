========
Examples
========

Currently the pyPTE package delivers three examples which shall demonstrate the usage of the phase transfer entropy method. 

Standard Kuramoto model
=======================

The standard Kuramoto model is a globally coupled system of linear differential equations of first order. It represents the phase behaviour of a set of coupled oscillators with respect to their intrinsinc frequencies and a global coupling strength.

Neural mass model
=================
This example incorporates an implementation of the stochastic non-linear dynamics of coupled cortical columns based on [Jansen1995]_ and [Wendling2000]_...

.. [Jansen1995] Ben H. Jansen and Vincent G. Rit. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biological Cybernetics*, 73:357-366, 1995. 

mne-python sample data set
==========================

This example illustrates how to extract data from a mne raw object, which can be fed into pyPTE. If you want to incorporate data from other software packages than MNE, please refer to the MNE documentation how to import raw data from other fileformats.
