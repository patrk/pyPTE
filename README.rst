===========
pyPTE
===========

pyPTE is an open-source implementation of the Phase Transfer Entropy method based on the publications [Lobier2014]_ and [Hillebrand2016]_. It is fully implemented in Python and requires only the Python libraries NumPy and SciPy.

.. [Lobier2014] Muriel Lobier, Felix Siebenhühner, Satu Palva and J. Matias Palva. Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions. *NeuroImage*, 2014. `doi:10.1016/j.neuroimage.2013.08.056 <http://dx.doi.org/10.1016/j.neuroimage.2013.08.056>`_

.. [Hillebrand2016] Arjan Hillebrand, Preejaas Tewarie, Edwin van Dellen, Meichen Yu, Ellen W. S. Carbo, Linda Douw, Alida A. Gouw, Elisabeth C.W. van Straaten and Cornelies J. Stam. Direction of information flow in large-scale resting-state networks is frequency-dependent. *Proceedings of the National Academy of Sciences of the United States of America*, 113(14):3867-72, 2016. `doi:10.1073/pnas.1515657113 <http://dx.doi.org/10.1073/pnas.1515657113>`_ 


==================
Installation Guide
==================

The prerequisites for this package are:

Mandantory:

- a working Python installation, version 3.6 or higher
- git
- NumPy
- SciPy

Optional:

- mne-python
- pandas
- seaborn

Recommended:

To prevent Python module incompatibilities using a virtual environment like 

- conda
- pyenv

is highly recommendable. If you are planning to use mne-python, an Anaconda3 installation is mandandory.

Step 1: Download pyPTE via GitHub
=================================

Clone into the public GitHub repository using:
::
	> git clone https://github.com/patrk/pyPTE.git

Step 2: Build pyPTE
===================
Build the pyPTE package and make it available to your Python interpreter by:
::	
	> cd pyPTE
	> python setup.py install

Step 3: Test pyPTE
==================
To test the installation of pyPTE simply run:
::
	> cd test
	> py.test
  
==================
Examples
==================
  
 
