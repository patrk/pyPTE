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

