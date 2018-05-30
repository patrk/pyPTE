.. ===========
.. About pyPTE
.. ===========

pyPTE is an open-source implementation of the Phase Transfer Entropy method based on the publications [Lobier2014]_ and [Hillebrand2016]_. It is fully implemented in Python and requires only the Python libraries NumPy and SciPy.

This implementation estimates the Phase Transfer Entropy which is defined for two given time-series X and Y with a known analysis lag delta as:

.. math::
 {PTE}_{X \rightarrow Y} &=&  H(\theta_Y(t), \theta_Y(t-\delta) \\ 
                         &+&  H(\theta_Y(t-\delta),\theta_X(t-\delta)) \\ 
                         &-&  H(\theta_Y(t-\delta)) \\
			 &-&  H(\theta_Y(t), \theta_Y(t-\delta), \theta_X(t-\delta))
, where the entropy terms H are defined as:

.. math::
 H(\theta_Y(t), \theta_Y(t)) = - \sum p(\theta_Y(t),\theta_Y(t-\delta)) log p(\theta_Y(t), \theta_Y(t-\delta)) \\
 H(\theta_Y(t-\delta), \theta_X(t-\delta)) = -\sum p(\theta_Y(t),\theta_X(t-\delta)) log p(\theta_Y(t), \theta_X(t-\delta)) \\
 H(\theta_Y(t-\delta)) = -\sum p(\theta_Y(t)) log p(\theta_Y(t)) \\
 H(\theta_Y(t-\delta), \theta_Y(t-\delta), \theta_X(t-\delta)) = -  \sum p(\theta_Y(t),\theta_Y(t-\delta), \theta_X(t-\delta)) log p(\theta_Y(t), \theta_Y(t-\delta), \theta_X(t-\delta)) \\
.. [Lobier2014] Muriel Lobier, Felix Siebenhühner, Satu Palva and J. Matias Palva. Phase transfer entropy: A novel phase-based measure for directed connectivity in networks coupled by oscillatory interactions. *NeuroImage*, 2014. `doi:10.1016/j.neuroimage.2013.08.056 <http://dx.doi.org/10.1016/j.neuroimage.2013.08.056>`_

.. [Hillebrand2016] Arjan Hillebrand, Preejaas Tewarie, Edwin van Dellen, Meichen Yu, Ellen W. S. Carbo, Linda Douw, Alida A. Gouw, Elisabeth C.W. van Straaten and Cornelies J. Stam. Direction of information flow in large-scale resting-state networks is frequency-dependent. *Proceedings of the National Academy of Sciences of the United States of America*, 113(14):3867-72, 2016. `doi:10.1073/pnas.1515657113 <http://dx.doi.org/10.1073/pnas.1515657113>`_ 
