Units
=====

We use cylindrical coordinates :math:`(r, \varphi, \xi)` for the axisymmetric 
geometry and Cartesian coordinates :math:`(x, y, \xi)` for the 3d geometry. 
The beam propagates in positive :math:`\xi`-direction.

The code works with dimensionless quantities. 
Units of measure depend on some basic plasma density :math:`n_0`. 
It is recommended to use the initial unperturbed plasma density as :math:`n_0`. 
All times are in units of 
:math:`\omega_p^{-1}`, where :math:`\omega_p = \sqrt{4 \pi n_0 e^2/m}` is the electron 
plasma frequency, :math:`e` is the elementary charge, and $m$ is the electron mass. 
All distances are in units of :math:`c/\omega_p`. The unit velocity is :math:`c`. 
The notation used and units of measure for various quantities are given 
in old `manual <https://lcode.info/site-files/manual.pdf>`_.
