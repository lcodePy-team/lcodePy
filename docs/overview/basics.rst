Basics
=======


In the code, the simulation window moves with the speed of light, 
and the quasistatic approximation is used to calculate the plasma response. 
Beams and plasma are modeled by macroparticles. 
The code is furnished with extensive diagnostic tools which include the ability of in-flight graphical 
presentation of the results.

The essence of the quasistatic approximation is illustrated by Figure 1 (cylindrical geometry). 
When we calculate the plasma response, 
the beam is considered as a ''rigid'' (non-evolving in time) distribution of 
charges and currents propagating at the speed of light :math:`c`. 
The fields produced by this beam depend on the longitudinal coordinate :math:`z` 
and time :math:`t` only in the combination :math:`\xi=z-ct` and can be found 
layer-by-layer starting from the beam head. 
Since the beam does not change, all particles starting from some transverse position :math:`r_0` 
copy the motion of each other, and their parameters (transverse coordinate and momenta) can be found as 
functions of :math:`\xi`.
Thus, a plasma macroparticle in the quasistatic model is not a ''big'' particle, 
but a ''string'' composed of real particles entering the simulation window 
at the same transverse coordinate and with the same initial momentum. 
This greatly reduces the memory required to store the plasma particles.

.. figure:: ../illustrations/f-quasistat.png
   :align: center

   Fig. 1: Geometry of the problem (a), and trajectory of 
   a plasma particle in the simulation window (b).

The calculated fields are then used to push the beam particles. 
For highly relativistic beams, the time step :math:`\Delta t` for the beam particles 
can be made large, which speeds up simulations by several orders of magnitude. 
The quasistatic approximation is thus useful if and only if the time scale 
of beam evolution is much longer than the period of the plasma wave.


Useful papers:
~~~~~~~~~~~~~~

Various details of LCODE and its underlying physics are described in the following papers:

.. #. K.V. Lotov, *Simulation of ultrarelativistic beam dynamics in plasma wake-field accelerator.* Phys. Plasmas **5**, 785 (1998). --- The fluid plasma model.

#. K.V. Lotov, *Fine wakefield structure in the blowout regime of plasma wakefield accelerators.* `Phys. Rev. ST - Accel. Beams <https://doi.org/10.1103/PhysRevSTAB.6.061301>`_ **6**, 061301 (2003). --- **The beam model and the 2d plasma model.**

#. K.V. Lotov, *Blowout regimes of plasma wakefield acceleration.* `Phys. Rev. E <https://doi.org/10.1103/PhysRevE.69.046405>`_ **69**, 046405 (2004). --- **Energy fluxes in the co-propagating window.**

#. A.P. Sosedkin, K.V. Lotov, *LCODE: A parallel quasistatic code for computationally heavy problems of plasma wakefield acceleration.* `Nuclear Instr. Methods A <http://dx.doi.org/10.1016/j.nima.2015.12.032>`_ **829**, 350 (2016). --- **Parallelization.**

#. P.V. Tuev, R.I. Spitsyn, K.V. Lotov, *Advanced Quasistatic Approximation.* `Plasma Physics Reports <https://doi.org/10.1134/S1063780X22601249>`_ **49**, 229 (2023). [`arxiv <https://doi.org/10.48550/arXiv.2205.04390>`_][`in Russian <https://sciencejournals.ru/cgi/getPDF.pl?jid=fizplaz&year=2023&vol=49&iss=2&file=FizPlaz2260143Tuev.pdf>`_] --- **Advantages of quasistatic approximation, its applicability area and possible extensions.**

#. R.N. Spitsyn, *Numerical realization of quasistatic model of laser driver for plasma wakefield acceleration* (in Russian). `Master theses <https://star.inp.nsk.su/~dep_plasma/dip/Spitsyn_m.pdf>`_, Novosibirsk State University (2016). --- **2d laser solver.**

#. I.Yu. Kargapolov, N.V. Okhotnikov, I.A. Shalimova, A.P. Sosedkin, and K.V. Lotov, *LCODE: Quasistatic code for simulating long-term evolution of three-dimensional plasma wakefields*. [`arxiv <https://doi.org/10.48550/arXiv.2401.11924>`_] --- **3d plasma and beam solvers, iteration loop of the plasma solver, declustering.**

#. K.V. Lotov, V.I. Maslov, I.N. Onishchenko, and E.N. Svistun, *Resonant excitation of plasma wakefields by a non-resonant train of short electron bunches.* `Plasma Phys. Control. Fusion <http://dx.doi.org/10.1088/0741-3335/52/6/065009>`_ **52**, 065009 (2010). --- **Discussion on applicability of quasistatic codes to simulations of long beams.**