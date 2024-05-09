.. |nbsp| unicode::  xa0
   :trim:

Configuration
=============

The config is a python dictionary that is passed to the `Simulation` 
class to configure simulation parameters. 

It is usually specified as follows:

.. code-block:: python

    config = {
        'geometry': 'circ',
        'time-limit': 1,
        'time-step': 1,
        # ...
    }
    sim = Simulation(config=config, ...)

This section describes the parameters that can be passed 
to the config and their default values.


The `runas.py` file contains the actual configuration parameters that were used 
to run the simulation. The file name can be specified as 
:code:`sim = Simulation(runas_filename='filename.py', ...)`.
If `runas_filename` is an empty string, this file is not generated.



.. role:: underlined

Geometry
---------------------------

* ``'geometry': 'circ'`` ('circ'|'3d') 
    Geometry of the problem being simulated. 

    * **'circ'** - cylindrical geometry with axial symmetry.
    * **'3d'** - 3d cartesian geometry.

Hardware
-----------

* ``'processing-unit-type': 'cpu'`` ('cpu'|'gpu')
    * **'cpu'** - central processor unit.
    * **'gpu'** - graphics processing unit. Avalible for 3d geometry only.

Grid parameters
---------------------------

* ``'window-width': 5.0`` (`float`)
    Transverse size of the window from the central axis to boundary. 
    The window width is adjusted to the closest "good" value
    to ensure fft performance for 3d.

* ``'transverse-step': 0.05`` (`float`)
    Transverse grid step: dr for cylindrical geometry and dx=dy for 3d.    

* ``'window-length': 15.0`` (`float`)
    Window length along :math:`\xi`-axis.

* ``'xi-step': 0.05`` (`float`)
    Grid step along :math:`\xi`-axis.

* ``'time-limit': 200.5`` (`float`)
    Beam evolution time. 

* ``'time-step': 25`` (`float`) 
    Time step size for beam evolution.
    Thus, the plasma response will be calculated 
    for each :math:`z=c` \* **'time-step'** up to the **'time-limit'**, 
    where :math:`c`` is speed of light. 

Parameters of plasma model
---------------------------

* ``'plasma-particles-per-cell': 10`` (`int`) 
    The number of plasma particles per one cell. It must be equal to the square 
    of an integer in 3d, otherwise it will be corrected to the nearest 
    suitable value (to 9 by default).


* ``'ion-model': 'mobile'`` **('mobile'|'background')**
    * **'mobile'** - ions are simulated as macroparticles.
    * **'background'** -  ions are stationary and affect the wave 
      as a background charge distribution.

* ``'ion-mass': 1836`` (`float`)
    Mass of plasma ions in unita of electron mass.


Profiling of the plasma distribution
---------------------------------------------

* ``'plasma-shape': 'legacy'`` **('legacy'|function(z, x, y))**
    * **function** - python `function(z, x, y) -> density_weights`, where
      z - position of plasma layer to be calculated, 
      x and y - transverse positions of the plasma macroparticles. 
      To set an arbitrary density distribution, the charges and masses 
      of all macroparticles are multiplied by the `density_weights`.
      This function will be called each time the plasma is initialized. 
      It is assumed that x = r and y = 0 in cylindrical geometry.


      Example of an arbitrary density function:

      .. code-block:: python

          import numpy as np

          def foo(z, x, y):
              """
              Exponential growth from 0 to 1 along z with a transverse Gaussian distribution.
              """
              z0, r0 = 100, 5 # profile parameters 
              r = np.sqrt(x**2 + y**2)
              density_weights = np.exp(-r**2 / r0**2) # transverse weights
              density_weights *= (1 - np.exp(-z**2 / z0**2)) # longitudinal weight
              return  density_weights

      .. Note::
          If you want to simulate with GPU, use ``cupy`` instead of ``numpy``.
    
    * **'legacy'** - the plasma profile is specified as described later in this subsection.

* ``'plasma-width': 2`` (`float`)
    Main parameter of initial transverse plasma density distributions, :math:`r_p`.

* ``'plasma-width-2': 0`` (`float`)
    Auxiliary parameter of initial transverse plasma density distributions, :math:`r_{p2}`.

* ``'plasma-density-2': 0.5`` (`float`)
    Auxiliary parameter of initial transverse plasma density distributions, :math:`n_{r2}`.

* ``'plasma-rshape': 1`` (`multiple choice`)
    The initial transverse profile of the plasma density.

    **In the case of a transverse profile independent of z (`int` or `string`):**

        * **1** or **'uniform'**: Uniform (:math:`n_r = 1`) up to the boundary.
        
        * **2** or **'stepwise'**: Uniform (:math:`n_r = 1`) up to :math:`r_p`.
          If :math:`r_{p2} < r_p`, then zero at :math:`r > r_p`, otherwise linear decreases
          from 1 to 0 for :math:`r_p < r < r_{p2}` and zero at :math:`r > r_{p2}`.
        
        * **3** or **'channel'**: :math:`n_r = n_{r2}` up to :math:`r_p`.
          If :math:`r_{p2} > r_p`, then uniform (:math:`n_r = 1`) at :math:`r > r_p`,
          otherwise linear increases from :math:`n_{r2}` to 1 for :math:`r_p < r < r_{p2}` 
          and :math:`n_r = 1` at :math:`r > r_{p2}`.
        
        * **4** or **'parabolic-channel'**: :math:`n_r = n_{r2} (1 + r^2/r_p^2)`
          as long as :math:`n_r < 1`, otherwise :math:`n_r = 1`.
        
        * **5** or **'gaussian'**: :math:`n_r = exp(-r^2/r_p^2)`, zero at
          :math:`r > 5r_p`.

    **In case of z-dependent transverse profile (`string`):** 
    
    The transverse profile 
    is specified in consecutive segments, one segment per line. 
    Each line consists of a segment length (`float`), 
    radial profile type as disciber above (`int` or `string`) and profile parameters
    :math:`r_p, r_{p2}, n_{p2}` (`float`). The plasma density distribution 
    is assumed to be **uniform** after passing all segments.
    
    Example: 
    
    .. code-block:: python

        config = {
            # ...
            'plasma-rshape': '''
                100 1 0 0 0 
                300 stepwise 1 2 0.5
                100 3 2 0 0.3
                '''
            # ...
        }

    The transverse profile of plasma density is **'uniform'** over :math:`100\, c/w_p`.  
    Then it is **'stepwise'** with :math:`n_r = 1` for :math:`r < 1`, 
    linear decrease from 1 to 0 for :math:`1 < r < 2` and :math:`n_r = 0` for :math:`r > 2` 
    over :math:`300\, c/w_p`. After it is **'channel'** with :math:`n_r = 0.3` 
    for :math:`r < 2` and :math:`n_r = 1` for :math:`r > 2` over :math:`100\, c/w_p`.




* ``'plasma-zshape': ''`` (`string`)
    This option defines a longitudinal plasma density profile, 
    i.e. the plasma density :math:`n_z(z)`. In the case of transversely non-uniform 
    plasmas, :math:`n_z` is multiplied by the transverse density profile 
    :math:`n_r`. The longitudinal profile is specified in 
    consecutive segments, one segment per line. Each line consists of a segment 
    length (`float`), density at the beginning of the segment (`float`), 
    a letter defining the segment shape (only **'L'** avalible) and 
    density at the end of the segment (`float`). :math:`n_z = 1` 
    after passing all segments.
    
    Example: 
    
    .. code-block:: python

        config = {
            # ...
            'plasma-zshape': '''
                100 0 L 1.2
                300 1.2 L 1.2
                100 1.2 L 0
                '''
            # ...
        }

    The plasma density increases linearly from 0 to 1.2 over :math:`100\, c/w_p`, 
    then remains constant over :math:`300\, c/w_p` and decreases from 1.2 to 0 
    over :math:`100\, c/w_p`.




Parameters of beam model
---------------------------
* ``'beam-substepping-energy': 2`` (`flaot`)
    Substepping energy for the beam, :math:`W_{ss}`
    The threshold of reducing the time step for beam particles. For each beam particle,
    the minimal integer :math:`l` is found that meets the condition  
    :math:`2^{2l}\sqrt{m_b^2 + p_{bz}^2} > W_{ss}`, 
    and the reduced :math:`dt'` = **'time-step'** / :math:`2^l` is then determined.
    Plasma fields are calculated with periodicity **'time-step'**, and each beam
    particle is propagating in these fields with its own time step :math:`dt'`.
    This feature is particularly useful if low energy beam particles are present in the
    system, which otherwise would require undesirable reduction of the main **'time-step'**. 



Advanced features
-----------------------

* ``'declustering-enabled': False`` (`bool`)
	Switches the declustering of plasma electrons on. 
	This feature is described in https://doi.org/10.48550/arXiv.2401.11924

	The following four parameters are used for 3d declustering. In 2d they are ignored. 
 

	* ``'declustering-averaging': 3`` (`int`)
		The sigma of the kernel in Gaussian Fourier filter used for averaging particle displacement inhomogeneities,
		measured in grid steps.

	* ``'declustering-force': 0.3`` (`float`)
		The coefficient (:math:`K`) characterizing the restoring force.

	* ``'declustering-damping': 0.1`` (`float`)
		The coefficient (:math:`\kappa`) characterizing the damping force.

	* ``'declustering-limit': 1e-3`` (`float`)
		The maximum noise displacement of particles (:math:`D_0`). 
		Particles with larger noise displacements are not affected by declustering.
		(To Do: in units of ``transverse-step``).


.. * ``field-solver-subtraction-coefficient`` (`float`) optional (default `1` + ``transverse-step`` ).
	A coefficient that controls stability and accuracy of the field solver. 
	It is described in https://doi.org/10.48550/arXiv.2401.11924 as :math:`\mu` in Eqs. (8)-(11).
	The solver becomes unstable at :math:`\mu \lessapprox 1` and lose accuracy at :math:`\mu \gg 1`.
	Changing this coefficient makes sense if the unperturbed plasma density is not equal to 1.


