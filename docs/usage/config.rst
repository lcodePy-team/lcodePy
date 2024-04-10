Config
======

The config is a python dictionary that is passed to the `Simulation` class to configure the simulation. 

It is usually specified as follows:

.. code-block:: python

    config = {
        'geometry': '2d',
        'time-limit': 1,
        'time-step': 1,
        # ...
    }
    sim = Simulation(config=config, ...)

This section describes the parameters that can be passed to the config and their default values.

Geometry
---------------------------

* ``geometry`` (`3d | circ`) optional (default `circ`)
    To Do


Grid parameters
---------------------------

* ``window-width`` (`float`) optional (default `5.0`)
    To Do

* ``window-width-step-size`` (`float`) optional (default `0.05`)
    To Do

* ``window-length`` (`float`) optional (default `15.0`)
    To Do

* ``xi-step`` (`float`) optional (default `0.05`)
    To Do

* ``time-limit`` (`float`) optional (default `200.5`)
    To Do

* ``time-step`` (`float`) optional (default `25`)
    To Do

* ``continuation`` (`n | y | Y`) optional (default `n`)
    To Do

Parameters of plasma model
---------------------------

* ``plasma-particles-per-cell`` (`int`) optional (default `10`)
    The number of plasma particles per one cell must be the square of a number in 3d. This parameter will be adjusted if 3d geometry is chosen by finding the closest square number to plasma-particles-per-cell parameter.



Parameters of beam model
---------------------------
* ``rigid-beam`` (`bool`) optional (default `False`)
    To Do

* ``beam-substepping-energy`` (`flaot`) optional (default `2`)
    To Do

* ``focusing`` (`n | y`) optional (default `n`)
    To Do

* ``foc-period`` (`float`) optional (default `100`)
    To Do

* ``foc-strength`` (`float`) optional (default `0.1`)
    To Do


CPU/GPU
--------------------

* ``processing-unit-type`` (`cpu | gpu`) optional (default `cpu`)
    Here we set the type of processing unit: CPU or GPU.

    For now, GPU can be used only for 3d simulations.

Noise reductor
-----------------------

Read more in the section :doc:`../details/noisereductor`

* ``enable-noise-filter`` (`bool`) optional (default `True`)

* ``filter-window-length`` (`int`) optional (default `5`)

* ``filter-polyorder`` (`int`) optional (default `3`)
    
* ``filter-coefficient`` (`float`) optional (default `0`)

* ``damping-coefficient`` (`float`) optional (default `0`)

* ``dx-max`` (`float`) optional (default `1`)

* ``field-solver-subtraction-coefficient`` (`float`) optional (default `1`)
