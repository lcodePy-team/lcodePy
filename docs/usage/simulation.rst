Simulation
===================================

A typical simulation run is organized using the Simulation class.  
It contains all information about plasma and beam parameters, 
solver configuration and required diagnostics.   

.. autoclass:: lcode.Simulation()
    
    .. .. method:: __init__(config, beam_parameters, diagnostics, runas_filename)
        
    ..     asd

    .. automethod:: __init__(config, beam_parametrs, diagnostics, runas_filename)
    .. automethod:: step


Example:

.. code-block:: python

    from lcode import Simulation # import 
    
    ### set simulation parametrs 
    config = {
        'geometry': 'circ',
        'time-limit': 100,
        'time-step': 10,
    }

    beam = {'current': 0.1}
    ###


    sim = Simulation(config=config, beam_parameters=beam) # init simulation
    sim.step() # run simulation
