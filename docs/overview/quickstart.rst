Quickstart
============

To start running your first simulation using lcode, all you need to do is run these three commands:

.. code-block:: shell

    pip install lcode
    python -m lcode get ssm
    python run.py


Explanation
-----------------

First, you need to install the `lcode` package along with its dependencies.

If you want to work with a virtual environment, then you need to create and activate it beforehand.

.. code-block:: shell

    # linux
    python -m venv venv
    source venv/bin/activate # linux

.. code-block:: powershell
    
    # Windows
    python -m venv venv
    .\venv\scripts\Activate

Then install lcode:

.. code-block:: shell

    pip install lcode

`lcode` provides a tool for generating example files for running simulations. The `ssm` file provides the raw data for the simulation described in the article. The following command generates a `run.py` file that specifies all the simulation data:

.. code-block:: shell

    python -m lcode get ssm

After generation, we can change the simulation parameters in the run.py file. More details about the parameters are written in the corresponding sections:

- :doc:`../usage/config` - About general simulation settings
- :doc:`../usage/beam` - About beam settings
- :doc:`../usage/diagnostics` - About selecting diagnostics
- :doc:`../usage/simulation` - About starting the simulation

.. warning::
    If you want to use GPU, you need to install the `cupy` library yourself [`Install Cupy`_].
.. _`Install Cupy`: https://docs.cupy.dev/en/stable/install.html
    

Now all that's left is to run the calculation:


.. code-block:: shell

    python run.py


.. raw:: html

    <script>
    window.addEventListener('load', function() {
        console.log('asd')
        document.querySelectorAll('a.reference.external').forEach(function(link) {

            link.target = '_blank';
        });
    });
    </script>
