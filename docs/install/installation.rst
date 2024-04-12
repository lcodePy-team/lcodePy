Installation
===================================

Pip
---------------

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


From source
---------------

Create and activate the virtual environment as at the beginning of the :ref:`pip` section.

Then clone the lcode source code to yourself:

.. code-block:: shell

    git clone https://github.com/lcodePy-team/lcodePy.git
    cd lcodepy

Now install all dependencies and lcode itself

.. code-block:: shell

    pip install -r requirements.txt
    pip install -e .

.. note::
    If you want to use GPU, you need to install the `cupy` library yourself [`Install Cupy`_].
.. _`Install Cupy`: https://docs.cupy.dev/en/stable/install.html

Conda
---------------


We recommend installing lcode in a separate environment. 
Any other python installation should work fine, but has not been tested. 


- Create a new environment and install the dependencies:

.. code-block::

    conda create -n lcode-env -c conda-forge numba numpy scipy matplotlib mpi4py

or 

.. code-block::

    conda env create -f conda-env.yml  

where `conda-env.yml` is avalible in sources.

- Acivate the new environment:

.. code-block::

    conda activate lcode-env


- **Optional**: in order to run simulations on GPU, add cupy and other necessary packages to the line of dependencies when creating a new eviroment. Check in advance if you have the drivers for your GPU installed. It is not necessary to install CUDA Toolkit in advance. For any other questions about cupy, please check https://docs.cupy.dev/en/stable/install.html

.. code-block::

    conda create -n lcode-env -c conda-forge numba numpy scipy matplotlib cupy


.. raw:: html

    <script>
    window.addEventListener('load', function() {
        console.log('asd')
        document.querySelectorAll('a.reference.external').forEach(function(link) {

            link.target = '_blank';
        });
    });
    </script>
