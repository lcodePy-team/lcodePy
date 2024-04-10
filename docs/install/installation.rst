Installation
===================================


We use https://www.continuum.io/why-anaconda and we recommend 
installing lcode in a separate environment. 
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


- Install lcode:

.. code-block::

    pip install lcode

or download sources from GitHub and run the forlowing command
in downloaded directory:

.. code-block::

    pip install .

