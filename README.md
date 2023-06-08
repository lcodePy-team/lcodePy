# lcodePy

## Overview 

LCODE is a free software for numerical simulation of
particle beam-driven plasma wakefield acceleration.
LCODE is based on the quasistatic approximation, capable
of simulation in 2D and 3D geometry, and can use GPUs and CPUs.

For now, this is new and experimental software. This is
a complete overhaul of the old C version in Python.

You can also find a more mature 2D version of LCODE at
http://lcode.info/.

## Instalation 

We use [Anaconda](https://www.continuum.io/why-anaconda)we and recommend 
installing lcode in a separate environment. 
Any other python installation should work fine, but has not been tested. 


- Create a new env and install the dependency:
```
conda create -n lcode-env -c conda-forge numba numpy scipy matplotlib mpi4py
```
or 
```
conda env create -f conda-env.yml  
```
where `conda-env.yml` is avalible in sources.

- Acivate new env:
```
conda activate lcode-env
```

- **Optional**: install cupy in order to run simulation on GPU. 
```
conda install -c conda-forge cupy
```

- Install lcode:
```
pip install lcode
```
or download sources from GitHub and run the forlowing command
in downloaded directory:
```
pip install .
```


----------

Fill free to contact `team@lcode.info` for assistance with it.

