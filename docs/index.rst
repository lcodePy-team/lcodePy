Welcome to LCODE's documentation!
=====

LCODE is a free software for numerical simulation of
particle beam-driven plasma wakefield acceleration.
LCODE is based on the quasistatic approximation, capable
of simulation in 2D and 3D geometry, and can use GPUs and CPUs.

For now, this is new and experimental software. This is
a complete overhaul of the old C version in Python.

You can also find a more mature 2D version of    LCODE at
http://lcode.info/.

.. note::

   This project is under active development.


.. raw:: html

   <style>
   /* front page: hide chapter titles
    * needed for consistent HTML-PDF-EPUB chapters
    */
   section#installation,
   section#usage {
       display:none;
   }
   </style>
   
Installation
-----
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:

   install/installation
   
Usage
-----
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/config
   usage/beam
   usage/diagnostics
   usage/simulation
