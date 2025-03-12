Welcome to LCODE's documentation!
=================================

LCODE is a free software for numerical simulation of
particle beam-driven plasma wakefield acceleration.
LCODE is based on the quasistatic approximation, is capable
of simulating in 2D and 3D geometries, and can use GPUs and CPUs.

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
   section#usage,
   section#overview,
   section#underlying-physics{
       display:none;
   }
   </style>
   

Overview
-----------------
.. toctree::
   :caption: OVERVIEW
   :maxdepth: 3
   :hidden:

   overview/basics
   overview/quickstart
   
Installation
-----------------
.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1
   :hidden:

   install/installation
   
Usage
-----------------
.. toctree::
   :caption: USAGE
   :maxdepth: 1
   :hidden:

   usage/units
   usage/simulation
   usage/configuration
   usage/beam
   usage/diagnostics
   usage/examples

Underlying physics
--------------------
.. toctree::
   :caption: UNDERLYING PHYSICS
   :maxdepth: 3
   :hidden: 

   underlying_physics/QSA
   underlying_physics/plasma_model
   underlying_physics/beam_model
