Plasma model
===============

.. warning::
  To be updated.

Plasma particles
----------------


3d geometry
~~~~~~~~~~~
.. warning::
  To be updated.


Cylindrical geometry
~~~~~~~~~~~~~~~~~~~~

Each plasma macro-particle is characterized by 7 quantities: 
transverse coordinate - :math:`r`, 
three components of the momentum - (:math:`p_r, p_\varphi, p_z`), 
mass - :math:`M`, charge - :math:`q`, and ordinal number. 
Parameters of plasma macro-particles are initialized ahead of the 
beam (at :math:`\xi=0`) and then advanced slice-by-slice according to equations

.. math::
  :nowrap:

  \begin{gather}
  \frac{d \vec{p}}{d \xi} = \frac{d \vec{p}}{d t}
  \frac{d t}{d \xi} = \frac{q}{v_z - 1} \left( \vec{E} +
  \left[ \vec{v} \times \vec{B} \right] \right),\\ 
   
  \frac{d r}{d \xi} = \frac{v_r}{v_z - 1}, \qquad
  \vec{v} = \frac{\vec{p}}{\sqrt{M^2+p^2}}.
  \end{gather}

If a particle hits the wall at :math:`r=r_\text{max}`, it is returned 
to the simulation area to some near-wall location with zero momentum. 
Plasma current and charge density are obtained by summation over plasma 
macro-particles lying within a given radial interval:

.. math::

  \vec{j} = A \sum_i \frac{q_i \vec{v}_i}{1-v_{z,i}}, \qquad
  \rho = A \sum_i \frac{q_i}{1-v_{z,i}},

where :math:`A` is a normalization factor. The denominator in deposition
appears since the contribution of a ''particle tube'' to density
and current depends on the macro-particle speed in the simulation
window.

Fields
------

The equations solved for the fields are Maxwell equations, which 
in the dimensionless variables take the form

.. math::
  :nowrap:

  \begin{gather}
  rot\,\vec{B} = \vec{j} + \vec{j}_b + \frac{\partial \vec{E}}{\partial t}, \qquad
  rot\,\vec{E} = - \frac{\partial \vec{B}}{\partial t}, \\ 
  div\,\vec{B} = 0, \qquad div\,\vec{E} = \rho + \rho_b.
  \end{gather}

Under the quasi-static assumption, the derivatives change as follows:

.. math::

    \frac{\partial}{\partial z} = - \frac{\partial}{\partial t} 
    = \frac{\partial}{\partial \xi},

3d geometry
~~~~~~~~~~~

Transverse fields
^^^^^^^^^^^^^^^^^

In QSA, transverse Maxwell's equations can be rewritten as:

.. math::

   \Delta_\perp E_x &= \frac{\partial \rho}{\partial x} - \frac{\partial j_x}{\partial \xi}

   \Delta_\perp E_y &= \frac{\partial \rho}{\partial y} - \frac{\partial j_y}{\partial \xi}

   \Delta_\perp B_x &= \frac{\partial j_y}{\partial \xi} - \frac{\partial j_z}{\partial y}

   \Delta_\perp B_y &= \frac{\partial j_z}{\partial x} - \frac{\partial j_x}{\partial \xi}

with mixed boundary conditions.


Longitudinal fields
^^^^^^^^^^^^^^^^^^^
In QSA, equation for :math:`B_z` can be rewritten as:

.. math::

   \Delta_\perp B_z = \frac{\partial j_x}{\partial y} - \frac{\partial j_y}{\partial x}

with Neumann boundary conditions (:math:`\partial B_z/ \partial r_\perp = 0`).

In QSA, equation for :math:`E_z` can be rewritten as:

.. math::

   \Delta_\perp E_z = \frac{\partial j_x}{\partial x} - \frac{\partial j_y}{\partial y}

with Dirichlet boundary conditions (:math:`E_z = 0`).


Cylindrical geometry
~~~~~~~~~~~~~~~~~~~~

In cylindrical geometry Maxwell's equations take the following form:

.. math:: 
  :nowrap:

  \begin{gather}
  \frac{1}{r} \frac{\partial}{\partial r} r E_r
  = \rho + \rho_b - \frac{\partial E_z}{\partial \xi}, \\
  \frac{1}{r} \frac{\partial}{\partial r} r B_r
  = - \frac{\partial B_z}{\partial \xi},\\ 

  \frac{1}{r} \frac{\partial}{\partial r} r (E_r - B_\varphi)
  = \rho - j_z, \\
  \frac{\partial E_z}{\partial r} = j_r, \\
  \frac{\partial B_z}{\partial r} = - j_\varphi, \\
  E_\varphi = -B_r.
  \end{gather}


Here we neglect the components :math:`j_{br}`` and :math:`j_{b\varphi}` 
of the beam current and put :math:`j_{bz} = \rho_b`, 
since beam particles are assumed to move mostly in :math:`z`-direction. 
To provide stability of the algorithm, we solve in finite
differencesd the following equations on :math:`E_r` and :math:`B_r` :

.. math::
  :nowrap:

  \begin{gather}
  \frac{\partial}{\partial r} \frac{1}{r}
  \frac{\partial}{\partial r} r E_r - E_r
  = \frac{\partial (\rho + \rho_b)}{\partial r}
  - \frac{\partial j_r}{\partial \xi} - \tilde{E}_r, \\
  \frac{\partial}{\partial r} \frac{1}{r}
  \frac{\partial}{\partial r} r B_r - B_r
  = \frac{\partial j_\varphi}{\partial \xi} - \tilde{B}_r,
  \end{gather}

where :math:`\tilde{E}_r` and :math:`\tilde{B}_r` are some predictions 
for fields :math:`E_r` and :math:`B_r`. 
Subtraction ofthe fields (with or without the tildes) from both sides 
of the equalities does
not produce a big error if the predictions are close to final fields.
The boundary conditions for equations are
those of a perfectly conducting tube of the radius :math:`r_\text{max}`:

.. math::
  :nowrap:

  \begin{gather}
  E_r (0) = B_r (0) = B_\varphi (0) = E_z (r_\text{max}) = B_r (r_\text{max}) = 0,\\ 
  
  \int_0^{r_\text{max}} 2 \pi r B_z \, dr = \pi r_\text{max}^2 B_0,
  \end{gather}

where :math:`B_0` is an external longitudinal magnetic field, if any (the
presence of this field does not change the axial symmetry of the system). 

