Beam model
==========


The beam is modeled by macro-particles. Each beam macro-particle is 
characterized by its longitudinal position :math:`\xi_b`, 
transverse position :math:`r_b` or (:math:`x, y`), three components 
of momentum :math:`\vec{p}_b`, charge :math:`q_b`, and mass :math:`m_b`. 
Equations of motion for the macro-particles are

.. math::
  :nowrap:
  
  \begin{gather}
  \frac{d r_b}{d t} = v_{br}, \qquad
  \frac{d \xi_b}{d t} = v_{bz} - 1, \\
  \frac{d \vec{p}_b}{d t} = q_b \vec{E} + q_b \left[ \vec v_b \times \vec B \right], 
  \qquad
  \vec{v}_b = \frac{\vec{p}_b}{\sqrt{m_b^2 + p_b^2}}.
  \end{gather}

These equations are solved with the modified Euler's method (midpoint method). 
The fields acting on the macro-particle are linearly interpolated to 
the predicted macro-particle location at the half time step.
If a particle has a small longitudinal momentum and thus a high frequency 
of betatron oscillations, then the time step for this particle is automatically reduced. 

With no external magnetic field (:math:`B_0=0`), 
the angular momentum of beam particles must conserve, so the azimuthal 
component of the momentum :math:`p_{b\varphi}` is not changed, 
and reconstructed  from the condition :math:`r_b p_{b\varphi} = const`.