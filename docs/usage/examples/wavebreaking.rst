Transverse wavebreaking
---------------------------------------------------

.. |rho| image:: wavebreaking/rho_-00060.00.jpg
   :height: 150

.. |Dmax| image:: wavebreaking/Dmax.png
   :height: 100

.. |ne| image:: wavebreaking/ne.png
   :height: 150

This example shows transverse wavebreaking of a positron-driven nonlinear wakefield with declustering on and off.
The runs reproduce two lines in Fig.20 of `this paper <https://doi.org/10.48550/arXiv.2401.11924>`_.

* Run with declustering:  :download:`wavebreaking-on.py <wavebreaking/wavebreaking-on.py>`.

* Run without declustering:  :download:`wavebreaking-off.py <wavebreaking/wavebreaking-off.py>`. Option ``enable-noise-filter`` is *True* for the displacement diagnostics to work.

* Post-processing (Jupyter notebook):  :download:`wavebreaking-figs.ipynb <wavebreaking/wavebreaking-figs.ipynb>` and images that it produces: 

|rho| |Dmax| |ne|

Run with declustering:

.. literalinclude:: wavebreaking/wavebreaking-on.py
   :language: python