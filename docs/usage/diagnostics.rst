Diagnostics
===================================

To use diagnostics, you must import corresponding classes and include a list of diagnostics in the Simulation:

.. code-block:: python

    from lcode.simulation import Simulation
    from lcode.diagnostics import FXiDiag, FXiType, OutputType, ParticlesDiag
    from lcode.diagnostics import SliceDiag, SliceType, SliceValue
    sim = Simulation(config=config, diagnostics = [Diag1(...), Diag2(...), ...])

This section will describe what can be transferred to the diagnostics list.

.. note::
    You can use several diagnostics of the same type, but be sure to specify a different *directory_name* for each.

FXiDiag
----------------------------

These keys control output of various quantities as functions of :math:`\xi`, either in the form of a graph, or in the form of a data array.

.. py:class:: lcode.diagnostics.FXiDiag(output_period, saving_xi_period, f_xi, output_type, probe_lines, directory_name)

    - **output_period** (default: *100*)
    - **saving_xi_period** (default: *100*)
    - **f_xi** (default: *FxiType.Ez*)
    - **output_type** (default: *OutputType.NUMBERS*)
    - **probe_lines** (default: *0*)
    - **directory_name** (default: *None*)


output_period
^^^^^^^^^^^^^^^^^^^^^^^^

The period at which the diagnostics is triggered. If ∆t\ :sub:`out`\  < ∆t, then the diagnostics is called at each time step.

saving_xi_period
^^^^^^^^^^^^^^^^^^^^^^^^

The period in :math:`\xi` at which data is dumped to disk.

.. warning::

    This property is experimental. Choosing a small value may slow down program execution.

f_xi
^^^^^^^^^^^^^^^^^^^^^^^^

Selecting a value to output.

- ``FXiType.E``: All components of the electric field. You can also select the component you want to display by using *Ex, Ey, Ez, Er, Ef* instead of *E*.
- ``FXiType.B``: All components of the magnetic field. You can also select the component you want to display by using *Bx, By, Bz, Bf* instead of *B*.
- ``FXiType.n``: Electron and ion density of the plasma. You can also select the component you want to display by using *ne* or *ni* instead of *n*.
- ``FxiType.rho_beam``: Charge density of the beam.
- ``FxiType.Phi``: Wakefield potential (currently only works in 3D).

``FXiType`` is imported from ``lcode.diagnostics``.

.. note::

    You can combine any values: ``FxiType.Ez | FxiType.Bx | FxiType.n``

output_type
^^^^^^^^^^^^^^^^^^^^^^^

Data output format. It can be ``OutputType.NUMBERS``, ``OutputType.PICTURES`` or ``OutputType.ALL``, where ``OutputType`` is imported from ``lcode.diagnostics``.

- ``OutputType.NUMBERS``: Output data in the form of numpy structured array.
- ``OutputType.PICTURES``: Output data in the form of png plots.
- ``OutputType.ALL``: Two above options at once.

probe_lines
^^^^^^^^^^^^^^^^^^^^^^^^

Selecting an offset in **x**, **y** (**r** for *2D*) from the axis.

Use a *number*, *list* or *np.ndarray*.

- When using the number **x**, the resulting shift is **(x, 0)** [or **x** in *r* for *2D*].
- When using a one-dimensional array **[x1, x2, ...]**, the diagnostics will output data for shifts **(x1, 0), (x2, 0), ...** [or x1, x2, ... for *2D*].
- When using a two-dimensional array **[[x1, y1], [x2, y2], ...]**, diagnostics will output data for shifts **(x1, y1), (x2, y2), ...** [or :math:`r_i = \sqrt{x_i^2 + y_i^2}`  for *2D*].

directory_name
^^^^^^^^^^^^^^^^^^^^^^^^

Select the folder in which the diagnostics will be saved. By default, the files will be saved in the ``diagnostics/`` folder. When you specify the ``example`` *directory_name*, the files will be saved along the ``diagnostics/example/`` path.




SliceDiag
------------------------------

These keys control output of various quantities as functions of 2 coordinates, either in the form of a colored map, or in the form of a data array.

.. py:class:: lcode.diagnostics.SliceDiag(slice_type, slice_value, output_type, limits, offset, output_period, saving_xi_period, directory_name)

    - **slice_type** (default: *SliceType.XI_X*)
    - **slice_value** (default: *SliceValue.Ez*)
    - **output_type** (default: *OutputType.NUMBERS*)
    - **limits** (default: *[[0, inf], [-inf, inf]]*)
    - **offset** (default: *0*)
    - **output_period** (default: *100*)
    - **saving_xi_period** (default: *100*)
    - **directory_name** (default: *None*)


slice_type
^^^^^^^^^^^^^^^^^^^^^^^^

Selecting a plane (2 coordinates) for output.

- ``SliceType.XI_X``
- ``SliceType.XI_Y``
- ``SliceType.X_Y``
- ``SliceType.XI_R`` *[Equivalent to XI_X]*

``SliceType`` is imported from ``lcode.diagnostics``.

slice_value
^^^^^^^^^^^^^^^^^^^^^^^^

Selecting a value to output.

- ``SliceValue.E``: All components of the electric field. You can also select the component you want to display by using *Ex* (*Ey, Ez, Er, Ef*) instead of *E*.
- ``SliceValue.B``: All components of the magnetic field. You can also select the component you want to display by using *Bx* (*By, Bz, Bf*) instead of *B*.
- ``SliceValue.n``: Electron and ion density of plasma. You can also select the component you want to display by using *ne* (*ni*) instead of *n*.
- ``SliceValue.rho_beam``: Density of beam.
- ``SliceValue.Phi``: Wake potential. (currently only works in 3D)

``SliceValue`` is imported from ``lcode.diagnostics``.

.. note::

    You can combine any values: ``SliceValue.Ez | SliceValue.Bx | SliceValue.n``

output_type
^^^^^^^^^^^^^^^^^^^^^^^

Data output format. It can be ``OutputType.NUMBERS``, ``OutputType.PICTURES`` or ``OutputType.ALL``, where ``OutputType`` is imported from ``lcode.diagnostics``.

- ``OutputType.NUMBERS``: Output data in the form of numpy structured array.
- ``OutputType.PICTURES``: Output data in the form of png plots.
- ``OutputType.ALL``: Two previous points at the same time.

limits
^^^^^^^^^^^^^^^^^^^^^^^^

Selecting a subwindow for output. By default, the entire window is displayed.

You need to use a two-dimensional list, np.ndarray like `[[from1, to1], [from2, to2]]`. Where 1 and 2 are the coordinates corresponding to the ``SliceType``.

If you go beyond the limits, the limit will be returned to the window border.

offset
^^^^^^^^^^^^^^^^^^^^^^^^

Responsible for shifting from the axis along a coordinate that is not specified in `SliceType`.

For example, offset set to **x** for slice **XI_X** means **x** shift along the **Y** coordinate.

output_period
^^^^^^^^^^^^^^^^^^^^^^^^

The period with which the diagnosis is triggered. If ∆t\ :sub:`out`\  < ∆t, then diagnostics will be called at every step.

saving_xi_period
^^^^^^^^^^^^^^^^^^^^^^^^

The intermediate period with which data is flushed to disk.

.. warning::

    This property is experimental. Changing it may slow down program execution.


directory_name
^^^^^^^^^^^^^^^^^^^^^^^^

Select the folder in which the diagnostics will be saved. By default, the files will be saved in the ``diagnostics/`` folder. When you specify the ``example`` *directory_name*, the files will be saved along the ``diagnsotics/example/`` path.


ParticlesDiag
------------------------------

Output plasma and beam particles

.. py:class:: lcode.diagnostics.SliceDiag(slice_type, slice_value, output_type, limits, offset, output_period, saving_xi_period, directory_name)

    - **save_beam** (default: *false*)
    - **save_plasma** (default: *false*)
    - **output_period** (default: *100*)
    - **saving_xi_period** (default: *100*)
    - **directory_name** (default: *None*)

save_beam
^^^^^^^^^^^^^^^^^^^^^^^^

If set to **true**, ``beam_<t>.npz`` files are output to ``diagnostics/directory_name`` every ``output_period``.

save_plasma
^^^^^^^^^^^^^^^^^^^^^^^^

If set to **true**, ``plasma_electrons_<t>.npz`` (and ``plasma_ions_<t>.npz`` for movable ions) files are output to ``diagnostics/directory_name`` every ``output_period``.

output_period
^^^^^^^^^^^^^^^^^^^^^^^^

The period with which the diagnosis is triggered. If ∆t\ :sub:`out`\  < ∆t, then diagnostics will be called at every step.

saving_xi_period
^^^^^^^^^^^^^^^^^^^^^^^^

The intermediate period with which plasma state is flushed to disk into ``diagnostic/snapshot`` folder.


directory_name
^^^^^^^^^^^^^^^^^^^^^^^^

Select the folder in which the diagnostics will be saved. By default, the files will be saved in the ``diagnostics/`` folder. When you specify the ``example`` *directory_name*, the files will be saved along the ``diagnsotics/example/`` path.