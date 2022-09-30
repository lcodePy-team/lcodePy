import numpy as np
import openpmd_api as io
from datetime import datetime as dt


class Diagnostics:
    def __init__(self, content, path='./diagnostics/', author="unknown"):
        self.description = {
            "author": author,
            "software": "LCODE",
            "softwareVersion": "1.0.0",
            "date": dt.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        }
        self.path = path
        self.content = content

    def before(self, t):
        self.series = io.Series(
            self.path + f"{t:010.3f}".replace('.', '_') + ".h5",
            io.Access.create
        )
        for attr, value in self.description.items():
            self.series.set_attribute(attr, value)
        self.i = self.series.iterations[0]

    def dxi(self, *params):
        for diag in self.content:
            diag.dxi(*params)

    def after(self):
        for diag in self.content:
            diag.dump(self.i)
        self.series.flush()
        del self.series


class Example:
    def __init__(self, cfg, period, time):
        self.period = cfg.getfloat('time-step') if period is None else period
        self.time = (0 if time[0] is None else time[0],
                     cfg.getfloat('time-limit')
                     if time[1] is None else time[1])

    def _check_time_borders(self, t):
        if t < self.time[0] or t > self.time[1]:
            return False
        if t < self._next_time:
            return False
        return True


class FieldsDiagnostics(Example):
    def __init__(self, cfg, *, period, time, select):

        super().__init__(cfg, period, time)

        for field in select.keys():
            if field not in ['E_z', 'E_f', 'E_r', 'B_f', 'B_z']:
                raise AttributeError("Name isn't corrected")
            if select[field] > cfg.getfloat('window-width'):
                raise AttributeError("r > window width")
            if select[field] < 0:
                select[field] = -1

        self.select = select
        self._next_time = self.time[0]

        r_step = cfg.getfloat('window-width-step-size')
        xi_step = cfg.getfloat('xi-step')
        width = cfg.getfloat('window-width')
        length = cfg.getfloat('window-length')

        self.window = (int(length/xi_step),
                       int(width/r_step) + 1)

    def dxi(self, t, layer_idx,
            plasma_particles, plasma_fields, rho_beam,
            beam_slice):
        if not self._check_time_borders(t):
            return

        if layer_idx == 0:
            self.data = {}
            for field in self.select.keys():
                self.data[field] = np.empty(
                    self.window if self.select[field] == -1 else
                    self.window[0], dtype=np.float64)

        for field in self.select.keys():
            r_idx = self.select[field]
            if r_idx == -1:
                self.data[field][layer_idx] = getattr(plasma_fields, field)
            else:
                self.data[field][layer_idx] = getattr(plasma_fields,
                                                      field)[r_idx]

    def dump(self, i):
        for field in self.select.keys():
            if field in ['E_z', 'E_f', 'E_r']:
                data = self.data[field]
                Ex = i.meshes['E'][field[-1]]
                Ex.reset_dataset(io.Dataset(data.dtype, data.shape))
                Ex.store_chunk(data)
            if field in ['B_f', 'B_z']:
                data = self.data[field]
                Bx = i.meshes['B'][field[-1]]
                Bx.reset_dataset(io.Dataset(data.dtype, data.shape))
                Bx.store_chunk(data)
        self._next_time += self.period


class PlasmaDiagnostics(Example):
    def __init__(self, cfg, *, period, time):
        super().__init__(cfg, period, time)

        self._next_time = self.time[0]

        r_step = cfg.getfloat('window-width-step-size')
        xi_step = cfg.getfloat('xi-step')
        width = cfg.getfloat('window-width')
        length = cfg.getfloat('window-length')
        ppil = cfg.getint('plasma-particles-per-cell')
        self.window = (int(length/xi_step), 8,
                       int(width/r_step)*ppil)
        self.xi_step = xi_step

    def dxi(self, t, layer_idx,
            plasma_particles, plasma_fields, rho_beam,
            beam_slice):
        if not self._check_time_borders(t):
            return

        if layer_idx == 0:
            self.data = np.empty(self.window, dtype='f8')
        pp = plasma_particles

        self.data[layer_idx] = \
            np.array((pp.age, pp.m,
                      pp.p_f, pp.p_r, pp.p_z,
                      pp.q, pp.r,
                      self.xi_step*layer_idx*np.ones_like(pp.age)))

    def dump(self, i):
        age, m, p_f, p_r, p_z, q, r, xi = \
            np.concatenate(self.data, axis=1)
        pp = i.particles['plasma_particles']

        ds = io.Dataset(age.dtype, age.shape)
        pp["age"]["data"].reset_dataset(ds)
        pp["mass"]['data'].reset_dataset(ds)
        pp['momentum']['f'].reset_dataset(ds)
        pp['momentum']['r'].reset_dataset(ds)
        pp['momentum']['z'].reset_dataset(ds)
        pp['position']['r'].reset_dataset(ds)
        pp['position']['xi'].reset_dataset(ds)

        pp["age"]['data'][()] = age
        pp["mass"]['data'][()] = m
        pp['momentum']['f'][()] = p_f
        pp['momentum']['r'][()] = p_r
        pp['momentum']['z'][()] = p_z
        pp['position']['r'][()] = r
        pp['position']['xi'][()] = xi
        self._next_time += self.period


class BeamDiagnostics(Example):
    def __init__(self, cfg, *, period, time):
        super().__init__(cfg, period, time)

        self._next_time = self.time[0]

        self.data = []

    def dxi(self, t, layer_idx,
            plasma_particles, plasma_fields, rho_beam,
            beam_slice):
        if not self._check_time_borders(t):
            return

        bs = beam_slice
        self.data.append((bs.M, bs.q_m, bs.q_norm,
                          bs.id, bs.p_r, bs.p_z,
                          bs.r, bs.xi))

    def dump(self, i):
        M, q_m, q_norm, id, p_r, p_z, r, xi = \
            np.concatenate(self.data, axis=1)
        bp = i.particles['beam_particles']

        ds = io.Dataset(xi.dtype, xi.shape)
        bp["M"]["data"].reset_dataset(ds)
        bp["q_m"]['data'].reset_dataset(ds)
        bp["q_norm"]['data'].reset_dataset(ds)
        bp["id"]['data'].reset_dataset(ds)
        bp['momentum']['r'].reset_dataset(ds)
        bp['momentum']['z'].reset_dataset(ds)
        bp['position']['r'].reset_dataset(ds)
        bp['position']['xi'].reset_dataset(ds)

        bp["M"]['data'][()] = M
        bp["q_m"]['data'][()] = q_m
        bp["q_norm"]['data'][()] = q_norm
        bp["id"]['data'][()] = id
        bp['momentum']['r'][()] = p_r
        bp['momentum']['z'][()] = p_z
        bp['position']['r'][()] = r
        bp['position']['xi'][()] = xi
        self._next_time += self.period
        self.data = []
