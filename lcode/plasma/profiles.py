from abc import ABC, abstractmethod

import numpy as np

_GAUSSIAN_CUTOFF_DISTANCE = 5.0


class BaseProfile(ABC):
    @abstractmethod
    def place_particles(self, particles_per_cell):
        """
        Compute positions of macroparticles for profile.

        Parameters
        ----------
        particles_per_cell: int
            Base density of macroparticles

        Returns
        -------
        np.ndarray
            Positions of macroparticles
        """
        pass

    @abstractmethod
    def weigh_particles(self, particle_positions):
        """
        Compute weights of macroparticles for profile.

        Parameters
        ----------
        particle_positions: np.ndarray
            Positions of macroparticles

        Returns
        -------
        np.ndarray
            Weights of macroparticles
        """
        pass


class CylindricalProfile(BaseProfile, ABC):
    def __init__(self, config):
        self.max_radius = config.getfloat('window-width')
        self.r_step = config.getfloat('window-width-step-size')

    def cylindrical_weights(self, particle_positions):
        r_step_p = particle_positions[1] - particle_positions[0]
        return 2 * np.pi * r_step_p * particle_positions


class UniformPlacedCylindricalProfile(CylindricalProfile, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.min_plasma_radius = 0
        self.max_plasma_radius = self.max_radius

    def place_particles(self, particles_per_cell):
        plasma_width = self.max_plasma_radius - self.min_plasma_radius
        n_particles = int(plasma_width / self.r_step) * particles_per_cell
        r_step_p = plasma_width / n_particles
        return (np.arange(n_particles) + 0.5) * r_step_p + self.min_plasma_radius


class UniformCylindricalProfile(UniformPlacedCylindricalProfile):
    def weigh_particles(self, particle_positions):
        return self.cylindrical_weights(particle_positions)


class StepwiseCylindricalProfile(UniformCylindricalProfile):
    def __init__(self, config):
        super().__init__(config)
        self.plasma_width = config.getfloat('plasma-width')
        self.max_plasma_radius = min(self.plasma_width, self.max_plasma_radius)


class ChannelCylindricalProfile(UniformCylindricalProfile):
    def __init__(self, config):
        super().__init__(config)
        self.min_plasma_radius = config.getfloat('plasma-width')


class GaussianCylindricalProfile(UniformPlacedCylindricalProfile):
    def __init__(self, config):
        super().__init__(config)
        self.plasma_width = config.getfloat('plasma-width')
        self.max_plasma_radius = min(
            _GAUSSIAN_CUTOFF_DISTANCE * self.plasma_width,
            self.max_plasma_radius,
        )

    def weigh_particles(self, particle_positions):
        gaussian_weights = np.exp(-particle_positions ** 2 / 2 / self.plasma_width ** 2)
        return self.cylindrical_weights(particle_positions) * gaussian_weights


class SubChannelCylindricalProfile(UniformPlacedCylindricalProfile):
    def __init__(self, config):
        super().__init__(config)
        self.plasma_width = config.getfloat('plasma-width')
        self.plasma_width2 = config.getfloat('plasma-width-2')
        self.plasma_density2 = config.getfloat('plasma-density-2')
        if self.plasma_density2 == 0.0:
            self.min_plasma_radius = self.plasma_width2

    def weigh_particles(self, particle_positions):
        weights = np.copy(particle_positions)
        weights[particle_positions < self.plasma_width2] = self.plasma_density2
        weights[particle_positions > self.plasma_width] = 1.0
        mask = np.logical_and(
            particle_positions >= self.plasma_width2,
            particle_positions <= self.plasma_width,
        )
        interp_w = self.plasma_width - particle_positions[mask]
        interp_w /= (self.plasma_width - self.plasma_width2)
        weights[mask] = self.plasma_density2 * interp_w + (1 - interp_w)
        return self.cylindrical_weights(particle_positions) * weights


_cylindrical_profiles = {
    'uniform': UniformCylindricalProfile,
    '1': UniformCylindricalProfile,
    'stepwise': StepwiseCylindricalProfile,
    '2': StepwiseCylindricalProfile,
    'gaussian': GaussianCylindricalProfile,
    '3': GaussianCylindricalProfile,
    'channel': ChannelCylindricalProfile,
    '5': ChannelCylindricalProfile,
    'sub-channel': SubChannelCylindricalProfile,
    '6': SubChannelCylindricalProfile,
}


def get_plasma_profile(config):
    profile = config.get('plasma-profile')
    try:
        profile = str(int(float(profile))) # process cases like '1.0'
    except:
        pass
    return _cylindrical_profiles[profile](config)
