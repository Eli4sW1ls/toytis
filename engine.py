import numpy as np
import logging
from potential import Potential
from cos_bump_series import CosBumpSeriesWalls
from flat_walls import FlatWall1D
from cos_dip_metastables import CosDipMetastableWalls

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class LangevinEngine:
    """ Representation of a Langevin engine in PPTIS

    Attributes
    ----------
    settings : dict
        Dictionary containing the settings of the engine
    dt : float
        Time step of the engine
    temperature : float
        Temperature of the engine
    friction : float
        Friction coefficient of the engine
    dim : int
        Dimension of the engine
    potential : :py:class:`Potential` object
        Potential of the engine
    phasepoint : tuple (x, v) of floats
        Phasepoint of the engine

    """
    def __init__(self, settings):
        self.settings = settings
        self.dt = self.settings["dt"]
        self.high_friction = settings.get("high_friction", False)
        self.T = self.settings["temperature"]
        self.gamma = self.settings["friction"]
        # self.potential = Potential()
        # self.potential = CosBumpSeriesWalls()
        self.potential = FlatWall1D()
        # self.potential = CosDipMetastableWalls()
        self.phasepoint = None
        self.kB = 1.0
        self.kT = self.kB * self.T
        self.beta = 1.0 / self.kT
        self.mass = self.settings["mass"]
        self.sigma = None
        self.bddt = None
        self.dim = self.settings["dim"]
        # self.gammadt = self.gamma * self.dt
        # self.dtdivmass = self.dt / self.mass
        # self.one_minus_gammadt = 1.0 - self.gammadt
        self.equipartition_sigma = np.sqrt(self.kT / self.mass)
        self.mdsteps = 0

        if self.high_friction:
            self.sigma = np.sqrt(2.0 * self.dt / (self.beta * self.mass * self.gamma))
            self.bddt = self.dt / (self.mass * self.gamma)
        else:
            gammadt = self.gamma * self.dt
            exp_gdt = np.exp(-gammadt)
            if self.gamma > 0.0:
                c_0 = exp_gdt
                c_1 = (1.0 - c_0) / gammadt
                c_2 = (1.0 - c_1) / gammadt
            else:
                raise ValueError(
                    'Langevin: Found gamma = {:6.2f} < 0.'.format(self.gamma)
                )
            
            self.c0 = c_0
            self.a1 = c_1 * self.dt
            self.a2 = c_2 * self.dt**2 / self.mass
            self.b1 = (c_1 - c_2) * self.dt / self.mass
            self.b2 = c_2 * self.dt / self.mass

            sig_ri2 = ((self.dt / (self.beta * self.gamma * self.mass)) *
                        (2. - (3. - 4.*exp_gdt + exp_gdt**2) / gammadt))
            sig_vi2 = (1.0 - exp_gdt**2) / (self.beta * self.mass)
            cov_rvi = (1./(self.mass * self.beta * self.gamma)) * (1.0 - exp_gdt)**2
            self.cov = np.array([[sig_ri2, cov_rvi],
                                    [cov_rvi, sig_vi2]])
            self.mean = np.zeros(2)

    def step(self, ph=None):
        """Performs a single step of the Langevin engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint at which to perform the step

        Returns
        -------
        ph : tuple (x, v) of floats
            Phasepoint after the step
        """
        x, v = ph
        
        force = self.potential.force((x, v))

        # Adapted from PyRETIS implementation, EW, May 2024
        if self.high_friction:
            rands = np.random.normal(loc = 0.0, scale=self.sigma, size=self.dim)
            x_new = x + self.bddt * force + rands
            v_new = rands
            return (x_new, v_new)
        else:
            randxv = np.random.multivariate_normal(self.mean, self.cov)
            x_rand = randxv[0]
            v_rand = randxv[1]
            x_new = x + self.a1 * v + self.a2 * force + x_rand
            
            v2 = self.c0 * v + self.b1 * force + v_rand

            force = self.potential.force((x_new, v))

            v_new = v2 + self.b2 * force

            self.mdsteps += 1

            return (x_new, v_new)

        # # Calculate the new position using the current velocity.
        # x_new = x + v * self.dt
        # force = self.potential.force((x_new, v))

        # # calculate the stochastic force
        # xi = np.random.normal(0, 1, self.dim)[0]

        # # update the velocity and position using the Langevin equation
        # v_new = v * self.one_minus_gammadt\
        #         + (force * self.dtdivmass)\
        #         + self.sigma * xi
        
        # self.mdsteps += 1

        # return (x_new, v_new)

    def draw_velocities(self):
        """
        Draw velocities from the Maxwell-Boltzmann distribution at temperature T
        for a particle of dimension dim. 

        Returns:
        v : np.array
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        # Maxwell-Boltzmann distribution for each component of the velocity
        return (np.random.normal(0, self.equipartition_sigma, self.dim))[0]

    def set_phasepoint(self, ph):
        """ Sets the phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to set the engine to

        """
        self.phasepoint = ph
