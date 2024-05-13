import numpy as np
import logging
from potential import Potential
from cos_bump_series import CosBumpSeriesWalls

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
        self.T = self.settings["temperature"]
        self.gamma = 1./self.settings["friction"]
        # self.potential = Potential()
        self.potential = CosBumpSeriesWalls()
        self.phasepoint = None
        self.mass = self.settings["mass"]
        self.dim = self.settings["dim"]
        self.kB = 1.0
        self.kT = self.kB * self.T
        self.beta = 1.0 / self.kT
        self.sigma = np.sqrt(2.0 * self.kT * self.gamma * self.dt / self.mass)
        self.gammadt = self.gamma * self.dt
        self.dtdivmass = self.dt / self.mass
        self.one_minus_gammadt = 1.0 - self.gammadt
        self.equipartition_sigma = np.sqrt(self.kT / self.mass)
        self.mdsteps = 0

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
        # Calculate the new position using the current velocity.
        x_new = x + v * self.dt
        _, force = self.potential.potential_and_force((x_new, v))

        # calculate the stochastic force
        xi = np.random.normal(0, 1, self.dim)[0]

        # update the velocity and position using the Langevin equation
        v_new = v * self.one_minus_gammadt\
                + (force * self.dtdivmass)\
                + self.sigma * xi
        
        self.mdsteps += 1

        return (x_new, v_new)

    def draw_velocities(self):
        """
        Draw velocities from the Maxwell-Boltzmann distribution at temperature T
        for a particle of dimension dim. 

        Returns:
        v : np.array
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        # Maxwell-Boltzmann distribution for each component of the velocity
        return (np.random.normal(0, self.equipartition_sigma, self.dim) *\
            np.sqrt(self.mass/(2*np.pi*self.kT)))[0]

    def set_phasepoint(self, ph):
        """ Sets the phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to set the engine to

        """
        self.phasepoint = ph
