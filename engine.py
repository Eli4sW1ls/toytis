"""
Langevin Dynamics Engine Module for ToyTIS

This module provides implementations of Langevin dynamics engines for molecular
simulations, including both regular and overdamped (high-friction) Langevin dynamics.
These engines are used to propagate the system according to the Langevin equation
of motion in path sampling simulations.
"""

import numpy as np
import logging
from potential import Potential
from potentials.cos_bump_series import CosBumpSeriesWalls
from potentials.flat_walls import FlatWall1D
from potentials.cos_dip_metastables import CosDipMetastableWalls
from potentials.cos_well import CosWellWalls

from potentials.mazepotential_mixed import Maze2D_color
from potentials.potential2channels_1 import PotentialTwoChannels
from potentials.pot_ibuprofen import Ibuprofen
from potentials.cffs2d import PotentialcFFS
from potentials.sjoelbak import RectangularGridWithBarrierPotential
from potentials.sjoelbak_slope import RectangularGridWithBarrierSlope

# Configure module logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class LangevinEngine:
    """
    Implementation of Langevin dynamics engine for 1D systems.
    
    This class implements a Langevin dynamics engine that can operate in both
    regular and overdamped (high-friction) regimes. It uses the BBK integrator for
    regular Langevin dynamics and a simpler Euler-Maruyama scheme for overdamped dynamics.
    
    Attributes
    ----------
    settings : dict
        Dictionary containing the settings of the engine
    dt : float
        Time step of the engine
    temperature : float
        Temperature of the engine (in reduced units)
    friction : float
        Friction coefficient of the engine (gamma)
    dim : int
        Dimension of the engine
    potential : :py:class:`Potential` object
        Potential energy function used in simulation
    phasepoint : tuple (x, v) of floats
        Current phase point (position, velocity) of the engine
    kB : float
        Boltzmann constant (set to 1.0 in reduced units)
    kT : float
        Thermal energy (kB * temperature)
    beta : float
        Inverse temperature (1/kT)
    mass : float
        Mass of the particle
    sigma : float
        Noise amplitude for stochastic term in Langevin equation
    bddt : float
        Precomputed factor for Brownian dynamics
    equipartition_sigma : float
        Standard deviation for Maxwell-Boltzmann velocity distribution
    mdsteps : int
        Counter for number of molecular dynamics steps performed
    c0, a1, a2, b1, b2 : float
        Precomputed coefficients for BBK Langevin integrator
    cov : ndarray
        Covariance matrix for correlated noise in Langevin integration
    mean : ndarray
        Mean values for noise terms (set to zeros)
    high_friction : bool
        Whether to use overdamped Langevin dynamics
    """
    def __init__(self, settings):
        """
        Initialize the Langevin dynamics engine.
        
        Parameters
        ----------
        settings : dict
            Dictionary containing simulation parameters:
            - dt : float
                Time step (mandatory)
            - temperature : float
                Temperature in reduced units (mandatory)
            - friction : float
                Friction coefficient (mandatory)
            - mass : float 
                Particle mass (mandatory)
            - dim : int
                System dimensionality (mandatory)
            - high_friction : bool
                Whether to use overdamped Langevin (optional, default: False)
        """
        # Store settings and extract basic parameters
        self.settings = settings
        self.dt = self.settings["dt"]
        self.high_friction = settings.get("high_friction", False)  # Default to regular Langevin if not specified
        self.T = self.settings["temperature"]
        self.gamma = self.settings["friction"]
        
        # Select potential function (currently using FlatWall1D by default)
        # Uncomment other options or add mechanism to select potential via settings
        # self.potential = Potential()
        # self.potential = CosBumpSeriesWalls()
        # self.potential = FlatWall1D()
        # self.potential = CosDipMetastableWalls()
        self.potential = CosWellWalls()
        
        # Initialize phase point and physical constants
        self.phasepoint = None
        self.kB = 1.0  # Boltzmann constant in reduced units
        self.kT = self.kB * self.T  # Thermal energy
        self.beta = 1.0 / self.kT  # Inverse temperature
        self.mass = self.settings["mass"]
        self.dim = self.settings["dim"]
        
        # Initialize integration parameters to be calculated in the setup
        self.sigma = None
        self.bddt = None
        
        # Standard deviation for Maxwell-Boltzmann velocity distribution
        self.equipartition_sigma = np.sqrt(self.kT / self.mass)
        self.mdsteps = 0

        # Set up integration scheme parameters based on friction regime
        if self.high_friction:
            # Overdamped Langevin (Brownian dynamics) parameters
            # sigma controls the noise amplitude in the position update
            self.sigma = np.sqrt(2.0 * self.dt / (self.beta * self.mass * self.gamma))
            # bddt is the coefficient for the deterministic force contribution
            self.bddt = self.dt / (self.mass * self.gamma)
        else:
            # Regular Langevin dynamics using BBK integrator
            # Calculate the integration coefficients based on Langevin equation
            gammadt = self.gamma * self.dt
            exp_gdt = np.exp(-gammadt)
            
            if self.gamma > 0.0:
                # Calculate BBK integrator coefficients for finite friction
                c_0 = exp_gdt  # Velocity damping factor
                c_1 = (1.0 - c_0) / gammadt  # Position update coefficient
                c_2 = (1.0 - c_1) / gammadt  # Force coefficient
            else:
                raise ValueError(
                    'Langevin: Found gamma = {:6.2f} < 0. Friction must be positive.'.format(self.gamma)
                )
            
            # Store coefficients for use in the step method
            self.c0 = c_0
            self.a1 = c_1 * self.dt
            self.a2 = c_2 * self.dt**2 / self.mass
            self.b1 = (c_1 - c_2) * self.dt / self.mass
            self.b2 = c_2 * self.dt / self.mass

            # Calculate variances and covariances for stochastic terms
            # These are derived from the fluctuation-dissipation theorem
            sig_ri2 = ((self.dt / (self.beta * self.gamma * self.mass)) *
                        (2. - (3. - 4.*exp_gdt + exp_gdt**2) / gammadt))
            sig_vi2 = (1.0 - exp_gdt**2) / (self.beta * self.mass)
            cov_rvi = (1./(self.mass * self.beta * self.gamma)) * (1.0 - exp_gdt)**2
            
            # Covariance matrix for correlated random numbers
            self.cov = np.array([[sig_ri2, cov_rvi],
                                 [cov_rvi, sig_vi2]])
            self.mean = np.zeros(2)  # Mean for the random numbers (zero)

    def step(self, ph=None):
        """
        Performs a single step of the Langevin dynamics integration.

        Updates the position and velocity according to the Langevin equation of motion.
        In the high friction regime, uses an overdamped approximation (Brownian dynamics).
        Otherwise, uses the BBK integrator for regular Langevin dynamics.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint (position, velocity) at which to perform the step

        Returns
        -------
        tuple (x_new, v_new) of floats
            Updated phasepoint after the integration step
        """
        x, v = ph
        
        # Get the force at the current position
        force = self.potential.force((x, v))
        
        # Adapted from PyRETIS implementation, EW, May 2024
        if self.high_friction:
            # Overdamped Langevin dynamics (Brownian dynamics)
            # Generate random noise for position update
            rands = np.random.normal(loc=0.0, scale=self.sigma, size=self.dim)
            
            # Update position: x' = x + (dt/mγ)F + √(2dt/mγβ)η
            x_new = x + self.bddt * force + rands[0]
            
            # In overdamped regime, velocity is proportional to random forces
            v_new = rands[0]
        else:
            # Regular Langevin dynamics using BBK integrator
            # Generate correlated random numbers for position and velocity updates
            randxv = np.random.multivariate_normal(self.mean, self.cov)
            x_rand = randxv[0]  # Random term for position update
            v_rand = randxv[1]  # Random term for velocity update
            
            # Update position: x' = x + a₁v + a₂F + η_x
            x_new = x + self.a1 * v + self.a2 * force + x_rand
            
            # First part of velocity update: v' = c₀v + b₁F + η_v
            v2 = self.c0 * v + self.b1 * force + v_rand

            # Calculate force at new position for second part of velocity update
            force = self.potential.force((x_new, v))

            # Complete velocity update: v_new = v' + b₂F_new
            v_new = v2 + self.b2 * force

        # Increment MD step counter
        self.mdsteps += 1

        return (x_new, v_new)

    def draw_velocities(self):
        """
        Draw velocities from the Maxwell-Boltzmann distribution.
        
        Generates velocities for a particle at temperature T according to the
        Maxwell-Boltzmann distribution, which is the equilibrium distribution
        for velocities in a system at constant temperature.

        Returns
        -------
        float
            Velocity drawn from the Maxwell-Boltzmann distribution
        """
        # Maxwell-Boltzmann distribution for each component of the velocity
        # Standard deviation is √(kT/m)
        return (np.random.normal(0, self.equipartition_sigma, self.dim))[0]

    def set_phasepoint(self, ph):
        """
        Sets the current phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to set the engine to, containing position and velocity
        """
        self.phasepoint = ph


class ndLangevinEngine:
    """
    Implementation of Langevin dynamics engine for n-dimensional systems.
    
    This class extends the Langevin dynamics to handle multidimensional systems,
    implementing both regular and overdamped Langevin dynamics with various potential
    energy functions.
    
    Attributes
    ----------
    settings : dict
        Dictionary containing simulation parameters
    dt : float
        Time step for integration
    high_friction : bool
        Whether to use overdamped Langevin dynamics
    T : float
        Temperature in reduced units
    gamma : float
        Friction coefficient
    potential : object
        Potential energy function object with force method
    phasepoint : tuple
        Current phase point (position, velocity)
    mass : float
        Particle mass
    dim : int
        System dimensionality
    kB : float
        Boltzmann constant (set to 1.0 in reduced units)
    kT : float
        Thermal energy
    beta : float
        Inverse temperature
    sigma : float
        Noise amplitude for overdamped Langevin
    bddt : float
        Coefficient for force term in overdamped Langevin
    equipartition_sigma : float
        Standard deviation for Maxwell-Boltzmann distribution
    mdsteps : int
        Counter for molecular dynamics steps
    c0, a1, a2, b1, b2 : float
        Coefficients for regular Langevin integration
    cov : ndarray
        Covariance matrix for correlated noise
    mean : ndarray
        Mean vector for noise (zeros)
    """
    def __init__(self, settings):
        """
        Initialize the n-dimensional Langevin dynamics engine.
        
        Parameters
        ----------
        settings : dict
            Dictionary containing simulation parameters:
            - dt : float
                Time step
            - temperature : float
                Temperature in reduced units
            - friction : float
                Friction coefficient
            - mass : float
                Particle mass
            - dim : int
                System dimensionality
            - high_friction : bool
                Whether to use overdamped Langevin (optional, default: False)
        """            
        # Store basic settings and parameters
        self.settings = settings
        self.dt = self.settings["dt"]
        self.high_friction = settings.get("high_friction", False)
        self.T = self.settings["temperature"]
        self.gamma = self.settings["friction"]
        
        # Select potential function - uncomment the desired potential
        # self.potential = Maze2D_color(mazefig="potentials/maze.png")
        # self.potential = Maze2D_color(mazefig="potentials/tunnelistarwalls.png")
        # self.potential = PotentialTwoChannels()
        self.potential = Ibuprofen()
        # self.potential = PotentialcFFS(4, np.pi/6)
        # self.potential = RectangularGridWithBarrierPotential(3*0.1, 9*0.1, 2000, (1.5*0.1, 4.5*0.1), -3.5*3/10, 0.2, 2*.1)
        # self.potential = RectangularGridWithBarrierPotential()
        # self.potential = RectangularGridWithBarrierSlope()
        #self.potential = RectangularGridWithRuggedPotential()
        
        # Initialize system state and physical parameters
        self.phasepoint = None
        self.mass = self.settings["mass"]
        self.dim = self.settings["dim"]
        self.kB = 1.0
        self.kT = self.kB * self.T
        self.beta = 1.0 / self.kT
        self.sigma = None
        self.bddt = None
        
        # Standard deviation for Maxwell-Boltzmann velocity distribution
        self.equipartition_sigma = np.sqrt(self.kT / self.mass)
        self.mdsteps = 0

        # Configure integration parameters based on friction regime
        if self.high_friction:
            # Overdamped Langevin dynamics parameters
            self.sigma = np.sqrt(2.0 * self.dt / (self.beta * self.mass * self.gamma))
            self.bddt = self.dt / (self.mass * self.gamma)
        else:
            # Regular Langevin dynamics using BBK integrator
            gammadt = self.gamma * self.dt
            exp_gdt = np.exp(-gammadt)
            
            if self.gamma > 0.0:
                # Calculate coefficients for the BBK integrator
                c_0 = exp_gdt
                c_1 = (1.0 - c_0) / gammadt
                c_2 = (1.0 - c_1) / gammadt
            else:
                raise ValueError(
                    'Langevin: Found gamma = {:6.2f} < 0. Friction must be positive.'.format(self.gamma)
                )
            
            # Store integration coefficients
            self.c0 = c_0
            self.a1 = c_1 * self.dt
            self.a2 = c_2 * self.dt**2 / self.mass
            self.b1 = (c_1 - c_2) * self.dt / self.mass
            self.b2 = c_2 * self.dt / self.mass

            # Calculate variances and covariances for the stochastic terms
            sig_ri2 = ((self.dt / (self.beta * self.gamma * self.mass)) *
                      (2. - (3. - 4.*exp_gdt + exp_gdt**2) / gammadt))
            sig_vi2 = (1.0 - exp_gdt**2) / (self.beta * self.mass)
            cov_rvi = (1./(self.mass * self.beta * self.gamma)) * (1.0 - exp_gdt)**2
            
            # Store covariance matrix for efficient random number generation
            self.cov = np.array([[sig_ri2, cov_rvi],
                                 [cov_rvi, sig_vi2]])
            self.mean = np.zeros(2)
            
            # Store precomputed covariance matrix for later use
            # This is more efficient than recomputing it on each step
            self.settings = settings
            
    def step(self, ph):
        """
        Perform a single integration step using the Langevin equation.
        
        This method implements either a high-friction (overdamped) Langevin integrator
        or a regular Langevin integrator based on the 'high_friction' setting.
        
        Parameters
        ----------
        ph : tuple (x, v)
            Current phase point where:
            - x: position array of shape (dim,)
            - v: velocity array of shape (dim,)

        Returns
        -------
        tuple (x_new, v_new)
            New phase point after the integration step where:
            - x_new: updated position array of shape (dim,)
            - v_new: updated velocity array of shape (dim,)
            
        Notes
        -----
        For the regular Langevin integrator, this uses a BBK scheme
        (Brooks, Brünger, and Karplus) integration algorithm.
        """
        x, v = ph

        # Get both potential and force to avoid duplicate calculation
        # This optimization reduces redundant force evaluations
        pot, force = self.potential.potential_and_force((x, v))

        # Adapted from PyRETIS implementation, EW, May 2024
        if self.high_friction:
            # Overdamped Langevin dynamics (Brownian dynamics)
            # In high friction limit, velocity correlations decay rapidly
            rands = np.random.normal(loc=0.0, scale=self.sigma, size=self.dim)
            
            # Position update: x' = x + (dt/mγ)F + √(2dt/mγβ)η
            x_new = x + self.bddt * force + rands
            
            # In overdamped regime, velocity is proportional to random force
            v_new = rands  
        else:
            # Regular Langevin dynamics using BBK integrator
            # Generate correlated random numbers for position and velocity updates
            # Using vectorized operations for efficiency
            randxv = np.random.multivariate_normal(self.mean, self.cov, self.dim)
            x_rand = randxv[:, 0]  # Random terms for position updates
            v_rand = randxv[:, 1]  # Random terms for velocity updates

            # Position update: x' = x + a₁v + a₂F + η_x
            x_new = x + self.a1 * v + self.a2 * force + x_rand
            
            # First part of velocity update: v' = c₀v + b₁F + η_v
            v2 = self.c0 * v + self.b1 * force + v_rand

            # Calculate force at new position for second part of velocity update
            # Reusing the potential_and_force method for efficiency
            _, force_new = self.potential.potential_and_force((x_new, v))
            
            # Complete velocity update: v_new = v' + b₂F_new
            v_new = v2 + self.b2 * force_new

        self.mdsteps += 1

        return (x_new, v_new)
    
    def draw_velocities(self):
        """
        Draw velocities from the Maxwell-Boltzmann distribution.
        
        Generates n-dimensional velocities according to the Maxwell-Boltzmann
        distribution for a system at temperature T.

        Returns
        -------
        numpy.ndarray
            Array of velocity components drawn from the Maxwell-Boltzmann distribution
        """
        # Generate velocities for each dimension from normal distribution
        # with standard deviation √(kT/m)
        return np.random.normal(0, self.equipartition_sigma, self.dim)

    def set_phasepoint(self, ph):
        """
        Sets the current phasepoint of the engine.

        Parameters
        ----------
        ph : tuple (x, v)
            Phasepoint to set the engine to, containing position and velocity arrays
        """
        self.phasepoint = ph