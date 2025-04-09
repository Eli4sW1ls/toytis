"""
Velocity Verlet integration engine for molecular dynamics simulations.

This module provides an implementation of the Velocity Verlet algorithm
for time integration in molecular dynamics simulations, specifically 
designed for path sampling in the PPTIS framework. It supports NVE 
(microcanonical ensemble) dynamics.

References
----------
[1] Swope, W. C., et al. "A computer simulation method for the calculation of 
    equilibrium constants for the formation of physical clusters of molecules: 
    Application to small water clusters." The Journal of Chemical Physics, 76(1), 637-649 (1982)
"""

import numpy as np
import logging
from potential import Potential

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class VelocityVerletEngine:
    """Velocity Verlet integration engine for molecular dynamics in PPTIS.
    
    This class implements the Velocity Verlet algorithm for time integration,
    performing NVE (constant energy) dynamics. It provides methods for 
    propagating the system forward in time and for sampling velocities from
    a Maxwell-Boltzmann distribution.
    
    Attributes
    ----------
    settings : dict
        Dictionary containing the settings of the engine
    dt : float
        Time step of the engine (integration time step)
    mass : float
        Mass of the particle (defaults to 1.0)
    temperature : float
        Temperature of the system (for velocity initialization)
    potential : :py:class:`Potential` object
        Potential energy function for force calculations
    phasepoint : tuple (x, v)
        Current phase point (position, velocity) of the system
        
    Examples
    --------
    >>> settings = {"dt": 0.001, "temperature": 1.0, "mass": 1.0}
    >>> engine = VelocityVerletEngine(settings)
    >>> engine.set_phasepoint((0.0, 1.0))  # Set initial position and velocity
    >>> for _ in range(1000):
    ...     engine.step()  # Propagate the system
    """
    def __init__(self, settings):
        """Initialize the VelocityVerlet engine with given settings.
        
        Parameters
        ----------
        settings : dict
            Dictionary containing simulation settings with keys:
            - 'dt': time step (required)
            - 'temperature': system temperature (required)
            - 'mass': particle mass (optional, default 1.0)
        """
        self.settings = settings
        # Extract required and optional settings
        self.dt = self.settings["dt"]
        self.mass = self.settings.get("mass", 1.0)
        self.temperature = self.settings["temperature"]
        
        # Initialize potential and phase point
        self.potential = Potential()
        self.phasepoint = None
        
        # Log initialization 
        logger.debug(f"Initialized VV Engine with dt={self.dt}, T={self.temperature}, mass={self.mass}")

    def step(self, ph=None):
        """Perform one integration step using the Velocity Verlet algorithm.
        
        This method advances the system by one time step using the Velocity Verlet
        integration scheme, which is symplectic and time-reversible.
        
        Algorithm steps:
        1. Update velocity by half a time step (v += 0.5*dt*a)
        2. Update position by a full time step (x += dt*v)
        3. Recalculate forces at new position
        4. Update velocity by another half time step (v += 0.5*dt*a)
        
        Parameters
        ----------
        ph : tuple (x, v), optional
            Phase point (position, velocity) at which to perform the step.
            If None, the current phase point of the engine is used.
            
        Returns
        -------
        tuple (x, v)
            Updated phase point after performing the integration step
            
        Notes
        -----
        The forces are calculated from the potential energy function.
        Force = -∇V(x), and acceleration a = F/m
        """
        if ph is None:
            # Use the current phase point if none provided
            # Copy to avoid modifying the original values
            x, v = self.phasepoint[0], self.phasepoint[1]
        else:
            x, v = ph
            
        # Step 1: Calculate the potential and force at the current position
        _, force = self.potential.potential_and_force((x, v))
        
        # Step 2: Update velocity by half a time step (v += 0.5*dt*a)
        # Force divided by mass gives acceleration
        a = force / self.mass
        v_half = v + 0.5 * self.dt * a
        
        # Step 3: Update position by a full time step using updated velocity
        x_new = x + self.dt * v_half
        
        # Step 4: Calculate force at the new position
        _, force_new = self.potential.potential_and_force((x_new, v_half))
        
        # Step 5: Update velocity by another half time step
        a_new = force_new / self.mass
        v_new = v_half + 0.5 * self.dt * a_new
        
        # Update the stored phase point
        self.phasepoint = (x_new, v_new)
        return self.phasepoint

    def draw_velocities(self, ph=None):
        """Sample velocities from the Maxwell-Boltzmann distribution.
        
        This method draws a random velocity from the Maxwell-Boltzmann 
        distribution corresponding to the system temperature, and adjusts
        it to conserve energy in the NVE ensemble.
        
        Parameters
        ----------
        ph : tuple (x, v), optional
            Current phase point (position, velocity). Used to calculate
            the potential energy for energy conservation.
            
        Returns
        -------
        float
            A velocity value drawn from the Maxwell-Boltzmann distribution
            and adjusted to conserve energy
            
        Notes
        -----
        For an NVE ensemble, total energy (kinetic + potential) must be conserved.
        This implementation first draws a random velocity, then scales it to
        maintain the same total energy as the current phase point.
        """
        if ph is None and self.phasepoint is None:
            # If no phase point is provided or set, just return a standard draw
            std_dev = np.sqrt(self.temperature / self.mass)
            velocity = std_dev * np.random.randn()
            logger.debug(f"Drew velocity {velocity} from Maxwell-Boltzmann (no adjustment)")
            return velocity
        
        # Get the current phase point and calculate its total energy
        current_pos, current_vel = ph if ph is not None else self.phasepoint
        potential_energy, _ = self.potential.potential_and_force((current_pos, current_vel))
        current_kinetic_energy = 0.5 * self.mass * current_vel**2
        total_energy = potential_energy + current_kinetic_energy
        
        # Draw velocity from Maxwell-Boltzmann distribution
        std_dev = np.sqrt(self.temperature / self.mass)
        new_velocity = std_dev * np.random.randn()
        
        # Calculate the new kinetic energy needed to conserve total energy
        new_kinetic_energy = total_energy - potential_energy
        
        # Ensure kinetic energy is non-negative
        if new_kinetic_energy < 0:
            logger.warning(f"Negative kinetic energy encountered at position {current_pos}. Using small positive value.")
            new_kinetic_energy = 1e-6
        
        # Calculate the scaling factor to adjust the velocity
        scaling_factor = np.sqrt(2 * new_kinetic_energy / (self.mass * new_velocity**2))
        adjusted_velocity = scaling_factor * new_velocity
        
        # Preserve the sign of the original drawn velocity
        adjusted_velocity = np.sign(new_velocity) * abs(adjusted_velocity)
        
        logger.debug(f"Drew velocity {adjusted_velocity} (adjusted from {new_velocity}) to conserve energy {total_energy}")
        return adjusted_velocity
    
    def set_phasepoint(self, ph):
        """Set the current phase point of the engine.
        
        Parameters
        ----------
        ph : tuple (x, v)
            Phase point (position, velocity) to set
            
        Examples
        --------
        >>> engine = VelocityVerletEngine({"dt": 0.01, "temperature": 1.0})
        >>> engine.set_phasepoint((0.0, 1.0))  # Set position=0.0, velocity=1.0
        """
        self.phasepoint = ph
        logger.debug(f"Set phase point to position={ph[0]}, velocity={ph[1]}")
