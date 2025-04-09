import numpy as np
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Potential:
    """
    Implementation of a one-dimensional double well potential with modulated sine bumps.
    
    The potential is defined as a sum of:
    1. A double well term: a*x^4 - b*(x-c)^2
    2. A modulated bump term: d * sin(2πx/p) * modulation_function
    
    The modulation function uses a sigmoid to reduce the effect of bumps at larger |x| values.
    """
    def __init__(self):
        """Initialize the Potential object with default parameters.

        The potential parameters are:
        a: Controls the steepness of the double well
        b: Controls the depth/width of the wells
        c: Horizontal offset of the double well
        d: Amplitude of the bumps
        p: Period of the bumps
        """
        # Core parameters defining the potential shape
        self.a = 1.       # Quartic term coefficient
        self.b = 3.5      # Quadratic term coefficient
        self.c = 0.       # Offset parameter
        self.d = .5       # Bump amplitude
        self.p = .3       # Bump period

        # Precompute constants to optimize calculations
        self.a4 = 4.*self.a                 # 4a term used in force calculation
        self.b2 = 2.*self.b                 # 2b term used in force calculation
        self.exp_p15 = np.exp(15.*self.p)   # exp(15p) term used in modulation
        self.p3 = 3.*self.p                 # 3p term used in modulation
        self.pi2 = 2.*np.pi                 # 2π term for sine/cosine calculations
        self.pi2divp = self.pi2/self.p      # 2π/p term for frequency calculation

    def potential(self, ph):
        """Calculate the potential energy at a given phase point.

        Parameters
        ----------
        ph : tuple (x, v)
            Phase point with position x and velocity v (velocity not used)

        Returns
        -------
        float
            Potential energy at the given position
        """
        x = ph[0]  # Extract position from phase point
        
        # Calculate double well component: a*x^4 - b*(x-c)^2
        doublewell = self.a*x**4 - self.b*(x - self.c)**2
        
        # Calculate sinusoidal bump: d*sin(2πx/p)
        bump = self.d * np.sin(self.pi2 * x / self.p)
        
        # Calculate sigmoid modulation to attenuate bumps at larger distances
        # This approaches 0 for |x| > 3p and 1 for |x| < 3p
        modulation = (1.-1. / (1 + np.exp(-5. * (np.abs(x) - self.p3))))
        
        # Combine components to get final potential
        pot = doublewell + bump * modulation

        return pot
    
    def force(self, ph):
        """Calculate the force (negative gradient of potential) at a given phase point.

        Parameters
        ----------
        ph : tuple (x, v)
            Phase point with position x and velocity v (velocity not used)

        Returns
        -------
        float
            Force at the given position
        """
        x = ph[0]  # Extract position from phase point
        
        # Calculate bump and modulation for force
        bump = self.d * np.sin(self.pi2 * x / self.p)
        modulation = (1.-1. / (1 + np.exp(-5. * (np.abs(x) - self.p3))))

        # Force from double well: -d/dx(a*x^4 - b*(x-c)^2) = -4a*x^3 + 2b*(x-c)
        f_doublewell = -self.a4*x**3 + self.b2*(x - self.c)
        
        # Derivative of bump: d/dx(d*sin(2πx/p)) = d*cos(2πx/p)*(2π/p)
        deriv_bump = self.d * np.cos(self.pi2 * x / self.p) * self.pi2divp
        
        # Derivative of modulation (using chain rule)
        deriv_modulation = \
           - 5. * np.sign(x) * np.exp(5.*(np.abs(x) + self.p3)) /\
           (np.exp(5.*np.abs(x)) + self.exp_p15)**2
        
        # Force contribution from bumps (using product rule)
        f_bump = -1. * deriv_bump * modulation - bump * deriv_modulation
        
        # Total force (negative gradient of potential)
        f = f_doublewell + f_bump

        return f

    def potential_and_force(self, ph):
        """Calculate both potential and force at a given phase point.
        
        This is more efficient than calling potential() and force() separately
        as it avoids duplicate calculations.

        Parameters
        ----------
        ph : tuple (x, v)
            Phase point with position x and velocity v (velocity not used)

        Returns
        -------
        tuple (float, float)
            (potential, force) at the given position
        """
        x = ph[0]  # Extract position from phase point
        
        # Calculate double well component
        doublewell = self.a*x**4 - self.b*(x - self.c)**2
        
        # Calculate bump and modulation (used for both potential and force)
        bump = self.d * np.sin(self.pi2 * x / self.p)
        modulation = (1.-1. / (1 + np.exp(-5. * (np.abs(x) - self.p3))))
        
        # Calculate potential energy
        pot = doublewell + bump * modulation

        # Calculate force (negative gradient of potential)
        # Force from double well
        f_doublewell = -self.a4*x**3 + self.b2*(x - self.c)
        
        # Derivative of bump
        deriv_bump = self.d * np.cos(self.pi2 * x / self.p) * self.pi2divp
        
        # Derivative of modulation
        deriv_modulation = \
           - 5. * np.sign(x) * np.exp(5.*(np.abs(x) + self.p3)) /\
           (np.exp(5.*np.abs(x)) + self.exp_p15)**2
           
        # Force contribution from bumps
        f_bump = -1. * deriv_bump * modulation - bump * deriv_modulation
        f = f_doublewell + f_bump

        return pot, f
    
    def plot_potential(self, ax):
        """Plot the potential energy function on the given axis.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        Returns
        -------
        None
            The function modifies the provided axis object
        """
        # Create position array covering the region of interest
        x = np.linspace(-1.5, 1.5, 1000)
        
        # Vectorized calculation of potential (more efficient than loop)
        pot = np.array([self.potential_and_force((xi, 0.))[0] for xi in x])
        
        # Plot the potential curve
        ax.plot(x, pot)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Potential energy')
        ax.set_title('Double well potential with modulated bumps')
    
    def plot_force(self, ax):
        """Plot the force function on the given axis.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the force

        Returns
        -------
        None
            The function modifies the provided axis object
        """
        # Create position array covering the region of interest
        x = np.linspace(-1.5, 1.5, 1000)
        
        # Vectorized calculation of force (more efficient than loop)
        f = np.array([self.potential_and_force((xi, 0.))[1] for xi in x])
        
        # Plot the force curve
        ax.plot(x, f)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Force')
        ax.set_title('Force in double well potential with modulated bumps')
