import logging
import math 
import numpy as np
import matplotlib.pyplot as plt  

logger = logging.getLogger(__name__)  # pylint: disable=C0103
logger.addHandler(logging.NullHandler())


class CosBumpSeriesWalls():
    """Series of cosine bump barriers
    AG, June 19, 2022"""

    def __init__(self, desc='1D cos_bump_series'):
        # these are the default parameters
        # bleft -- first bump starts at x=bleft, and V=0 for x<bleft
        # bright -- first bump ends at x=bright, and then:
        #             either second bump starts at x=bright
        #             or V=0 for x>bright
        # deltaf -- height of bumps is deltaf, in units kBT
        # nbump -- number of bumps
        self.params = {'xleft': -0.2, 
                       'xright': 0.2, 
                       'k': 100., 
                       'bleft': -0.1, 
                       'bright': 0.1 ,
                       'deltaf': 1,
                        'nbump': 1}

    def potential(self, ph):

        x = ph[0]  # this is array, assume 1D (x)
        xleft  = self.params['xleft']
        xright = self.params['xright']
        k = self.params['k']
        bleft  = self.params['bleft']
        bright = self.params['bright']
        deltaf = self.params['deltaf']
        nbump  = self.params['nbump']

        assert bleft < bright 
        assert bleft >= xleft
        assert bright <= xright
        assert xright > xleft
        L = bright - bleft     # width of 1 bump
        bright2 = bleft + nbump*L   # V=0 after all bumps, for x>bright2
        xright2 = bright2 + (xright-bright)

        if x < xleft:
            v_pot = k*(x-xleft)**2/2.
        elif x > xright2:
            v_pot = k*(x-xright2)**2/2.
        elif (x < bleft) or (x > bright2):
            v_pot = np.zeros_like(x)
        else:
            v_pot = deltaf/2 * (1 - np.cos(2*np.pi*(x-bleft)/L))

        return v_pot.sum()

    def force(self, ph):
        x = ph[0]  # this is array, assume 1D (x)
        xleft  = self.params['xleft']
        xright = self.params['xright']
        k = self.params['k']
        bleft  = self.params['bleft']
        bright = self.params['bright']
        deltaf = self.params['deltaf']
        nbump  = self.params['nbump']

        assert bleft < bright
        assert bleft >= xleft
        assert bright <= xright
        assert xright > xleft
        L = bright - bleft     # width of 1 bump
        bright2 = bleft + nbump*L   # V=0 after all bumps, for x>bright2
        xright2 = bright2 + (xright-bright)

        if x < xleft:
            force = -k*(x-xleft)
        elif x > xright2:
            force = -k*(x-xright2)
        elif (x < bleft) or (x > bright2):
                force = 0
        else:
            force = -deltaf*np.pi/L * np.sin(2*np.pi*(x-bleft)/L)

        return force

    def potential_and_force(self, ph):
        v_pot = self.potential(ph)
        force = self.force(ph)
         
        return v_pot, force
    
    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        x = np.linspace(-1.5, 1.5, 1000)
        pot = np.zeros_like(x)
        for i in range(len(x)):
            pot[i], _ = self.potential_and_force((x[i], 0.))
        ax.plot(x, pot)
    
    def plot_force(self, ax):
        """Plots the force.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the force

        """
        x = np.linspace(-1.5, 1.5, 1000)
        f = np.zeros_like(x)
        for i in range(len(x)):
            _, f[i] = self.potential_and_force((x[i], 0.))
        ax.plot(x, f)

