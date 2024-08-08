import logging
import math 
import numpy as np
import matplotlib.pyplot as plt  
from pyretis.forcefield.potential import PotentialFunction
logger = logging.getLogger(__name__)  # pylint: disable=C0103
logger.addHandler(logging.NullHandler())

class CosWellWalls(PotentialFunction):
    """Cos-dip with modulation
        EW - March 2024"""

    def __init__(self, desc='1D cos_bump_series'):
        # super().__init__(dim=1, desc=desc)
        # these are the default parameters
        # bleft -- first bump starts at x=bleft, and V=0 for x<bleft
        # bright -- first bump ends at x=bright, and then:
        #             either second bump starts at x=bright
        #             or V=0 for x>bright
        # deltaf -- height of bumps is deltaf, in units kBT
        # 
        self.params = {'wall_dx': 0.1, 'k': 100., 'bleft': -0.25, 'bright': 0.25 ,'deltaf': 1.2,
                        'height': 0.15, 'width': 0.1, 'db': 0}

    def potential(self, ph):
        x = ph[0]  # this is array, assume 1D (x)
        wall_dx  = self.params['wall_dx']
        k = self.params['k']
        bleft  = self.params['bleft']
        bright = self.params['bright']
        deltaf = self.params['deltaf']
        height = self.params['height']
        width = self.params['width']
        dB = self.params['db']

        assert bleft < bright
        L = bright - bleft     # width of 1 bump
        xleft = bleft - wall_dx - width
        xright = bright + wall_dx + width

        if x < xleft:
            v_pot = k*(x-xleft)**2/2.
        elif x > xright:
            v_pot = k*(x-xright)**2/2.
        elif x < bleft-width:
            v_pot = np.zeros_like(x)
        elif x > bright+width:
            v_pot = np.zeros_like(x) + dB*height
        elif x < bleft:
            v_pot = height/2 * (1 - np.cos(np.pi*(x-bleft+width)/width))
        elif x > bright:
            v_pot = (1-dB)*height/2 * (1 - np.cos(np.pi*(x-bright-width)/width)) + dB*height
        else:
            v_pot = height + deltaf/2 * (-1 + np.cos(2*np.pi*(x-bleft)/L))

        return v_pot.sum()

    def force(self, ph):
        x = ph[0]  # this is array, assume 1D (x)
        wall_dx  = self.params['wall_dx']
        k = self.params['k']
        bleft  = self.params['bleft']
        bright = self.params['bright']
        deltaf = self.params['deltaf']
        height = self.params['height']
        width = self.params['width']
        dB = self.params['db']

        assert bleft < bright
        L = bright - bleft     # width of 1 bump
        xleft = bleft - wall_dx - width
        xright = bright + wall_dx + width

        if x < xleft:
            force = -k*(x-xleft)
        elif x > xright:
            force = -k*(x-xright)
        elif (x < bleft-width) or (x > bright+width):
                force = 0
        elif x < bleft:
            force = -height*np.pi / (2*width) * np.sin(np.pi*(x-bleft+width)/width)
        elif x > bright:
            force = -(1-dB)*height*np.pi / (2*width) * np.sin(np.pi*(x-bright-width)/width)
        else:
            force =  np.pi * deltaf/L * np.sin(2*np.pi*(x-bleft)/L) 

        # forces = np.ones(1) * force
        # virial = np.zeros((1,1))

        return force

    def potential_and_force(self, ph):
        v_pot = self.potential(ph)
        forces = self.force(ph)
         
        return v_pot, forces
    
    
    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        x = np.linspace(-0.5, 0.5, 1000)
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
        x = np.linspace(-0.5, 0.5, 1000)
        f = np.zeros_like(x)
        for i in range(len(x)):
            _, f[i] = self.potential_and_force((x[i], 0.))
        ax.plot(x, f)
