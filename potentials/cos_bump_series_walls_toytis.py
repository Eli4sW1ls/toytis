import logging
import numpy as np
import matplotlib.pyplot as plt
from pyretis.forcefield.potential import PotentialFunction

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CosBumpSeriesWalls(PotentialFunction):
    """Series of cosine bump barriers with walls
       Reformatted to match cos_well.py style"""

    def __init__(self, desc='1D cos_bump_series'):
        self.params = {
            'xleft': -1.5*np.pi,
            'xright': 1.5*np.pi,
            'k': 100.,
            'bleft': -np.pi,
            'bright': np.pi,
            'deltaf': 1,
            'nbump': 1
        }

    def potential(self, ph):
        x = ph[0]

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

        L = bright - bleft
        bright2 = bleft + nbump * L
        xright2 = bright2 + (xright - bright)

        if x < xleft:
            v_pot = k * (x - xleft)**2 / 2.
        elif x > xright2:
            v_pot = k * (x - xright2)**2 / 2.
        elif (x < bleft) or (x > bright2):
            v_pot = 0.0
        else:
            v_pot = deltaf / 2 * (1 - np.cos(2 * np.pi * (x - bleft) / L))

        return np.array(v_pot).sum()

    def force(self, ph):
        x = ph[0]

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

        L = bright - bleft
        bright2 = bleft + nbump * L
        xright2 = bright2 + (xright - bright)

        if x < xleft:
            force = -k * (x - xleft)
        elif x > xright2:
            force = -k * (x - xright2)
        elif (x < bleft) or (x > bright2):
            force = 0.0
        else:
            force = -deltaf * np.pi / L * np.sin(2 * np.pi * (x - bleft) / L)

        return force

    def potential_and_force(self, ph):
        v_pot = self.potential(ph)
        force = self.force(ph)
        return v_pot, force

    def plot_potential(self, ax):
        x = np.linspace(-5, 5, 1000)
        pot = np.zeros_like(x)
        for i in range(len(x)):
            pot[i], _ = self.potential_and_force((x[i],))
        ax.plot(x, pot)

    def plot_force(self, ax):
        x = np.linspace(-5, 5, 1000)
        f = np.zeros_like(x)
        for i in range(len(x)):
            _, f[i] = self.potential_and_force((x[i],))
        ax.plot(x, f)
