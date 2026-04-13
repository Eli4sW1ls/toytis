import logging
import math 
import numpy as np
import matplotlib.pyplot as plt  
# from pyretis.forcefield.potential import PotentialFunction
logger = logging.getLogger(__name__)  # pylint: disable=C0103
logger.addHandler(logging.NullHandler())


class CosDipMetastableWalls():
    """Cos-dip with modulation
        EW - March 2024"""

    def __init__(self, desc='1D cos_bump_series'):
        # these are the default parameters
        # bleft -- first bump starts at x=bleft, and V=0 for x<bleft
        # bright -- first bump ends at x=bright, and then:
        #             either second bump starts at x=bright
        #             or V=0 for x>bright
        # deltaf -- height of bumps is deltaf, in units kBT
        # 
        self.params = {'wall_dx': 0.1, 'k': 100., 'bleft': -0.05, 'bright': 0.45, 'deltaf': 1.2,
                        'height': 0.15, 'width': 0.05, 'dfmod': 0.12, 'db': 0}

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
        dfmod = self.params['dfmod']

        assert bleft < bright
        L = bright - bleft     # width of 1 bump
        xleft = bleft - wall_dx - width
        xright = bright + wall_dx

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
            v_pot = height + deltaf/2 * (-1 + np.cos(2*np.pi*(x-bleft)/L)) -\
            (dfmod * np.cos(2. * np.pi * x / 0.06)) * (1 - np.cos(2*np.pi*(x[np.logical_and((x >= bleft), (x < bright))]-bleft)/L))

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
        dfmod = self.params['dfmod']

        assert bleft < bright
        L = bright - bleft     # width of 1 bump
        xleft = bleft - wall_dx
        xright = bright + wall_dx

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
            force =  np.pi * deltaf/L * np.sin(2*np.pi*(x-bleft)/L) -\
            ((2*np.pi*dfmod/0.06 * np.sin(2. * np.pi * x / 0.06)) * (1 - np.cos(2*np.pi*(x-bleft)/L)) + \
            (dfmod * np.cos(2. * np.pi * x / 0.06))*(-2*np.pi/L * np.sin(2*np.pi*(x-bleft)/L)))


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


# x = np.array([i*0.001 for i in range(-2000,2000)])
# y = np.zeros_like(x)
# zz = np.zeros_like(x)

# wall_dx = 0.1
# bleft = -0.05
# bright = 0.45
# k = 100
# deltaf = 1.5
# height = 0.15
# width = 0.05
# db = 0
# dfmod = 0.12

# L = bright - bleft     # width of 1 bump
# xleft = bleft - wall_dx
# xright = bright + wall_dx

# y[np.logical_and((x >= bleft), (x < bright))] = height + deltaf/2 * (-1 + np.cos(2*np.pi*(x[np.logical_and((x >= bleft), (x < bright))]-bleft)/L)) +(-dfmod * np.cos(2. * np.pi * x[np.logical_and((x >= bleft), (x < bright))] / 0.06)) * (1 - np.cos(2*np.pi*(x[np.logical_and((x >= bleft), (x < bright))]-bleft)/L))
# y[x >= bright] = (1-db)*height/2 * (1 - np.cos(np.pi*(x[x >= bright]-bright-width)/width)) + db*height
# y[x < bleft] = height/2 * (1 - np.cos(np.pi*(x[x < bleft]-bleft+width)/width))
# y[x < bleft-width] = np.zeros_like(x[x < bleft-width])
# y[x > bright+width] = np.zeros_like(x[x > bright+width]) + db*height
# y[x<xleft] = k*(x[x<xleft]-xleft)**2/2.
# y[x > xright] = k*(x[x > xright]-xright)**2/2. + db*height

# # zz[np.logical_or((x > bleft), (x < bright2))] = deltaf*np.pi / (2*L) * np.sin(2*np.pi*(x[np.logical_or((x > bleft), (x < bright2))]-bleft)/L)
# # zz[x > bright2] = -height*np.pi / (2*width) * np.sin(np.pi*(x[x > bright2]-bright2-width)/width)
# # zz[x < bleft] = -height*np.pi / (2*width) * np.sin(np.pi*(x[x < bleft]-bleft+width)/width)

# zz[np.logical_and((x >= bleft), (x < bright))] = np.pi * deltaf/L * np.sin(2*np.pi*(x[np.logical_and((x >= bleft), (x < bright))]-bleft)/L) -\
#             ((2*np.pi*dfmod/0.06 * np.sin(2. * np.pi * x[np.logical_and((x >= bleft), (x < bright))] / 0.06)) * (1 - np.cos(2*np.pi*(x[np.logical_and((x >= bleft), (x < bright))]-bleft)/L)) + \
#             (dfmod * np.cos(2. * np.pi * x[np.logical_and((x >= bleft), (x < bright))] / 0.06))*(-2*np.pi/L * np.sin(2*np.pi*(x[np.logical_and((x >= bleft), (x < bright))]-bleft)/L)))
# zz[x >= bright] = -(1-db)*height*np.pi / (2*width) * np.sin(np.pi*(x[x >= bright]-bright-width)/width)
# zz[x < bleft] = -height*np.pi / (2*width) * np.sin(np.pi*(x[x < bleft]-bleft+width)/width)
# zz[np.logical_or(x < bleft-width, x > bright+width)] = np.zeros_like(zz[np.logical_or(x < bleft-width, x > bright+width)])
# zz[x<xleft] = -k*(x[x<xleft]-xleft)
# zz[x > xright] = -k*(x[x > xright]-xright)

# z = -np.gradient(y)  # force

# plt.rc( 'text', usetex=True )
# plt.rc('font',family = 'serif',  size=20)
# plt.xlim([-0.4,0.8])
# plt.ylim([-1,1])
# plt.plot(x,y)
# plt.vlines([-0.1,0,0.1,0.2,0.3,0.4,0.5],-2,5, linestyles='dashed', colors="green")
# plt.show()
