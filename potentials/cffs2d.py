# -*- coding: utf-8 -*-
# Copyright (c) 2019, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""This is a 2D example potential."""
import logging
import numpy as np
import matplotlib.pyplot as plt
from pyretis.forcefield.potential import PotentialFunction
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class PotentialcFFS(PotentialFunction):
    r"""Hyst2D(PotentialFunction).

    This class defines a 2D dimensional potential with two
    reactions channels.
    The potential energy (:math:`V_\text{pot}`) is given by

    TODO -- adapt.... TODO

    .. math::

       V_\text{pot}(x, y) = \exp(-cx**2)
         ( Vmin/4 + Vmin2/4 + Vmax/2

         + (Vmin2-Vmin1)/2 \sin(2\pi y/L)
         + (Vmax/2-Vmin1/2-Vmin2/2) \cos(4\pi y/L) )

    where :math:`x` and :math:`y` gives the positions and :math:`\gamma_1`,
    :math:`\gamma_2`, :math:`\gamma_3`, :math:`\alpha_1`, :math:`\alpha_2`,
    :math:`\beta_1`, :math:`\beta_2`, :math:`x_0` and :math:`y_0` are
    potential parameters.

    Attributes
    ----------
    params : dict
        Contains the parameters. The keys are:

       *`gamma1`: The :math:`\gamma_1` parameter for the potential.
       *`gamma2`: The :math:`\gamma_2` parameter for the potential.
       *`gamma3`: The :math:`\gamma_3` parameter for the potential.
       *`alpha1`: The :math:`\alpha_1` parameter for the potential.
       *`alpha2`: The :math:`\alpha_2` parameter for the potential.
       *`beta1`: The :math:`\beta_1` parameter for the potential.
       *`beta2`: The :math:`\beta_2` parameter for the potential.
       *`x0`: The :math:`x_0` parameter for the potential.
       *`y0`: The :math:`y_0` parameter for the potential.

    """

    def __init__(self, n_PES=1, angle=0, desc='2D surface with 2 reaction channels'):
        """Set up the potential.

        Parameters
        ----------
        a : float, optional
            Parameter for the potential.
        b : float, optional
            Parameter for the potential.
        c : float, optional
            Parameter for the potential.
        desc : string, optional
            Description of the force field.

        """
        # super().__init__(dim=2, desc=desc)
        self.params = {'n': n_PES, 'angle': angle,
                       'a1': 0.02, 'k1': 4, 'b1': 0.3, 'dz1': 0.0026, 
                       'a4': 2.93, 'b4': 4.2, 'c4': 0.005, 'dz4': -0.627}

    def potential(self, ph):
        """Evaluate the potential.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out : float
            The potential energy.

        """
        xi = ph[0][1]
        yi = ph[0][0]

        n_PES = self.params['n']
        angle = self.params['angle']

        if n_PES == 1:
            a = self.params['a1']
            b = self.params['b1']
            k = self.params['k1']
            dz = self.params['dz1']
            x = xi*np.cos(angle) - yi*np.sin(angle)
            y = xi*np.sin(angle) + yi*np.cos(angle)
            v_pot = a*(x**4+y**4)-k*np.exp(-((x+2)**2+(y+2)**2))-k*np.exp(-((x-2)**2+(y-2)**2))+b*(x-y)**2+dz
        elif n_PES == 4:
            x = xi*np.cos(angle) - yi*np.sin(angle)
            y = xi*np.sin(angle) + yi*np.cos(angle)
            a = self.params['a4']
            b = self.params['b4']
            c = self.params['c4']
            dz = self.params['dz4']
            # --- 4.1 ---
            # v_pot = -a*np.exp(-((x-3)**2/2+(y-2.5)**2/2))-a*np.exp(-((x+2)**2/2+(y+2)**2/2))+\
            #         b*np.exp(-0.32*((x+1)**2+(y-2)**2+12*(x+y-2.7)**2-1))+2*b*np.exp(-0.1*((x-3)**2+(y-0.35)**2+10*(x+y)**2-1))+\
            #         c*(x**4+y**4)+dz
            # --- 4.2 ---
            v_pot = 2*b*np.exp(-0.05*(45*(x + y)**2 + (x - 4)**2 + (y + 0.2)**2 - 1)) + \
                    b*np.exp(-0.12*(35*(x + y - 2.7)**2 + (x + 1)**2 + (y - 3)**2 - 1)) - \
                    a*np.exp(-(x + 2)**2 / 2 - (y + 2)**2 / 2) - a*np.exp(-(x - 3)**2 / 2 - (y - 2.5)**2 / 2) + \
                    c*(x**4 + y**4) - dz
        else:
            raise ValueError("Please choose a number from 1 to 4 to select the potential.")

        return v_pot.sum()

    def force(self, ph):
        """Evaluate forces.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out[0] : numpy.array
            The calculated force.
        out[1] : numpy.array
            The virial, currently not implemented for this potential.

        """
        xi = ph[0][1]  # pylint: disable=invalid-name
        yi = ph[0][0] # pylint: disable=invalid-name

        n_PES = self.params['n']
        angle = self.params['angle']

        forces = np.zeros_like(ph[0])
        if n_PES == 1:
            x = xi*np.cos(angle) - yi*np.sin(angle)
            y = xi*np.sin(angle) + yi*np.cos(angle)
            a = self.params['a1']
            b = self.params['b1']
            k = self.params['k1']
            dz = self.params['dz1']
            forces[1] = 4*a*x**3 + 2*b*(x - y) + 2*k*(x - 2)*np.exp(-(x - 2)**2 - \
                     (y - 2)**2) + 2*k*(x + 2)*np.exp(-(x + 2)**2 - (y + 2)**2) # d/dx
            forces[0] = 4*a*y**3 - 2*b*(x - y) + 2*k*(y - 2)*np.exp(-(x - 2)**2 - (y - 2)**2) + \
                        2*k*(y + 2)*np.exp(-(x + 2)**2 - (y + 2)**2) # d/dy
        elif n_PES == 4:
            x = xi*np.cos(angle) - yi*np.sin(angle)
            y = xi*np.sin(angle) + yi*np.cos(angle)
            a = self.params['a4']
            b = self.params['b4']
            c = self.params['c4']
            dz = self.params['dz4']
            # --- 4.1 ---
            # forces[1] = -0.2*b*(20*(x + y) + 2*(x - 3))*np.exp(-0.1*(10*(x + y)**2 + (x - 3)**2 + (y - 0.35)**2 - 1)) - \
            #             0.32*b*(24*(x + y - 2.7) + 2*(x + 1))*np.exp(-0.32*(12*(x + y - 2.7)**2 + (x + 1)**2 + (y - 2)**2 - 1)) - \
            #                 a*(-x - 2)*np.exp(-(x + 2)**2 / 2 - (y + 2)**2 / 2) - a*(3 - x)*np.exp(-(x - 3)**2 / 2 - (y - 2.5)**2 / 2) + 4*c*x**3 # d/dx
            # forces [0] = -0.2*b*(20*(y + x) + 2*(y - 0.35))*np.exp(-0.1*(10*(y + x)**2 + (y - 0.35)**2 + (x - 3)**2 - 1)) - \
            #                 0.32*b*(24*(y + x - 2.7) + 2*(y - 2))*np.exp(-0.32*(12*(y + x - 2.7)**2 + (y - 2)**2 + (x + 1)**2 - 1)) - \
            #                     a*(-y - 2)*np.exp(-(y + 2)**2 / 2 - (x + 2)**2 / 2) - a*(2.5 - y)*np.exp(-(y - 2.5)**2 / 2 - (x - 3)**2 / 2) + 4*c*y**3 #d/dy
            # --- 4.2 ---
            forces[1] = -0.1*b*(90*(x + y) + 2*(x - 4))*np.exp(-0.05*(45*(x + y)**2 + (x - 4)**2 + (y + 0.2)**2 - 1)) - \
                        0.12*b*(70*(x + y - 2.7) + 2*(x + 1))*np.exp(-0.12*(35*(x + y - 2.7)**2 + (x + 1)**2 + (y - 3)**2 - 1)) - \
                            a*(-x - 2)*np.exp(-(x + 2)**2 / 2 - (y + 2)**2 / 2) - a*(3 - x)*np.exp(-(x - 3)**2 / 2 - (y - 2.5)**2 / 2) + 4*c*x**3 # d/dx
            forces [0] = -0.1*b*(90*(y + x) + 2*(y + 0.2))*np.exp(-0.05*(45*(y + x)**2 + (y + 0.2)**2 + (x - 4)**2 - 1)) - \
                            0.12*b*(70*(y + x - 2.7) + 2*(y - 3))*np.exp(-0.12*(35*(y + x - 2.7)**2 + (y - 3)**2 + (x + 1)**2 - 1)) - \
                            a*(-y - 2)*np.exp(-(y + 2)**2 / 2 - (x + 2)**2 / 2) - a*(2.5 - y)*np.exp(-(y - 2.5)**2 / 2 - (x - 3)**2 / 2) + 4*c*y**3 #d/dy
        else:
            raise ValueError("Please choose a number from 1 to 4 to select the potential.")
        
        # virial = np.zeros((self.dim, self.dim))  # just return zeros here
        return -forces

    def potential_and_force(self, ph):
        """Evaluate the potential and the force.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out[0] : float
            The potential energy as a float.
        out[1] : numpy.array
            The force as a numpy.array of the same shape as the
            positions in `particles.pos`.
        out[2] : numpy.array
            The virial, currently not implemented for this potential.

        """
        # virial = np.zeros((self.dim, self.dim))  # just return zeros here
        vpot = self.potential(ph)
        forces = self.force(ph)

        return vpot, forces

    def plot_pot(self, intfs):
        lim = 4 if self.params["n"] != 4 else 7 
        fig1, ax1 = plt.subplots()
        x_y = np.zeros([500,500])
        i = 0
        for y in np.linspace(-lim, lim, 500):
            x_y[i] = np.array([self.potential((np.array([y, xx]), np.array([0,0]))) for xx in np.linspace(-lim, lim, 500)])
            i+=1
        c1 = ax1.pcolorfast((-lim,lim),(-lim,lim), x_y, vmax=4)
        for intf in intfs:
            ax1.axvline(intf, ymin=-4, ymax=4, color='red')
        fig1.colorbar(c1)
        plt.show()

        # plt.figure()
        fig2, ax2 = plt.subplots()
        i=0
        for y in np.linspace(-lim, lim, 500):
            x_y[i] = np.array([np.sum(self.force((np.array([y, xx]), np.array([0,0])))) for xx in np.linspace(-lim, lim, 500)])
            i+=1
        c2 = ax2.pcolorfast((-lim,lim),(-lim,lim),  x_y, vmax=9)
        for intf in intfs:
            ax2.axvline(intf, ymin=-3, ymax=3, color='red')
        fig2.colorbar(c2)
        plt.show(block=True)