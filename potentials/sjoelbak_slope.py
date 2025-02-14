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


class RectangularGridWithBarrierSlope:
    def __init__(self, Lx=0.3, Ly=0.9, k=2000, M=(0.15, 0.45), r=-1.0, A=0.2, L=0.2):
        
        
        """
        Initialize the potential with given parameters.
        
        Parameters
        ----------
        Lx : float
            The length of the grid in the x direction.
        Ly : float
            The length of the grid in the y direction.
        k : float
            The coefficient for the harmonic potential term outside the boundaries.
        M : tuple
            A point (x, y) through which the line passes.
        r : float
            The slope of the line.
        A : float
            The amplitude of the barrier potential.
        L : float
            The length scale for the barrier potential.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.k = k
        self.M = M
        self.r = r
        self.A = A
        self.L = L
        
    def potential_and_force(self, ph):
        """
        Calculate the potential and force at the given phasepoint.
        
        Parameters
        ----------
        ph : tuple (x, v) of numpy float arrays
            The phasepoint at which to calculate.
        
        Returns
        -------
        pot : float
            The potential at ph
        f : numpy float array
            The force vector acting in ph
        """
        x, v = ph
        
        # Calculate the harmonic potential for the rectangular grid boundaries
        pot_x = 0.0
        pot_y = 0.0
        fx = 0.0
        fy = 0.0
        
        if x[0] < 0:
            pot_x = 0.5 * self.k * (0 - x[0])**2
            fx = self.k * (0 - x[0])
        elif x[0] > self.Lx:
            pot_x = 0.5 * self.k * (x[0] - self.Lx)**2
            fx = -self.k * (x[0] - self.Lx)
        
        if x[1] < 0:
            pot_y = 0.5 * self.k * (0 - x[1])**2
            fy = self.k * (0 - x[1])
        elif x[1] > self.Ly:
            pot_y = 0.5 * self.k * (x[1] - self.Ly)**2
            fy = -self.k * (x[1] - self.Ly)
        
        pot = pot_x + pot_y
        f = np.array([fx, fy])
        
        # Calculate the distance to the line
        a = self.r
        b = self.M[0] - self.r * self.M[1]
        D = abs(a * x[0] - x[1] + b) / np.sqrt(a**2 + 1)
        Dsgn = np.sign(x[1] - a * x[0] - b)
        
        # Add the linear barrier potential if D < L/4
        if 0 >= Dsgn*D >= -self.L / 4:
            barrier_pot = self.A * np.cos(2 * np.pi * D / self.L)
            pot += barrier_pot
            
            # Force due to the barrier is by definition perpendicular to the line
            dV_dD = -2 * np.pi * self.A / self.L * np.sin(2 * np.pi * D / self.L)
            # the slope of the line is a, so the gradient is (-a, 1)
            dD_dx = a / np.sqrt(a**2 + 1)
            dD_dy = -1 / np.sqrt(a**2 + 1)
            fx_barrier = dV_dD * dD_dx
            fy_barrier = dV_dD * dD_dy
            f += Dsgn * np.array([fx_barrier, fy_barrier])
        elif self.Ly > Dsgn*D > 0:
            barrier_pot = self.A * np.cos(2 * np.pi * D / (0.6*self.Ly))
            pot += barrier_pot
            
            # Force due to the barrier is by definition perpendicular to the line
            dV_dD = -2 * np.pi * self.A / self.L * np.sin(2 * np.pi * D / (0.6*self.Ly))
            # the slope of the line is a, so the gradient is (-a, 1)
            dD_dx = a / np.sqrt(a**2 + 1)
            dD_dy = -1 / np.sqrt(a**2 + 1)
            fx_barrier = dV_dD * dD_dx
            fy_barrier = dV_dD * dD_dy
            f += Dsgn * np.array([fx_barrier, fy_barrier])

        # we have hardcoded state A and state B potentials here. 
        # these are cosine bump lines, like before but with a negative amplitude
        Y1, Y2 = .25*.1, 8.75*.1
        A1, A2 = -0.15, -0.15
        L1, L2 = 2*.1, 2*.1 
        D1 = np.abs(x[1] - Y1)
        D2 = np.abs(x[1] - Y2)
        if D1 < L1 / 4:
            pot += A1 * np.cos(2 * np.pi * D1 / L1)
            dV_dD1 = A1 * 2 * np.pi / L1 * np.sin(2 * np.pi * D1 / L1)
            dD1_dx = 0
            dD1_dy = np.sign(x[1] - Y1)
            f += dV_dD1 * np.array([dD1_dx, dD1_dy])

        if D2 < L2 / 4:
            pot += A2 * np.cos(2 * np.pi * D2 / L2)
            dV_dD2 = A2 * 2 * np.pi / L2 * np.sin(2 * np.pi * D2 / L2)
            dD2_dx = 0
            dD2_dy = np.sign(x[1] - Y2)
            f += dV_dD2 * np.array([dD2_dx, dD2_dy])

        f += np.array([0., -0.2]) # gravity

        return pot, f
    
    def potential(self, ph):
        """
        Calculate the potential at the given phasepoint.
        
        Parameters
        ----------
        ph : tuple (x, v) of numpy float arrays
            The phasepoint at which to calculate.
        
        Returns
        -------
        pot : float
            The potential at ph
        """
        x, v = ph
        
        # Calculate the harmonic potential for the rectangular grid boundaries
        pot_x = 0.0
        pot_y = 0.0
        
        if x[0] < 0:
            pot_x = 0.5 * self.k * (0 - x[0])**2
        elif x[0] > self.Lx:
            pot_x = 0.5 * self.k * (x[0] - self.Lx)**2
        
        if x[1] < 0:
            pot_y = 0.5 * self.k * (0 - x[1])**2
        elif x[1] > self.Ly:
            pot_y = 0.5 * self.k * (x[1] - self.Ly)**2
        
        pot = pot_x + pot_y
        
        # Calculate the distance to the line
        a = self.r
        b = self.M[1] - self.r * self.M[0]
        D = abs(a * x[0] - x[1] + b) / np.sqrt(a**2 + 1)
        
        # Add the linear barrier potential if D < L/4
        if D < self.L / 4:
            barrier_pot = self.A * np.cos(2 * np.pi * D / self.L)
            pot += barrier_pot
        # we have hardcoded state A and state B potentials here. 
        # these are cosine bump lines, like before but with a negative amplitude
        Y1, Y2 = .25*.1, 8.75*.1
        A1, A2 = -0.15, -0.15
        L1, L2 = 2*.1, 2*.1 
        D1 = np.abs(x[1] - Y1)
        D2 = np.abs(x[1] - Y2)
        if D1 < L1 / 4:
            pot += A1 * np.cos(2 * np.pi * D1 / L1)

        if D2 < L2 / 4:
            pot += A2 * np.cos(2 * np.pi * D2 / L2)

        return pot
    
    def force(self, ph):
        """
        Calculate the force at the given phasepoint.
        
        Parameters
        ----------
        ph : tuple (x, v) of numpy float arrays
            The phasepoint at which to calculate.
        
        Returns
        -------
        f : numpy float array
            The force vector acting in ph
        """
        x, v = ph
        
        # Calculate the harmonic potential for the rectangular grid boundaries
        fx = 0.0
        fy = 0.0
        
        if x[0] < 0:
            fx = self.k * (0 - x[0])
        elif x[0] > self.Lx:
            fx = -self.k * (x[0] - self.Lx)
        
        if x[1] < 0:
            fy = self.k * (0 - x[1])
        elif x[1] > self.Ly:
            fy = -self.k * (x[1] - self.Ly)
        
        f = np.array([fx, fy])
        
        # Calculate the distance to the line
        a = self.r
        b = self.M[1] - self.r * self.M[0]
        D = abs(a * x[0] - x[1] + b) / np.sqrt(a**2 + 1)
        Dsgn = np.sign(x[1] - a * x[0] - b)
        
        # Add the linear barrier potential if D < L/4
        if D < self.L / 4:
            # Force due to the barrier is by definition perpendicular to the line
            dV_dD = -2 * np.pi * self.A / self.L * np.sin(2 * np.pi * D / self.L)
            # the slope of the line is a, so the gradient is (-a, 1)
            dD_dx = a / np.sqrt(a**2 + 1)
            dD_dy = -1 / np.sqrt(a**2 + 1)
            fx_barrier = dV_dD * dD_dx
            fy_barrier = dV_dD * dD_dy
            f += Dsgn * np.array([fx_barrier, fy_barrier])

        # we have hardcoded state A and state B potentials here. 
        # these are cosine bump lines, like before but with a negative amplitude
        Y1, Y2 = .25*.1, 8.75*.1
        A1, A2 = -0.15, -0.15
        L1, L2 = 2*.1, 2*.1 
        D1 = np.abs(x[1] - Y1)
        D2 = np.abs(x[1] - Y2)
        if D1 < L1 / 4:
            dV_dD1 = A1 * 2 * np.pi / L1 * np.sin(2 * np.pi * D1 / L1)
            dD1_dx = 0
            dD1_dy = np.sign(x[1] - Y1)
            f += dV_dD1 * np.array([dD1_dx, dD1_dy])

        if D2 < L2 / 4:
            dV_dD2 = A2 * 2 * np.pi / L2 * np.sin(2 * np.pi * D2 / L2)
            dD2_dx = 0
            dD2_dy = np.sign(x[1] - Y2)
            f += dV_dD2 * np.array([dD2_dx, dD2_dy])

        f += np.array([0., -0.2]) # gravity

        return f

    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        xvals, yvals = np.meshgrid(np.linspace(-.3*self.Ly, 1.3*self.Ly, 1000), 
                                   np.linspace(-.3*self.Lx, 1.3*self.Lx, 1000))
        potvals = np.array([[self.potential_and_force((np.array([x,y]),
            np.array([0,0])))[0] for x in yvals[:,0]] for y in xvals[0]]).T
        g =ax.contourf(xvals, yvals, potvals)
        ax.contour(xvals, yvals, potvals, levels=[-5, 0, 5, 10, 15, 20], colors="black")
        return g

    def get_potential_plot(self,N=100):
        """Returns U(x,y) for given x,y grid."""
        xvals, yvals = np.meshgrid(np.linspace(-.1*self.Lx, 1.1*self.Lx, N), 
                                   np.linspace(-.1*self.Ly, 1.1*self.Ly, N))
        potvals = np.array([[self.potential_and_force((np.array([x,y]),
            np.array([0,0])))[0] for x in xvals[0]] for y in yvals[:,0]])
        return xvals, yvals, potvals

    def plot_pot(self, intfs):
        # plt.figure()
        fig0, ax0 = plt.subplots()
        self.plot_potential(ax0)
        plt.show()

        fig1, ax1 = plt.subplots()
        x_y = np.zeros([500,500])
        i = 0
        for y in np.linspace(-.02*self.Lx, 1.02*self.Lx, 500):
            x_y[i] = np.array([self.potential_and_force((np.array([y, xx]), np.array([0,0])))[0] for xx in np.linspace(-.02*self.Ly, 1.02*self.Ly, 500)])
            i+=1
        c1 = ax1.pcolorfast((-.02*self.Ly, 1.02*self.Ly),(-.02*self.Lx, 1.02*self.Lx), x_y, vmax=0.52)
        for intf in intfs:
            ax1.axvline(intf, ymin=-.1*self.Ly, ymax=1.1*self.Ly, color='orange', linewidth=0.5)
        fig1.colorbar(c1)
        plt.show()

        # plt.figure()
        fig2, ax2 = plt.subplots()
        i=0
        for y in np.linspace(-.2*self.Lx, 1.2*self.Lx, 500):
            x_y[i] = np.array([np.average(self.potential_and_force((np.array([y, xx]), np.array([0,0])))[1]) for xx in np.linspace(-.2*self.Ly, 1.2*self.Ly, 500)])
            i+=1
        c2 = ax2.pcolorfast((-.2*self.Ly, 1.2*self.Ly),(-.2*self.Lx, 1.2*self.Lx),  x_y)
        for intf in intfs:
            ax2.axvline(intf, ymin=-.2*self.Lx, ymax=1.2*self.Lx, color='red')
        fig2.colorbar(c2)
        plt.show(block=True)