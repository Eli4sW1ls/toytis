import logging
import numpy as np
from pyretis.forcefield.potential import PotentialFunction
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())
np.set_printoptions(threshold=np.inf)

class Ibuprofen(PotentialFunction):
    """
    The ibuprofen-DOPC potential.
    """
    def __init__(self):
        # description = 'The ibuprofen-DOPC potential.'
        # super().__init__(dim=1,desc=description)
        # Locate the the coefficients in the input folder
        folder = "potentials/input/"
        # z-dependent slice reconstruction by polynomial approximation 
        # (deg = 15)
        self.ptrans = np.poly1d(np.load(folder+'ptrans.npy'))
        self.pcis = np.poly1d(np.load(folder+'pcis.npy'))
        self.ptrans_deriv = np.poly1d(np.load(folder+'ptrans_deriv.npy'))
        self.pcis_deriv = np.poly1d(np.load(folder+'pcis_deriv.npy'))
        # Theta-dependent slice reconstruction by fourier approximation 
        # (N = 2*5+1)
        self.Xbulk = np.load(folder+'Xbulk.npy')
        self.Xwat = np.load(folder+'Xwat.npy')

        self.Nfourier = 5
        self.omega = 1. # period of 2pi for both
        self.Lwat = 716 # Original signal length
        self.Lbulk = 713 # Original signal length
    
    def potential(self,ph):
        """
        Calculate the potential energy.
        Parameters
        ----------
        system : object like :py:class:`.System`
            This object is used for the actual calculation, typically
            only `system.particles.pos` and/or `system.particles.vel`
            will be used. In some cases `system.forcefield` can also be
            used to include specific energies for the order parameter.
        Returns
        -------
        out : float
            The potential energy.
        """
        # Extracting the coordinates
        # x  --  the position in the membrane (z)
        # theta  --  the angle, is periodic

        # 2 particles, where each is 1-dimensional
        # particle 1 is x, particle 2 is theta
        # pos = system.particles.pos

        # compute coords
        xabs = np.abs(ph[0][0]) # Sign doesn't matter for potential
        theta_full = ph[0][1] # unwrapped theta value
        # Wrap theta, and remember whether to flip the force sign
        thet, fliptheta = get_theta(theta_full)
            
        return F2D(self.pcis, self.ptrans, self.Xwat, self.Xbulk, \
            self.Nfourier, self.Nfourier, self.Lwat, self.Lbulk, \
            self.omega, self.omega, xabs,thet)
    
    def force(self,ph):
        """
        Calculate the force.
        Parameters
        ----------
        system : object like :py:class:`.System`
            This object is used for the actual calculation, typically
            only `system.particles.pos` and/or `system.particles.vel`
            will be used. In some cases `system.forcefield` can also be
            used to include specific energies for the order parameter.
        Returns
        -------
        out : numpy.ndarray
            The force vector.
        """

        # 2 particles, each 1-dimensional
        # particle 1 is x, particle 2 is theta
        # pos = system.particles.pos
        # print("force function, pos:", pos.shape)

        x = ph[0][0]  # Sign matters for force, not for potential
        xabs = np.abs(x)
        theta_full = ph[0][1] # unwrapped theta value
        # Wrap theta, and remember whether to flip the force sign
        thet, fliptheta = get_theta(theta_full)

        Fz, Ftheta = Force2D(self.pcis, self.pcis_deriv, self.ptrans, \
            self.ptrans_deriv, self.Xwat, self.Xbulk, self.Nfourier, \
            self.Nfourier, self.Lwat, self.Lbulk, self.omega, self.omega, \
            xabs, thet)

        # Determine the sign of the force components
        F = np.zeros_like(ph[0])
        if x > 0:
            F[0] = Fz
        else:
            F[0] = -Fz
        if fliptheta:
            F[1] = -Ftheta
        else:
            F[1] = Ftheta

        # We don't care about the virial for this potential
        # virial = np.zeros((self.dim,self.dim))

        return F

    def potential_and_force(self,ph):
        """
        Calculate the potential energy and force.
        Parameters
        ----------
        system : object like :py:class:`.System`
            This object is used for the actual calculation, typically
            only `system.particles.pos` and/or `system.particles.vel`
            will be used. In some cases `system.forcefield` can also be
            used to include specific energies for the order parameter.
        Returns
        -------
        out : tuple
            The potential energy and force vector.
        """

        # 2 particles, each 1-dimensional
        # particle 1 is x, particle 2 is theta
        # pos = system.particles.pos
        # print("potforce function, pos:", pos.shape)

        x = ph[0][0]  # Sign matters for force, not for potential
        xabs = np.abs(x)
        full_theta = ph[0][1] # unwrapped theta value
        # Wrap theta, and remember whether to flip the force sign
        thet, fliptheta = get_theta(full_theta)
        pot, Fz, Ftheta = F2D_and_Force2D(self.pcis, self.pcis_deriv, \
            self.ptrans, self.ptrans_deriv, self.Xwat, self.Xbulk, \
            self.Nfourier, self.Nfourier, self.Lwat, self.Lbulk, \
            self.omega, self.omega, xabs, thet)

        # Determine the sign of the force components
        F = np.zeros_like(ph[0])
        if x > 0:
            F[0] = Fz
        else:
            F[0] = -Fz
        if fliptheta:
            F[1] = -Ftheta
        else:
            F[1] = Ftheta

        # We don't care about the virial for this potential
        # virial = np.zeros((self.dim,self.dim))

        return pot, F
    
    def plot_pot(self, intfs):
        # plt.figure()
        fig1, ax1 = plt.subplots()
        x_y = np.zeros([500,500])
        i = 0
        for y in np.linspace(-np.pi, np.pi, 500):
            x_y[i] = np.array([self.potential((np.array([xx, y]), np.array([0,0]))) for xx in np.linspace(-3.5, 3.5, 500)])
            i+=1
        c1 = ax1.pcolorfast((-3.5,3.5),(-np.pi, np.pi), x_y)
        for intf in intfs:
            ax1.axvline(intf, ymin=-3, ymax=3, color='red')
        fig1.colorbar(c1)
        plt.show()

        # plt.figure()
        fig2, ax2 = plt.subplots()
        i=0
        for y in np.linspace(-np.pi, np.pi, 500):
            x_y[i] = np.array([np.average(self.force((np.array([xx, y]), np.array([0,0])))) for xx in np.linspace(-3.5, 3.5, 500)])
            i+=1
        c2 = ax2.pcolorfast((-3.5,3.5),(-np.pi, np.pi),  x_y)
        for intf in intfs:
            ax2.axvline(intf, ymin=-3, ymax=3, color='red')
        fig2.colorbar(c2)
        plt.show()


def get_theta(theta_full):
    """
    Wraps the angle theta_full to the range [0,pi], and checks whether 
    the force needs to be flipped for the current theta_full value.
    Parameters
    ----------
    theta_full : float
        The angle theta, unwrapped (]-inf,inf[)
    Returns
    -------
    theta : float
        The angle theta, wrapped to the range [0,pi].
    flip_theta : bool
        Whether the force needs to be flipped
    """
    # periodic boundary conditions:
    #   1) 2pi periodicity 
    #   2) mirror symmetry around theta=pi
    # If we flipped theta, we need to flip the sign of the force

    flip_theta = False
    # Get the angle in the range [0,2pi]
    theta = theta_full%(2*np.pi)

    # If it's in the range [pi,2pi], flip it
    if theta > np.pi:
        theta = 2*np.pi - theta
        flip_theta = True

    return theta, flip_theta

def fourier_approx_eval_func(X,omega,L,N,t):
    """
    evaluate the fourier approximation of function x at time t
    """
    phi = np.array([np.exp(1j*omega*k*t) for k in range(-N+1,N)])
    return np.real(np.dot(X,phi)/L)

def fourier_approx_eval_deriv(X,omega,L,N,t):
    """
    evaluate the derivative of the fourier approximation of x at time t
    """
    phi = np.array([1j*omega*k*np.exp(1j*omega*k*t) for k in range(-N+1,N)])
    return np.real(np.dot(X,phi)/L)

def F2D(fcis,ftrans,Xwat,Xbulk,Nwat,Nbulk,Lwat,Lbulk,omegawat,omegabulk,x,thet):
    """
    Evaluate 2D potential at (x,thet). Constructed by linear combination of 
    the 1D slice potentials

    It is assumed here that x >= 0
    """
    fcisval = np.polyval(fcis,x)
    ftransval = np.polyval(ftrans,x)
    gwatval = fourier_approx_eval_func(Xwat,omegawat,Lwat,Nwat,thet)
    gbulkval = fourier_approx_eval_func(Xbulk,omegabulk,Lbulk,Nbulk,thet)
    fcismult = (np.pi-thet)/np.pi
    ftransmult = thet/np.pi
    gwatmult = x/3.5
    gbulkmult = (3.5-x)/3.5
    return fcisval*fcismult + \
        ftransval*ftransmult + \
        gwatval*gwatmult + \
        gbulkval*gbulkmult

def Force2D(fcis,fcis_deriv,ftrans,ftrans_deriv,
            Xwat,Xbulk,Nwat,Nbulk,Lwat,Lbulk,omegawat,
            omegabulk,x,thet):
    """
    Evaluate 2D force at (x,thet). Constructed by linear combination of 
    the derivatives of the 1D slice potentials

    It is assumed here that x>=0.
    """
    fcis_derivval = np.polyval(fcis_deriv,x)
    ftrans_derivval = np.polyval(ftrans_deriv,x)

    fcisval = np.polyval(fcis,x)
    ftransval = np.polyval(ftrans,x)

    gwat_derivval = fourier_approx_eval_deriv(Xwat,omegawat,Lwat,Nwat,thet)
    gbulk_derivval = fourier_approx_eval_deriv(Xbulk,omegabulk,Lbulk,Nbulk,thet)

    gwatval = fourier_approx_eval_func(Xwat,omegawat,Lwat,Nwat,thet)
    gbulkval = fourier_approx_eval_func(Xbulk,omegabulk,Lbulk,Nbulk,thet)

    fcismult = (np.pi-thet)/np.pi
    ftransmult = thet/np.pi

    gwatmult = x/3.5
    gbulkmult = (3.5-x)/3.5

    fcis_derivmult = -1/np.pi
    ftrans_derivmult = 1/np.pi

    gwat_derivmult = 1/3.5
    gbulk_derivmult = -1/3.5

    # return -dV/dz, -dV/dtheta
    return -1.*(fcis_derivval*fcismult + ftrans_derivval*ftransmult \
            + gwatval*gwat_derivmult + gbulkval*gbulk_derivmult), \
            -1*(fcisval*fcis_derivmult + ftransval*ftrans_derivmult \
            + gwat_derivval*gwatmult + gbulk_derivval*gbulkmult)


def F2D_and_Force2D(fcis,fcis_deriv,ftrans,ftrans_deriv,
        Xwat,Xbulk,Nwat,Nbulk,Lwat,Lbulk,omegawat,
        omegabulk,x,thet):
    """
    Evaluate 2D potential and force at (x,thet). 
    Constructed by linear combination of the 1D slice potentials and 
    their derivatives

    It is assumed here that x>=0.
    """
    fcis_derivval = np.polyval(fcis_deriv,x)
    ftrans_derivval = np.polyval(ftrans_deriv,x)

    fcisval = np.polyval(fcis,x)
    ftransval = np.polyval(ftrans,x)

    gwat_derivval = fourier_approx_eval_deriv(Xwat,omegawat,Lwat,Nwat,thet)
    gbulk_derivval = fourier_approx_eval_deriv(Xbulk,omegabulk,Lbulk,Nbulk,thet)

    gwatval = fourier_approx_eval_func(Xwat,omegawat,Lwat,Nwat,thet)
    gbulkval = fourier_approx_eval_func(Xbulk,omegabulk,Lbulk,Nbulk,thet)

    fcismult = (np.pi-thet)/np.pi
    ftransmult = thet/np.pi

    gwatmult = x/3.5
    gbulkmult = (3.5-x)/3.5

    fcis_derivmult = -1/np.pi
    ftrans_derivmult = 1/np.pi

    gwat_derivmult = 1/3.5
    gbulk_derivmult = -1/3.5

    # return pot, -dV/dz, -dV/dtheta
    return fcisval*fcismult + ftransval*ftransmult + gwatval*gwatmult + \
            gbulkval*gbulkmult, \
            -1.*(fcis_derivval*fcismult + ftrans_derivval*ftransmult \
            + gwatval*gwat_derivmult + gbulkval*gbulk_derivmult), \
            -1.*(fcisval*fcis_derivmult + ftransval*ftrans_derivmult \
            + gwat_derivval*gwatmult + gbulk_derivval*gbulkmult)