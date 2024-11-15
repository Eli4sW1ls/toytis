import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Potential:
    def __init__(self):
        """Initialize the Potential object.

        This one is a dummy potenttial representing a double well with
        sigmoid modulated sine bumps.

        """
        self.a = 1.
        self.b = 3.5
        self.c = 0.
        self.d = .5
        self.p = .3

        # some extra constants to speed up the calculation
        self.a4 = 4.*self.a
        self.b2 = 2.*self.b
        self.exp_p15 = np.exp(15.*self.p)
        self.p3 = 3.*self.p
        self.pi2 = 2.*np.pi
        self.pi2divp = 2.*np.pi/self.p

    def potential_and_force(self, ph):
        """Returns the potential and force at phasepoint ph.

        Parameters
        ----------
        ph : tuple (x, v) of floats containing the position and velocity of the
             Langevin particle at phasepoint ph

        Returns
        -------
        pot : float
            Potential at phasepoint ph
        force : float
            Force at phasepoint ph

        """
        x = ph[0]
        doublewell = self.a*x**4 - self.b*(x - self.c)**2
        bump = self.d * np.sin(2. * np.pi * x / self.p)
        modulation = (1.-1. / (1 + np.exp(-5. * (np.abs(x) - 3.*self.p))))
        pot = doublewell + bump * modulation

        # force
        f_doublewell = -self.a4*x**3 + self.b2*(x - self.c)
        deriv_bump = self.d * np.cos(self.pi2 * x / self.p) * self.pi2divp
        deriv_modulation = \
            - 5. * np.sign(x) * np.exp(5.*(np.abs(x) + self.p3)) /\
            (np.exp(5.*np.abs(x)) + self.exp_p15)**2
        f_bump = -1. * deriv_bump * modulation - bump * deriv_modulation
        f = f_doublewell + f_bump

        return pot, f
    
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

class SingleWell2D:
    def __init__(self, k=2.):
        self.k = k
        
    def potential_and_force(self, ph):
        """Calculate the potential and force at the given phasepoint.
        
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

        # potential is simply given by pot(x,y) = k/2 * (x**2 + y**2)
        # force is simply given by f = np.array(k*x, k*y)
        pot = self.k/2 * (ph[0][0]**2 + ph[0][1]**2)
        f = -1. * self.k * ph[0]
        return pot, f
    

class RectangularGridWithBarrierPotential:
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
        b = self.M[1] - self.r * self.M[0]
        D = abs(a * x[0] - x[1] + b) / np.sqrt(a**2 + 1)
        Dsgn = np.sign(x[1] - a * x[0] - b)
        
        # Add the linear barrier potential if D < L/4
        if D < self.L / 4:
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

        # we have hardcoded state A and state B potentials here. 
        # these are horizontal lines at y = 2 and y = 13, with L = 1 and A = -10
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

    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        xvals, yvals = np.meshgrid(np.linspace(-.1*self.Lx, 1.1*self.Lx, 100), 
                                   np.linspace(-.1*self.Ly, 1.1*self.Ly, 100))
        potvals = np.array([[self.potential_and_force((np.array([x,y]),
            np.array([0,0])))[0] for x in xvals[0]] for y in yvals[:,0]]).T
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


class RectangularGridWithRuggedPotential:
    def __init__(self, Lx=0.9, Ly=0.9, k=250, N=500, A=0.075, L=0.05,
                #   positions=None, amplitudes=None, widths=None):
                 positions=np.load("/home/wouter/onesnake/ruggeda_positions.npy"),
                 amplitudes=np.load("/home/wouter/onesnake/ruggeda_amplitudes.npy"),
                    widths=np.load("/home/wouter/onesnake/ruggeda_widths.npy")):
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
        N : int
            The number of Gaussian wells/bumps.
        A : float
            The maximum absolute amplitude of the Gaussian wells/bumps.
        L : float
            The maximum width of the Gaussian wells/bumps.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.k = k
        self.N = N
        self.A = A
        self.L = L
        self.positions = positions 
        self.amplitudes = amplitudes
        self.widths = widths
        
        # # Generate random positions, amplitudes, and widths for the Gaussian wells/bumps
        # if self.positions is None:
        #     self.positions = np.random.rand(N, 2) * np.array([Lx*1.5, 0.8*Ly])
        #     self.positions[:, 0] -= 0.25 * Lx
        #     self.positions[:, 1] += 0.1 * Ly
        #     print(self.positions.shape)
        #     #print(self.positions)

        #     np.save("/home/wouter/onesnake/rugged_MD/rugged_positions.npy", self.positions)
        # else: logger.info("Using predefined positions for gaussians")
        # if self.amplitudes is None:
        #     self.amplitudes = np.random.uniform(-A, A, N) + A/4
        #     np.save("/home/wouter/onesnake/rugged_MD/rugged_amplitudes.npy", self.amplitudes)
        # else: logger.info("Using predefined amplitudes for gaussians")
        # if self.widths is None:
        #     self.widths = np.random.uniform(L/3, L, N)
        #     np.save("/home/wouter/onesnake/rugged_MD/rugged_widths.npy", self.widths)
        #     # lengths must be at least 
        # else: logger.info("Using predefined widths for gaussians")

        
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
        
        # Vectorized calculation of the Gaussian wells/bumps potentials
        pos_diffs = x - self.positions  # Shape: (N, 2)
        distances_squared = np.sum(pos_diffs**2, axis=1)  # Shape: (N,)
        exp_terms = np.exp(-distances_squared / (2 * self.widths**2))  # Shape: (N,)
        gaussian_pots = self.amplitudes * exp_terms  # Shape: (N,)
        pot += np.sum(gaussian_pots)
        
        # The force is the negative gradient of the potential
        grad_gaussian_pots = -self.amplitudes[:, np.newaxis] * pos_diffs / (self.widths[:, np.newaxis]**2) * exp_terms[:, np.newaxis]  # Shape: (N, 2)
        f -= np.sum(grad_gaussian_pots, axis=0)

        # we have hardcoded state A and state B potentials here. 
        # these are horizontal lines at y = 2 and y = 13, with L = 1 and A = -10
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

        # Adding gravity as a constant force
        f += np.array([0., -0.2]) # gravity

        return pot, f

    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        xvals, yvals = np.meshgrid(np.linspace(-.1*self.Lx, 1.1*self.Lx, 100), 
                                   np.linspace(-.1*self.Ly, 1.1*self.Ly, 100))
        potvals = np.array([[self.potential_and_force((np.array([x,y]),
            np.array([0,0])))[0] for x in xvals[0]] for y in yvals[:,0]]).T
        
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



class Maze2D_color:
    r"""Maze2D(PotentialFunction).
    
    This class definies a two-dimensional maze potential, built from a 
    pixel-drawing of a maze. 
    
    THE MAZE PNG MUST BE A SQUARE. (Maybe it works for rectangles too, but I'm not sure)

    Make sure that at least D pixels are free space around the maze. Otherwise, the particle
    may visit areas where the detection window reaches outside the maze, causing a crash.

    Note that extra whitespace does not affect performance, as it uses a detection window. 
    Make sure you have at least D+1 white pixels on the edges of the .png image,
    where 2D+1 is the size of the square detection window side.
    
    Attributes
    ----------
    * `mazefig`: The filename of the maze figure (png image)
    
    """
    
    def __init__(self, desc = '2D maze potential', mazefig = None, mazearr = None,
                lenx=1., leny=1., gauss_a = 500., gauss_b = 0., gauss_c = 1.5, D = 4,
                dw=0.5, gauss_a2 = 25., gauss_b2 = 0., gauss_c2 = 1.5, dw2=0.5, 
                global_pot = "global_slope", global_pot_params = [0.,0.5], slope_exit=0.2):

        """Set up the potential. 

        Attributes
        ----------
        * `mazefig`: String: filename of image (png,jpg)
        * `mazearr`: String: filename of numpy array
        * `lenx`: Physical length of the maze's horizontal edge (i.e. independent
                  of the maze array size)
        * `leny`: Physical length of the maze's vertical edge (i.e. independent
                  of the maze array size)
        * `N_x`: Amount of maze-pixels in x-direction (x-period)
        * `N_y`: Amount of maze-pixels in y-direction (y-period)
        * `gauss_a,b,c`: Parameters that determine the gaussian (or exponential)
                         potential of the maze walss. See the potential definitions.
        * `dw`: Width of the gaussian potential wall 
        * `D`: Detection window size. The particle can feel walls this many pixels
                up/down/left/right.
        * `global_slope_x`: Slope of the global potential in x-direction
        * `global_slope_y`: Slope of the global potential in y-direction

        PyRETIS will internally always use coordinates in the range [0,lenx] and [0,leny].
        I find no usecase for lenx or leny different from 1, but it is possible...

        Distances calculated are using pixel units. Therefore, the wall potential uses
        pixel units as well. HOWEVER: the global potential uses the same units as lenx and leny.
        """
        
        # self.maze_primitive contains 0's, 1's and 2's. 0's are free space, 1's are walls,
        # and 2's are the soft walls. The walls get the gaussian parameters (gauss_a,b,c and dw),
        # and the soft walls get the parameters (gauss_a2,b2,c2 and dw2).
        if mazearr is not None:
            self.maze_primitive = mazearr
        else:
            assert mazefig is not None
            self.maze_primitive = colormazefig_to_maze(mazefig)
        self.maze = evolve_maze(self.maze_primitive)
        #self.segments, self.segdirecs = segment_maze(self.maze)
        self.segments, self.segdirecs, self.walltypes = evolve_colormaze_easy(self.maze_primitive)
        
        #print("segments:",self.segments)
        #print("segdirecs:",self.segdirecs)
        self.N_x, self.N_y = np.shape(self.maze)
        
        assert lenx != 0.
        assert leny != 0.
        
        self.lenx = lenx
        self.leny = leny
        
        # Black maze wall potentials
        self.gauss_a = gauss_a
        self.gauss_b = gauss_b
        self.gauss_c = gauss_c
        self.dw = dw

        # Red maze wall potentials
        self.gauss_a2 = gauss_a2
        self.gauss_b2 = gauss_b2
        self.gauss_c2 = gauss_c2
        self.dw2 = dw2
        self.slope_exit = slope_exit
        
        # Global potential. Here just hardcoded to use slope. A manager could be used
        # to make this more flexible. (as in calculate_glob_pot(global_pot, global_pot_params))
        if global_pot == "global_slope":
            global_slope_x = global_pot_params[0]
            global_slope_y = global_pot_params[1]
            self.global_slope = np.array([global_slope_x,global_slope_y])
        
        # Detection windows size
        self.D = D  # How many pixels up/down/left/right can the particle feel the
                    # outer wall. Detection window (sqaure with side 2*D+1)
        
        print("##############################################")
        print("############## SIMULATION-SETUP ##############")
        print("##############################################")
        print("MAZE:")
        for i in range(len(self.maze)):
            print("".join([str(int(x)) if x != 0. else "-" for x in self.maze[i,:]]))
        print("SEGMENTED MAZE:")
        segmetn_check = np.zeros_like(self.maze)
        for i in range(len(self.maze)):
            for j in range(len(self.maze)):
                if self.segments[i,j]:
                   segmetn_check[i,j] = 1 
        for i in range(len(segmetn_check)):
            print("".join([str(int(x)) if x != 0. else "-" for x in segmetn_check[i,:]]))
        print("PARAMETERS:")
        print("gauss_a:",self.gauss_a)
        print("gauss_b:",self.gauss_b)
        print("gauss_c:",self.gauss_c)
        print("lenx:", self.lenx)
        print("leny:", self.leny)
        
    def potential(self, system):
        """Evaluate the potential.
        
        Particle (x,y) is within ([0;1],[0,1]). First rescale to the maze coordinate 
        system (mazex,mazey) within ([0;self.lenx], [0;self.leny]).
        """
        x = system.particles.pos[0,0] 
        y = system.particles.pos[0,1]
        
        mazex = x*self.N_x
        mazey = y*self.N_y
        
        mazei = int(np.floor(mazex))
        mazej = int(np.floor(mazey))
        
        # LOCAL potential (maze walls in detection window)
        x_distances = [] # List distances to horizontal segments in the detec window
        y_distances = [] # List distances to vertical segments in the detec window
        x_walltypes = [] # List walltypes of the horizontal segments in the detec window
        y_walltypes = [] # List walltypes of the vertical segments in the detec window
        D = self.D # particle can feel walls this many pixels up/down/left/right
        for i in range(-D,D+1):
            for j in range(-D,D+1):
                if self.segments[mazei+i,mazej+j]: # if that maze element has segments
                    # Keep track whether this is a soft wall or a hard wall
                    for el,eldirec,walltype in zip(self.segments[mazei+i,mazej+j],
                        self.segdirecs[mazei+i,mazej+j],self.walltypes[mazei+i,mazej+j]):
                        dist = point_to_lineseg_dist(np.array([mazex,mazey]),el)
                        if eldirec == 0: # Horizontal segment
                            x_distances.append(dist)
                            x_walltypes.append(walltype)
                        elif eldirec == 1: # Vertical segment
                            y_distances.append(dist)
                            y_walltypes.append(walltype)
                        else:
                            raise Exception("segment direction (eldirec) is neither 0 or 1, something went wrong.")
        if not x_distances: #if no x-segment is found within the detection window
            F_horizontal = 0.
        else:
            x_idx = x_distances.index(min(x_distances))
            x_d = x_distances[x_idx]
            if x_walltypes[x_idx] == 1:
                F_horizontal = gaussian_solid(x_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
            elif x_walltypes[x_idx] == 2:
                F_horizontal = gaussian_solid(x_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")
        if not y_distances:
            F_vertical = 0.
        else:
            y_idx = y_distances.index(min(y_distances))
            y_d = y_distances[y_idx]
            if y_walltypes[y_idx] == 1:
                F_vertical = gaussian_solid(y_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
            elif y_walltypes[y_idx] == 2:
                F_vertical = gaussian_solid(y_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")

        # GLOBAL Potential (uses x and y, not mazex and mazey))
        F_glob = self.global_slope[0]*x + self.global_slope[1]*((y-self.slope_exit)/(1-self.slope_exit))
        if y < self.slope_exit:
            F_glob = 0
    
        return F_horizontal + F_vertical + F_glob


    def force(self, system):
        """Evaluate the force.
        
        Particle (x,y) is within ([0;1],[0,1]). First rescale to the maze coordinate 
        system (mazex,mazey) within ([0;self.lenx], [0;self.leny]).
        """
        x = system.particles.pos[0,0] 
        y = system.particles.pos[0,1]
        
        assert x>=0 and y>=0

        forces = np.zeros_like(system.particles.pos)
        
        mazex = x*self.N_x
        mazey = y*self.N_y
        
        mazei = int(np.floor(mazex))
        mazej = int(np.floor(mazey))

        #print([x,y,mazex,mazey,mazei,mazej])
        
        # LOCAL forces (maze walls in detection window)
        x_distances = [] # List distances to horizontal segments in the detec window
        y_distances = [] # List distances to vertical segments in the detec window
        x_fvecs = [] # List force vectors of the horizontal segments in the detec window
        y_fvecs = [] # List force vectors of the vertical segments in the detec window
        x_walltypes = [] # List walltypes of the horizontal segments in the detec window
        y_walltypes = [] # List walltypes of the vertical segments in the detec window
        D = self.D # particle can feel walls this many pixels up/down/left/right
        for i in range(-D,D+1): 
            for j in range(-D,D+1):
                if self.segments[mazei+i,mazej+j]: # if that maze element has segments
                    # Keep track whether this is a soft wall or a hard wall
                    for el,eldirec,walltype in zip(self.segments[mazei+i,mazej+j],
                        self.segdirecs[mazei+i,mazej+j],self.walltypes[mazei+i,mazej+j]):
                        dist, fvec = point_to_lineseg_dist_with_normed_force_vector(np.array([mazex,mazey]),el)
                        if eldirec == 0: # Horizontal segment
                            x_distances.append(dist)
                            x_fvecs.append(fvec)
                            x_walltypes.append(walltype)
                        elif eldirec == 1: # Vertical segment
                            y_distances.append(dist)
                            y_fvecs.append(fvec)
                            y_walltypes.append(walltype)
                        else:
                            raise Exception("segment direction (eldirec) is neither 0 or 1, something went wrong.")
        
        # x_ (y_) stands for 'stemming from horizontal (vertical) wall component'
        # _x (_y) stands for 'x (y) component of a vector'
        # change this naming convention .......
        if not x_distances: # First calculate force by closest horizontal wall
            x_forces_x = 0.
            x_forces_y = 0.
        else:
            x_idx = x_distances.index(min(x_distances))
            x_d = x_distances[x_idx]
            if x_walltypes[x_idx] == 1:
                x_force_mag = gaussian_solid_force(x_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
            elif x_walltypes[x_idx] == 2:
                x_force_mag = gaussian_solid_force(x_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")
            x_f_vector = x_fvecs[x_idx]
            x_forces_x = x_f_vector[0]*x_force_mag #x component
            x_forces_y = x_f_vector[1]*x_force_mag #y_component
            
        if not y_distances: # Second calculate force by closest vertical wall
            y_forces_x = 0.
            y_forces_y = 0.
        else:
            y_idx = y_distances.index(min(y_distances))
            y_d = y_distances[y_idx]
            if y_walltypes[y_idx] == 1:
                y_force_mag = gaussian_solid_force(y_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
            elif y_walltypes[y_idx] == 2:
                y_force_mag = gaussian_solid_force(y_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")                
            y_f_vector = y_fvecs[y_idx]
            y_forces_x = y_f_vector[0]*y_force_mag #x component
            y_forces_y = y_f_vector[1]*y_force_mag #y_component
            
        # GLOBAL forces (uses x and y, not mazex and mazey)
        force_glob = -1.*(self.global_slope/(1-self.slope_exit))
        if y < self.slope_exit:
            force_glob[0] = 0
            force_glob[1] = 0
        
        forces[0,0] = x_forces_x + y_forces_x + force_glob[0]
        forces[0,1] = x_forces_y + y_forces_y + force_glob[1]

        #print("forces: ",forces)
        #print(mazex,mazey,forces[0,0],forces[0,1])
        return forces

    def potential_and_force(self, ph):
        # We don't just call potential() and force() here because that would
        # result in two calls to a similar function... 
        """Evaluate the potential and the force.
        
        Particle (x,y) is within ([0;1],[0,1]). First rescale to the maze coordinate 
        system (mazex,mazey) within ([0;self.lenx], [0;self.leny]).
        """
        pos, vel = ph
        x = pos[0]
        y = pos[1]
        
        forces = np.zeros((1,2))
        
        mazex = x*self.N_x
        mazey = y*self.N_y
        
        mazei = int(np.floor(mazex))
        mazej = int(np.floor(mazey))

        #print([x,y,mazex,mazey,mazei,mazej])
        
        # LOCAL forces (maze walls in detection window)
        x_distances = [] # List distances to horizontal segments in the detec window
        y_distances = [] # List distances to vertical segments in the detec window
        x_fvecs = [] # List force vectors of the horizontal segments in the detec window
        y_fvecs = [] # List force vectors of the vertical segments in the detec window
        x_walltypes = [] # List walltypes of the horizontal segments in the detec window
        y_walltypes = [] # List walltypes of the vertical segments in the detec window
        D = self.D # particle can feel walls this many pixels up/down/left/right
        for i in range(-D,D+1): 
            for j in range(-D,D+1):
                if self.segments[mazei+i,mazej+j]: # if that maze element has segments
                    # Keep track whether this is a soft wall or a hard wall
                    for el,eldirec,walltype in zip(self.segments[mazei+i,mazej+j],
                        self.segdirecs[mazei+i,mazej+j],self.walltypes[mazei+i,mazej+j]):
                        dist, fvec = point_to_lineseg_dist_with_normed_force_vector(np.array([mazex,mazey]),el)
                        if eldirec == 0: # Horizontal segment
                            x_distances.append(dist)
                            x_fvecs.append(fvec)
                            x_walltypes.append(walltype)
                        elif eldirec == 1: # Vertical segment
                            y_distances.append(dist)
                            y_fvecs.append(fvec)
                            y_walltypes.append(walltype)
                        else:
                            raise Exception("segment direction (eldirec) is neither 0 or 1, something went wrong.")
       
        if not x_distances: # First calculate force by closest horizontal wall
            x_forces_x = 0.
            x_forces_y = 0.
            F_horizontal = 0.
        else:
            x_idx = x_distances.index(min(x_distances))
            x_d = x_distances[x_idx]
            x_f_vector = x_fvecs[x_idx]
            if x_walltypes[x_idx] == 1:
                F_horizontal = gaussian_solid(x_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
                x_force_mag = gaussian_solid_force(x_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
            elif x_walltypes[x_idx] == 2:
                F_horizontal = gaussian_solid(x_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
                x_force_mag = gaussian_solid_force(x_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
            x_forces_x = x_f_vector[0]*x_force_mag #x component
            x_forces_y = x_f_vector[1]*x_force_mag #y_component

        if not y_distances: # Second calculate force by closest vertical wall
            y_forces_x = 0.
            y_forces_y = 0.
            F_vertical = 0.
        else:
            y_idx = y_distances.index(min(y_distances))
            y_d = y_distances[y_idx]
            y_f_vector = y_fvecs[y_idx]
            if y_walltypes[y_idx] == 1:
                F_vertical = gaussian_solid(y_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
                y_force_mag = gaussian_solid_force(y_d,self.gauss_a,self.gauss_b,self.gauss_c,self.dw)
            elif y_walltypes[y_idx] == 2:
                F_vertical = gaussian_solid(y_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
                y_force_mag = gaussian_solid_force(y_d,self.gauss_a2,self.gauss_b2,self.gauss_c2,self.dw2)
            y_forces_x = y_f_vector[0]*y_force_mag #x component
            y_forces_y = y_f_vector[1]*y_force_mag #y_component
        
        # GLOBAL forces (maze walls outside detection window)
        F_glob = self.global_slope[0]*x + self.global_slope[1]*((y-self.slope_exit)/(1-self.slope_exit))
        force_glob = -1.*(self.global_slope/(1-self.slope_exit))
        if y < self.slope_exit:
            F_glob = 0
            force_glob[0] = 0
            force_glob[1] = 0

        # Sum up all forces and potentials
        pot = F_horizontal + F_vertical + F_glob
 
        forces[0,0] = x_forces_x + y_forces_x + force_glob[0]
        forces[0,1] = x_forces_y + y_forces_y + force_glob[1]
    
        return pot, forces[0]
    
    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        xvals, yvals = np.meshgrid(np.linspace(0.025,0.975,250),
                                   np.linspace(0.025,0.975,250))
        potvals = np.array([[self.potential_and_force((np.array([x,y]),
            np.array([0,0])))[0] for x in xvals[0]] for y in yvals[:,0]])
        g =ax.contourf(xvals, yvals, potvals)
        ax.contour(xvals, yvals, potvals, levels=[1,10, 20, 30,40, 100], colors="black")
        return g

    def get_potential_plot(self):
        """Returns U(x,y) for given x,y grid."""
        xvals, yvals = np.meshgrid(np.linspace(0.025,0.975,250),
                                   np.linspace(0.025,0.975,250))
        potvals = np.array([[self.potential_and_force((np.array([x,y]),
            np.array([0,0])))[0] for x in xvals[0]] for y in yvals[:,0]])
        return xvals, yvals, potvals


###################################   
# Calculate wall distance vectors #
###################################
def point_to_lineseg_dist_with_normed_force_vector(r,l):
    """We heavily abuse the fact that the line is either in the x or y direction,
    as we assume that either dx or dy is equivalent for the two p-to-endpoint dists.
    Here, also the normalized force-vector is calculated. For distance-based functions,
    this is just the vector connecting the wallpoint that is closest to the walker-point.
    
    r: np.arr([rx,ry])
    l: [np.arr([p1x,p1y]),np.arr([p2x,p2y])]
    
    """
    d1 = r - l[0]
    d2 = r - l[1]
    
    if d1[0] == d2[0]: # x-distance 
        if np.sign(d1[1]) != np.sign(d2[1]): # point is 'in between'
            return abs(d1[0]), np.array([np.sign(-d1[0]),0])
        else:
            if abs(d1[1]) <= abs(d2[1]):
                return ((d1[0])**2 + (d1[1])**2)**(.5), (l[0]-r)/np.linalg.norm(l[0]-r)
            else:
                return ((d1[0])**2 + (d2[1])**2)**(.5), (l[1]-r)/np.linalg.norm(l[1]-r)
    else: 
        assert d1[1] == d2[1] # y-distance
        if np.sign(d1[0]) != np.sign(d2[0]): # point is 'in between'
            return abs(d1[1]), np.array([0,np.sign(-d1[1])])
        else:
            if abs(d1[0]) <= abs(d2[0]):
                return ((d1[1])**2 + (d1[0])**2)**(.5), (l[0]-r)/np.linalg.norm(l[0]-r)
            else:
                return ((d1[1])**2 + (d2[0])**2)**(.5), (l[1]-r)/np.linalg.norm(l[1]-r)    

def point_to_lineseg_dist(r,l):
    """We heavily abuse the fact that the line is either in the x or y direction,
    as we assume that either dx or dy is equivalent for the two p-to-endpoint dists
    r: np.arr([rx,ry])
    l: [np.arr([p1x,p1y]),np.arr([p2x,p2y])]

    """
    d1 = r - l[0]
    d2 = r - l[1]

    if d1[0] == d2[0]: # x-distance 
        if np.sign(d1[1]) != np.sign(d2[1]): # point is 'in between'
            return abs(d1[0])
        else:
            return ((d1[0])**2 + (min(abs(d1[1]),abs(d2[1])))**2)**(.5)

    else: 
        assert d1[1] == d2[1] # y-distance
        if np.sign(d1[0]) != np.sign(d2[0]): # point is 'in between'
            return abs(d1[1])
        else:
            return ((d1[1])**2 + (min(abs(d1[0]),abs(d2[0])))**2)**(.5)   



###########################
# Manipulate maze figures #
###########################
def segment_maze(maze):
    N,M = np.shape(maze)
    line_arr = np.zeros_like(maze,dtype='object')
    line_dir = np.zeros_like(maze,dtype='object')
    """
      (i,j+1) -->.--.--. --> (i+1,j+1)
                 |  |  |
    (i,j+1/2) -->.--p2-. --> (i+1,j+1/2)
                 |  |  |
        (i,j) -->.--.--. --> (i+1,j)
                    ^     
                    |       
                (i+1/2,j) 

    p2 is always the midpoint. For straight lines, we don't need 
    the midpoint, and we use only one line segment from p1 to p3. 
    """

    for i in range(N):
        for j in range(M):

            p2 = np.array([i+.5,j+.5])

            blocktype = maze[i,j]

            if blocktype == 0:
                line_arr[i,j] = []
                line_dir[i,j] = []

            elif blocktype == 1: 
                """
                p1--p2
                    |
                    p3
                """
                p1 = np.array([i+0.,j+.5])
                p3 = np.array([i+.5,j+0.])

                line_arr[i,j] = [[p1,p2],[p2,p3]]
                line_dir[i,j] = [0,1] #horizontal=0, vertical=1

            elif blocktype == 2: 
                """
                    p1
                    |
                p3--p2
                """
                p1 = np.array([i+.5,j+1.])
                p3 = np.array([i+0.,j+.5])

                line_arr[i,j] = [[p1,p2],[p2,p3]]
                line_dir[i,j] = [1,0]

            elif blocktype == 3:
                """
                p1
                |
                p2--p3
                """
                p1 = np.array([i+.5,j+1.])
                p3 = np.array([i+1.,j+.5])

                line_arr[i,j] = [[p1,p2],[p2,p3]]
                line_dir[i,j] = [1,0]

            elif blocktype == 4: 
                """
                p2--p1
                |
                p3
                """
                p1 = np.array([i+1.,j+.5])
                p3 = np.array([i+.5,j+0.])

                line_arr[i,j] = [[p1,p2],[p2,p3]]
                line_dir[i,j] = [0,1]

            elif blocktype == 5:
                """
                p1
                |
                p2
                |
                p3
                """
                p1 = np.array([i+.5,j+1.])
                p3 = np.array([i+.5,j+0.])

                line_arr[i,j] = [[p1,p3]]
                line_dir[i,j] = [1]

            elif blocktype == 6:
                """
                p1--p2--p3
                """
                p1 = np.array([i+0.,j+.5])
                p3 = np.array([i+1.,j+.5])

                line_arr[i,j] = [[p1,p3]]
                line_dir[i,j] = [0]

            else:
                print("WARNING: Undefined block-type during maze segmentation")

    return line_arr,line_dir

def evolve_maze(prim):
    """Make sure the first and last columns and rows are zeros!
    If not, those maze-walls will NOT be implemented. 
    This searches for _, |, |_, _|, |- and -| types of wall segments,
    and allocates a blocktype number to them, which is used in the
    segment_maze function.
    """
    N,M = np.shape(prim) 
    maze = np.zeros_like(prim)
    for i in range(1,N-1):
        for j in range(1,M-1):
            if prim[i,j] != 1:
                maze[i,j] = 0
            else:
                if prim[i-1,j] == 1 and prim[i,j-1] == 1:
                    maze[i,j] = 1
                if prim[i-1,j] == 1 and prim[i,j+1] == 1:
                    maze[i,j] = 2
                if prim[i,j+1] == 1 and prim[i+1,j] == 1:
                    maze[i,j] = 3
                if prim[i+1,j] == 1 and prim[i,j-1] == 1:
                    maze[i,j] = 4
                if prim[i,j+1] == 1 and prim[i,j-1] == 1:
                    maze[i,j] = 5
                if prim[i-1,j] == 1 and prim[i+1,j] == 1:
                    maze[i,j] = 6
    return maze

def evolve_colormaze_easy(prim):
    """
    As the first/last columns and rows are zeros, no periodic segments shoudl be detected...
    Searches for -- and | types of wall segments for each pixel of the maze.
    Saves these segments in a list for each pixel, and assigns a similarly sized
    list of directions to each pixel (0 for --, 1 for |).
    examples for pixel .:
    --.-- will get a list of two segments, and a list of two directions [0,0].
    |._ will get a list of two segments, and a list of two directions [1,0].
    We also add a list that keeps track of which walltype the segment is associated with.
    """
    N,M = np.shape(prim)
    edges = np.zeros_like(prim,dtype='object')
    direcs = np.zeros_like(prim,dtype='object')
    walltypes = np.zeros_like(prim,dtype='object')
    # Perhaps we can do it faster, but I just do a (N-1)*(M-1)*4 complexity search...
    for i in range(N-1):
        for j in range(M-1):
            edges_on_this_point = []
            direcs_on_this_point = []
            walltypes_on_this_point = []
            if prim[i,j] != 0 and prim[i,j+1] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+.5,j+1.])])
                direcs_on_this_point.append(1)
                if prim [i,j] == 2 or prim[i,j+1] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            if prim[i,j] != 0 and prim[i,j-1] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+.5,j+0.])])
                direcs_on_this_point.append(1)
                if prim [i,j] == 2 or prim[i,j-1] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            if prim[i,j] != 0 and prim[i+1,j] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+1.,j+.5])])
                direcs_on_this_point.append(0)
                if prim [i,j] == 2 or prim[i+1,j] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            if prim[i,j] != 0 and prim[i-1,j] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+0.,j+.5])])
                direcs_on_this_point.append(0)
                if prim [i,j] == 2 or prim[i-1,j] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            edges[i,j] = edges_on_this_point
            direcs[i,j] = direcs_on_this_point
            walltypes[i,j] = walltypes_on_this_point
    return edges, direcs, walltypes

def evolve_maze_easy(prim):
    N,M = np.shape(prim)
    edges = np.zeros_like(prim,dtype='object')
    direcs = np.zeros_like(prim,dtype='object')
    # Perhaps we can do it faster, but I just do a (N-1)*(M-1)*4 complexity search...
    for i in range(N-1):
        for j in range(M-1):
            edges_on_this_point = []
            direcs_on_this_point = []
            if prim[i,j] == 1 and prim[i,j+1] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+.5,j+1.])])
                direcs_on_this_point.append(1)
            if prim[i,j] == 1 and prim[i,j-1] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+.5,j+0.])])
                direcs_on_this_point.append(1)
            if prim[i,j] == 1 and prim[i+1,j] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+1.,j+.5])])
                direcs_on_this_point.append(0)
            if prim[i,j] == 1 and prim[i-1,j] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]),np.array([i+0.,j+.5])])
                direcs_on_this_point.append(0)
            edges[i,j] = edges_on_this_point
            direcs[i,j] = direcs_on_this_point
    return edges, direcs

#####################
# Read maze figures #
#####################
def read_maze_fig(mazefig):
    """Convert an image (jpg or png) to a numpy array (dim x,y,3[4])
    """
    import imageio as im
    return im.imread(mazefig)

# Black-white mazes
def mazefig_to_maze(mazefig):
    """
    Read an image, convert it to a numpy array with dim (x,y,3[4]. 
    Then convert it to an array with dim (x,y) with 0's and 1's,
    where 1 is for a black pixel (wall) and 0 is for a white pixel (free space)
    """
    mazearr = read_maze_fig(mazefig)
    return rgb2black(mazearr)

def rgb2black(pic):
    """
    pic (x,y,3) or (x,y,4) numpy array
    """
    return ((pic[:,:,0] == 0) & (pic[:,:,1] == 0) & (pic[:,:,2] == 0)).astype(int)

# Colormazes
def colormazefig_to_maze(mazefig):
    """
    Read an image, convert it to a numpy array with dim (x,y,3[4]. 
    Then convert it to an array with dim (x,y) with 0's , 1's and 2's.
    where 1 is for a black pixel (wall), 0 is for a white pixel (free space)
    and 2 is for a red pixel (soft wall).
    """
    mazearr = read_maze_fig(mazefig)
    return rgb2color(mazearr)

def rgb2color(mazearr):
    """Convert an image (jpg or png) to a numpy array (dim x,y,3[4])
    accepts red, black and white pixels.
    """
    N,M,_ = np.shape(mazearr)
    maze = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if mazearr[i,j,0] == 0 and mazearr[i,j,1] == 0 and mazearr[i,j,2] == 0:
                maze[i,j] = 1
            elif mazearr[i,j,0] == 255 and mazearr[i,j,1] == 0 and mazearr[i,j,2] == 0:
                maze[i,j] = 2
            else:
                maze[i,j] = 0
    return maze

# Some other color extractors
def rgb2bw(pic,threshold=255):
    """
    pic: (x,y,3) or (x,y,4) numpy array
    threshold: int

    Check whether the average of the R G B values of a pixel is 
    less than \treshold. If true, turn the pixel white. If false,
    turn the pixel black. Thus, if you chose 
    """
    return (np.mean(pic[:,:,:3],axis=-1) < threshold/3 +1).astype(int)

def rgb2r(pic):
    """
    pic (x,y,3) or (x,y,4) numpy array
    """
    return ((pic[:,:,0] == 255) & (pic[:,:,1] == 0) & (pic[:,:,2] == 0)).astype(int)

def rgb2g(pic):
    """
    pic (x,y,3) or (x,y,4) numpy array
    """
    return ((pic[:,:,0] == 0) & (pic[:,:,1] == 255) & (pic[:,:,2] == 0)).astype(int)

def rgb2b(pic):
    """
    pic (x,y,3) or (x,y,4) numpy array
    """
    return ((pic[:,:,0] == 0) & (pic[:,:,1] == 0) & (pic[:,:,2] == 255)).astype(int)



##############
# Potentials #
##############
def sigmoid(x,shift=0.,scale_x=1.,scale_y=1.):
    z = np.exp(-scale_x*(x-shift))
    sig = 1 / (1 + z)
    return scale_y*sig    

def exponential(d,a,b,c):
    return a*np.exp(b-c*d)

def exponential_force(d,a,b,c):
    return -a*c*np.exp(b-c*d)

def gaussian(d,a,b,c):
    return a*np.exp((-(d - b)**2)/(2*c**2))

def gaussian_force(d,a,b,c):
    return -1*a/c**2*(d-b)*np.exp((-(d - b)**2)/(2*c**2))

def gaussian_solid(d,a,b,c,dw):
    if d > dw:
        return a*np.exp((-((d-dw) - b)**2)/(2*c**2))
    else: 
        return a

def gaussian_solid_force(d,a,b,c,dw):
    if d > dw:
        return -1*a/c**2*((d-dw) - b)*np.exp((-(d - b)**2)/(2*c**2))
    else:
        return 0
