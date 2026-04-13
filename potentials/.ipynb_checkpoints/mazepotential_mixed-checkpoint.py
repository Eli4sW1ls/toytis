# created by Wouter Vervust ~ August 2022
import logging
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())
np.set_printoptions(threshold=np.inf)
class Maze2D_color():
    r"""Maze2D(PotentialFunction).
    
    This class definies a two-dimensional maze potential, built from a 
    pixel-drawing of a maze. 
    
    The pixel maze must be square. (Maybe it works for rectangles too, 
    but I'm not sure)

    Note that extra whitespace does not affect performance, as it uses a 
    detection window. Make sure you have at least D+1 white pixels on the edges 
    of the .png image, where 2D+1 is the size of the square detection window 
    side.
    """
    
    def __init__(self, desc = '2D maze potential', mazefig = None, \
        mazearr = None, lenx=1., leny=1., gauss_a = 500., gauss_b = 0., \
        gauss_c = 1.0, D = 4, dw = 0.5, gauss_a2 = 25., gauss_b2 = 0., \
        gauss_c2 = 1.0, dw2 = 0.5, global_pot = "global_slope", \
        global_pot_params = [0.,0.5], slope_exit=0.2):

        """Set up the potential. 

        Attributes
        ----------
        * `mazefig`: String: filename of image (png,jpg)
        * `mazearr`: String: filename of numpy array
        * `lenx`: Physical length of the maze's horizontal edge 
                (i.e. independent of the maze array size)
        * `leny`: Physical length of the maze's vertical edge 
                (i.e. independent of the maze array size)
        * `N_x`: Amount of maze-pixels in x-direction (x-period)
        * `N_y`: Amount of maze-pixels in y-direction (y-period)
        * `gauss_a,b,c`: Parameters that determine the gaussian (or exponential)
                         potential of the maze wals. See the potential defs.
        * `dw`: Width of the gaussian potential wall 
        * `D`: Detection window size. The particle can feel walls this many 
                pixels up/down/left/right.
        * `global_slope_x`: Slope of the global potential in x-direction
        * `global_slope_y`: Slope of the global potential in y-direction

        PyRETIS will internally always use coordinates in the range [0,lenx] 
        and [0,leny]. I find no usecase for lenx or leny different from 1, 
        but it is possible...

        Distances calculated are using pixel units. Therefore, the wall 
        potential uses pixel units as well. HOWEVER: the global potential uses 
        the same units as lenx and leny. Might be optimized in the future, if 
        this function is used more often...
        """
        
        # super().__init__(dim=2, desc=desc)
        # self.maze_primitive contains 0's, 1's and 2's: 
        # 0's are free space, 
        # 1's are hard walls,
        # 2's are soft walls. 
        # The hard walls get the gaussian parameters (gauss_a,b,c and dw),
        # The soft walls get the gaussian parameters (gauss_a2,b2,c2 and dw2).
        if mazearr is not None:
            self.maze_primitive = mazearr
        else:
            assert mazefig is not None
            self.maze_primitive = colormazefig_to_maze(mazefig)
        self.maze = evolve_maze(self.maze_primitive)
        # self.segments, self.segdirecs = segment_maze(self.maze)
        self.segments, self.segdirecs, self.walltypes = \
            evolve_colormaze_easy(self.maze_primitive)
        
        self.N_x, self.N_y = np.shape(self.maze)
        
        assert lenx != 0.
        assert leny != 0.
        
        self.lenx = lenx
        self.leny = leny
        
        # Black maze wall potentials (hard walls)
        self.gauss_a = gauss_a
        self.gauss_b = gauss_b
        self.gauss_c = gauss_c
        self.dw = dw

        # Red maze wall potentials (soft walls)
        self.gauss_a2 = gauss_a2
        self.gauss_b2 = gauss_b2
        self.gauss_c2 = gauss_c2
        self.dw2 = dw2
        self.slope_exit = slope_exit
        
        # Global potential. Here just hardcoded to use slope. A manager could be
        # used to make this more flexible. (as in calculate_glob_pot(global_pot,
        # global_pot_params))
        if global_pot == "global_slope":
            global_slope_x = global_pot_params[0]
            global_slope_y = global_pot_params[1]
            self.global_slope = np.array([global_slope_x,global_slope_y])
        
        # Detection windows size
        self.D = D  # How many pixels up/down/left/right can the particle feel 
                    # the walls. Detection window (square with side 2*D+1)
        
        logger.info("##############################################")
        logger.info("############## SIMULATION-SETUP ##############")
        logger.info("##############################################")
        logger.info("MAZE:")
        for i in range(len(self.maze)):
            logger.info(self.maze[i,:])
        logger.info("SEGMENTED MAZE:")
        segment_check = np.zeros_like(self.maze)
        for i in range(len(self.maze)):
            for j in range(len(self.maze)):
                if self.segments[i,j]:
                   segment_check[i,j] = 1 
        logger.info(str(segment_check))
        logger.info("PARAMETERS:")
        logger.info("gauss_a:",self.gauss_a)
        logger.info("gauss_b:",self.gauss_b)
        logger.info("gauss_c:",self.gauss_c)
        logger.info("lenx:", self.lenx)
        logger.info("leny:", self.leny)
        
    def potential(self, ph):
        """Evaluate the potential.
        
        Particle (x,y) is in ([0;1],[0;1]). First rescale to the maze coordinate 
        system (mazex,mazey) within ([0;self.lenx], [0;self.leny]).
        """
        x = ph[0][0]
        y = ph[0][1]
        
        mazex = x*self.N_x
        mazey = y*self.N_y
        
        mazei = int(np.floor(mazex))
        mazej = int(np.floor(mazey))
        
        # LOCAL potential (maze walls in detection window)
        x_distances = []  # List distances to horizontal segments within Dwindow
        y_distances = []  # List distances to vertical segments within Dwindow
        x_walltypes = []  # List walltypes of horizontal segments within Dwindow
        y_walltypes = []  # List walltypes of vertical segments within Dwindow
        D = self.D # particle can feel walls this many pixels up/down/left/right
        for i in range(-D,D+1):
            for j in range(-D,D+1):
                if self.segments[mazei+i,mazej+j]:  
                    # if that maze el has segments, we do the distance calc.
                    # Keep track whether this is a soft wall or a hard wall
                    for el,eldirec,walltype in \
                        zip(self.segments[mazei+i,mazej+j], \
                        self.segdirecs[mazei+i,mazej+j], \
                            self.walltypes[mazei+i,mazej+j]):
                        dist = point_to_lineseg_dist(np.array([mazex,mazey]),el)
                        if eldirec == 0:  # Horizontal segment
                            x_distances.append(dist)
                            x_walltypes.append(walltype)
                        elif eldirec == 1:  # Vertical segment
                            y_distances.append(dist)
                            y_walltypes.append(walltype)
                        else:
                            raise Exception("segment direction (eldirec) is " +\
                                "neither 0 or 1, something went wrong.")
        if not x_distances:  # if no x-segment is found within Dwindow
            F_horizontal = 0.
        else:
            x_idx = x_distances.index(min(x_distances))
            x_d = x_distances[x_idx]
            if x_walltypes[x_idx] == 1:
                F_horizontal = gaussian_solid(x_d, self.gauss_a, self.gauss_b, \
                    self.gauss_c, self.dw)
            elif x_walltypes[x_idx] == 2:
                F_horizontal = gaussian_solid(x_d, self.gauss_a2, \
                    self.gauss_b2, self.gauss_c2, self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")
        if not y_distances:
            F_vertical = 0.
        else:
            y_idx = y_distances.index(min(y_distances))
            y_d = y_distances[y_idx]
            if y_walltypes[y_idx] == 1:
                F_vertical = gaussian_solid(y_d, self.gauss_a, self.gauss_b, \
                    self.gauss_c, self.dw)
            elif y_walltypes[y_idx] == 2:
                F_vertical = gaussian_solid(y_d, self.gauss_a2, self.gauss_b2, \
                    self.gauss_c2, self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")

        # GLOBAL Potential (uses x and y, not mazex and mazey))
        F_glob = self.global_slope[0]*x + \
            self.global_slope[1]*((y-self.slope_exit)/(1-self.slope_exit))
        if y < self.slope_exit:
            F_glob = 0
    
        return F_horizontal + F_vertical + F_glob


    def force(self, ph):
        """Evaluate the force.
        
        Particle (x,y) is within ([0;1],[0,1]). 
        First rescale to the maze coordinate system (mazex,mazey) 
        within ([0;self.lenx], [0;self.leny]).
        """
        x = ph[0][0]
        y = ph[0][1]
        
        assert x>=0 and y>=0

        forces = np.zeros_like(ph[0])
        
        mazex = x*self.N_x
        mazey = y*self.N_y
        
        mazei = int(np.floor(mazex))
        mazej = int(np.floor(mazey))
        
        # LOCAL forces (maze walls in detection window)
        x_distances = [] # List distances to horizontal segments within Dwindow
        y_distances = [] # List distances to vertical segments within Dwindow
        x_fvecs = [] # List force vectors of horizontal segments within Dwindow
        y_fvecs = [] # List force vectors of vertical segments within Dwindow
        x_walltypes = [] # List walltypes of horizontal segments within Dwindow
        y_walltypes = [] # List walltypes of vertical segments within Dwindow
        D = self.D # particle can feel walls this many pixels up/down/left/right
        for i in range(-D,D+1): 
            for j in range(-D,D+1):
                if self.segments[mazei+i,mazej+j]: 
                    # if that maze element has segments, we do the distance calc
                    # Keep track whether this is a soft wall or a hard wall
                    for el, eldirec, walltype in \
                        zip(self.segments[mazei+i,mazej+j], \
                            self.segdirecs[mazei+i,mazej+j], \
                            self.walltypes[mazei+i,mazej+j]):
                        dist, fvec = \
                            point_to_lineseg_dist_with_normed_force_vector( \
                                np.array([mazex,mazey]),el)
                        if eldirec == 0:  # Horizontal segment
                            x_distances.append(dist)
                            x_fvecs.append(fvec)
                            x_walltypes.append(walltype)
                        elif eldirec == 1:  # Vertical segment
                            y_distances.append(dist)
                            y_fvecs.append(fvec)
                            y_walltypes.append(walltype)
                        else:
                            raise Exception("segment direction (eldirec) is " +\
                                "neither 0 or 1, something went wrong.")
        
        # x_ (y_) stands for 'stemming from horizontal (vertical) wall'
        # _x (_y) stands for 'x (y) component of a vector'
        # Horrible naming convention. Open for change if used more...
        if not x_distances: # First calculate force by closest horizontal wall
            x_forces_x = 0.
            x_forces_y = 0.
        else:
            x_idx = x_distances.index(min(x_distances))
            x_d = x_distances[x_idx]
            if x_walltypes[x_idx] == 1:
                x_force_mag = gaussian_solid_force(x_d, self.gauss_a, \
                    self.gauss_b, self.gauss_c, self.dw)
            elif x_walltypes[x_idx] == 2:
                x_force_mag = gaussian_solid_force(x_d, self.gauss_a2, \
                    self.gauss_b2, self.gauss_c2, self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")
            x_f_vector = x_fvecs[x_idx]
            x_forces_x = x_f_vector[0]*x_force_mag  # x component
            x_forces_y = x_f_vector[1]*x_force_mag  # y_component
            
        if not y_distances:  # Second calculate force by closest vertical wall
            y_forces_x = 0.
            y_forces_y = 0.
        else:
            y_idx = y_distances.index(min(y_distances))
            y_d = y_distances[y_idx]
            if y_walltypes[y_idx] == 1:
                y_force_mag = gaussian_solid_force(y_d, self.gauss_a, \
                    self.gauss_b, self.gauss_c, self.dw)
            elif y_walltypes[y_idx] == 2:
                y_force_mag = gaussian_solid_force(y_d, self.gauss_a2, \
                    self.gauss_b2, self.gauss_c2, self.dw2)
            else:
                raise Exception("Illegal walltype, something went wrong.")                
            y_f_vector = y_fvecs[y_idx]
            y_forces_x = y_f_vector[0]*y_force_mag  # x component
            y_forces_y = y_f_vector[1]*y_force_mag  # y_component
            
        # GLOBAL forces (uses x and y, not mazex and mazey)
        force_glob = -1.*(self.global_slope/(1-self.slope_exit))
        if y < self.slope_exit:
            force_glob[0] = 0
            force_glob[1] = 0
        
        forces[0] = x_forces_x + y_forces_x + force_glob[0]
        forces[1] = x_forces_y + y_forces_y + force_glob[1]
        
        # We don't care about the virial here
        # virial = np.zeros((self.dim, self.dim))  

        return forces

    def potential_and_force(self, ph):
        # We don't just call potential() and force() here because that would
        # result in two calls to a similar function, which is inefficient.
        """Evaluate the potential and the force.
        
        Particle (x,y) is within ([0;1],[0,1]). First rescale to the maze 
        coordinate system (mazex,mazey) within ([0;self.lenx], [0;self.leny]).
        """
        x = ph[0][0]
        y = ph[0][1]
        
        forces = np.zeros_like(ph[0])
        
        mazex = x*self.N_x
        mazey = y*self.N_y
        
        mazei = int(np.floor(mazex))
        mazej = int(np.floor(mazey))
        
        # LOCAL forces (maze walls in detection window)
        x_distances = [] # List distances to horizontal segments within Dwindow
        y_distances = [] # List distances to vertical segments within Dwindow
        x_fvecs = [] # List force vectors of horizontal segments within Dwindow
        y_fvecs = [] # List force vectors of vertical segments within Dwindow
        x_walltypes = [] # List walltypes of horizontal segments within Dwindow
        y_walltypes = [] # List walltypes of vertical segments within Dwindow
        D = self.D # particle can feel walls this many pixels up/down/left/right
        for i in range(-D,D+1): 
            for j in range(-D,D+1):
                try: self.segments[mazei+i,mazej+j] 
                except: j+=0
                if self.segments[mazei+i,mazej+j]: 
                    # if that maze element has segments, do dist calculation
                    # Keep track whether this is a soft wall or a hard wall
                    for el,eldirec,walltype in \
                        zip(self.segments[mazei+i,mazej+j], \
                            self.segdirecs[mazei+i,mazej+j], \
                            self.walltypes[mazei+i,mazej+j]):
                        dist, fvec = \
                            point_to_lineseg_dist_with_normed_force_vector( \
                                np.array([mazex,mazey]),el)
                        if eldirec == 0:  # Horizontal segment
                            x_distances.append(dist)
                            x_fvecs.append(fvec)
                            x_walltypes.append(walltype)
                        elif eldirec == 1:  # Vertical segment
                            y_distances.append(dist)
                            y_fvecs.append(fvec)
                            y_walltypes.append(walltype)
                        else:
                            raise Exception("segment direction (eldirec) is " +\
                                "neither 0 or 1, something went wrong.")
       
        if not x_distances:  # First calculate force by closest horizontal wall
            x_forces_x = 0.
            x_forces_y = 0.
            F_horizontal = 0.
        else:
            x_idx = x_distances.index(min(x_distances))
            x_d = x_distances[x_idx]
            x_f_vector = x_fvecs[x_idx]
            if x_walltypes[x_idx] == 1:
                F_horizontal = gaussian_solid(x_d, self.gauss_a, self.gauss_b, \
                    self.gauss_c, self.dw)
                x_force_mag = gaussian_solid_force(x_d, self.gauss_a, \
                    self.gauss_b, self.gauss_c, self.dw)
            elif x_walltypes[x_idx] == 2:
                F_horizontal = gaussian_solid(x_d, self.gauss_a2, \
                    self.gauss_b2, self.gauss_c2, self.dw2)
                x_force_mag = gaussian_solid_force(x_d, self.gauss_a2, \
                    self.gauss_b2, self.gauss_c2, self.dw2)
            x_forces_x = x_f_vector[0]*x_force_mag  # x component
            x_forces_y = x_f_vector[1]*x_force_mag  # y_component

        if not y_distances:  # Second calculate force by closest vertical wall
            y_forces_x = 0.
            y_forces_y = 0.
            F_vertical = 0.
        else:
            y_idx = y_distances.index(min(y_distances))
            y_d = y_distances[y_idx]
            y_f_vector = y_fvecs[y_idx]
            if y_walltypes[y_idx] == 1:
                F_vertical = gaussian_solid(y_d, self.gauss_a, self.gauss_b, \
                    self.gauss_c, self.dw)
                y_force_mag = gaussian_solid_force(y_d, self.gauss_a, \
                    self.gauss_b, self.gauss_c, self.dw)
            elif y_walltypes[y_idx] == 2:
                F_vertical = gaussian_solid(y_d, self.gauss_a2, self.gauss_b2, \
                    self.gauss_c2, self.dw2)
                y_force_mag = gaussian_solid_force(y_d, self.gauss_a2, \
                    self.gauss_b2, self.gauss_c2, self.dw2)
            y_forces_x = y_f_vector[0]*y_force_mag  # x component
            y_forces_y = y_f_vector[1]*y_force_mag  # y_component
        
        # GLOBAL forces (maze walls outside detection window)
        F_glob = self.global_slope[0]*x + \
            self.global_slope[1]*((y-self.slope_exit)/(1-self.slope_exit))
        force_glob = -1.*(self.global_slope/(1-self.slope_exit))
        if y < self.slope_exit:
            F_glob = 0
            force_glob[0] = 0
            force_glob[1] = 0

        # Sum up all forces and potentials
        pot = F_horizontal + F_vertical + F_glob
 
        forces[0] = x_forces_x + y_forces_x + force_glob[0]
        forces[1] = x_forces_y + y_forces_y + force_glob[1]
    
        # We don't care about the virial here
        # virial = np.zeros((self.dim, self.dim))

        return pot, forces
    
    def plot_pot(self, intfs):
        # plt.figure()
        fig1, ax1 = plt.subplots()
        x_y = np.zeros([500,1000])
        i = 0
        for y in np.linspace(0.1, 0.93, 500):
            x_y[i] = np.array([self.potential((np.array([y, xx]), np.array([0,0]))) for xx in np.linspace(0.1, 0.93, 1000)])
            # plt.imshow([np.linspace(0.1, 0.93, 1000), x_y + y], color=(min(abs(y-0.1)/0.83,1),0,min(abs(1-(y-0.1)/0.83), 1)))
            i+=1
        c1 = ax1.pcolorfast((0.1,0.93),(0.1,0.93), x_y)
        for intf in intfs:
            ax1.axvline(intf, ymin=0.1, ymax=0.93, color='red')
        fig1.colorbar(c1)
        plt.show()

        # plt.figure()
        fig2, ax2 = plt.subplots()
        i=0
        for y in np.linspace(0.1, 0.93, 500):
            x_y[i] = np.array([np.average(self.force((np.array([y, xx]), np.array([0,0])))) for xx in np.linspace(0.1, 0.93, 1000)])
            # plt.plot(np.linspace(0.1, 0.93, 1000), x_y + y, color=(min(abs(y-0.1)/0.83,1),0,min(abs(1-(y-0.1)/0.83), 1)))
            i+=1
        c2 = ax2.pcolorfast((0.1,0.93),(0.1,0.93),  x_y)
        for intf in intfs:
            ax2.axvline(intf, ymin=0.1, ymax=0.93, color='red')
        fig2.colorbar(c2)
        plt.show()

###################################   
# Calculate wall distance vectors #
###################################
def point_to_lineseg_dist_with_normed_force_vector(r,l):
    """
    We heavily abuse the fact that the line is either in the x or y direction,
    as we assume that either dx or dy is equivalent for the two p-to-endpoint 
    dists. Here, also the normalized force-vector is calculated. 
    For distance-based functions, this is just the vector connecting the 
    wallpoint that is closest to the walker-point.
    
    r: np.arr([rx,ry])
    l: [np.arr([p1x,p1y]),np.arr([p2x,p2y])]

    There are only two cases:

             p1-----p2              p1-----p2
                |                  /
                |         or      /  
                r                r 
    
    returns:
        d: distance
        f: normalized force vector
    """
    d1 = r - l[0]
    d2 = r - l[1]
    
    if d1[0] == d2[0]: # x-distance 
        if np.sign(d1[1]) != np.sign(d2[1]): # point is 'in between'
            return abs(d1[0]), np.array([np.sign(-d1[0]),0])
        else:
            if abs(d1[1]) <= abs(d2[1]):
                return ((d1[0])**2 + (d1[1])**2)**(.5), \
                    (l[0]-r)/np.linalg.norm(l[0]-r)
            else:
                return ((d1[0])**2 + (d2[1])**2)**(.5), \
                    (l[1]-r)/np.linalg.norm(l[1]-r)
    else: 
        assert d1[1] == d2[1] # y-distance
        if np.sign(d1[0]) != np.sign(d2[0]): # point is 'in between'
            return abs(d1[1]), np.array([0,np.sign(-d1[1])])
        else:
            if abs(d1[0]) <= abs(d2[0]):
                return ((d1[1])**2 + (d1[0])**2)**(.5), \
                    (l[0]-r)/np.linalg.norm(l[0]-r)
            else:
                return ((d1[1])**2 + (d2[0])**2)**(.5), \
                    (l[1]-r)/np.linalg.norm(l[1]-r)    

def point_to_lineseg_dist(r,l):
    """
    We heavily abuse the fact that the line is either in the x or y direction,
    as we assume that either dx or dy is equivalent for the two p-to-endpoint 
    dists.

    r: np.arr([rx,ry])
    l: [np.arr([p1x,p1y]),np.arr([p2x,p2y])]

    There are only two cases:

             p1-----p2              p1-----p2
                |                  /
                |         or      /  
                r                r 

    returns:
        d: distance
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


def evolve_colormaze_easy(prim):
    """
    As the first/last columns and rows must be zeros (detection window 
    requirements), no periodic segments should be detected.

    This function searches for -- and | types of wall segments for each pixel of
    the maze. It saves these segments in a list for each pixel, and assigns a 
    similarly sized list of directions to each pixel (0 for --, 1 for |).
    examples for pixel .:
        1) --.-- : gets a list of 2 segments, and a list of 2 directions [0,0].
        2)  |._  : gets a list of 2 segments, and a list of 2 directions [1,0].
    We also add a list that keeps track of which walltype the segment is 
    associated with.
    """
    N,M = np.shape(prim)
    edges = np.zeros_like(prim,dtype='object')
    direcs = np.zeros_like(prim,dtype='object')
    walltypes = np.zeros_like(prim,dtype='object')
    # Perhaps we can do it faster, 
    # but I just do a (N-1)*(M-1)*4 complexity search...
    for i in range(N-1):
        for j in range(M-1):
            edges_on_this_point = []
            direcs_on_this_point = []
            walltypes_on_this_point = []
            if prim[i,j] != 0 and prim[i,j+1] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+.5,j+1.])])
                direcs_on_this_point.append(1)
                if prim [i,j] == 2 or prim[i,j+1] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            if prim[i,j] != 0 and prim[i,j-1] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+.5,j+0.])])
                direcs_on_this_point.append(1)
                if prim [i,j] == 2 or prim[i,j-1] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            if prim[i,j] != 0 and prim[i+1,j] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+1.,j+.5])])
                direcs_on_this_point.append(0)
                if prim [i,j] == 2 or prim[i+1,j] == 2:
                    walltypes_on_this_point.append(2)
                else:
                    walltypes_on_this_point.append(1)
            if prim[i,j] != 0 and prim[i-1,j] != 0:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+0.,j+.5])])
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
    """ 
    See evolve_colormaze_easy for details. Same function, but without
    different walltypes...
    """
    N,M = np.shape(prim)
    edges = np.zeros_like(prim,dtype='object')
    direcs = np.zeros_like(prim,dtype='object')
    # Perhaps we can do it faster, 
    # but I just do a (N-1)*(M-1)*4 complexity search...
    for i in range(N-1):
        for j in range(M-1):
            edges_on_this_point = []
            direcs_on_this_point = []
            if prim[i,j] == 1 and prim[i,j+1] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+.5,j+1.])])
                direcs_on_this_point.append(1)
            if prim[i,j] == 1 and prim[i,j-1] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+.5,j+0.])])
                direcs_on_this_point.append(1)
            if prim[i,j] == 1 and prim[i+1,j] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+1.,j+.5])])
                direcs_on_this_point.append(0)
            if prim[i,j] == 1 and prim[i-1,j] == 1:
                edges_on_this_point.append([np.array([i+.5,j+.5]), \
                                            np.array([i+0.,j+.5])])
                direcs_on_this_point.append(0)
            edges[i,j] = edges_on_this_point
            direcs[i,j] = direcs_on_this_point
    return edges, direcs

# OLDER maze segmentation and evolution functions. Not used anymore. 
# Left here for reference.
def segment_maze(maze):
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
    N,M = np.shape(maze)
    line_arr = np.zeros_like(maze,dtype='object')
    line_dir = np.zeros_like(maze,dtype='object')

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
    """
    Make sure the first and last columns and rows are zeros! If not, those 
    maze-walls will NOT be implemented. Should be okay already due to Dwindow
    requirements.
    This function searches for _, |, |_, _|, |- and -| types of wall segments,
    and allocates a blocktype number to them, which is then used in the
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
    return ((pic[:,:,0] == 0) & \
            (pic[:,:,1] == 0) & \
            (pic[:,:,2] == 0)).astype(int)

# Colormazes
def colormazefig_to_maze(mazefig):
    """
    Read an image, convert it to a numpy array with dim (x,y,3[4]). 
    Then convert it to an array with dim (x,y) with 0's , 1's and 2's, where
        0 is for a white pixel (free space),
        1 is for a black pixel (hard wall),
        2 is for a red pixel (soft wall).
    """
    mazearr = read_maze_fig(mazefig)
    return rgb2color(mazearr)

def rgb2color(mazearr):
    """
    Convert an image (jpg or png) to a numpy array (dim x,y,3[4])
    accepts red, black and white pixels.
    """
    N,M,_ = np.shape(mazearr)
    maze = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if mazearr[i,j,0] == 0 and \
                mazearr[i,j,1] == 0 and \
                mazearr[i,j,2] == 0:
                maze[i,j] = 1
            elif mazearr[i,j,0] == 255 and \
                mazearr[i,j,1] == 0 and \
                mazearr[i,j,2] == 0:
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
    return ((pic[:,:,0] == 255) & (pic[:,:,1] == 0) & \
        (pic[:,:,2] == 0)).astype(int)

def rgb2g(pic):
    """
    pic (x,y,3) or (x,y,4) numpy array
    """
    return ((pic[:,:,0] == 0) & (pic[:,:,1] == 255) & \
        (pic[:,:,2] == 0)).astype(int)

def rgb2b(pic):
    """
    pic (x,y,3) or (x,y,4) numpy array
    """
    return ((pic[:,:,0] == 0) & (pic[:,:,1] == 0) & \
        (pic[:,:,2] == 255)).astype(int)

##############
# Potentials #
##############
# The ones below are used in the paper
def gaussian_solid(d,a,b,c,dw):
    from math import erf
    # if d > dw:
    #     return a*np.exp((-((d-dw) - b)**2)/(2*c**2))
    # else: 
    #     return a
    if d > dw:
        return a*np.exp((-(d - b)**2)/(2*c**2)) - \
            a*dw/c*np.sqrt(np.pi/2)*erf((b-d)/np.sqrt(2)/c) - \
            a*dw*np.sqrt(np.pi/2)
    else:
        return a*np.exp((-(dw - b)**2)/(2*c**2)) - \
            a*dw/c*np.sqrt(np.pi/2)*erf((b-dw)/np.sqrt(2)/c) - \
            a*dw*np.sqrt(np.pi/2)

def gaussian_solid_force(d,a,b,c,dw):
    if d > dw:
        return -1*a/c**2*((d-dw) - b)*np.exp((-(d - b)**2)/(2*c**2))
    else:
        return 0

# The ones below are not used in the paper
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