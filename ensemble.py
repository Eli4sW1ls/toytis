import logging
import numpy as np
from path import Path
from filehandler import make_ens_dirs_and_files
from engine import LangevinEngine, ndLangevinEngine
from order import OrderParameter, OrderX
import pickle as pkl
from funcs import remove_lines_from_file, validate_staple
from moves import kick_retis
from pgmoves import kick_star

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Format string for writing path data to pathensemble.txt
PATH_FMT = (
    '{0:>10d} {1:>10d} {2:>10d} {3:1s} {4:1s} {5:1s} {6:>7d} '
    '{7:3s} {8:2s} {9:>16.9e} {10:>16.9e} {11:>7d} {12:>7d} '
    '{13:>16.9e} {14:>7d} {15:7d} {16:>16.9e}'
)
# Format string for writing order parameter data
ORDER_FMT = ('{:>10d}', '{:>12.6f}')

class Ensemble:
    """ 
    Representation of an ensemble in path sampling simulations.
    
    This class manages the generation, validation, and storage of paths
    within an ensemble defined by its interfaces and boundary conditions.
    It handles different types of ensembles used in RETIS, PPTIS, and i*
    simulations.

    Attributes
    ----------
    id : int
        Unique identifier of the ensemble
    intfs : dict of floats
        Dictionary of the L M R interfaces of the ensemble
    paths : list of :py:class:`Path` objects
        List of previously accepted paths in the ensemble
    data : list
        List containing data on the paths generated for each cycle in the
        ensemble
    max_len : int
        Maximum length of a path in the ensemble
    settings : dict
        Dictionary containing the settings of the ensemble
    start_conditions : set of str
        Set containing the start conditions of the ensemble
    end_conditions : set of str
        Set containing the end conditions of the ensemble
    cross_conditions : set of str
        Set containing the cross conditions of the ensemble
    ens_type : str
        String indicating the type of ensemble
    engine : :py:class:`LangevinEngine` object
        Engine of the ensemble
    paths : list of :py:class:`Path` objects
        List of paths in the ensemble
    max_paths : int
        Maximum number of paths that are kept in memory for the ensemble
    cycle : int
        Cycle number of the ensemble, i.e. the number of paths that have been
        generated for the ensemble (including rejected paths).
    cycle_acc : int
        How many cycles resulted in an accepted path?
    cycle_md : int
        How many cycles included the MD engine to generate the accepted paths?
    last_path : :py:class:`Path` object
        Last path that was **accepted** for the ensemble
    name : str
        Name of the ensemble
    orderparameter : :py:class:`OrderParameter` object
        Order parameter of the ensemble
    extremal_conditions : set of str
        Set containing the extremal conditions of the ensemble. This is the
        union of the start and end conditions.
    simtype : str
        Type of simulation to perform. This is either "repptis", "retis" or "i*"
    prime_both_starts : bool
        Whether to prime both starts in ensembles where this is applicable
    high_friction : bool
        Whether to use high friction in the Langevin dynamics
    v_ord : bool
        Whether to include velocity in the order parameter output
    illegal_pathtypes : dict
        Dictionary containing the illegal path types for this ensemble
    """

    def __init__(self, settings):
        """
        Initialize the Ensemble object.

        Parameters
        ----------
        settings : dict
            Dictionary containing the settings of the simulation, which should include:
            - id: Ensemble identifier
            - intfs: Dictionary of interfaces (L, M, R)
            - max_len: Maximum path length
            - ens_type: Type of ensemble
            - max_paths: Maximum number of paths to keep in memory
            - simtype: Simulation type ("repptis", "retis", "i*")
            - dim: Dimensionality of the system
            Optional settings:
            - name: Name of the ensemble (defaults to id)
            - prime_both_starts: Whether to prime both starts (defaults to False)
            - high_friction: Whether to use high friction (defaults to False)
            - v_ord: Whether to include velocity in order output (defaults to False)
        """

        self.id = settings["id"]
        # Create ensemble directories and files
        make_ens_dirs_and_files(self.id)
        
        self.settings = settings
        self.intfs = self.settings["intfs"]
        self.max_len = self.settings["max_len"]
        self.ens_type = self.settings["ens_type"]
        self.paths = []
        self.data = []
        self.max_paths = self.settings["max_paths"]
        self.name = settings.get("name", str(self.id))  # Default to id if name not provided
        self.cycle = 0
        self.cycle_acc = 0
        self.cycle_md = 0
        self.simtype = settings["simtype"]
        self.prime_both_starts = settings.get("prime_both_starts", False)
        self.high_friction = settings.get("high_friction", False)
        self.v_ord = settings.get("v_ord", False)
        self.last_path = None  # Will be set when first path is accepted

        # Dictionary to store illegal path types for this ensemble
        self.illegal_pathtypes = {}

        # Set the start, end and cross conditions of the ensemble
        self.set_conditions()
        # extremal conditions is the union of start and end conditions
        self.extremal_conditions = self.start_conditions.union(
            self.end_conditions)

        # Set the engine of the ensemble
        self.set_engine()

        # Set the order parameter of the ensemble
        self.set_order_parameter()

    def set_engine(self):
        """ 
        Sets the appropriate engine for the ensemble based on dimensionality.
        
        For multi-dimensional systems, uses ndLangevinEngine;
        for one-dimensional systems, uses LangevinEngine.
        """
        if self.settings["dim"] > 1:
            self.engine = ndLangevinEngine(self.settings)
        else:
            self.engine = LangevinEngine(self.settings)

    def set_order_parameter(self):
        """
        Sets the appropriate order parameter for the ensemble based on dimensionality.
        
        For multi-dimensional systems, uses OrderX;
        for one-dimensional systems, uses OrderParameter.
        """
        if self.settings["dim"] > 1:
            self.orderparameter = OrderX(self.settings)
        else:
            self.orderparameter = OrderParameter(self.settings)

    def update_data(self, status, trial, gen, simcycle, update_paths=True):
        """
        Updates the data of the path ensemble after a move has been performed.
        
        If the path is accepted, the last_path and paths attributes are updated.
        Records path information to the ensemble's data files.

        Parameters
        ----------
        status : str
            Status of the move: "ACC" for accepted or any "REJ*" flag for rejected
        trial : :py:class:`Path` object or tuple
            Trial path, possibly with additional metadata
        gen : str
            Generation method: swap (s-, s+), shoot (sh), null (00), or load (ld)
        simcycle : int
            Cycle number of the simulation
        update_paths : bool, optional
            Whether to update the last_path and paths attributes. Default is True.
            Some moves handle path management themselves.
        """
        dir=0
        # Extract path type if provided as tuple (path, type)
        if isinstance(trial, tuple):
            ptype = trial[1]
            trial = trial[0]
        else:
            ptype = self.get_ptype(trial)
            
        # Find minimum and maximum order parameter values and their indices
        ordermin = (min([op[0] for op in trial.orders]), np.argmin([op[0] for op in trial.orders]))
        ordermax = (max([op[0] for op in trial.orders]), np.argmax([op[0] for op in trial.orders]))
        plen = len(trial.phasepoints)
        
        # Update cycle counters
        self.cycle += 1
        if status == "ACC":
            self.cycle_acc += 1
            if update_paths:  # Some moves do path management themselves
                self.last_path = trial
                # Insert the path at the beginning of the list
                self.paths.insert(0, trial)
                if len(self.paths) > self.max_paths:
                    # Remove the last path from the list to maintain max_paths limit
                    self.paths.pop()
                    
            # Update MD cycle counter based on simulation type and generation method
            if self.simtype == "retis":
                if gen == "sh":
                    self.cycle_md += 1
            elif self.simtype == "i*":
                if gen == "sh":
                    self.cycle_md += 1
    
                # Determine path direction
                if ptype == "LMR":
                    dir = 1  # Forward direction
                elif ptype == "RML":
                    dir = -1  # Backward direction
                elif status == "ACC" and self.id > 1:
                    # For accepted paths in non-first ensembles, determine direction from staple validation
                    _, start_ext, end_ext = validate_staple(self, trial)
                    if start_ext < end_ext:
                        dir = 1
                    else:
                        dir = -1
                    
                # Update path metadata with direction information
                if trial.meta is not None:
                    trial.meta[1] = dir
            elif self.simtype == "repptis":
                if gen in ["s-", "s+", "sh"]:
                    self.cycle_md += 1
        else:  # Rejected path
            if update_paths:
                # Insert copy of current path at beginning
                self.paths.insert(0, self.paths[0])  # TODO: this was copy_path
                if len(self.paths) > self.max_paths:
                    self.paths.pop()

        # Initialize star index if not set
        if trial.staridx is None:
            trial.staridx = (0, 0)
        
        # Write path data to ensemble file
        self.write_to_pe_file(simcycle, self.cycle_acc, self.cycle_md, ptype,
                              plen, status, gen, ordermin, ordermax, dir, trial.staridx)
                              
        # Write to order parameter file
        # To reduce file size, only write orders of accepted paths
        if status == "ACC":
            self.write_to_order_file(trial, simcycle, ptype, plen, status, gen, dir, trial.staridx)
        else:
            # For rejected paths, write minimal data with zero order parameters
            dummy_path = Path(
                orders=[[0 for __ in range(len(trial.orders[0]))] for _ in range(1)], 
                ens_id=trial.ens_id, 
                phasepoints=[tuple([0 for __ in range(2*len(trial.orders[0]))]) for _ in range(1)]
            )
            self.write_to_order_file(dummy_path, simcycle, ptype, plen, status, gen, dir, trial.staridx)


    def jump_back(self, n=1):
        """
        Jump back n cycles in the ensemble.
        
        This removes the n most recent paths from the ensemble and corresponding
        lines from the pathensemble.txt file, effectively undoing the last n cycles.
        
        Parameters
        ----------
        n : int, optional
            Number of cycles to jump back. Default is 1.
        """
        # Remove the last n paths from the ensemble
        for _ in range(n):
            if self.paths:
                self.paths.pop(0)
                
        # Remove the last n lines from the pathensemble.txt file
        remove_lines_from_file(str(self.id) + "/pathensemble.txt", n=n)
        
        # Update the cycle counter but preserve acc_cycle and md_cycle
        self.cycle -= n

    def write_to_pe_file(self, simcycle, cycle_acc, cycle_md, ptype, plen,
                         status, gen, ordermin, ordermax, dir, staridx):
        """
        Write path data to the pathensemble.txt file.
        
        The file format includes columns for simulation cycle, accepted cycle count,
        MD cycle count, path type, path length, status, generation method,
        minimum and maximum order parameters, and star indices.

        Parameters
        ----------
        simcycle : int
            Simulation cycle number
        cycle_acc : int
            Number of accepted cycles so far
        cycle_md : int
            Number of MD cycles so far
        ptype : str
            Path type (e.g., "LMR", "RML")
        plen : int
            Path length (number of phase points)
        status : str
            Status of the move ("ACC" or "REJ*")
        gen : str
            Generation method (e.g., "sh", "s+", "s-", "00")
        ordermin : tuple
            (minimum order parameter value, index in path)
        ordermax : tuple
            (maximum order parameter value, index in path)
        dir : int
            Path direction (1=forward, -1=backward, 0=unknown)
        staridx : tuple
            Star indices (start, end) for i* simulations
        """
        with open(str(self.id).zfill(3) + "/pathensemble.txt", "a") as f:
            f.write(PATH_FMT.format(
                simcycle, cycle_acc, cycle_md, ptype[0], ptype[1], ptype[2],
                plen, status, gen,
                ordermin[0], ordermax[0], ordermin[1], ordermax[1], staridx[0], staridx[1], dir, 1.) + "\n")
    
    def write_to_order_file(self, path, simcycle, ptype, plen, status, gen, dir=0, staridx=(0,0)):
        """
        Write order parameter data to the order.txt file.
        
        Records detailed information about order parameters along the path,
        as well as velocities if v_ord is True.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path whose order parameters will be written
        simcycle : int
            Simulation cycle number
        ptype : str
            Path type (e.g., "LMR", "RML")
        plen : int
            Path length (number of phase points)
        status : str
            Status of the move ("ACC" or "REJ*")
        gen : str
            Generation method (e.g., "sh", "s+", "s-", "00")
        dir : int
            Path direction (1=forward, -1=backward, 0=unknown)
        staridx : tuple
            Star indices (start, end) for i* simulations
        """
        dim = len(path.orders[0])
        with open(str(self.id).zfill(3) + "/order.txt", "a") as f:
            # Write header with path information
            f.write(f"# Cycle: {simcycle}, status: {status}, move: {gen}, path length: {plen}, "
                    f"path type: {ptype}, direction: {'fw' if dir==1 else 'bw'}, staridx: {staridx}\n")
            
            # Write column headers for order parameters
            f.write("#     Time" + "        Orderp")
            for j in range(1, dim):
                f.write(f"        Orderp{j}")
            f.write("\n")
            
            # Write order parameter values for each point in the path
            for i, ord in enumerate(path.orders):
                # Write time index and first order parameter
                f.write((ORDER_FMT[0] + "  " + ORDER_FMT[1]).format(i, ord[0]))
                
                # Write remaining order parameters
                for j in range(1, dim):
                    f.write(("  " + ORDER_FMT[1]).format(ord[j]))
                if self.v_ord:
                    for j in range(dim):
                        if dim > 1:
                            f.write(("  " + ORDER_FMT[1]).format(path.phasepoints[i][1][j]))
                        else:
                            f.write(("  " + ORDER_FMT[1]).format(path.phasepoints[i][1]))
                            
                f.write("\n")


    def set_conditions(self):
        """ 
        Determines the start, end and cross conditions of the ensemble based on its type.
        
        Sets attributes:
        - start_conditions: where paths must begin
        - end_conditions: where paths must end
        - cross_conditions: which interfaces paths must cross
        - illegal_pathtypes: paths that are not allowed
        
        Ensemble types include:
        - body_TIS: Regular [i^+] ensemble in (PP)TIS simulations
        - state_A: Regular [0^-] ensemble in (PP)TIS simulations
        - state_A_lambda_min_one: [0^-'] ensemble with lambda_{-1}
        - body_PPTIS: Regular [i^{+-}] ensemble in PPTIS simulations
        - PPTIS_0plusmin_primed: [0^{+-}'] ensemble in PPTIS simulations
        - RETIS_0plus: [0^+] ensemble in RETIS simulations
        - state_B: [N^-] ensemble in snakeTIS simulations
        - state_B_lambda_plus_one: [N^-'] ensemble with lambda_{N+1}
        - body_i*: Regular ensemble in i* simulations
        - i*_0star: [0^*] ensemble in i* simulations
        """
        self.illegal_pathtypes = {}

        if self.ens_type == "body_TIS":
            # Regular TIS ensemble: paths start at L interface and end at either L or R
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "state_A":
            # State A ensemble: paths start and end in state A (right of R interface)
            self.start_conditions = {"R"}
            self.end_conditions = {"R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_A_lambda_min_one":
            # State A with lambda_{-1}: paths can start and end at either L or R
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "body_PPTIS":
            # PPTIS body ensemble: paths can start and end at either L or R
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}

        elif self.ens_type == "PPTIS_0plusmin_primed":
            # PPTIS 0^{+-}' ensemble
            if self.prime_both_starts:
                self.start_conditions = {"L", "R"}
            else:
                self.start_conditions = {"L"}  # no R because repptis_swap!!
            self.illegal_pathtypes = {"RMR"}  # Paths starting at R and returning to R are illegal
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "RETIS_0plus":
            # RETIS 0^+ ensemble: paths start at L and end at either L or R
            self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_B":
            # State B ensemble: paths start and end in state B (left of L interface)
            self.start_conditions = {"L"}
            self.end_conditions = {"L"}
            self.cross_conditions = {}

        elif self.ens_type == "PPTIS_Nplusmin_primed":
            # PPTIS N^{+-}' ensemble
            if self.prime_both_starts:
                self.start_conditions = {"L", "R"}
            else:
                self.start_conditions = {"R"}
            self.illegal_pathtypes = {"LML"}  # Paths starting at L and returning to L are illegal
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        elif self.ens_type == "state_B_lambda_plus_one":
            # State B with lambda_{N+1}: paths can start and end at either L or R
            self.start_conditions = {"L", "R"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}
        
        elif self.ens_type == "body_i*":
            # i* body ensemble: start conditions depend on ensemble position
            if self.id < len(self.intfs["all"])-1 or self.prime_both_starts:
                self.start_conditions = {"L", "R"}
            else:
                self.start_conditions = {"L"}
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {"M"}
        
        elif self.ens_type == "i*_0star":
            # i* 0^* ensemble
            if self.prime_both_starts:
                self.start_conditions = {"L", "R"}
            else:
                self.start_conditions = {"L"}  # no R because repptis_swap!!
            self.illegal_pathtypes = {"RMR"}  # Paths starting at R and returning to R are illegal
            self.end_conditions = {"L", "R"}
            self.cross_conditions = {}

        else:
            raise ValueError("Unknown ensemble type: {}".format(self.ens_type))

    def check_ph_in_ensemble(self, ph):
        """ 
        Checks whether a phasepoint could be valid for a path in the pathensemble.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to check, where x is position and v is velocity

        Returns
        -------
        bool
            True if the phasepoint could be valid for a path in this ensemble
        """
        x = ph[0]
        if self.ens_type in \
            ["body_TIS", "body_PPTIS", "state_A_lambda_min_one",
             "state_B_lambda_plus_one", "PPTIS_0plusmin_primed"]:
            # For these ensemble types, x must be between L and R interfaces
            return x >= self.intfs["L"] and x <= self.intfs["R"]

        elif self.ens_type in ["state_A"]:
            # For state A, x must be to the left of L interface
            return x <= self.intfs["L"]

        elif self.ens_type in ["state_B"]:
            # For state B, x must be to the right of R interface
            return x >= self.intfs["R"]

        else:
            raise ValueError("Unknown ensemble type: {}".format(self.ens_type))
        
    def check_position(self, ph, L, R):
        """ 
        Determines the position region of a phasepoint relative to boundaries.

        Parameters
        ----------
        ph : tuple (x, v) of floats
            Phasepoint to check
        L : float
            Left boundary of the interval
        R : float
            Right boundary of the interval

        Returns
        -------
        str
            Position region: "M" if between boundaries, "L" if left of L, "R" if right of R
        """
        # Calculate order parameter for the phasepoint
        op = self.orderparameter.calculate(ph)[0]
        
        # Return position region based on order parameter value
        if L <= op <= R:
            return "M"  # Middle (between boundaries)
        elif op < L:
            return "L"  # Left of L boundary
        else:
            return "R"  # Right of R boundary

    def check_path(self, path):
        """ 
        Checks whether a path is valid for the ensemble by verifying all conditions.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets all validation criteria for the ensemble
        """
        # Path must satisfy start conditions, end conditions, and cross conditions
        return (self.check_start(path) and 
                self.check_end(path) and 
                self.check_cross(path))

    def check_cross(self, path):
        """ 
        Checks whether a path meets the crossing conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets all crossing conditions
        """
        # If there are no cross conditions, the check passes automatically
        if not self.cross_conditions:
            return True
            
        # Find minimum and maximum order parameter values
        ordermax = max([op[0] for op in path.orders])
        ordermin = min([op[0] for op in path.orders])
        
        # Check if path crosses all required interfaces
        crossed = True
        for cross_condition in self.cross_conditions:
            # A successful crossing means the minimum value is below the interface
            # and the maximum value is above the interface
            crossed = crossed and ordermin < self.intfs[cross_condition] <= ordermax
            
        return crossed

    def check_start_and_end(self, path):
        """ 
        Combined check of start and end conditions.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets both start and end conditions
        """
        return self.check_start(path) and self.check_end(path)

    def check_start(self, path):
        """ 
        Checks whether a path meets the start conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the start conditions
        """
        # If there are no start conditions, the check passes automatically
        if not self.start_conditions:
            return True
            
        # Get order parameter value at the start of the path
        op = path.orders[0][0]
        
        # Check if the start position satisfies any of the start conditions
        start = False
        for start_condition in self.start_conditions:
            if start_condition == "R":
                # For R condition, order parameter must be >= interface value
                start = start or op >= self.intfs[start_condition]
            elif start_condition == "L":
                # For L condition, order parameter must be <= interface value
                start = start or op <= self.intfs[start_condition]
            else:
                raise ValueError(f"Unknown start condition: {start_condition}")
                
        return start

    def check_end(self, path):
        """ 
        Checks whether a path meets the end conditions of the ensemble.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        bool
            True if the path meets the end conditions
        """
        # If there are no end conditions, the check passes automatically
        if not self.end_conditions:
            return True
            
        # Get order parameter value at the end of the path
        op = path.orders[-1][0]
        
        # Check if the end position satisfies any of the end conditions
        end = False
        for end_condition in self.end_conditions:
            if end_condition == "R":
                # For R condition, order parameter must be >= interface value
                end = end or op >= self.intfs[end_condition]
            elif end_condition == "L":
                # For L condition, order parameter must be <= interface value
                end = end or op <= self.intfs[end_condition]
            else:
                raise ValueError(f"Unknown end condition: {end_condition}")
                
        return end

    def check_start_end_positions(self, path):
        """ Returns the start and end positions of a path.
        These are either "L", "R", or "M" for left, right, or middle,
        respectively. R denotes right of the right interface, L denotes left of
        the left interface, and M denotes between the left and right interface.

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        tuple of strings
            (startpos, endpos) where each is "L", "R", or "*" (unknown)
        """
        # Get order parameter values at start and end
        start_op = path.orders[0][0]
        end_op = path.orders[-1][0]
        
        # Determine position regions
        startpos = "L" if start_op <= self.intfs["L"] else \
                   "R" if start_op >= self.intfs["R"] else "*"
        endpos = "L" if end_op <= self.intfs["L"] else \
                 "R" if end_op >= self.intfs["R"] else "*"
                 
        return startpos, endpos

    def get_ptype(self, path):
        """ Returns the type of the path. This is a three letter combination of
        L, M, R and * (for left, middle, right and unknown, respectively)

        Parameters
        ----------
        path : :py:class:`Path` object
            Path to check

        Returns
        -------
        str
            Three-letter code representing path type (e.g., "LMR", "RML")
        """
        # Get start and end positions
        startpos, endpos = self.check_start_end_positions(path)
        
        # Determine if path crosses an interface in the middle
        middlepos = "M" if self.check_cross(path) else "*"
        
        # Return combined path type
        return startpos + middlepos + endpos

    def create_initial_path(self, N=31):
        """
        Create an initial path for the ensemble.
        
        Constructs a path that satisfies the ensemble's conditions by creating
        appropriate start, middle, and end points based on ensemble type.

        Parameters
        ----------
        N : int, optional
            Half the number of phasepoints in the initial path. Default is 31.
        """
        logger.info(f"Creating initial path for ensemble {self.name}")
        
        # For certain ensemble types, use specialized path creation functions
        if self.ens_type == "body_TIS":
            # For TIS body ensembles, can scale path length based on ensemble ID
            # kick_retis(self)
            # return
            N += int(N * np.random.rand() * (self.id-1))
            start = self.intfs["L"] * (1 - np.sign(self.intfs["L"]) * 0.001)  # Just inside L
            mid = self.intfs["M"] * (1 + np.sign(self.intfs["M"]) * 0.001)    # Just past M
            stop = self.intfs["L"] * (1 - np.sign(self.intfs["L"]) * 0.001)   # Back to L
        elif self.ens_type in ["state_A", "state_A_lambda_min_one", "body_PPTIS", 
                               "PPTIS_0plusmin_primed", "RETIS_0plus"]:
            # Use specialized kick move for these ensemble types
            kick_retis(self)
            return
            N += int(N*np.random.rand()*(self.id-1))
            start = self.intfs["R"]+0.000001
            mid = self.intfs["R"]*(1 - np.sign(self.intfs["R"])*0.15)
            stop = self.intfs["R"]+0.000001
        elif self.ens_type == "state_A_lambda_min_one":
            kick_retis(self)
            return
            # For state A ensembles, we start at the right interface
            start = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
        elif self.ens_type == "body_PPTIS":
            # For body ensembles, we start at the left interface
            kick_retis(self)
            return
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = self.intfs["M"]
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "PPTIS_0plusmin_primed":
            # For body ensembles, we start at the left interface
            kick_retis(self)
            return
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = (self.intfs["R"] + self.intfs["L"]) / 2 
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "RETIS_0plus":
            # For body ensembles, we start at the left interface
            kick_retis(self)
            return
            N += int(N*np.random.rand()*(self.id-1))
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = (self.intfs["L"] + self.intfs["R"]) / 2 - np.random.rand()/6
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
        elif self.ens_type == "PPTIS_Nplusmin_primed":
            # PPTIS N+- primed ensemble
            start = self.intfs["R"] * (1 + np.sign(self.intfs["R"]) * 0.001)  # Just outside R
            mid = (self.intfs["R"] + self.intfs["L"]) / 2                     # Halfway between
            stop = self.intfs["L"] * (1 - np.sign(self.intfs["L"]) * 0.001)   # Just inside L
        elif self.ens_type in ["body_i*", "i*_0star"]:
            # Use specialized star kick move for i* ensembles
            kick_star(self)
            return
            rand_stop = np.random.randint(self.id, len(self.intfs["all"]))
            rand_stop = len(self.intfs["all"])-1
            rand_start = np.random.randint(self.id-1)
            rand_start = 0
            start = self.intfs["all"][rand_start]- 0.0001
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["all"][rand_stop]+0.0001
        elif self.ens_type == "i*_0star":
            # For body ensembles, we start at the left interface
            kick_star(self)
            return
            rand_stop = np.random.randint(2, len(self.intfs["all"]))
            rand_stop = len(self.intfs["all"])-1
            start = self.intfs["L"]-0.0001
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["all"][rand_stop]+0.0001
        elif self.ens_type == "state_B":
            # For state B ensembles, we start at the left interface
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = self.intfs["L"]*(1 + np.sign(self.intfs["L"])*0.1)
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            N += int(N*np.random.rand()*(self.id-1))
            start = self.intfs["R"]+0.000001
            mid = self.intfs["R"]*(1 - np.sign(self.intfs["R"])*0.15)
            stop = self.intfs["R"]+0.000001
        elif self.ens_type == "state_A_lambda_min_one":
            kick_retis(self)
            return
            # For state A ensembles, we start at the right interface
            start = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
            mid = (self.intfs["R"] + self.intfs["L"]) / 2
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
        elif self.ens_type == "body_PPTIS":
            # For body ensembles, we start at the left interface
            kick_retis(self)
            return
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = self.intfs["M"]
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "PPTIS_0plusmin_primed":
            # For body ensembles, we start at the left interface
            kick_retis(self)
            return
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = (self.intfs["R"] + self.intfs["L"]) / 2 
            stop = self.intfs["R"]*(1 + np.sign(self.intfs["R"])*0.001)
        elif self.ens_type == "RETIS_0plus":
            # For body ensembles, we start at the left interface
            kick_retis(self)
            return
            N += int(N*np.random.rand()*(self.id-1))
            start = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
            mid = (self.intfs["L"] + self.intfs["R"]) / 2 - np.random.rand()/6
            stop = self.intfs["L"]*(1 - np.sign(self.intfs["L"])*0.001)
        else:
            raise ValueError(f"Unknown ensemble type: {self.ens_type}")
            
        # Create phasepoints depending on system dimensionality
        if self.settings["dim"] > 1:
            # maze_entry = 0.64
            # maze_entry = 0.353187488
            maze_entry = 0.15
            phasepoints1 = [(np.array([maze_entry]*(self.settings["dim"]-1) + [i]), np.zeros(self.settings["dim"])) for i in np.linspace(start, mid, N)]
            phasepoints2 = [(np.array([maze_entry]*(self.settings["dim"]-1) + [i]), np.zeros(self.settings["dim"])) for i in np.linspace(mid, stop, N)]
            if self.simtype == "i*":
                last_ph = (
                    np.array([maze_entry] * (self.settings["dim"]-1) + 
                             [self.intfs["all"][rand_stop-1]-0.001]), 
                    np.zeros(self.settings["dim"])
                )
                first_ph = (
                    np.array([maze_entry] * (self.settings["dim"]-1) + 
                             [self.intfs["all"][rand_start+1]+0.002 if self.ens_type != "i*_0star" 
                              else start-0.001]), 
                    np.zeros(self.settings["dim"])
                )
        else:
            # For 1D systems, simpler phase points
            phasepoints1 = [(i, 0.) for i in np.linspace(start, mid, N)]
            phasepoints2 = [(i, 0.) for i in np.linspace(mid, stop, N)]
            
            # For i* simulations, prepare special phase points
            if self.simtype == "i*":
                last_ph = (self.intfs["all"][rand_stop-1]-0.001, 0)
                first_ph = (self.intfs["all"][rand_start+1]+0.002, 0)

        # For i* ensembles, adjust paths to include turns
        # Count points below certain interfaces
        p2 = len([ph for ph in phasepoints2 if self.orderparameter.calculate(ph)[0] <= self.intfs["R"]])
        p1 = len([ph for ph in phasepoints1 if self.orderparameter.calculate(ph)[0] <= self.intfs["L"]])
        pp1 = N - p1
        
        # Modify paths for specific i* ensemble types
        if self.ens_type == "i*_0star":
            if rand_stop < len(self.intfs["all"])-1:
                # Add return path segment
                phasepoints2 += list(reversed([
                    ph for ph in phasepoints2 
                    if self.orderparameter.calculate(ph)[0] >= self.intfs["all"][rand_stop-1]
                ])) + [last_ph]
        elif self.ens_type == "body_i*":
            if rand_start > 0:
                # Adjust p1 count and add initial path segment
                p1 += len([
                    ph for ph in phasepoints1 
                    if self.orderparameter.calculate(ph)[0] <= self.intfs["all"][rand_start+1]
                ]) + 1
                phasepoints1 = [first_ph] + list(reversed([
                    ph for ph in phasepoints1 
                    if self.orderparameter.calculate(ph)[0] <= self.intfs["all"][rand_start+1]
                ])) + phasepoints1
            if rand_stop < len(self.intfs["all"])-1:
                # Add return path segment
                phasepoints2 += list(reversed([
                    ph for ph in phasepoints2 
                    if self.orderparameter.calculate(ph)[0] >= self.intfs["all"][rand_stop-1]
                ])) + [last_ph]
        
        # Calculate order parameters for all phasepoints
        orders1 = [self.orderparameter.calculate(ph) for ph in phasepoints1]
        orders2 = [self.orderparameter.calculate(ph) for ph in phasepoints2]
        
        # Combine phasepoints and orders (skipping duplicate at join point)
        phasepoints = phasepoints1 + phasepoints2[1:]
        orders = orders1 + orders2[1:]
        
        # Create the path object
        path = Path(phasepoints, orders, self.id, ["LMR", 0, "ACC", orders1[0]])

        # Set star indices based on ensemble type
        if self.ens_type == "i*_0star":
            path.staridx = (1, int(N)+p2-2)
        else:
            path.staridx = (int(p1), int(p1 + pp1 + p2-2))

        # Store the path and update ensemble data
        self.paths.append(path)
        self.last_path = path
        self.update_data("ACC", 
                         path if self.ens_type not in ["i*_0star", "body_i*"] else (path, "LMR"), 
                         "ld", 0)

    def plot_min_max_distributions(self, flag="ACC"):
        """
        Plot distributions of minimum and maximum order parameters for paths.
        
        Parameters
        ----------
        flag : str, optional
            Filter for paths: "ACC" for accepted paths, "REJ" for rejected paths.
            Default is "ACC".
        """
        import matplotlib.pyplot as plt
        
        # Load data from pathensemble.txt file
        if flag == "ACC":
            # Load status flag (column 7) and min/max order parameters (columns 9-10)
            data = np.loadtxt(str(self.id).zfill(3) + "/pathensemble.txt",
                              usecols=(7, 9, 10), dtype=str)
            # Filter for accepted paths only
            data = data[data[:, 0] == "ACC"]
            # Convert order parameter columns to float
            data = data[:, 1:].astype(float)
        elif flag == "REJ":
            # Same as above but for rejected paths
            data = np.loadtxt(str(self.id).zfill(3) + "/pathensemble.txt",
                              usecols=(7, 9, 10), dtype=str)
            data = data[data[:, 0] != "ACC"]
            data = data[:, 1:].astype(float)
        else:
            raise ValueError(f"Unknown flag: {flag}")
            
        # Create histogram plot
        fig, ax = plt.subplots()
        ax.hist(data[:, 0], bins=100, density=True, label="min")
        ax.hist(data[:, 1], bins=100, density=True, label="max")
        ax.legend()
        ax.set_title(f"{flag} ensemble {self.name}")
        fig.show()


    def write_restart_pickle(self):
        """
        Writes the ensemble state to a pickle file for later restart.
        """
        with open(str(self.id) + "/restart.pkl", "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load_restart_pickle(cls, id):
        """
        Loads an ensemble state from a pickle file.
        
        Parameters
        ----------
        id : int
            Ensemble identifier
            
        Returns
        -------
        Ensemble
            Loaded ensemble object
        """
        with open(str(id) + "/restart.pkl", "rb") as f:
            return pkl.load(f)