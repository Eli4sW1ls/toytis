import logging
import numpy as np
from ensemble import Ensemble
from moves import shooting_move, swap, swap_zero, repptis_swap
from snakemove import snake_move, forced_extension
from pgmoves import pg_shooting_move, pg_swap_zero, pg_swap, shooting_move_old
import pickle as pkl
from funcs import plot_paths
import matplotlib.pyplot as plt

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Simulation:
    """ Path sampling simulation using (RE)PPTIS or i* algorithms
    
    This class manages the simulation of rare events using path sampling techniques,
    specifically implementing RETIS (Replica Exchange Transition Interface Sampling),
    REPPTIS (Replica Exchange Partial Path TIS), and i* methods.
    
    The simulation consists of multiple ensembles, each with its own path collection,
    and different Monte Carlo moves to sample the path space efficiently.

    Attributes
    ----------
    settings : dict
        Dictionary of simulation settings including temperature, interfaces, etc.
    intfs : list of floats
        List of interfaces defining the boundaries between states
    cycle : int
        Current cycle number of the simulation
    max_cycles : int
        Maximum number of cycles to run
    ensembles : list of :py:class:`Ensemble` objects
        List of ensembles maintained during the simulation
    simtype : str
        Type of simulation to perform: "repptis", "retis" or "i*"
    method : str
        How the simulation is initialized: "restart" for loading previous state
        or any other value for creating new ensembles
    permeability: bool
        Whether to use a lambda_{-1} interface for the [0-] ensemble
    zero_left: float
        The left interface of the [0-] ensemble (only used for permeability)
    p_shoot: float
        Probability of performing shooting moves (vs. swap moves)
    include_stateB: bool
        Whether to include dedicated state B ensembles
    prime_both_starts: bool
        Whether to prime both start points in shooting moves
    high_friction: bool
        Whether to use high friction for the simulation
    v_ord: bool
        Whether to include velocity in the order.txt file for paths for additional analysis
    """
    def __init__(self, settings):
        """Initialize the Simulation object with the provided settings.

        Parameters
        ---------
        settings : dict
            Dictionary containing all simulation parameters and configuration
            Required keys: interfaces, simtype, method
            Optional keys: cycle, max_cycles, permeability, zero_left, p_shoot,
                          include_stateB, prime_both_starts, high_friction, v_ord
        """
        # Store the simulation settings
        self.settings = settings
        self.intfs = settings["interfaces"]
        
        # Initialize simulation state parameters
        self.cycle = settings.get("cycle", 0)  # Default to 0 if not provided
        self.max_cycles = settings.get("max_cycles", 1000000)
        
        # Set simulation type and method
        self.simtype = settings.get("simtype", "retis")
        self.method = settings["method"]
        
        # Additional configuration options
        self.permeability = settings.get("permeability", False)
        self.zero_left = settings.get("zero_left", None)
        self.p_shoot = settings.get("p_shoot", 0.9)
        self.include_stateB = settings.get("include_stateB", False)
        self.prime_both_starts = settings.get("prime_both_starts", False)
        self.high_friction = settings.get("high_friction", False)
        self.v_ord = settings.get("v_ord", False)

        logger.info("Initializing the %s simulation.", self.simtype)
        
        # Initialize the list of ensembles
        self.ensembles = []
        
        # Create ensembles or load from restart based on method
        if self.method != "restart":
            # Create fresh ensembles
            logger.info("Making the ensembles")
            self.create_ensembles()
            logger.info("Done making the ensembles")

            # plt.rcParams['text.usetex'] = True
            # p = self.ensembles[0].engine.potential
            # p.plot_pot(self.intfs)
            # p.plot_potential()
            # plt.close()
            # Create initial paths for each ensemble
            logger.info("Creating dummy initial paths for the ensembles")
            for ens in self.ensembles:
                ens.create_initial_path(N=6)  # Start with 6 time steps
            logger.info("Done creating dummy initial paths for the ensembles")
        else:
            # Load the ensembles from restart pickles
            logger.info("Loading restart pickles for the ensembles")
            self.load_ensembles_from_restart()
            logger.info("Done loading restart pickles for ensembles")

    def do_shooting_moves(self):
        """Perform shooting moves in all the ensembles.
        
        A shooting move attempts to generate a new path by selecting a point
        on an existing path, modifying it slightly, and then integrating forward
        and/or backward in time to create a new path.
        
        The appropriate shooting move is selected based on the ensemble type.
        Results are recorded in each ensemble's data structure.
        """
        self.cycle += 1
        
        # Perform shooting moves for each ensemble
        for ens in self.ensembles:
            # Choose appropriate shooting method based on ensemble type
            if ens.ens_type in ["body_i*", "i*_0star"]:
                # Use the path-graph shooting move for i* ensembles
                status, trial = pg_shooting_move(ens)
            else:
                # Use standard shooting move for other ensemble types
                status, trial = shooting_move(ens)
                
            logger.info("Shooting move in %s resulted in %s", ens.name, status)
            
            # Update the ensemble data with the result of the shooting move
            ens.update_data(status, trial, "sh", self.cycle)


    def do_swap_moves(self):
        """ Perform swap moves between adjacent ensembles.
        
        Swap moves exchange paths between ensembles to enhance sampling.
        Two different schemes are randomly chosen:
        
        Scheme 1:
        - Null move in ensemble 0
        - Swap ensembles 1↔2, 3↔4, etc.
        - If even number of ensembles, also do null move in last ensemble
        
        Scheme 2:
        - Swap ensembles 0↔1, 2↔3, etc.
        - If odd number of ensembles, do null move in the last ensemble
        """
        self.cycle += 1

        # Randomly choose which swap scheme to use
        scheme = np.random.choice([1, 2])
        
        # Check if we have an odd or even number of ensembles
        odd = len(self.ensembles) % 2 != 0
        
        if scheme == 1:
            # Scheme 1: Null move for ensemble 0, then swap 1↔2, 3↔4, etc.
            self.do_null_move(0, "00")
            
            # If even number of ensembles, null move for last ensemble too
            if not odd:
                self.do_null_move(-1, "00")
                
            # Swap pairs of ensembles (1,2), (3,4), etc.
            for i in range(1, len(self.ensembles) - 1, 2):
                self.do_swap_move(i)
                
        elif scheme == 2:
            # Scheme 2: Swap 0↔1, 2↔3, etc.
            
            # If odd number of ensembles, null move for last ensemble
            if odd:
                self.do_null_move(-1, "00")
                
            # Swap pairs of ensembles (0,1), (2,3), etc.
            for i in range(0, len(self.ensembles) - 1, 2):
                self.do_swap_move(i)


    def do_swap_move(self, i):
        """ Perform a swap move between ensemble i and i+1.
        
        This function attempts to exchange paths between ensembles i and i+1,
        using the appropriate swap function based on the simulation type.

        Parameters
        ----------
        i : int
            Index of the first ensemble to swap with the next one (i+1)
        """
        # Special case for the first ensemble (i=0)
        if i == 0:
            if self.simtype == "i*":
                # Use the path-graph swap for i* simulations
                status, trial1, trial2 = pg_swap_zero(self.ensembles)
            else:
                # Use standard swap for RETIS/REPPTIS
                status, trial1, trial2 = swap_zero(self.ensembles)
                
            logger.info("Swap move %s <-> %s resulted in %s",
                self.ensembles[i].name, self.ensembles[i+1].name, status)
                
            # Update data for both ensembles involved in the swap
            self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
            self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)
            return

        # For ensembles other than the first one
        if self.simtype == "retis":
            # Regular TIS swap
            status, trial1, trial2 = swap(self.ensembles, i)
            
        elif self.simtype == "repptis":
            # Replica Exchange PPTIS swap
            status, trial1, trial2 = repptis_swap(self.ensembles, i)
        
        elif self.simtype == "i*":
            # Path-graph swap for i* simulations
            status, trial1, trial2 = pg_swap(self.ensembles, i)
        
        # Log the result of the swap attempt
        logger.info("Swap move %s <-> %s resulted in %s",
            self.ensembles[i].name, self.ensembles[i+1].name, status)
            
        # Update data for both ensembles involved in the swap
        self.ensembles[i].update_data(status, trial1, "s+", self.cycle)
        self.ensembles[i+1].update_data(status, trial2, "s-", self.cycle)
    

    def do_null_move(self, i, gen="00"):
        """ Perform a null move (identity move) in ensemble i.
        
        A null move keeps the current path unchanged but updates the statistics.
        This is needed to maintain proper balance in the Monte Carlo scheme.

        Parameters
        ----------
        i : int
            Index of the ensemble in which to perform the null move
        gen : str, optional
            Generator identifier for the null move, default is "00"
        """
        # Get the appropriate path based on simulation type
        if self.simtype == "i*":
            # For i* simulations, we need to include the metadata
            path = (self.ensembles[i].last_path, 
                   "RMR" if i==0 else self.ensembles[i].last_path.meta[0])
        else:
            # For other simulation types, just use the path itself
            path = self.ensembles[i].last_path
            
        # Update the ensemble data with an accepted null move
        self.ensembles[i].update_data("ACC", path, gen, self.cycle)
        

    def create_ensembles(self):
        """ Create all the ensembles needed for the simulation.
        
        This function initializes the different ensemble types based on the
        simulation settings. The specific ensembles created depend on:
        - Simulation type (RETIS, REPPTIS, i*)
        - Interface positions
        - Whether permeability calculation is enabled
        - Whether state B ensembles are included
        """
        # Base settings for all ensembles
        ens_set = {
            "id": 0,
            "max_len": self.settings["max_len"],
            "simtype": self.simtype,
            "temperature": self.settings["temperature"],
            "friction": self.settings["friction"],
            "dt": self.settings["dt"],
            "prime_both_starts": self.prime_both_starts,
            "max_paths": self.settings["max_paths"],
            "mass": self.settings["mass"],
            "dim": self.settings["dim"],
            "high_friction": self.settings["high_friction"],
            "v_ord": self.v_ord
        }

        # Create the zero minus ensemble [0-]
        if self.permeability:
            # Ensure zero_left is specified for permeability calculations
            if self.zero_left is None:
                raise ValueError("No zero_left specified for permeability calculation")
                
            ens_set["intfs"] = {
                "L": self.zero_left,
                "M": None,  # Not needed for permeability
                "R": self.intfs[0]
            }
            ens_set["ens_type"] = "state_A_lambda_min_one"
            ens_set["name"] = "[0-']"
        else:
            # Standard state A ensemble
            ens_set["intfs"] = {
                "L": -np.inf,
                "M": None,  # Not defined for state A
                "R": self.intfs[0]
            }
            ens_set["ens_type"] = "state_A"
            ens_set["name"] = "[0-]"
            
        logger.info("Making ensemble %s", ens_set["name"])
        self.ensembles.append(Ensemble(ens_set))

        # Create the zero plus ensemble [0+]
        ens_set["id"] = 1
        
        # Configure based on simulation type
        if self.simtype == "repptis":
            ens_set["intfs"] = {
                "L": self.intfs[0],
                "M": None,  # Not defined for this ensemble type
                "R": self.intfs[1]
            }
            ens_set["ens_type"] = "PPTIS_0plusmin_primed"
            ens_set["name"] = "[0+-']"
            
        elif self.simtype == "retis":
            ens_set["intfs"] = {
                "L": self.intfs[0],
                "M": None,  # Not defined for this ensemble type
                "R": self.intfs[-1]
            }
            ens_set["ens_type"] = "RETIS_0plus"
            ens_set["name"] = "[0+]"
            
        elif self.simtype == "i*":
            ens_set["intfs"] = {
                "L": self.intfs[0],
                "M": None,  # Not defined for this ensemble type
                "R": self.intfs[1],
                "all": self.intfs  # All interfaces are needed for i* algorithm
            }
            ens_set["ens_type"] = "i*_0star"
            ens_set["name"] = "[0*]"
            
        logger.info("Making ensemble %s", ens_set["name"])
        self.ensembles.append(Ensemble(ens_set))

        # Create the body ensembles
        for i in range(len(self.intfs) - 2):
            ens_set["id"] = i + 2
            
            if self.simtype == "repptis":
                ens_set["intfs"] = {
                    "L": self.intfs[i],
                    "M": self.intfs[i + 1],
                    "R": self.intfs[i + 2]
                }
                ens_set["ens_type"] = "body_PPTIS"
                ens_set["name"] = f"[{i+1}+-]"
                
            elif self.simtype == "retis":
                ens_set["intfs"] = {
                    "L": self.intfs[0],
                    "M": self.intfs[i+1],
                    "R": self.intfs[-1]
                }
                ens_set["ens_type"] = "body_TIS"
                ens_set["name"] = f"[{i+1}+]"
                
            elif self.simtype == "i*":
                ens_set["intfs"] = {
                    "L": self.intfs[i],
                    "M": self.intfs[i + 1],
                    "R": self.intfs[i+2],
                    "all": self.intfs  # All interfaces for i* algorithm
                }
                ens_set["ens_type"] = "body_i*"
                ens_set["name"] = f"[{i+1}*]"
                
            logger.info("Making ensemble %s", ens_set["name"])
            self.ensembles.append(Ensemble(ens_set))

        # Add state B ensembles if requested
        if self.include_stateB:
            logger.info("Making state B ensembles")
            
            # Check that we're using the appropriate simulation type for state B
            if self.simtype != "repptis":
                raise ValueError("State B ensembles only implemented for repptis")
                
            # Create the N+-' ensemble (connects to state B)
            ens_set["id"] = len(self.intfs)
            ens_set["intfs"] = {
                "L": self.intfs[-2],
                "M": None,  # Not defined for this ensemble type
                "R": self.intfs[-1]
            }
            ens_set["ens_type"] = "PPTIS_Nplusmin_primed"
            ens_set["name"] = f"[{len(self.intfs)-1}+-']"
            logger.info("Making ensemble %s", ens_set["name"])
            self.ensembles.append(Ensemble(ens_set))
            
            # Create the state B ensemble
            ens_set["id"] = len(self.intfs) + 1
            ens_set["intfs"] = {
                "L": self.intfs[-1],
                "M": None,  # Not defined for state B
                "R": np.inf
            }
            ens_set["ens_type"] = "state_B"
            ens_set["name"] = f"[{len(self.intfs)-1}-]"
            logger.info("Making ensemble %s", ens_set["name"])
            self.ensembles.append(Ensemble(ens_set))


    def load_ensembles_from_restart(self):
        """ Load all ensembles from previously saved restart files.
        
        This method loads the pickle files for each ensemble to restart a simulation.
        """
        # Use the number of interfaces to determine how many ensembles to load
        for i in range(len(self.interfaces)):
            self.ensembles.append(Ensemble.load_restart_pickle(i))

    def save_simulation(self, filename):
        """ Save the complete simulation state to a pickle file.
        
        Parameters
        ----------
        filename : str
            Path to the file where the simulation will be saved
        """
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    def run(self):
        """Run the main simulation loop.
        
        This method performs the Monte Carlo sampling by alternating between
        shooting moves and swap moves based on the p_shoot probability.
        The simulation continues until max_cycles is reached or interrupted.
        """
        p_shoot = self.p_shoot  # Store locally for faster access
        
        while self.cycle < self.max_cycles:
            try:
                # Log the current cycle
                logger.info("-" * 80)
                logger.info("Cycle %d", self.cycle)
                logger.info("-" * 80)
                
                # Choose which move to perform based on p_shoot
                if np.random.rand() < p_shoot:
                    # Shooting move
                    self.do_shooting_moves()
                else:
                    # Swap move
                    self.do_swap_moves()
                # if self.cycle % 12 == 0 or self.cycle == 1:
                #     ps = []
                #     # for i in range(self.settings["max_paths"]):
                #     #     ps += [self.ensembles[1].paths[min(len(self.ensembles[1].paths)-1,i)]]
                #     for i in range(len(self.intfs)):
                #         if self.ensembles[i].ens_type != "state_A":
                #             ps += [self.ensembles[i].last_path]
                #             # ps += [self.ensembles[i].paths[np.argmax([max([op[0] for op in self.ensembles[i].paths[j].orders]) for j in range(len(self.ensembles[i].paths))])]]
                #             # if len([p for p in self.ensembles[i].paths[:-2] if p.ptype[2] != "ACC"])>0:
                #             #      ps += [p for p in self.ensembles[i].paths[:-2] if p.ptype[2] != "ACC"]
                #     plot_paths(ps, self.intfs)
                #             #plot_paths([path for path in self.ensembles[i].paths if self.ensembles[i].get_ptype(path) in ["LMR","RML"]][-7:], self.ensembles[i].intfs["all"])                    
                #     print(self.cycle)
                #     plt.close()
            except KeyboardInterrupt:
                # Allow for graceful interruption of the simulation
                print('\nPausing...  (Hit ENTER to continue, type quit to exit.)')
                try:
                    response = input()
                    if response == 'q':
                        break
                    print('Resuming...')
                except KeyboardInterrupt:
                    print('Resuming...')
                    continue
        if self.cycle >= self.max_cycles:
            logger.info("Reached maximum cycles (%d). Stopping simulation.", self.max_cycles)
            logger.info("Total amount of MD steps: %d", sum(ens.engine.mdsteps for ens in self.ensembles))

    @classmethod
    def load_simulation(cls, filename):
        """Load a simulation object from a pickle file.
        
        Parameters
        ----------
        filename : str
            Path to the pickle file containing a saved simulation
            
        Returns
        -------
        Simulation
            The loaded simulation object
        """
        with open(filename, "rb") as f:
            return pkl.load(f)
        

    ####### [i*] SIMULATION FUNCTIONS ########

    # If we would ever need a separate run function
    def run_pg(self):
        """Run the path graph (i*) simulation.
        
        This method implements the main loop for i* simulations,
        which is similar to the standard run() but may have specific
        requirements for the i* algorithm.
        """
        p_shoot = self.p_shoot  # Store locally for faster access
        
        while self.cycle < self.max_cycles:
            try:
                # Log the current cycle
                logger.info("-" * 80)
                logger.info("Cycle %d", self.cycle)
                logger.info("-" * 80)
                
                # Choose which move to perform based on p_shoot
                if np.random.rand() < p_shoot:
                    # Shooting move for i* simulation
                    self.do_shooting_moves()
                    # if self.cycle % 1 == 0 or self.cycle == 1:
                    #     ps = []
                    #     for i in range(len(self.intfs)):
                    #         if self.ensembles[i].ens_type != "state_A":
                    #             ps += [self.ensembles[i].last_path]
                    #             # if len([p for p in self.ensembles[i].paths[:-2] if p.ptype[2] != "ACC"])>0:
                    #             #      ps += [p for p in self.ensembles[i].paths[:-2] if p.ptype[2] != "ACC"]
                    #     plot_paths(ps, self.ensembles[i].intfs["all"])
                    #             #plot_paths([path for path in self.ensembles[i].paths if self.ensembles[i].get_ptype(path) in ["LMR","RML"]][-7:], self.ensembles[i].intfs["all"])                    
                    #     print(self.cycle)
                    #     plt.close('all')
                else:
                    # Swap move for i* simulation
                    self.do_swap_moves()
                    
            except KeyboardInterrupt:
                # Allow for graceful interruption of the simulation
                print('\nPausing...  (Hit ENTER to continue, type quit to exit.)')
                try:
                    response = input()
                    if response == 'q':
                        break
                    print('Resuming...')
                except KeyboardInterrupt:
                    print('Resuming...')
                    continue

    ############################################################
    # Legacy functions for snake moves and multi-level shooting
    #
    # Note: These methods are maintained for historical reasons but
    # detailed balance considerations have shown that memoryless
    # approaches are more rigorous for Monte Carlo simulations.
    ############################################################
    
    def do_snake_move(self, tail_id=None):
        """Perform a snake move starting from a random or specified ensemble.
        
        A snake move attempts to create a continuous sequence of paths that
        connect different interfaces, allowing for more efficient exploration 
        of the transition path space.
        
        Parameters
        ----------
        tail_id : int, optional
            If provided, start the snake from this ensemble ID. Otherwise,
            a random ensemble is chosen.
        """
        self.cycle += 1
        logger.info("Cycle %d: Performing a snake move.", self.cycle)
        
        # Execute the snake move algorithm
        snake_skeleton, snake_paths, visits = snake_move(
            self.ensembles, 
            Lmax=self.snake_Lmax,
            simcycle=self.cycle,
            tail_id=tail_id
        )
        
        # Calculate statistics about the snake move
        tot_visits = np.sum(visits)
        unique_visits = np.sum(visits > 0)
        logger.info("Snake made %d visits to %d unique ensembles.",
            tot_visits, unique_visits)

    def do_multi_level_shooting_moves(self):
        """Perform shooting moves with level selection in all ensembles.
        
        This method implements a variant of shooting where the shooting point
        can be chosen from different "levels" of stored paths, potentially
        allowing for more diverse path generation.
        """
        self.cycle += 1
        
        for ens in self.ensembles:
            # Randomly choose level 0 or 1 for the shooting point
            level = np.random.choice([0, 1])
            
            # Perform the shooting move at the chosen level
            status, trial = shooting_move(ens, level=level)
            
            logger.info("Shooting move in %s at level %d resulted in %s",
                ens.name, level, status)
                
            # Update ensemble data, but don't update paths automatically
            ens.update_data(status, trial, "sh", self.cycle, update_paths=False)
            # update the ensemble paths list. (If the trial path is accepted!!)
            #
            # If we shot from level 0, then the trial path is the new level 1 
            # path. The original level 1 path is now the level 0 path. The 
            # original level 0 path becomes the level 3 path. For the other
            # paths, level N becomes level N+1. 
            # We can accomplish this quickly by just inserting the trial path
            # # at position 1, and then flipping level 0 with level 2. 
            # if level == 0 and status == "ACC":
            #     ens.paths.insert(1, trial)
            #     ens.paths[0], ens.paths[2] = ens.paths[2], ens.paths[0]
            #     if len(ens.paths) > ens.max_paths:
            #         ens.paths.pop()
            # # If we shot from level 1, then the trial path is the new level 0
            # # path. All other paths are just shifted one level up.
            # # We can accomplish this quickly by just inserting the trial path
            # # at position 0. 
            # if level == 1 and status == "ACC":
            #     ens.paths.insert(0, trial)
            #     if len(ens.paths) > ens.max_paths:
            #         ens.paths.pop()
            # The above didn't work, so we'll make it easier: 
            # If we shot from level 0, then trial = new level 1, and old level 1
            # is now level 0. We first make two empty spots in the paths list, 
            # and fill them up. 
            if level == 0 and status == "ACC":
                ens.paths.insert(0, trial)
                ens.paths.insert(0, ens.paths[2])  # old level 1 becomes level 0
            elif level == 1 and status == "ACC":
                ens.paths.insert(0, ens.paths[0])  # keep old level 0
                ens.paths.insert(0, trial)
            elif status != 'ACC':
                # If rejected, keep the existing levels but update the order
                ens.paths.insert(0, ens.paths[1])  # old level 1
                ens.paths.insert(0, ens.paths[1])  # old level 0
                
            # Trim path list if it exceeds max_paths
            while len(ens.paths) > ens.max_paths:
                ens.paths.pop()

    def run_snake(self):
        """Run simulation using snake moves.
        
        This method implements the main simulation loop using a combination
        of shooting moves and snake moves based on the p_shoot probability.
        """
        p_shoot = self.p_shoot  # Store locally for faster access
        
        while self.cycle < self.max_cycles:
            # Log the current cycle
            logger.info("-" * 80)
            logger.info("Cycle %d", self.cycle)
            logger.info("-" * 80)
            
            # Choose between shooting and snake moves
            if np.random.rand() < p_shoot:
                self.do_shooting_moves()
            else:
                self.do_snake_move()

    def do_forced_extension_move(self, idx=None):
        """Perform a forced extension move starting from ensemble idx.
        
        A forced extension move attempts to extend a path from one ensemble
        to another ensemble, allowing for more efficient exploration across
        different interface regions.
        
        Parameters
        ----------
        idx : int, optional
            Index of the ensemble to start the forced extension from.
            If None, a random ensemble is chosen.
        """
        self.cycle += 1
        
        # Choose a random ensemble if none specified
        if idx is None:
            idx = np.random.randint(len(self.ensembles))
            
        logger.info("Forced extension move in %s (%d)", 
                   self.ensembles[idx].name, idx)
        
        # Perform the forced extension
        status, trial, newidx = forced_extension(self.ensembles, idx)
        logger.info("Forced ext from {} ({}) to {} ({}) resulted in {}".format(
            self.ensembles[idx].name, idx,
            self.ensembles[newidx].name, newidx,
            status))
        # Update the data. 
        # Sadly, we will waste some computational time to enforce detailed
        # balance. The ensemble which we extend from will remove its last path
        # and also the path_data for that path. The ensemble which we extend to
        # gains the newly generated path, whose path_data is added to the 
        # pathensemble.txt file. 
        self.ensembles[newidx].update_data(status, trial, 'fn', self.cycle)
        
        # Remove the last path from the source ensemble to maintain detailed balance
        self.ensembles[idx].jump_back(n=1)
        # That's it..
        
        # ## OLD UPDATE CODE ##
        # # update the data
        # # new ensemble gets the trial to pathensemble.txt, while the old 
        # # ensemble gets its second_last path to pathensemble.txt. We update the
        # # ensemble paths list for the new ensemble using update_data (as this 
        # # also manages the max length of the paths list)
        # self.ensembles[newidx].update_data(status, trial, 'fn', self.cycle)
        # self.ensembles[idx].update_data(status, self.ensembles[idx].paths[1],
        #                                 'fo', self.cycle, update_paths=False)
        # # manage the ensemble path list of the old ensemble
        # # for the old ensemble, we pop the first path, and as a safety measure
        # # we insert the lowest level path again (such that we never run out of
        # # paths) TODO: this may lead to biases in extreme cases?
        # self.ensembles[idx].paths.pop(0)
        # self.ensembles[idx].paths.append(self.ensembles[idx].paths[-1])
        # # If the trial path is not accepted (which should not happen, unless
        # # FTX or BTX occurs), we are screwed.
        # if status != "ACC":
        #     logger.warning("Forced extension move failed somehow...")
        # # If we use the pe2 file, we need to write the paths to it. 
        # if self.save_pe2:
        #     self.ensembles[newidx].write_to_pe2(gen="fn")
        #     self.ensembles[idx].write_to_pe2(gen="fo")
        # # Let's plot the level 0 and level 1 paths of both ensembles before and 
        # # after this move has been finished. 
        # # fig, ax = plt.subplots()
        # # l0paths = [self.ensembles[idx].paths[0], self.ensembles[newidx].paths[0]]
        # # l1paths = [self.ensembles[idx].paths[1], self.ensembles[newidx].paths[1]]
        # # l2paths = [self.ensembles[idx].paths[2], self.ensembles[newidx].paths[2]]
        # # plot_paths(l0paths, ax=ax, start_ids=[0,0], color="r")
        # # plot_paths(l1paths, ax=ax, start_ids=[1000,1000], color="g")
        # # plot_paths(l2paths, ax=ax, start_ids=[2000,2000], color="b")
        # # ax.set_title("After the forced extension...")
        # # fig.show()
        # # Let's do it again, but plot make two subplots, one for idx and one for head
        # # fig, (ax1, ax2) = plt.subplots(1,2)
        # # idxpaths = [self.ensembles[idx].paths[0], self.ensembles[idx].paths[1],
        # #             self.ensembles[idx].paths[2]]
        # # headpaths = [self.ensembles[newidx].paths[0],
        # #              self.ensembles[newidx].paths[1],
        # #              self.ensembles[newidx].paths[2]]
        # # plot_paths(idxpaths, ax=ax1, start_ids="staggered")
        # # plot_paths(headpaths, ax=ax2, start_ids="staggered")
        # # ax1.set_title("{}: After the forced extension...".format(
        # #     self.ensembles[idx].name))
        # # ax2.set_title("{}: After the forced extension...".format(
        # #     self.ensembles[newidx].name))
        # # fig.show()

    def run_force(self):
        """Run simulation using forced extension moves.
        
        This method implements the main simulation loop using a combination
        of shooting moves and forced extension moves based on the p_shoot probability.
        """
        p_shoot = self.p_shoot  # Store locally for faster access
        
        while self.cycle < self.max_cycles:
            # Log the current cycle
            logger.info("-" * 80)
            logger.info("Cycle %d", self.cycle)
            logger.info("-" * 80)
            
            # Choose between shooting and forced extension moves
            if np.random.rand() < p_shoot:
                self.do_shooting_moves()
            else:
                self.do_forced_extension_move()