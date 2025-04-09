from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union, Any

class Path:
    """ Representation of a path in Transition Interface Sampling (TIS).
    
    This class stores phase space points along a trajectory path and their 
    associated order parameters. Each path is assigned a unique identifier
    through a global counter to enable tracking in the simulation.

    Attributes
    ----------
    phasepoints : list of tuples (x, v) of floats
        List of phase space points in the path, where each point is a tuple
        containing position (x) and velocity (v) coordinates.
    orders : list of floats
        List of order parameters for each phasepoint, used to determine
        the progress of the transition.
    path_id : int
        Unique identifier of the path, assigned automatically.
    ens_id : int
        Unique identifier of the ensemble to which the path belongs.
    meta : dict or None
        Optional metadata associated with the path.
    staridx : int or None
        Optional index marking a special point in the path (e.g., transition state).
    """
    path_counter = 0

    def __init__(self, phasepoints=None, orders=None, ens_id=None, metadata=None, staridx=None):
        """Initialize the Path object with phase space points and related data.

        Parameters
        ----------
        phasepoints : list, optional
            List of phase space points in the path, where each point is typically
            a tuple containing position and velocity coordinates.
            Default is an empty list.
        orders : list, optional
            List of order parameters corresponding to each phase point.
            Default is an empty list.
        ens_id : int, optional
            Unique identifier of the ensemble to which the path belongs.
            Default is None.
        metadata : dict, optional
            Additional information about the path that might be useful for
            analysis or tracking. Default is None.
        staridx : int, optional
            Index marking a special point in the path, such as a transition state
            or a crossing point. Default is None.
        
        Notes
        -----
        The path_id is automatically assigned using the class-level counter.
        """
        # Store the phase space points representing the trajectory (initialize as empty list if None)
        self.phasepoints = phasepoints if phasepoints is not None else []
        # Store the order parameters for each point in the trajectory (initialize as empty list if None)
        self.orders = orders if orders is not None else []
        # Store the ensemble identifier this path belongs to
        self.ens_id = ens_id
        # Assign a unique identifier to this path
        self.path_id = Path.path_counter
        # Increment the counter for future paths
        Path.path_counter += 1
        # Store additional metadata
        self.meta = metadata
        # Store special index if provided
        self.staridx = staridx
        
        # Validate input parameters
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate the input parameters to ensure consistency.
        
        Raises
        ------
        ValueError
            If the phasepoints and orders lists have different lengths
        TypeError
            If the provided parameters are not of the expected types
        """
        if len(self.phasepoints) != len(self.orders) and len(self.phasepoints) > 0 and len(self.orders) > 0:
            raise ValueError("The phasepoints and orders lists must have the same length")
        
        if self.staridx is not None and not isinstance(self.staridx, int):
            raise TypeError("staridx must be an integer or None")
            
        if self.ens_id is not None and not isinstance(self.ens_id, int):
            raise TypeError("ens_id must be an integer or None")
            
        if self.meta is not None and not isinstance(self.meta, dict):
            raise TypeError("metadata must be a dictionary or None")

    def copy_path(self):
        """Returns a deep copy of the current path.
        
        Creates a new Path object with copies of all attributes,
        ensuring that modifications to the new path do not affect the original.
        
        Returns
        -------
        Path
            A new Path object that is a copy of the current path.
        """
        return Path(
            self.phasepoints.copy(),
            self.orders.copy(),
            self.ens_id,
            self.meta.copy() if self.meta is not None else None,
            self.staridx
        )