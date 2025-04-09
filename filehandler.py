"""
File handling utilities for ToyTIS simulations.

This module provides functionality for creating and managing directory structures
and files needed for ensemble-based simulations in the ToyTIS framework.
"""
import logging
import os
from typing import Union, List

# Set up module-level logger with null handler to prevent "No handler found" warnings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _create_empty_file(filepath: str) -> bool:
    """
    Helper function to create an empty file if it doesn't exist.
    
    Parameters
    ----------
    filepath : str
        The full path of the file to create
        
    Returns
    -------
    bool
        True if a new file was created, False if file already existed
    """
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("")
        return True
    return False

def make_ens_dirs_and_files(id: Union[int, str]) -> str:
    """ Makes the directory for ensemble with id, and creates the
    pathensemble.txt and order.txt files within it.
    
    This function ensures that the necessary directory structure and
    initialization files exist for a simulation ensemble. It creates
    a zero-padded directory name and initializes empty text files
    that will later store simulation data.

    Parameters
    ----------
    id : int or str
        Unique identifier of the ensemble. Will be zero-padded to 3 digits.
        
    Returns
    -------
    str
        The created directory name (zero-padded ID)
        
    Examples
    --------
    >>> make_ens_dirs_and_files(7)
    '007'
    """
    # Convert ID to zero-padded string (e.g., 7 becomes '007')
    id = str(id).zfill(3)
    
    # Create ensemble directory if it doesn't exist
    if not os.path.exists(id):
        os.makedirs(id)
        logger.info("Created directory for ensemble %s", id)
    
    # Define required files for this ensemble
    required_files = [
        "pathensemble.txt",   # Main path ensemble data file
        "pathensemble2.txt",  # Secondary path ensemble data file
        "order.txt"           # Order parameters tracking file
    ]
    
    # Create each required file and log results
    for filename in required_files:
        filepath = os.path.join(id, filename)
        if _create_empty_file(filepath):
            logger.info("Made file %s for ensemble %s", filepath, id)
        else:
            logger.info("File %s already exists for ensemble %s", filepath, id)
    
    return id