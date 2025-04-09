import os
import matplotlib.pyplot as plt
import numpy as np

def check_position(op, L, R):
    """ Checks whether a phasepoint is in, left of, or right of a specified interval.
    
    This function determines the position of a point relative to an interval [L, R]
    and categorizes it into three states: inside ('M'), left ('L'), or right ('R').
    
    Parameters
    ----------
    op : tuple or list
        Order parameter, where op[0] contains the position coordinate to check
    L : float
        Left boundary of the interval
    R : float
        Right boundary of the interval

    Returns
    -------
    str
        'M': if the point is in the interval [L, R]
        'L': if the point is left of L
        'R': if the point is right of R
    """
    # Assumes the first element of the order parameter is the position coordinate in 1D
    return "M" if L <= op[0] <= R else "L" if op[0] < L else "R"

def validate_staple(ens, path):
    """
    Determines whether a path has valid turns at its start and end points.
    
    A valid path should have turns at both ends that cross at least two interfaces
    in the ensemble. This optimized implementation efficiently checks for valid turns
    by scanning from both ends of the path.
    
    Parameters
    ----------
    ens : :py:class:`Ensemble` object
        Ensemble containing interface definitions to check the path against
    path : :py:class:`Path` object
        Path to validate
        
    Returns
    -------
    bool
        True if the path has valid turns at both ends, False otherwise
    float
        Extremal value at the start of the path (maximum or minimum deviation)
    float
        Extremal value at the end of the path (maximum or minimum deviation)
    """
    # Cache frequently accessed values for performance
    orders = path.orders
    intfs = ens.intfs["all"]
    intfs_min, intfs_max = min(intfs), max(intfs)
    path_length = len(orders)
    
    # Early exit for paths that are too short to have meaningful turns
    if path_length <= 1:
        return False, orders[0][0], orders[0][0]
    
    start_op = orders[0][0]
    end_op = orders[-1][0]
    
    # Check if start/end points are already beyond interface boundaries (automatic turns)
    start_turn = start_op <= intfs_min or start_op >= intfs_max
    end_turn = end_op <= intfs_min or end_op >= intfs_max
    
    # Initialize extremal values to the start and end points
    start_extr = start_op
    end_extr = end_op
    
    # Early exit if both turns are already valid (outside interface boundaries)
    if start_turn and end_turn:
        return True, start_extr, end_extr
    
    last_idx = 0  # Last index of the start turn
    first_idx = path_length - 1  # First index of the end turn
    
    # Check start turn validity by scanning forward
    if not start_turn and path_length > 1:
        next_val = orders[1][0]
        start_increasing = start_op < next_val  # Direction of the initial segment
        max_deviation = start_op  # Track maximum deviation from start point
        
        # Pre-identify the first interface crossed by the initial segment for efficiency
        initial_interface = None
        for i in range(len(intfs)):
            # Check which interface is crossed first based on direction
            if (start_increasing and start_op < intfs[-1-i] <= next_val) or (not start_increasing and next_val <= intfs[i] < start_op):
                initial_interface = intfs[-1-i] if start_increasing else intfs[i]
                break
        
        # Scan forward from start to find valid turn
        for idx in range(1, path_length):
            current_val = orders[idx][0]
            
            # Update maximum deviation from start point
            if (start_increasing and current_val > max_deviation) or (not start_increasing and current_val < max_deviation):
                max_deviation = current_val
            
            # Check if we've recrossed the initial interface (turn completed)
            recrossed = (initial_interface is not None) and (
                (start_increasing and current_val <= initial_interface < max_deviation) or 
                (not start_increasing and current_val >= initial_interface > max_deviation)
            )
            
            if recrossed:
                # Only calculate interfaces crossed when a potential turn is found (optimization)
                min_val, max_val = min(start_op, max_deviation), max(start_op, max_deviation)
                interfaces_crossed = np.sum((intfs > min_val) & (intfs < max_val))
                
                # A valid turn must cross at least 2 interfaces
                if interfaces_crossed >= 2:
                    start_turn = True
                    start_extr = max_deviation
                    last_idx = idx
                    break
    
    # Check end turn validity by scanning backward
    if not end_turn and path_length > 1:
        prev_val = orders[-2][0]
        end_increasing = end_op < prev_val  # Direction from end point to previous point
        max_deviation = end_op  # Track maximum deviation from end point
        
        # Pre-identify the first interface crossed by the final segment for efficiency
        initial_interface = None
        for i in range(len(intfs)):
            # Check which interface is crossed first based on direction
            if (end_increasing and prev_val > intfs[-1-i] >= end_op) or (not end_increasing and end_op >= intfs[i] > prev_val):
                initial_interface = intfs[-1-i] if end_increasing else intfs[i]
                break
        
        # Scan backward from end to find valid turn
        for idx in range(path_length-2, -1, -1):
            current_val = orders[idx][0]
            
            # Update maximum deviation from end point
            if (end_increasing and current_val > max_deviation) or (not end_increasing and current_val < max_deviation):
                max_deviation = current_val
            
            # Check if we've recrossed the initial interface (turn completed)
            recrossed = (initial_interface is not None) and (
                (end_increasing and current_val <= initial_interface < max_deviation) or 
                (not end_increasing and current_val >= initial_interface > max_deviation)
            )
            
            if recrossed:
                # Only calculate interfaces crossed when a potential turn is found (optimization)
                min_val, max_val = min(end_op, max_deviation), max(end_op, max_deviation)
                interfaces_crossed = np.sum((intfs > min_val) & (intfs < max_val))
                
                # A valid turn must cross at least 2 interfaces
                if interfaces_crossed >= 2:
                    end_turn = True
                    end_extr = max_deviation
                    first_idx = idx
                    break
    
    # Path is valid only if both turns meet criteria
    valid = start_turn and end_turn
    
    # Check for turn overlap - turns can't be the same turn unless they're both at the same extreme
    if valid and path_length > 2:
        # If both endpoints are on the same extreme side, overlapping is allowed
        both_min = (start_op <= intfs_min and end_op <= intfs_min)
        both_max = (start_op >= intfs_max and end_op >= intfs_max)
        
        if not (both_min or both_max):
            # Otherwise ensure they're not the same turn
            # (this happens if the turns overlap completely)
            valid = not (start_extr == end_extr and last_idx == path_length - 1 and first_idx == 0)
    
    return valid, start_extr, end_extr


def plot_paths(paths, intfs=None, ax=None, start_ids=0, **kwargs):
    """ 
    Plots a collection of paths in 2D or 3D with optional interfaces.
    
    This function visualizes multiple paths, optionally with interfaces, and
    highlights important features such as start/end points and path segments.
    
    Parameters
    ----------
    paths : list of :py:class:`Path` objects
        Paths to plot
    intfs : list of floats, optional
        Interface values to plot as horizontal lines or planes
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, a new 3D figure is created.
    start_ids : int or str or list, optional
        Starting indices for each path:
        - 0: All paths start at index 0
        - 'staggered': Paths are staggered along y-axis
        - list: Explicit starting indices for each path
    **kwargs : dict
        Additional keyword arguments passed to matplotlib plotting functions
        
    Returns
    -------
    None
        The function shows the plot directly
    """
    # Initialize start_ids based on input type
    if start_ids == 0:
        # All paths start at index 0
        start_ids = [0 for _ in paths]
    elif start_ids == "staggered":
        # Stagger paths based on their lengths
        start_ids = [0]
        for path in paths[:-1]:
            start_ids.append(start_ids[-1] + len(path.phasepoints))
    
    # Ensure we have the right number of start indices
    assert len(start_ids) == len(paths), "Number of start_ids must match number of paths"
    
    # Create a new 3D figure if no axes provided
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    
    # Plot each path
    for path, start_idx in zip(paths, start_ids):
        # Check if we have multi-dimensional order parameters (x,v)
        if len(path.orders[0]) > 1:
            # Plot 3D path: v (x-axis), path index (y-axis), x (z-axis)
            x_coords = [path.orders[i + start_idx][1] for i in range(len(path.orders))]  # v coordinate
            y_coords = [i + start_idx for i in range(len(path.orders))]      # path index
            z_coords = [op[0] for op in path.orders]                         # x coordinate
            
            # Plot the main path
            ax.plot(x_coords, y_coords, z_coords, "-x", **kwargs)
            
            # Highlight the star index region if defined
            if path.staridx is not None:
                start_star, end_star = path.staridx[0], path.staridx[1]
                
                # Extract indices that lie within the star region
                star_indices = [i for i in range(len(path.orders)) if (i >= start_star and i <= end_star)]
                
                # Get coordinates for the star region
                star_x = [path.orders[i + start_idx][1] for i in star_indices]
                star_y = [i + start_idx for i in star_indices]
                
                # Handle different ensemble IDs (they may have different orders structures)
                if path.ens_id == 1:
                    star_z = [path.orders[i + start_idx][0] for i in star_indices]
                    ax.plot(star_x, star_y, star_z, ".-", **kwargs)
                elif path.ens_id > 1:
                    # Use phasepoints for ens_id > 1 (different structure)
                    star_z = [path.orders[i + start_idx][0] for i in range(len(path.phasepoints)) if (i >= start_star and i <= end_star)]
                    ax.plot(star_x, star_y, star_z, ".-", **kwargs)
            
            # Highlight start/end points with arrows
            # Use the same color as the path for consistency
            if path.meta is not None:
                # Plot start point (up arrow)
                ax.plot(path.orders[0][1], 0, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms=7, 
                        label=str((path.meta[:2], path.ens_id)))
                
                # Plot end point (down arrow)
                ax.plot(path.orders[-1][1], len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms=7)
                
                print(path.meta)
                try:
                    # Plot meta point (circle)
                    ax.plot(path.meta[-1][1], path.orders.index(path.meta[-1]), 
                            path.meta[-1][0], "o", **kwargs)
                except:
                    return
            else:
                # No meta data - plot start and end points without labels
                ax.plot(path.orders[0][1], 0, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms=7)
                ax.plot(path.orders[-1][1], len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms=7)
                
                # Plot interfaces as planes if provided
                if intfs is not None:
                    for intf in intfs:
                        # Create a mesh grid for the interface plane
                        xx, yy = np.meshgrid(range(2,10), 
                                            range(max([len(path.orders) for path in paths])))
                        # Plot semi-transparent plane at interface value
                        ax.plot_surface(np.asarray(xx)/10, np.asarray(yy), 
                                      intf*np.ones_like(xx), color='black', alpha=0.15)
        else:
            # 2D plot for 1D order parameters
            # Plot basic path: path index (x-axis), order parameter (y-axis)
            ax.plot([i + start_idx for i in range(len(path.orders))],
                    [ph[0] for ph in path.orders], "-x", **kwargs)
            
            # Highlight the star index region if defined
            if path.staridx is not None:
                start_star, end_star = path.staridx[0], path.staridx[1]
                
                # Extract indices that lie within the star region
                star_indices = [i for i in range(len(path.orders)) if (i >= start_star and i <= end_star)]
                
                # Get coordinates for the star region
                if path.ens_id == 1:
                    ax.plot([i + start_idx for i in star_indices],
                            [path.orders[i + start_idx][0] for i in star_indices], ".-", **kwargs)
                elif path.ens_id > 1:
                    # Use phasepoints for ens_id > 1 (different structure)
                    ax.plot([i + start_idx for i in range(len(path.phasepoints)) if (i >= start_star and i <= end_star)],
                            [path.orders[i + start_idx][0] for i in range(len(path.phasepoints)) if (i >= start_star and i <= end_star)], 
                            ".-", **kwargs)
            
            # Highlight start/end points with arrows
            if path.meta is not None:
                # Plot start point (up arrow) with label
                ax.plot(start_idx, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms=7, 
                        label=str((path.meta[:2], path.ens_id)))
                
                # Plot end point (down arrow)
                ax.plot(start_idx + len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms=7)
                
                print(path.meta)
                # Plot meta point (circle)
                ax.plot(path.orders.index(path.meta[-1]), path.meta[-1][0], "o", **kwargs)
            else:
                # No meta data - plot start and end points without labels
                ax.plot(start_idx, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms=7)
                ax.plot(start_idx + len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms=7)
            
            # Plot interfaces as horizontal lines if provided
            if intfs is not None:
                for intf in intfs:
                    ax.axhline(intf, color="k", ls="--", lw=.5)
    
    # Add legend and optimize layout
    ax.legend()
    plt.tight_layout()
    
    # Display the plot
    if ax is not None:
        plt.show(block=True)
    

def overlay_paths(path, paths):
    """
    Identifies common segments between a reference path and a list of paths.
    
    This function searches for identical sequences of phasepoints between a reference
    path and multiple comparison paths. It returns information about the longest
    matching sequence for each comparison path.
    
    Parameters:
    -----------
    path: Path object
        Reference path to compare against. Contains phasepoints as tuples (x, v).
    paths: list of Path objects
        List of paths to compare with the reference path.

    Returns:
    --------
    list of tuples
        For each path in paths, returns a tuple containing:
        - Length of the longest identical sequence
        - Start index of the sequence in the reference path
        - Start index of the sequence in the comparison path
    """
    result = []
    
    # For each path in the comparison list
    for p in paths:
        max_length = 0       # Length of longest matching sequence found
        start_index = -1     # Start index in reference path
        path_start_idx = -1  # Start index in comparison path

        # Optimization: Pre-compute lengths once
        p_len = len(p.phasepoints)
        path_len = len(path.phasepoints)
        
        # Try all possible starting positions in comparison path
        for i in range(p_len):
            # Try all possible starting positions in reference path
            for j in range(path_len):
                # Use optimized approach with early exit for impossible matches
                remaining_p = p_len - i
                remaining_path = path_len - j
                
                # Skip if remaining elements can't exceed max_length
                if min(remaining_p, remaining_path) <= max_length:
                    continue
                    
                # Count matching elements
                length = 0
                while (i + length < p_len and 
                       j + length < path_len and 
                       p.phasepoints[i + length] == path.phasepoints[j + length]):
                    length += 1

                # Update if this is the longest match so far
                if length > max_length:
                    max_length = length
                    start_index = j
                    path_start_idx = i

        # Record results for this comparison path
        result.append((max_length, start_index, path_start_idx))

    return result

def remove_lines_from_file(fn, n=1):
    """
    Removes the specified number of lines from the end of a file.
    
    This function efficiently removes lines without reading the entire file into memory,
    making it suitable for large files.
    
    Parameters
    ----------
    fn : str
        Path to the file to modify
    n : int, optional
        Number of lines to remove from the end, default is 1
    
    Returns
    -------
    None
        The file is modified in place
    
    Notes
    -----
    After removal, a newline character is appended to ensure the file ends properly.
    This implementation is based on an efficient approach from Stack Overflow.
    """
    # Process requested number of lines
    for i in range(n):
        with open(fn, "r+", encoding="utf-8") as f:
            # Move file pointer to the end of the file
            f.seek(0, os.SEEK_END)
            
            # Handle edge case of empty file
            if f.tell() == 0:
                return
                
            # Position pointer at the last character
            pos = f.tell() - 1
            
            # Read backwards until finding a newline character or reaching start of file
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, os.SEEK_SET)
                
            # Truncate the file at this position if not at the beginning
            if pos > 0:
                f.seek(pos, os.SEEK_SET)
                f.truncate()
                
            # Add a newline character to ensure proper file ending
            f.write("\n")