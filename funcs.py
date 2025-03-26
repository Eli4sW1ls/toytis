import os
import matplotlib.pyplot as plt
import numpy as np
def check_position(op, L, R):
    """ Checks whether a phasepoint is:
    - in the interval [L, R] : M
    - left of L              : L
    - right of R             : R

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
        String representing the condition of the phasepoint

    """
    return "M" if L <= op[0] <= R else "L" if op[0] < L else "R"        # Assume that 1st index of orderparameter list is the determining 1D order parameter

def validate_staple(ens, path):
    """
    Optimized version for determining the validity of turns in a path.
    
    Parameters
    ----------
    ens : :py:class:`Ensemble` object
        Ensemble to check the path against
    path : :py:class:`Path` object
        Path to check
        
    Returns
    -------
    bool
        True if the path has valid turns, False otherwise
    float
        Extremal value at the start of the path
    float
        Extremal value at the end of the path
    """
    # Cache frequently accessed values
    orders = path.orders
    intfs = ens.intfs["all"]
    intfs_min, intfs_max = min(intfs), max(intfs)
    path_length = len(orders)
    
    # Early exit for short paths
    if path_length <= 1:
        return False, orders[0][0], orders[0][0]
    
    start_op = orders[0][0]
    end_op = orders[-1][0]
    
    # Pre-compute conditions for early exit
    start_turn = start_op <= intfs_min or start_op >= intfs_max
    end_turn = end_op <= intfs_min or end_op >= intfs_max
    
    # Initialize extremal values
    start_extr = start_op
    end_extr = end_op
    
    # Early exit if both turns are already valid
    if start_turn and end_turn:
        return True, start_extr, end_extr
    
    last_idx = 0
    first_idx = path_length - 1
    
    # Check start turn validity
    if not start_turn and path_length > 1:
        next_val = orders[1][0]
        start_increasing = start_op < next_val
        max_deviation = start_op
        
        # Precompute interfaces crossed by initial segment
        initial_interface = None
        for i in range(len(intfs)):
            if (start_increasing and start_op < intfs[-1-i] <= next_val) or (not start_increasing and next_val <= intfs[i] < start_op):
                initial_interface = intfs[-1-i] if start_increasing else intfs[i]
                break
        
        # Scan forward from start
        for idx in range(1, path_length):
            current_val = orders[idx][0]
            
            # Update maximum deviation more efficiently
            if (start_increasing and current_val > max_deviation) or (not start_increasing and current_val < max_deviation):
                max_deviation = current_val
            

            recrossed = (initial_interface is not None) and (
                (start_increasing and current_val <= initial_interface < max_deviation) or 
                (not start_increasing and current_val >= initial_interface > max_deviation)
            )
            
            if recrossed:
                # Count interfaces crossed - only do this when necessary
                min_val, max_val = min(start_op, max_deviation), max(start_op, max_deviation)
                interfaces_crossed = np.sum((intfs > min_val) & (intfs < max_val))
                
                if interfaces_crossed >= 2:
                    start_turn = True
                    start_extr = max_deviation
                    last_idx = idx
                    break
    
    # Check end turn validity
    if not end_turn and path_length > 1:
        prev_val = orders[-2][0]
        end_increasing = end_op < prev_val
        max_deviation = end_op
        
        # Precompute interfaces crossed by final segment
        initial_interface = None
        for i in range(len(intfs)):
            if (end_increasing and prev_val > intfs[-1-i] >= end_op) or (not end_increasing and end_op >= intfs[i] > prev_val):
                initial_interface = intfs[-1-i] if end_increasing else intfs[i]
                break
        
        # Scan backward from end
        for idx in range(path_length-2, -1, -1):
            current_val = orders[idx][0]
            
            # Update maximum deviation more efficiently
            if (end_increasing and current_val > max_deviation) or (not end_increasing and current_val < max_deviation):
                max_deviation = current_val
            
            # Check if we've recrossed the initial interface
            recrossed = (initial_interface is not None) and (
                (end_increasing and current_val <= initial_interface < max_deviation) or 
                (not end_increasing and current_val >= initial_interface > max_deviation)
            )
            
            if recrossed:
                # Count interfaces crossed - only do this when necessary
                min_val, max_val = min(end_op, max_deviation), max(end_op, max_deviation)
                interfaces_crossed = np.sum((intfs > min_val) & (intfs < max_val))
                
                if interfaces_crossed >= 2:
                    end_turn = True
                    end_extr = max_deviation
                    first_idx = idx
                    break
    
    # Check for turn validity - both turns must be valid
    valid = start_turn and end_turn
    
    # Check for turn overlap - turns can overlap, but can't be the same turn
    # Exception: If both first and last order parameters are either <= intfs_min or >= intfs_max
    if valid and path_length > 2:
        # If both are on the same extreme side, it's allowed
        both_min = (start_op <= intfs_min and end_op <= intfs_min)
        both_max = (start_op >= intfs_max and end_op >= intfs_max)
        
        if not (both_min or both_max):
            # Otherwise ensure they're not the same turn
            valid = not (start_extr == end_extr and last_idx == path_length - 1 and first_idx == 0)
    
    if not valid:
        valid = valid
    
    return valid, start_extr, end_extr


def plot_paths(paths, intfs=None, ax=None, start_ids=0, **kwargs):
    """ Plots the paths in the list paths, with optional interfaces intfs.
    
    Parameters
    ----------
    paths : list of :py:class:`Path` objects
        Paths to plot
        
    intfs : list of floats, optional
        Interfaces to plot

    """
    if start_ids == 0:
        start_ids = [0 for _ in paths]
    elif start_ids == "staggered":
        start_ids = [0]
        for path in paths[:-1]:
            start_ids.append(start_ids[-1] + len(path.phasepoints))
    assert len(start_ids) == len(paths)
    if ax is None:
        # ax = plt.figure().add_subplot()
        ax = plt.figure().add_subplot(projection='3d')
    for path, start_idx in zip(paths, start_ids):
        if len(path.orders[0]) > 1:
            ax.plot([path.orders[i + start_idx][1] for i in range(len(path.orders))], [i + start_idx for i in range(len(path.orders))],
                    [op[0] for op in path.orders], "-x", **kwargs)
            if path.staridx is not None:
                if path.ens_id == 1:
                    ax.plot([path.orders[i + start_idx][1] for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])],
                            [i + start_idx for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])],
                            [path.orders[i+start_idx][0] for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])], ".-", **kwargs)
                elif path.ens_id > 1:
                    ax.plot([path.orders[i + start_idx][1] for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])],
                            [i + start_idx for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])],
                            [path.orders[i+start_idx][0] for i in range(len(path.phasepoints)) if (i >= path.staridx[0] and i <= path.staridx[1])], ".-", **kwargs)
            # plot the first and last point again to highlight start/end phasepoints
            # it must have the same color as the line for the path
            if path.meta is not None:
                ax.plot(path.orders[0][1], 0, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms = 7, label=str((path.meta[:2], path.ens_id)))
                ax.plot(path.orders[len(path.orders) - 1][1], len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms = 7)
                print(path.meta)
                try:
                    ax.plot(path.meta[-1][1], path.orders.index(path.meta[-1]), path.meta[-1][0], "o", **kwargs)
                except:
                    return
            else:
                ax.plot(path.orders[0][1], 0, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms = 7)
                ax.plot(path.orders[-1][1], len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms = 7)
                if intfs is not None:
                    for intf in intfs:
                        # ax.axhline(intf, color="k", ls="--", lw=.5)
                        xx, yy = np.meshgrid(range(2,10), range(max([len(path.orders) for path in paths])))
                        ax.plot_surface(np.asarray(xx)/10, np.asarray(yy), intf*np.ones_like(xx), color='black', alpha=0.15)
        else:
            ax.plot([i + start_idx for i in range(len(path.orders))],
                    [ph[0] for ph in path.orders], "-x", **kwargs)
            if path.staridx is not None:
                if path.ens_id == 1:
                    ax.plot([i + start_idx for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])],
                            [path.orders[i+start_idx][0] for i in range(len(path.orders)) if (i >= path.staridx[0] and i <= path.staridx[1])], ".-", **kwargs)
                elif path.ens_id > 1:
                    ax.plot([i + start_idx for i in range(len(path.phasepoints)) if (i >= path.staridx[0] and i <= path.staridx[1])],
                    [path.orders[i+start_idx][0] for i in range(len(path.phasepoints)) if (i >= path.staridx[0] and i <= path.staridx[1])], ".-", **kwargs)
            # plot the first and last point again to highlight start/end phasepoints
            # it must have the same color as the line for the path
            if path.meta is not None:
                ax.plot(start_idx, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms = 7, label=str((path.meta[:2], path.ens_id)))
                ax.plot(start_idx + len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms = 7)
                print(path.meta)
                ax.plot(path.orders.index(path.meta[-1]), path.meta[-1][0], "o", **kwargs)
            else:
                ax.plot(start_idx, path.orders[0][0], "^",
                        color=ax.lines[-1].get_color(), ms = 7)
                ax.plot(start_idx + len(path.orders) - 1,
                        path.orders[-1][0], "v",
                        color=ax.lines[-1].get_color(), ms = 7)
            if intfs is not None:
                for intf in intfs:
                    ax.axhline(intf, color="k", ls="--", lw=.5)
    ax.legend()
    plt.tight_layout()
    if ax is not None:
        plt.show(block=True)
    

def overlay_paths(path, paths):
    """Searches for sequences of phasepoints that are identical in path and
    the list paths contained in paths. Returns a list containing a tuple for 
    each element of paths. 
    The first element of a tuple is the length of the largest sequence of 
    identical phasepoints, the second element is the index where the identical
    sequence starts in the path of paths.

    Parameters:
    -----------
    path: Path object
        Path to compare to. path.phasepoints are the phasepoints. A phasepoint
        is a tuple (x, v) of floats. We care only about the x coordinate. 
    paths: list of Path objects
        Paths to compare to path. Each path.phasepoints is a list of phasepoints
        (see above).

    Returns:
    --------
    list of tuples
        List of tuples (length of identical sequence, start index of identical
        sequence in path.phasepoints) for each path in paths.
    """
    result = []

    for p in paths:
        max_length = 0
        start_index = -1
        path_start_idx = -1  # Initialize the start index w.r.t. paths

        for i in range(len(p.phasepoints)):
            for j in range(len(path.phasepoints)):
                length = 0
                while i + length < len(p.phasepoints) and j + length < len(path.phasepoints) and \
                        p.phasepoints[i + length] == path.phasepoints[j + length]:
                    length += 1

                if length > max_length:
                    max_length = length
                    start_index = j
                    path_start_idx = i

        result.append((max_length, start_index, path_start_idx))

    return result

def remove_lines_from_file(fn, n=1):
    """Remove the last n lines from a file.
    
    Parameters
    ----------
    fn : str
        File name
    n : int, optional
        Number of lines to remove, by default 1

    """
    # We use: https://stackoverflow.com/questions/1877999/
    for i in range(n):
        with open(fn, "r+", encoding="utf-8") as f:
            # Move the pointer (similar to a cursor in a text editor) to the
            # end of the file
            f.seek(0, os.SEEK_END)
            # This code means the following code skips the very last character
            # in the file - i.e. in the case the last line is null we delete
            # the last line and the penultimate one
            pos = f.tell() - 1
            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, os.SEEK_SET)
            # So long as we're not at the start of the file, delete all the
            # characters ahead of this position
            if pos > 0:
                f.seek(pos, os.SEEK_SET)
                f.truncate()
            # After truncating, we need to position the pointer at the end, 
            # on a new blank line..
            f.write("\n")