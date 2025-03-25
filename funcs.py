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
    start_op = orders[0][0]
    end_op = orders[-1][0]
    
    # Pre-compute conditions for early exit
    start_turn = start_op <= intfs[0] or start_op >= intfs[-1]
    end_turn = end_op <= intfs[0] or end_op >= intfs[-1]
    
    # Initialize extremal values
    start_extr = start_op
    end_extr = end_op
    
    # Early exit if both turns are already valid
    if start_turn and end_turn:
        return True, start_extr, end_extr
    
    # Pre-compute additional values to avoid recalculation in loop
    path_length = len(orders)
    midpoint = path_length // 2
    
    next_val = orders[1][0] if path_length > 1 else start_op
    prev_val = orders[-2][0] if path_length > 1 else end_op
    
    # Pre-compute direction for start and end segments
    start_increasing = start_op < next_val
    end_increasing = end_op < prev_val
    
    # Pre-compute relevant interfaces for start and end segments
    # For start segment
    if not start_turn:
        start_relevant_intfs = []
        min_val, max_val = (start_op, next_val) if start_increasing else (next_val, start_op)
        for intf in intfs:
            if min_val < intf < max_val:
                start_relevant_intfs.append(intf)
    else:
        start_relevant_intfs = []
    
    # For end segment
    if not end_turn:
        end_relevant_intfs = []
        min_val, max_val = (end_op, prev_val) if end_increasing else (prev_val, end_op)
        for intf in intfs:
            if min_val < intf < max_val:
                end_relevant_intfs.append(intf)
    else:
        end_relevant_intfs = []
    
    # Only iterate until midpoint or until both turns are found
    for idx in range(1, midpoint + 1):
        if start_turn and end_turn:
            break
            
        # Process start segment if needed
        if not start_turn and start_relevant_intfs:
            # Get current value
            current_val = orders[idx][0]
            
            # Check if we've crossed required interfaces
            if start_increasing:
                # Check whether current value is below interface and we've crossed at least 2 interfaces
                if current_val <= start_relevant_intfs[0]:
                    # Count interfaces crossed using direct comparison instead of np operations
                    interfaces_crossed = 0
                    for intf in intfs:
                        if (intf <= max(start_op, current_val)) != (intf <= start_op):
                            interfaces_crossed += 1
                    
                    if interfaces_crossed >= 2:
                        start_turn = True
                        # Calculate extremal value more efficiently with direct loop
                        start_extr = orders[0][0]
                        for i in range(1, idx+1):
                            if (start_increasing and orders[i][0] > start_extr) or \
                               (not start_increasing and orders[i][0] < start_extr):
                                start_extr = orders[i][0]
            else:
                # Similar logic for decreasing segment
                if current_val >= start_relevant_intfs[0]:
                    interfaces_crossed = 0
                    for intf in intfs:
                        if (intf >= min(start_op, current_val)) != (intf >= start_op):
                            interfaces_crossed += 1
                    
                    if interfaces_crossed >= 2:
                        start_turn = True
                        # Calculate extremal value
                        start_extr = orders[0][0]
                        for i in range(1, idx+1):
                            if (not start_increasing and orders[i][0] < start_extr) or \
                               (start_increasing and orders[i][0] > start_extr):
                                start_extr = orders[i][0]
        
        # Process end segment if needed
        if not end_turn and end_relevant_intfs:
            # Calculate index from the end
            end_idx = path_length - idx - 1
            current_val = orders[end_idx][0]
            
            if end_increasing:
                if current_val <= end_relevant_intfs[0]:
                    interfaces_crossed = 0
                    for intf in intfs:
                        if (intf <= max(end_op, current_val)) != (intf <= end_op):
                            interfaces_crossed += 1
                    
                    if interfaces_crossed >= 2:
                        end_turn = True
                        # Calculate extremal value
                        end_extr = orders[-1][0]
                        for i in range(path_length-2, end_idx-1, -1):
                            if (end_increasing and orders[i][0] > end_extr) or \
                               (not end_increasing and orders[i][0] < end_extr):
                                end_extr = orders[i][0]
            else:
                if current_val >= end_relevant_intfs[0]:
                    interfaces_crossed = 0
                    for intf in intfs:
                        if (intf >= min(end_op, current_val)) != (intf >= end_op):
                            interfaces_crossed += 1
                    
                    if interfaces_crossed >= 2:
                        end_turn = True
                        # Calculate extremal value
                        end_extr = orders[-1][0]
                        for i in range(path_length-2, end_idx-1, -1):
                            if (not end_increasing and orders[i][0] < end_extr) or \
                               (end_increasing and orders[i][0] > end_extr):
                                end_extr = orders[i][0]
                        
    return start_turn and end_turn, start_extr, end_extr


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