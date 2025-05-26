# This Gnuplot script visualizes path ensemble data from PyRETIS simulations.
# It plots order parameters for paths, highlighting start, end, min, and max points.
# Interfaces can also be displayed if found in RST or logging files.
#
# Usage:
#   gnuplot -e "VAR1=value1;VAR2=value2" plot_paths.gp
#
# Available variables (passed via -e):
#   MINOP:    Minimum order parameter value for a path to be included.
#             Paths are filtered based on their recorded min/max OP values
#             from pathensemble.txt (columns 10 and 11).
#             Default: -99999.0 (effectively no lower bound).
#   MAXOP:    Maximum order parameter value for a path to be included.
#             Default: 99999.0 (effectively no upper bound).
#   MAXPATHS: Maximum number of paths to plot. If more paths match MINOP/MAXOP,
#             a subset is selected (randomly or sequentially).
#             Default: 1000000.
#   RANDOM:   Set to 1 to select paths randomly if the number of matching paths
#             exceeds MAXPATHS. Set to 0 for sequential selection (first N paths).
#             Default: 1 (random selection).
#
# Example: Plot up to 100 random paths where the path's order parameter
#          stayed between -1.0 and 1.0:
#   gnuplot -e "MINOP=-1.0;MAXOP=1.0;MAXPATHS=100;RANDOM=1" plot_paths.gp

if (!exists("MINOP")) MINOP=-99999.0
if (!exists("MAXOP")) MAXOP=99999.0
if (!exists("MAXPATHS")) MAXPATHS=1000000
if (!exists("RANDOM")) RANDOM=1

# Set up plot configuration
set terminal pngcairo size 1200,800 enhanced font "Arial,11"
set output 'paths.png'
set style data lines
set key outside right box title "Ensembles" spacing 1.5

# Get a sample order.txt to check columns
system("ls -d [0-9][0-9][0-9]/order.txt | head -n1 > first_order.tmp")
first_order = system("cat first_order.tmp") 
ncols = system(sprintf("awk '!/^#/ {print NF; exit}' %s", first_order)) + 0

# Check if we're plotting 1D or 2D order parameters
is_1d = (ncols <= 2 ? 1 : 0)

# Set appropriate labels and using statement based on number of columns
if (!is_1d) {
    print sprintf("Found %d columns, plotting columns 2:3", ncols)
    set xlabel "Order parameter 1"
    set ylabel "Order parameter 2"
    using_str = "using 2:3"
} else {
    print sprintf("Found %d columns, plotting columns 1:2", ncols)
    set xlabel "Order parameter"
    set ylabel "Time"
    using_str = "using 1:2"
}

set title sprintf("Order parameters for paths with %.2f ≤ OP ≤ %.2f (max %d paths)", MINOP, MAXOP, MAXPATHS)

# Find interfaces in RST files or logging.log
print "Looking for interface values..."
system("find . -name '*tis.rst' -o -name 'retis.rst' | xargs grep -l 'interfaces = ' | head -n1 > rstfile.tmp")
has_interfaces = int(system("[ -s rstfile.tmp ] && echo 1 || echo 0"))

if (!has_interfaces) {
    print "No RST file with interfaces found, trying logging.log..."
    system("grep -l 'interfaces = ' logging.log > logfile.tmp 2>/dev/null")
    has_interfaces = int(system("[ -s logfile.tmp ] && echo 1 || echo 0"))
    
    if (has_interfaces) {
        system("grep 'interfaces = ' logging.log | tail -n1 | sed 's/.*interfaces = \\[\\(.*\\)\\]/\\1/' > interfaces.tmp")
    }
} else {
    system("grep 'interfaces = ' $(cat rstfile.tmp) | sed 's/.*interfaces = \\[\\(.*\\)\\]/\\1/' > interfaces.tmp")
}

# Process interfaces if found
interface_values = ""
n_interfaces = 0
if (has_interfaces) {
    # Convert comma-separated list to space-separated values
    system("cat interfaces.tmp | tr ',' ' ' | tr -s ' ' > interface_values.tmp")
    interface_values = system("cat interface_values.tmp")
    n_interfaces = words(interface_values)
    print sprintf("Found %d interface values", n_interfaces)
    
    # Debug: Print interface values
    do for [i=1:n_interfaces] {
        val = word(interface_values, i) + 0
        print sprintf("Interface %d: %f", i, val)
    }
} else {
    print "No interface values found"
    n_interfaces = 0
}

# Get number of ensemble directories
n_ensembles = system("ls -d [0-9][0-9][0-9] | wc -l") + 0
print sprintf("Found %d ensemble directories", n_ensembles)

if (n_ensembles == 0) {
    print "No ensemble directories found!"
    exit
}

# First collect all matching paths
system("rm -f all_paths.tmp")
print "\n=== DEBUG: Starting path collection process ==="
print sprintf("Looking for paths with order parameters in range [%f, %f]", MINOP, MAXOP)

do for [i=1:n_ensembles] {
    ens_dir = system(sprintf("ls -d [0-9][0-9][0-9] | sed -n '%dp' | tr -d '\n'", i))
    print sprintf("\n--- DEBUG: Processing ensemble directory: %s ---", ens_dir)
    
    # Check if pathensemble.txt exists
    file_exists_cmd = sprintf("[ -f %s/pathensemble.txt ] && echo 1 || echo 0", ens_dir)
    file_exists_result = system(file_exists_cmd)
    # Trim any whitespace/newlines to ensure clean conversion
    file_exists_result = system(sprintf("echo '%s' | tr -d '\n' | tr -d ' '", file_exists_result))
    file_exists_result = (file_exists_result eq "1" ? 1 : 0)
    
    print sprintf("DEBUG: Command executed: %s", file_exists_cmd)
    print sprintf("DEBUG: File exists check result: %d", file_exists_result)
    
    if (file_exists_result) {
        print sprintf("DEBUG: Found pathensemble.txt in %s", ens_dir)
        
        # First save the filtered paths to a temp file - only ACC paths with order parameters in range
        temp_filtered_file = sprintf("temp_filtered_%s.txt", ens_dir)
        filter_cmd = sprintf("awk '$8 == \"ACC\" && $10 >= %f && $11 <= %f {print \"%s \"$1}' %s/pathensemble.txt > %s", \
                          MINOP, MAXOP, ens_dir, ens_dir, temp_filtered_file)
        
        print sprintf("DEBUG: Executing filter command: %s", filter_cmd)
        print sprintf("DEBUG: (Filtering for status='ACC' in column 8)")
        system(filter_cmd)
        
        # Count the matching paths from the temp file
        count_cmd = sprintf("wc -l < %s", temp_filtered_file)
        matching_paths_str = system(count_cmd)
        # Clean up the output to ensure it's just a number
        matching_paths_str = system(sprintf("echo '%s' | tr -d '\n' | tr -d ' '", matching_paths_str))
        matching_paths = int(matching_paths_str)
        
        print sprintf("DEBUG: ACC paths matching criteria: %d", matching_paths)
        
        # Now append the filtered results to the main collection file
        system(sprintf("cat %s >> all_paths.tmp", temp_filtered_file))
        
        # Clean up temp file
        system(sprintf("rm -f %s", temp_filtered_file))
        
        # Verify the append worked
        if (matching_paths > 0) {
            print sprintf("DEBUG: %d ACC paths from %s added to selection", matching_paths, ens_dir)
            
            # Print first few matching paths with their order parameters (for verification)
            print "DEBUG: Sample of matching paths (first 3 or fewer):"
            system(sprintf("awk '$8 == \"ACC\" && $10 >= %f && $11 <= %f {printf \"  Path %%s: status=%%s, min_op=%%s, max_op=%%s\\n\", $1, $8, $10, $11}' %s/pathensemble.txt | head -3", \
                         MINOP, MAXOP, ens_dir))
        } else {
            print sprintf("DEBUG: No ACC paths from %s matched the criteria", ens_dir)
            
            # Show a few ACC paths and their order parameters to understand why
            print "DEBUG: Sample of ACC paths that didn't match (first 3):"
            system(sprintf("awk '$8 == \"ACC\"' %s/pathensemble.txt | head -3 | awk '{printf \"  Path %%s: status=%%s, min_op=%%s, max_op=%%s, match=(%%s >= %f && %%s <= %f)=%%s\\n\", $1, $8, $10, $11, $10, $11, ($10 >= %f && $11 <= %f) ? \"true\" : \"false\"}' ", \
                         ens_dir, MINOP, MAXOP, MINOP, MAXOP))
        }
    } else {
        print sprintf("DEBUG: No pathensemble.txt found in %s", ens_dir)
    }
}

# Total collected paths - handle empty file case
total_collected_cmd = "[ -f all_paths.tmp ] && wc -l < all_paths.tmp || echo 0"
total_collected_str = system(total_collected_cmd)
total_collected_str = system(sprintf("echo '%s' | tr -d '\n' | tr -d ' '", total_collected_str))
total_collected = int(total_collected_str)

print sprintf("\n=== DEBUG: Collection complete. Total paths collected: %d ===\n", total_collected)

# Check if any paths matched our criteria
paths_exist_cmd = "[ -s all_paths.tmp ] && echo 1 || echo 0"
paths_exist_str = system(paths_exist_cmd)
paths_exist_str = system(sprintf("echo '%s' | tr -d '\n' | tr -d ' '", paths_exist_str))
paths_exist = int(paths_exist_str)

print sprintf("DEBUG: Check if paths exist: %d", paths_exist)

if (paths_exist == 0) {
    print "DEBUG: No paths found matching criteria - exiting plot routine"
    set title sprintf("No paths found with %.2f ≤ OP ≤ %.2f", MINOP, MAXOP)
    plot -1 notitle
    exit
}

# Select paths (randomly or sequentially)
if (RANDOM) {
    system(sprintf("shuf -n %d all_paths.tmp > selected_paths.tmp 2>/dev/null || head -n %d all_paths.tmp > selected_paths.tmp", \
                  MAXPATHS, MAXPATHS))
} else {
    system(sprintf("head -n %d all_paths.tmp > selected_paths.tmp", MAXPATHS))
}

# Process paths
do for [i=1:n_ensembles] {
    ens_dir = system(sprintf("ls -d [0-9][0-9][0-9] | sed -n '%dp' | tr -d '\n'", i))
    
    system(sprintf("awk '$1==\"%s\" {print $2}' selected_paths.tmp > %s_cycles.tmp", ens_dir, ens_dir))
    
    if (int(system(sprintf("[ -s %s_cycles.tmp ] && echo 1 || echo 0", ens_dir)))) {
        system(sprintf("awk 'NR==FNR {cycles[$1]=1; next} \
        /^# Cycle:/ {in_path=0; match($0,/Cycle: ([0-9]+)/,a); \
        if(a[1] in cycles){in_path=1; print \"\"}} \
        /^#[ ]+Time/{next} \
        in_path && /^[ ]*[0-9]/{print $0}' \
        %s_cycles.tmp %s/order.txt > %s_paths.tmp", ens_dir, ens_dir, ens_dir))
    }
}

# Collect all data points to determine plot ranges
system("cat /dev/null > all_data.tmp")
do for [i=1:n_ensembles] {
    ens_dir = system(sprintf("ls -d [0-9][0-9][0-9] | sed -n '%dp' | tr -d '\n'", i))
    
    if (int(system(sprintf("[ -s %s_paths.tmp ] && echo 1 || echo 0", ens_dir)))) {
        system(sprintf("cat %s_paths.tmp >> all_data.tmp", ens_dir))
    }
}

# Calculate axes ranges
# X axis range (order parameter for 1D, order parameter 1 for 2D)
if (is_1d) {
    system("awk '{if($1+0==$1)print $1}' all_data.tmp | sort -n | head -n1 > xmin.tmp")
    system("awk '{if($1+0==$1)print $1}' all_data.tmp | sort -n | tail -n1 > xmax.tmp")
} else {
    system("awk '{if($2+0==$2)print $2}' all_data.tmp | sort -n | head -n1 > xmin.tmp")
    system("awk '{if($2+0==$2)print $2}' all_data.tmp | sort -n | tail -n1 > xmax.tmp")
}

# Y axis range (time for 1D, order parameter 2 for 2D)
if (is_1d) {
    system("awk '{if($2+0==$2)print $2}' all_data.tmp | sort -n | head -n1 > ymin.tmp")
    system("awk '{if($2+0==$2)print $2}' all_data.tmp | sort -n | tail -n1 > ymax.tmp")
} else {
    system("awk '{if($3+0==$3)print $3}' all_data.tmp | sort -n | head -n1 > ymin.tmp")
    system("awk '{if($3+0==$3)print $3}' all_data.tmp | sort -n | tail -n1 > ymax.tmp")
}

# Read the ranges
xmin = system("cat xmin.tmp") + 0
xmax = system("cat xmax.tmp") + 0
ymin = system("cat ymin.tmp") + 0
ymax = system("cat ymax.tmp") + 0

# Add some margin to the ranges (5%)
xrange = xmax - xmin
yrange = ymax - ymin
xmin = xmin - 0.05 * xrange
xmax = xmax + 0.05 * xrange
ymin = ymin - 0.05 * yrange
ymax = ymax + 0.05 * yrange

print sprintf("Plot ranges: X=[%f:%f], Y=[%f:%f]", xmin, xmax, ymin, ymax)

# Set the ranges
set xrange [xmin:xmax]
set yrange [ymin:ymax]

# Extract start and end points of each path
do for [i=1:n_ensembles] {
    ens_dir = system(sprintf("ls -d [0-9][0-9][0-9] | sed -n '%dp' | tr -d '\n'", i))
    
    if (int(system(sprintf("[ -s %s_paths.tmp ] && echo 1 || echo 0", ens_dir)))) {
        # Split paths into separate files
        system(sprintf("awk '/^$/{n++; next} {print > \"%s_path_\"n\".tmp\"}' n=0 %s_paths.tmp", ens_dir, ens_dir))
        
        # Count how many path files we have
        num_paths_cmd = sprintf("ls -1 %s_path_*.tmp 2>/dev/null | wc -l", ens_dir)
        num_paths = system(num_paths_cmd) + 0
        
        # Create empty output files
        system(sprintf("cat /dev/null > %s_start_points.tmp", ens_dir))
        system(sprintf("cat /dev/null > %s_end_points.tmp", ens_dir))
        system(sprintf("cat /dev/null > %s_min_points.tmp", ens_dir))
        system(sprintf("cat /dev/null > %s_max_points.tmp", ens_dir))
        
        # Process each path file to get start, end, min, max
        j = 0
        while (j < num_paths) {
            path_file = sprintf("%s_path_%d.tmp", ens_dir, j)
            
            # Extract first line (start point)
            system(sprintf("head -n1 %s >> %s_start_points.tmp", path_file, ens_dir))
            
            # Extract last line (end point)
            system(sprintf("tail -n1 %s >> %s_end_points.tmp", path_file, ens_dir))
            
            # Find min value line
            system(sprintf("sort -nk2 %s | head -n1 >> %s_min_points.tmp", path_file, ens_dir))
            
            # Find max value line
            system(sprintf("sort -rnk2 %s | head -n1 >> %s_max_points.tmp", path_file, ens_dir))
            
            j = j + 1
        }
    }
}

# Define vibrant colors for paths
# These colors are chosen to be distinct and visible
colors = "dark-blue red forest-green dark-violet orange dark-cyan navy purple goldenrod coral \
          sea-green magenta midnight-blue dark-orange salmon dark-red web-green blue-violet \
          yellow-orange turquoise royal-blue orchid olive brown light-coral dark-khaki \
          steel-blue crimson dark-olive-green medium-purple dark-goldenrod"

# Build the plot command
plot_cmd = "plot "
first = 1

# Add paths to plot command
do for [i=1:n_ensembles] {
    ens_dir = system(sprintf("ls -d [0-9][0-9][0-9] | sed -n '%dp' | tr -d '\n'", i))
    
    if (int(system(sprintf("[ -s %s_paths.tmp ] && echo 1 || echo 0", ens_dir)))) {
        # Get color for this ensemble
        ensemble_color = word(colors, (i % words(colors)) + 1)
        
        # First add the paths with distinct colors
        if (first) {
            plot_cmd = plot_cmd.sprintf("'%s_paths.tmp' %s title '%s' with lines lw 0.7 lc rgb '%s'", \
                                      ens_dir, using_str, ens_dir, ensemble_color)
            first = 0
        } else {
            plot_cmd = plot_cmd.sprintf(", '%s_paths.tmp' %s title '%s' with lines lw 0.7 lc rgb '%s'", \
                                      ens_dir, using_str, ens_dir, ensemble_color)
        }
        
        # Then add special points
        if (int(system(sprintf("[ -s %s_start_points.tmp ] && echo 1 || echo 0", ens_dir)))) {
            plot_cmd = plot_cmd.sprintf(", '%s_start_points.tmp' %s notitle with points pt 7 ps 0.6 lc rgb 'dark-green'", \
                                      ens_dir, using_str)
        }
        
        if (int(system(sprintf("[ -s %s_end_points.tmp ] && echo 1 || echo 0", ens_dir)))) {
            plot_cmd = plot_cmd.sprintf(", '%s_end_points.tmp' %s notitle with points pt 5 ps 0.6 lc rgb 'red'", \
                                      ens_dir, using_str)
        }
        
        if (int(system(sprintf("[ -s %s_min_points.tmp ] && echo 1 || echo 0", ens_dir)))) {
            plot_cmd = plot_cmd.sprintf(", '%s_min_points.tmp' %s notitle with points pt 11 ps 0.6 lc rgb 'blue'", \
                                      ens_dir, using_str)
        }
        
        if (int(system(sprintf("[ -s %s_max_points.tmp ] && echo 1 || echo 0", ens_dir)))) {
            plot_cmd = plot_cmd.sprintf(", '%s_max_points.tmp' %s notitle with points pt 9 ps 0.6 lc rgb 'magenta'", \
                                      ens_dir, using_str)
        }
    }
}

# Add interfaces to plot command if available
if (has_interfaces && n_interfaces > 0) {
    if (is_1d) {
        # For 1D plot: Add interfaces as black dashed vertical lines
        do for [i=1:n_interfaces] {
            val = word(interface_values, i) + 0
            # Only add interfaces that are within or close to the plot range
            if (val >= xmin - 0.1*xrange && val <= xmax + 0.1*xrange) {
                plot_cmd = plot_cmd.sprintf(", %f notitle with lines lt -1 lw 1 dt 2", val)
            }
        }
    } else {
        # For 2D plot: Add interfaces as black dashed vertical lines across order parameter 1 range
        # Make vertical data files for each interface
        do for [i=1:n_interfaces] {
            val = word(interface_values, i) + 0
            # Only add interfaces that are within or close to the plot range
            if (val >= xmin - 0.1*xrange && val <= xmax + 0.1*xrange) {
                system(sprintf("echo '%f %f\n%f %f' > interface_%d.tmp", val, ymin, val, ymax, i))
                plot_cmd = plot_cmd.sprintf(", 'interface_%d.tmp' using 1:2 notitle with lines lt -1 lw 1 dt 2", i)
            }
        }
    }
}

# Execute plots
if (first) {
    print "No paths found to plot"
    set title sprintf("No paths found with %.2f ≤ OP ≤ %.2f", MINOP, MAXOP)
    plot -1 notitle
} else {
    print "Creating plots..."
    eval(plot_cmd)
    
    # Add legend for the symbols
    set label "● Start point" at graph 0.02, graph 0.97 tc rgb 'dark-green'
    set label "■ End point" at graph 0.02, graph 0.94 tc rgb 'red'
    set label "▼ Min OP" at graph 0.02, graph 0.91 tc rgb 'blue'
    set label "▲ Max OP" at graph 0.02, graph 0.88 tc rgb 'magenta'
    
    print "Creating PDF output..."
    set terminal pdfcairo size 8,6 enhanced font "Arial,11"
    set output 'paths.pdf'
    replot
    
    print "Creating interactive display..."
    set terminal qt persist
    set output
    replot
}

print "Cleaning up..."
system("rm -f all_paths.tmp selected_paths.tmp *_cycles.tmp *_paths.tmp *_start_points.tmp *_end_points.tmp *_minmax_points.tmp *.tmp")
system("rm -f first_order.tmp rstfile.tmp logfile.tmp interfaces.tmp interface_values.tmp")
system("rm -f bounds.tmp interface_*.tmp all_data.tmp xmin.tmp xmax.tmp ymin.tmp ymax.tmp")

print "Done!"
