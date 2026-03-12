# Import libraries
from pathlib import Path
from vips_data_analysis import Burst

# Define the date
date = "2025-11-20"

# Define the column names
col_names = {
    "TIME":  "Time",
    "FLOW":  "Flow Unit #2 [Flowboard (340)]",
    "PRESS": "MFCS-EZ (884) #2",
}

# Init the generic class
burst = Burst(Path.cwd().__str__(), date, col_names)

# Load the data
burst.load_data()

# Clean the data
burst.data_cleaning()

# Plot the data
burst.plot_all()
