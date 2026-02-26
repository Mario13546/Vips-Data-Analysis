# Import libraries
from pathlib import Path
from vips_data_analysis import Push_Pull

# Define the date
date = "12-19-2025"

# Define the column names
col_names = {
    "TIME"   : "Time",
    "FLOW"   : "Flow Unit #2 [Flowboard (340)]",
    "PRESS1" : "MFCS-EZ (884) #2",
    "PRESS2" : "MFCS-EZ (884) #3",
}

# Init the generic class
push_pull = Push_Pull(Path.cwd().__str__(), date, col_names)

# Load the data
push_pull.load_data()

# Clean the data
push_pull.data_cleaning()

# Plot the data
push_pull.plot_all()
