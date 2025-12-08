# Import libraries
from pathlib import Path
from vips_data_analysis import Push

# Define the column names
col_names = {
    "X_COL_1": "Flow EZ #2 (12807)",
    "Y_COL_1": "Flow Unit #1 [Flow EZ #2 (12807)]",
}

# Init the generic class
push = Push(Path.cwd().__str__(), "08-27-2025", col_names)

# Load the data
push.load_data()

# Clean the data
push.data_cleaning()

# Plot the data
push.plot_all()
