# Import libraries
from pathlib import Path
from vips_data_analysis import Transistor

# Define the date
date = "2026-03-31"

# Init the transistor analysis class
# Defaults for this dataset:
# - input = column F
# - output = column J
transistor = Transistor(Path.cwd().__str__(), date)

# Load the data
transistor.load_data()

# Clean the data
transistor.data_cleaning()

# Dictionary to define graph properties
properties = {
    "x_label": "Gate Pressure (mbar)",
    "y_label": "Flow Rate (µL/min)"
}

# Plot the data
transistor.plot_all(properties)
