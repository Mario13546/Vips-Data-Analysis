# Import libraries
from pathlib import Path
from vips_data_analysis import Transistor

# Init the transistor analysis class
# Defaults for this dataset:
# - input = column F
# - output = column J
transistor = Transistor(Path.cwd().__str__(), "02-22-2026")

# Load the data
transistor.load_data()

# Clean the data
transistor.data_cleaning()

# Plot the data
transistor.plot_all()
