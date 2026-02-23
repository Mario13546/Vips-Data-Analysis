# Import libraries
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vips_data_analysis import Transistor

# Init the transistor analysis class
# Defaults for this dataset:
# - input = column F
# - output = column J
transistor = Transistor(Path.cwd().__str__(), "2-22-2026")

# Load the data
transistor.load_data()

# Clean the data
transistor.data_cleaning()

# Plot the data
transistor.plot_all()
