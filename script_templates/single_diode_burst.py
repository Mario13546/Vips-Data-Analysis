from pathlib import Path
from vips_data_analysis import Burst

col_names = {
    "TIME":  "Time",
    "FLOW":  "Flow Unit #2 [Flowboard (340)]",
    "PRESS": "MFCS-EZ (884) #2",
}

burst = Burst(Path.cwd().__str__(), "11-20-2025", col_names)
burst.load_data()
burst.data_cleaning()
burst.plot_all()
