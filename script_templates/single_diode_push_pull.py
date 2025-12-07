from pathlib import Path
from vips_data_analysis import Push_Pull  # assuming you export it

col_names = {
    "TIME"   : "Time",
    "FLOW"   : "MFCS-EZ (884) #2",
    "PRESS1" : "MFCS-EZ (884) #2",
    "PRESS2" : "MFCS-EZ (884) #3",
}

push_pull = Push_Pull(Path.cwd().__str__(), "10-24-2025", col_names)
push_pull.load_data()
push_pull.data_cleaning()
push_pull.plot_all()
