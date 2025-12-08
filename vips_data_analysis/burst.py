import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import Analysis


class Burst:
    """
    Burst pressure testing class.

    Responsibilities:
    - Load burst test CSV files into data_dict.
    - Clean and normalize data into filtered_dict.
    - Plot Flow vs Time and Pressure vs Time on twin y-axes.
    - Use Analysis.estimate_burst_pressure to estimate burst pressure
      for each run and write a small text report.
    """

    def __init__(self, parent_directory: str, date: str, col_names: dict):
        """
        Parameters
        ----------
        parent_directory : str
            Root directory (expects data/graphs/analysis under this).
        date : str
            Date string used to construct subdirectories (e.g., "11-20-2025").
        col_names : dict
            Mapping for time / flow / pressure columns, e.g.:

            {
                "TIME"  : "Time",
                "FLOW"  : "MFCS-EZ (884) #2",
                "PRESS" : "Pressure Sensor #1",
            }
        """
        self.COL_NAMES     = col_names
        self.data_dict     = {}
        self.filtered_dict = {}

        # Directory layout mirrors Push / Push_Pull
        self.graph_str     = parent_directory + "/graphs/"   + date + "/"
        self.analysis_str  = parent_directory + "/analysis/" + date + "/"
        self.directory_str = parent_directory + "/data/"     + date + "/"

        # Init Analysis (reused for ANOVA + burst estimation)
        self.analysis = Analysis(self.analysis_str, col_names)

        # Ensure top-level graph dir exists
        if not os.path.isdir(self.graph_str):
            os.makedirs(self.graph_str)

        # Ensure per-chip graph dirs exist if data dir exists
        if os.path.isdir(self.directory_str):
            for chip in os.listdir(self.directory_str):
                chip_graph_dir = self.graph_str + chip + "/"
                if not os.path.isdir(chip_graph_dir):
                    os.makedirs(chip_graph_dir)

    # ── I/O: load + clean ────────────────────────────────────────────────────
    def load_data(self) -> None:
        """
        Load CSVs from /data/<date>/<chip>/ into self.data_dict[chip] as DataFrames.
        """
        self.data_dict = {}

        if not os.path.isdir(self.directory_str):
            return

        for chip in os.listdir(self.directory_str):
            chip_dir = self.directory_str + chip + "/"
            if not os.path.isdir(chip_dir):
                continue

            self.data_dict[chip] = []
            for fname in os.listdir(chip_dir):
                if not fname.lower().endswith(".csv"):
                    continue
                path = chip_dir + fname
                self.data_dict[chip].append(pd.read_csv(path, delimiter=";"))

    def data_cleaning(self) -> None:
        """
        Process raw data into a cleaned, analysis-ready form:
        - Parse time as datetime and convert to elapsed seconds.
        - Convert flow and pressure to numeric.
        - Clamp negative pressure values to 0 (invalid for burst tests).
        - Drop rows with missing values.
        """
        time_col  = self.COL_NAMES["TIME"]
        flow_col  = self.COL_NAMES["FLOW"]
        press_col = self.COL_NAMES["PRESS"]

        self.filtered_dict = {}

        for chip, df_list in self.data_dict.items():
            self.filtered_dict[chip] = []

            for df in df_list:
                df = df.copy()

                # Parse time column as datetime
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

                # Convert to elapsed seconds (relative to first timestamp)
                df[time_col] = (
                    df[time_col] - df[time_col].iloc[0]
                ).dt.total_seconds()

                # Convert flow + pressure to numeric
                for col in [flow_col, press_col]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Clamp negative flows and pressures to 0 (invalid readings in burst tests)
                df[flow_col] = df[flow_col].clip(lower=0)
                df[press_col] = df[press_col].clip(lower=0)

                # Keep only the relevant columns and drop rows with NaNs
                df_clean = df[[time_col, flow_col, press_col]].dropna()

                self.filtered_dict[chip].append(df_clean)

    # ── plotting (matplotlib) ────────────────────────────────────────────────
    def _build_plot(self, df: pd.DataFrame, chip: str, index: int) -> None:
        """
        Matplotlib-based plot: Flow vs Time and Pressure vs Time
        with separate y-axes (twin axes). No forced centering or zero line.
        """
        time_col  = self.COL_NAMES["TIME"]
        flow_col  = self.COL_NAMES["FLOW"]
        press_col = self.COL_NAMES["PRESS"]

        t     = df[time_col].values
        flow  = df[flow_col].values
        press = df[press_col].values

        # Create figure and twin axes
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=400)
        ax2 = ax1.twinx()

        # Flow on left axis
        flow_line = ax1.plot(
            t,
            flow,
            linestyle="-",
            linewidth=1.5,
            color="tab:blue",
            label="Flow",
        )[0]

        # Pressure on right axis
        press_line = ax2.plot(
            t,
            press,
            linestyle="-",
            linewidth=1.5,
            color="tab:red",
            label="Pressure",
        )[0]

        # Labels and title
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Flow rate (µL/min)")
        ax2.set_ylabel("Pressure (mbar)")
        ax1.set_title("Burst Test: Flow & Pressure vs Time")

        # Combined legend
        lines = [flow_line, press_line]
        labels = ["Flow", "Pressure"]
        ax1.legend(lines, labels, loc="upper left")

        fig.tight_layout()

        out_path = f"{self.graph_str}{chip}/graph_{index}.png"
        fig.savefig(out_path)
        plt.close(fig)

    # ── public plotting + analysis API ───────────────────────────────────────
    def plot_all(self) -> None:
        """
        Loop through all cleaned dataframes in self.filtered_dict, build
        plots, estimate burst pressure for each run, and run a one-way ANOVA
        on flow across all traces.
        """
        all_groups = []

        for chip, df_list in self.filtered_dict.items():
            for i, df in enumerate(df_list, start=1):
                # Ensure per-chip graph directory exists
                chip_dir = self.graph_str + chip + "/"
                if not os.path.isdir(chip_dir):
                    os.makedirs(chip_dir)

                # Plot this run
                self._build_plot(df, chip, i)

                # Collect flow data for ANOVA
                all_groups.append(df[self.COL_NAMES["FLOW"]].values)

                # Estimate burst pressure for this particular run
                # Uses Analysis.estimate_burst_pressure (must be defined there)
                self.analysis.estimate_burst_pressure(
                    df,
                    chip=chip,
                    press_key="PRESS",   # resolves via self.COL_NAMES["PRESS"]
                    time_key="TIME",     # resolves via self.COL_NAMES["TIME"]
                    run_label=str(i),    # e.g., <chip>_burst_pressure_1.txt
                )

        # Run ANOVA on flow across all runs (optional but consistent)
        if all_groups:
            self.analysis.anova_test(all_groups)

    def plot(
        self,
        df: pd.DataFrame,
        chip: str = "single",
        index: int = 1,
        run_stats: bool = False,
        estimate_burst: bool = True,
    ) -> None:
        """
        Plot from a single dataframe (e.g., for ad-hoc burst tests).

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataframe containing TIME, FLOW, PRESS columns.
        chip : str
            Label used for the output subdirectory.
        index : int
            Index used in the filename (graph_<index>.png).
        run_stats : bool
            If True, run ANOVA on just this trace's flow data.
        estimate_burst : bool
            If True, estimate burst pressure for this single run.
        """
        chip_dir = self.graph_str + chip + "/"
        if not os.path.isdir(chip_dir):
            os.makedirs(chip_dir)

        # Plot the single run
        self._build_plot(df, chip, index)

        # Optional ANOVA on just this run (mainly for consistency/debug)
        if run_stats:
            self.analysis.anova_test([df[self.COL_NAMES["FLOW"]].values])

        # Optional burst estimation
        if estimate_burst:
            self.analysis.estimate_burst_pressure(
                df,
                chip=chip,
                press_key="PRESS",
                time_key="TIME",
                run_label=str(index),
            )

    # GETTER to match Push / Push_Pull style
    def get_data_dict(self):
        """
        Return the cleaned data dictionary if available, otherwise the raw.
        """
        return self.filtered_dict if self.filtered_dict else self.data_dict
