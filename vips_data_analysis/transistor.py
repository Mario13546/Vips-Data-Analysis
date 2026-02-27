import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import Analysis


class Transistor:
    """
    Transistor test workflow.

    Defaults are set for your 02-22-2026 dataset:
    - Input signal from column F (index 5)
    - Output signal from column J (index 9)
    """

    def __init__(
        self,
        parent_directory: str,
        date: str = "02-22-2026",
        input_col_index: int = 5,
        output_col_index: int = 9,
        time_col_index: int = 0,
    ):
        self.data_dict = {}
        self.filtered_dict = {}

        self.input_col_index = input_col_index
        self.output_col_index = output_col_index
        self.time_col_index = time_col_index

        self.graph_str = parent_directory + "/graphs/" + date + "/"
        self.analysis_str = parent_directory + "/analysis/" + date + "/"
        self.directory_str = parent_directory + "/data/" + date + "/"

        self.COL_NAMES = {
            "TIME": "TimeSeconds",
            "INPUT": "Input_F",
            "OUTPUT": "Output_J",
        }

        self.analysis = Analysis(self.analysis_str, self.COL_NAMES)

        if not os.path.isdir(self.graph_str):
            os.makedirs(self.graph_str)

    def load_data(self) -> None:
        """
        Load CSV files from /data/<date>/.

        Supports both:
        - flat file layout: /data/<date>/*.csv
        - chip subfolders: /data/<date>/<chip>/*.csv
        """
        self.data_dict = {}

        if not os.path.isdir(self.directory_str):
            return

        # Flat layout in the date folder
        flat_csvs = [
            f for f in os.listdir(self.directory_str)
            if f.lower().endswith(".csv")
        ]
        if flat_csvs:
            self.data_dict["chip_1"] = []
            for fname in flat_csvs:
                path = self.directory_str + fname
                self.data_dict["chip_1"].append(pd.read_csv(path))

        # Optional chip-subfolder layout
        for chip in os.listdir(self.directory_str):
            chip_dir = self.directory_str + chip + "/"
            if not os.path.isdir(chip_dir):
                continue

            self.data_dict[chip] = []
            for fname in os.listdir(chip_dir):
                if not fname.lower().endswith(".csv"):
                    continue
                path = chip_dir + fname
                self.data_dict[chip].append(pd.read_csv(path))

    @staticmethod
    def _parse_time_to_seconds(series: pd.Series) -> pd.Series:
        """
        Parse time values (e.g. '35:42.8') to elapsed seconds.
        """
        s = series.astype(str).str.strip()
        out = pd.Series(np.nan, index=series.index, dtype=float)

        # Handle MM:SS(.sss) explicitly (common in Fluigent exports).
        mmss_mask = s.str.match(r"^\d+:\d+(?:\.\d+)?$")
        if mmss_mask.any():
            parts = s[mmss_mask].str.split(":", n=1, expand=True)
            mins = pd.to_numeric(parts[0], errors="coerce")
            secs = pd.to_numeric(parts[1], errors="coerce")
            out.loc[mmss_mask] = mins * 60.0 + secs

        # Handle standard HH:MM:SS(.sss) style values.
        remaining = out.isna()
        if remaining.any():
            td = pd.to_timedelta(s[remaining], errors="coerce")
            out.loc[remaining] = td.dt.total_seconds()

        # Handle full datetimes if present.
        remaining = out.isna()
        if remaining.any():
            dt = pd.to_datetime(s[remaining], errors="coerce", format="mixed")
            if dt.notna().any():
                first_valid = dt.dropna().iloc[0]
                out.loc[remaining] = (dt - first_valid).dt.total_seconds()

        # Final fallback: numeric time column.
        remaining = out.isna()
        if remaining.any():
            out.loc[remaining] = pd.to_numeric(s[remaining], errors="coerce")

        if out.notna().any():
            return out - out.dropna().iloc[0]

        return out

    def data_cleaning(self) -> None:
        """
        Build cleaned dataframes with only:
        - TIME (seconds)
        - INPUT (column F)
        - OUTPUT (column J)
        """
        self.filtered_dict = {}

        for chip, df_list in self.data_dict.items():
            self.filtered_dict[chip] = []

            for raw_df in df_list:
                if raw_df.shape[1] <= max(self.time_col_index, self.input_col_index, self.output_col_index):
                    continue

                df = raw_df.copy()

                t = self._parse_time_to_seconds(df.iloc[:, self.time_col_index])
                x = pd.to_numeric(df.iloc[:, self.input_col_index], errors="coerce")
                y = pd.to_numeric(df.iloc[:, self.output_col_index], errors="coerce")

                clean_df = pd.DataFrame({
                    self.COL_NAMES["TIME"]: t,
                    self.COL_NAMES["INPUT"]: x,
                    self.COL_NAMES["OUTPUT"]: y,
                }).dropna(subset=[self.COL_NAMES["INPUT"], self.COL_NAMES["OUTPUT"]])

                self.filtered_dict[chip].append(clean_df)

    def _build_plot(self, df: pd.DataFrame, chip: str, index: int) -> None:
        """
        Plot output vs input for one run.
        """
        x_col = self.COL_NAMES["INPUT"]
        y_col = self.COL_NAMES["OUTPUT"]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(df[x_col].values, df[y_col].values, ".", markersize=2, color="tab:blue")
        ax.set_xlabel("Input (column F)")
        ax.set_ylabel("Output (column J)")
        ax.set_title("Transistor Transfer: Output vs Input")
        ax.grid(True, alpha=0.3)

        chip_dir = self.graph_str + chip + "/"
        if not os.path.isdir(chip_dir):
            os.makedirs(chip_dir)

        out_path = chip_dir + f"graph_{index}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def plot_all(self, run_anova: bool = True) -> None:
        """
        Plot all cleaned runs and optionally run one-way ANOVA on output.
        """
        anova_groups = []

        for chip, df_list in self.filtered_dict.items():
            for i, df in enumerate(df_list, start=1):
                self._build_plot(df, chip, i)
                anova_groups.append(df[self.COL_NAMES["OUTPUT"]].values)

        if run_anova and anova_groups:
            self.analysis.anova_test(anova_groups)

    def plot(
        self,
        df: pd.DataFrame,
        chip: str = "single",
        index: int = 1,
        run_stats: bool = False,
    ) -> None:
        """
        Plot a single cleaned dataframe.
        """
        self._build_plot(df, chip, index)

        if run_stats:
            self.analysis.anova_test([df[self.COL_NAMES["OUTPUT"]].values])

    def get_data_dict(self):
        return self.filtered_dict if self.filtered_dict else self.data_dict
