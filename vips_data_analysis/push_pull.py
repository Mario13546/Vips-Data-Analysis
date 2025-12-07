import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple

from scipy.optimize import curve_fit
from plotnine import (
    ggplot, aes, geom_point, geom_line, labs, theme_bw, theme,
    scale_color_manual, guides, guide_legend, element_text
)

from . import Analysis


class Push_Pull:
    """
    Variant of the Push-style class for time-series experiments with
    sinusoidal flow and pressure (flow & pressure vs time with sine fits).
    """

    def __init__(self, parent_directory: str, date: str, col_names: dict):
        self.COL_NAMES     = col_names
        self.data_dict     = {}
        self.filtered_dict = {}

        # Directory layout mirrors Push
        self.graph_str     = parent_directory + "/graphs/"   + date + "/"
        self.analysis_str  = parent_directory + "/analysis/" + date + "/"
        self.directory_str = parent_directory + "/data/"     + date + "/"

        # Init Analysis (we reuse the existing class just for ANOVA here)
        self.analysis = Analysis(self.analysis_str, col_names)

        # Create per-chip graph subdirectories if needed
        if os.path.isdir(self.graph_str) is False:
            os.makedirs(self.graph_str)

        if os.path.isdir(self.directory_str):
            for chip in os.listdir(self.directory_str):
                chip_graph_dir = self.graph_str + chip + "/"
                if not os.path.isdir(chip_graph_dir):
                    os.makedirs(chip_graph_dir)

    # ── helpers: sinusoidal fit ───────────────────────────────────────────────

    @staticmethod
    def _sinusoid(t: np.ndarray, A: float, f: float, phi: float, C: float) -> np.ndarray:
        """A simple sinusoidal model: A·sin(2π·f·t + φ) + C"""
        return A * np.sin(2 * np.pi * f * t + phi) + C

    def fit_sine(self, t, y) -> Tuple[np.ndarray, str]:
        """
        Fit a sinusoidal curve to time-series data.
        Returns (fit_line, equation_string).
        """
        # Drop NaNs and align arrays
        mask = ~np.isnan(t) & ~np.isnan(y)
        t = np.asarray(t[mask], dtype=float)
        y = np.asarray(y[mask], dtype=float)

        if len(y) == 0:
            return np.zeros_like(t), "No valid data"

        # Initial guesses
        guess_A = (y.max() - y.min()) / 2
        guess_C = y.mean()

        # Estimate frequency using FFT for a better initial guess
        dt = np.mean(np.diff(t))
        freqs = np.fft.fftfreq(len(t), d=dt)
        fft_magnitude = np.abs(np.fft.fft(y - guess_C))

        # Ignore the zero-frequency component (DC)
        guess_f = abs(freqs[np.argmax(fft_magnitude[1:]) + 1])
        if guess_f == 0 or np.isnan(guess_f):
            # Fallback: 1 period across full time span
            guess_f = 1.0 / (t.max() - t.min())

        guess_phi = 0.0
        p0 = [guess_A, guess_f, guess_phi, guess_C]

        try:
            popt, _ = curve_fit(self._sinusoid, t, y, p0=p0, maxfev=10000)
        except RuntimeError:
            fit_line = self._sinusoid(t, *p0)
            eqn = (
                f"Fit failed — using guess: "
                f"{guess_A:.2f}·sin(2π·{guess_f:.3f}·t+{guess_phi:.2f})+{guess_C:.2f}"
            )
            return fit_line, eqn

        A, f, phi, C = popt
        fit_line = self._sinusoid(t, *popt)
        eqn = f"{A:.2f}·sin(2π·{f:.3f}·t+{phi:.2f})+{C:.2f}"
        return fit_line, eqn

    # ── I/O: load + clean ────────────────────────────────────────────────────

    def load_data(self) -> None:
        """Load CSVs into self.data_dict[chip] as DataFrames."""
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
        - Combine two pressure columns into a single bipolar signal.
        - Convert flow and pressure to numeric.
        - Drop rows with missing values.
        """
        time_col   = self.COL_NAMES["TIME"]
        flow_col   = self.COL_NAMES["FLOW"]
        press1_col = self.COL_NAMES["PRESS1"]
        press2_col = self.COL_NAMES["PRESS2"]

        # Store the combined pressure under a consistent name in COL_NAMES
        self.COL_NAMES["PRESS"] = "CombinedPressure"
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

                # Combine the two pressure columns into a single bipolar signal
                df[press_col] = df[press1_col] - df[press2_col]

                # Convert flow + pressure to numeric
                for col in [flow_col, press_col]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Keep only the relevant columns and drop rows with NaNs
                df_clean = df[[time_col, flow_col, press_col]].dropna()
                self.filtered_dict[chip].append(df_clean)

    # ── plotting (matplotlib) ────────────────────────────────────────────────
    def _build_dual_axis_plot(
        self,
        df: pd.DataFrame,
        chip: str,
        index: int,
        with_regression: bool = False,
    ) -> None:
        """
        Matplotlib-based plot: Flow vs Time and Pressure vs Time
        with separate y-axes (twin axes), zero-centered limits,
        bold zero line, and data drawn as lines instead of points.
        """
        time_col  = self.COL_NAMES["TIME"]
        flow_col  = self.COL_NAMES["FLOW"]
        press_col = self.COL_NAMES["PRESS"]

        t     = df[time_col].values
        flow  = df[flow_col].values
        press = df[press_col].values

        # Optional sinusoidal fits
        if with_regression:
            flow_fit,  flow_eq  = self.fit_sine(t, flow)
            press_fit, press_eq = self.fit_sine(t, press)
        else:
            flow_fit = press_fit = None
            flow_eq = press_eq = ""

        # Zero-centered axis limits
        flow_max_mag  = max(abs(flow.min()), abs(flow.max()))
        press_max_mag = max(abs(press.min()), abs(press.max()))

        flow_limits  = (-flow_max_mag * 1.05,  flow_max_mag * 1.05)
        press_limits = (-press_max_mag * 1.05, press_max_mag * 1.05)

        # Create figure and twin axes
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=400)
        ax2 = ax1.twinx()

        # ----------------------------
        # Draw bold zero lines on BOTH axes
        # ----------------------------
        ax1.axhline(0, color="black", linewidth=1.0)
        ax2.axhline(0, color="black", linewidth=1.0)

        # ----------------------------
        # Flow data (LINE instead of points)
        # ----------------------------
        flow_data_line = ax1.plot(
            t,
            flow,
            linestyle="-",
            linewidth=1.5,
            color="tab:blue",
            label="Flow Data",
        )[0]

        # Optional regression fit for flow
        if with_regression:
            flow_fit_line = ax1.plot(
                t,
                flow_fit,
                linestyle="--",
                linewidth=1.5,
                color="tab:cyan",
                label="Flow Fit",
            )[0]
        else:
            flow_fit_line = None

        # ----------------------------
        # Pressure data (LINE instead of points)
        # ----------------------------
        press_data_line = ax2.plot(
            t,
            press,
            linestyle="-",
            linewidth=1.5,
            color="tab:red",
            label="Pressure Data",
        )[0]

        # Optional regression fit for pressure
        if with_regression:
            press_fit_line = ax2.plot(
                t,
                press_fit,
                linestyle="--",
                linewidth=1.5,
                color="tab:orange",
                label="Pressure Fit",
            )[0]
        else:
            press_fit_line = None

        # Apply axis limits
        ax1.set_ylim(flow_limits)
        ax2.set_ylim(press_limits)

        # Labels and title
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Flow rate (µL/min)")
        ax2.set_ylabel("Pressure (mbar)")
        ax1.set_title("Flow & Pressure vs Time")

        # Build combined legend
        lines = [flow_data_line, press_data_line]
        labels = ["Flow Data", "Pressure Data"]

        if with_regression:
            lines.extend([flow_fit_line, press_fit_line])
            labels.extend(["Flow Fit", "Pressure Fit"])

        ax1.legend(lines, labels, loc="best")

        # Caption with regression equations
        if with_regression:
            caption = f"Flow fit: {flow_eq}    Pressure fit: {press_eq}"
            fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8)

        fig.tight_layout(rect=[0, 0.03 if with_regression else 0, 1, 1])

        out_path = f"{self.graph_str}{chip}/graph_{index}.png"
        fig.savefig(out_path)
        plt.close(fig)


    # ── original plotnine version (kept for future use) ──────────────────────
    def _build_dual_axis_plot_plotnine(self, df: pd.DataFrame, chip: str, index: int) -> None:
        """
        Original plotnine-based Flow & Pressure vs Time plot.
        Kept here so it can be re-enabled when plotnine supports multiple axes.
        """
        t        = df[self.COL_NAMES["TIME"]]
        flow     = df[self.COL_NAMES["FLOW"]]
        press    = df[self.COL_NAMES["PRESS"]]

        flow_fit,  flow_eq  = self.fit_sine(t, flow)
        press_fit, press_eq = self.fit_sine(t, press)

        dplot = pd.DataFrame({
            "t"            : t,
            "Flow"         : flow,
            "Flow-fit"     : flow_fit,
            "Pressure"     : press,
            "Pressure-fit" : press_fit,
        })

        p = (
            ggplot(dplot, aes("t"))
            + geom_point(aes(y="Flow",     color="'Flow Data'"),     size=0.8)
            + geom_line (aes(y="Flow-fit", color="'Flow Fit'"),      linetype="dashed")
            + geom_point(aes(y="Pressure", color="'Pressure Data'"), size=0.8)
            + geom_line (aes(y="Pressure-fit", color="'Pressure Fit'"), linetype="dashed")
            + scale_color_manual(
                values={
                    "Flow Data"     : "#1fa3b4",
                    "Flow Fit"      : "#08519c",
                    "Pressure Data" : "#d62728",
                    "Pressure Fit"  : "#7f0000",
                }
            )
            + labs(
                x="Time (s)",
                y="Flow rate (µL/min) / Pressure (mbar)",
                title="Flow & Pressure vs Time",
                color="Data Series",
                caption=f"Flow fit: {flow_eq}    Pressure fit: {press_eq}",
            )
            + guides(
                color=guide_legend(title="Legend")
            )
            + theme_bw()
            + theme(
                plot_caption=element_text(size=8, color="gray", ha="center")
            )
        )

        out_path = f"{self.graph_str}{chip}/graph_{index}.png"
        p.save(out_path, width=12, height=6, dpi=400)

    # ── public plotting API ───────────────────────────────────────────────────
    def plot_all(self, with_regression: bool = False) -> None:
        """
        Loop through all cleaned dataframes in self.filtered_dict, build
        plots, and run a one-way ANOVA on flow across all traces.

        Set with_regression=True to include sinusoidal fits.
        """
        all_groups = []

        for chip, df_list in self.filtered_dict.items():
            for i, df in enumerate(df_list, start=1):
                self._build_dual_axis_plot(df, chip, i, with_regression=with_regression)
                all_groups.append(df[self.COL_NAMES["FLOW"]].values)

        if all_groups:
            self.analysis.anova_test(all_groups)

    def plot(
        self,
        df: pd.DataFrame,
        chip: str = "single",
        index: int = 1,
        run_stats: bool = False,
        with_regression: bool = False,
    ) -> None:
        """
        Plot from a single dataframe (e.g., for ad-hoc tests).

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
        with_regression : bool
            If True, overlay sinusoidal fits.
        """
        chip_dir = self.graph_str + chip + "/"
        if not os.path.isdir(chip_dir):
            os.makedirs(chip_dir)

        self._build_dual_axis_plot(df, chip, index, with_regression=with_regression)

        if run_stats:
            self.analysis.anova_test([df[self.COL_NAMES["FLOW"]].values])

    # GETTER to match Push style
    def get_data_dict(self):
        """Return the cleaned data dictionary if available, otherwise the raw."""
        return self.filtered_dict if self.filtered_dict else self.data_dict
