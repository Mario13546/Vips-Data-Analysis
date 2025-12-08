# Import libraries
import os
import math
import numpy as np
import pandas as pd
import scipy.stats as stats

from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_line,
    ggtitle,
    labs,
    annotate,
    xlim,
    ylim,
    scale_color_manual,
)
from typing import Tuple
from scipy.optimize import curve_fit

# Import custom classes
from . import Analysis


# Define the Push class
class Push:
    def __init__(self, parent_directory: str, date: str, col_names: dict):
        self.COL_NAMES      = col_names
        self.data_dict      = {}
        self.graph_str      = parent_directory + "/graphs/" + date + "/"
        self.analysis_str   = parent_directory + "/analysis/" + date + "/"
        self.directory_str  = parent_directory + "/data/" + date + "/"

        # Init the Analysis class
        self.analysis = Analysis(self.analysis_str, col_names)

        for dir in os.listdir(self.directory_str):
            tmp_graph_str     = self.graph_str + dir + "/"

            if os.path.isdir(tmp_graph_str) == False:
                os.makedirs(tmp_graph_str)

    def load_data(self):
        for dir in os.listdir(self.directory_str):
            self.data_dict[dir] = []

            tmp_directory_str = self.directory_str + dir + "/"

            for file in os.listdir(tmp_directory_str):
                self.data_dict[dir].append(pd.read_csv(tmp_directory_str + file, delimiter=";"))

    def data_cleaning(self):
        self.i_range = self.analysis.get_i_range()

        self.filtered_dict = {}

        COL_LIST = list(self.COL_NAMES.values())
        COL_LIST.append("Time")

        for key in self.data_dict.keys():
            self.filtered_dict[key] = []

            for df in self.data_dict[key]:
                self.filtered_dict[key].append(df[COL_LIST])

    def find_lin_fit(self, df: pd.DataFrame) -> list[str]:
        """
        Use scipy to find the linear fit and fitting parameters
        """
        lin_eqs = []

        for i in self.i_range:
            x_col = self.COL_NAMES[f"X_COL_{i}"]
            y_col = self.COL_NAMES[f"Y_COL_{i}"]
            lin_fit = f"lin_fit_{i}"

            a_lin, b_lin, r_lin, _, _ = stats.linregress(df[x_col], df[y_col])

            df[lin_fit] = a_lin * df[x_col] + b_lin

            y_true     = df[y_col]
            y_pred_lin = df[lin_fit]
            ss_res_lin = np.sum((y_true - y_pred_lin) ** 2)
            ss_tot_lin = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_lin = 1 - (ss_res_lin / ss_tot_lin)

            lin_eqs.append((
                f"Linear: y = {a_lin:.3f}x + {b_lin:.3f}\n"
                f"R² = {r2_lin:.3f}"
            ))

        return lin_eqs

    def find_logarithmic_fit(self, df: pd.DataFrame) -> list[str]:
        """
        Use scipy to find the logarithmic fit and fitting parameters
        """
        log_eqs = []

        for i in self.i_range:
            x_col = self.COL_NAMES[f"X_COL_{i}"]
            y_col = self.COL_NAMES[f"Y_COL_{i}"]
            log_fit = f"log_fit_{i}"

            log_x = np.log(df[x_col])
            y = df[y_col]

            a_log, b_log, r_log, _, _ = stats.linregress(log_x, y)

            df[log_fit] = a_log * log_x + b_log

            y_true     = df[y_col]
            y_pred_log = df[log_fit]
            ss_res_log = np.sum((y_true - y_pred_log) ** 2)
            ss_tot_log = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_log = 1 - (ss_res_log / ss_tot_log)

            if not np.isnan(a_log):
                log_eqs.append((
                    f"Log: y = {a_log:.3f} ln(x) + {b_log:.3f}\n"
                    f"R² = {r2_log:.3f}"
                ))
            else:
                df['log_fit'] = np.nan
                log_eqs.append("Logarithmic fit failed")

        return log_eqs
    
    def find_logistic_fit(self, df: pd.DataFrame) -> list[str]:
        """
        Use scipy to find the logistic fit and fitting parameters
        """
        logistic_eqs = []

        # Logistic function definition
        def logistic(x, L, k, x0):
            return L / (1 + np.exp(-k * (x - x0)))

        for i in self.i_range:
            x_col = self.COL_NAMES[f"X_COL_{i}"]
            y_col = self.COL_NAMES[f"Y_COL_{i}"]
            logistic_fit = f"logistic_fit_{i}"

            # Initial parameter guess: [max y, slope, midpoint]
            initial_guess = [df[y_col].max(), 1, df[x_col].median()]

            # Fit the logistic model
            try:
                params, _ = curve_fit(logistic, df[x_col], df[y_col], p0=initial_guess, maxfev=10000)
                L, k, x0_log = params
                df[logistic_fit] = logistic(df[x_col], L, k, x0_log)

                # Calculate R²
                y_true = df[y_col]
                y_pred = df[logistic_fit]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2_logistic = 1 - (ss_res / ss_tot)

                logistic_eqs.append((
                    f"Logistic: y = {L:.2f} / (1 + e^(-{k:.2f}(x - {x0_log:.2f})))\n"
                    f"R² = {r2_logistic:.3f}"
                ))
            except RuntimeError:
                df[logistic_fit] = np.nan
                logistic_eqs.append("Logistic fit failed")

        return logistic_eqs
    
    def _melt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal helper to melt regression-fit columns for legend support.
        Uses self.i_range and self.COL_NAMES.
        """
        # Build labels and value columns for linear + logistic fits
        name_vars = (
            [f"Linear Fit {i}" for i in self.i_range] +
            [f"Logistic Fit {i}" for i in self.i_range]
        )
        value_vars = (
            [f"lin_fit_{i}" for i in self.i_range] +
            [f"logistic_fit_{i}" for i in self.i_range]
        )

        df_melt = pd.melt(
            df,
            id_vars=list(self.COL_NAMES.values()),
            value_vars=value_vars,
            var_name="Fit Type",
            value_name="Fit Value"
        )

        # Map the raw column names to readable fit labels
        fit_labels = {value_vars[i]: name_vars[i] for i in range(len(name_vars))}
        df_melt["Fit Type"] = df_melt["Fit Type"].map(fit_labels)

        return df_melt

    def plot_all(self, with_regression: bool = False) -> None:
        """
        Plot all dataframes in self.filtered_dict(), preserving the original
        behavior (including chi-square per dataframe and a single ANOVA).
        """
        anova_data = []

        X_LABEL = "Pressure In (mb)"
        Y_LABEL = "Flow Rate Out (µL/min)"

        # Graph all the data
        for key, chip_list in self.filtered_dict.items():
            for i, df in enumerate(chip_list):
                if with_regression:
                    # Plot with regression fits
                    data_list, chi_df = self._plot_single_df_with_regression(
                        df, key, i, X_LABEL, Y_LABEL
                    )
                else:
                    # Plot plain scatter
                    data_list = self._plot_single_df(
                        df, key, i, X_LABEL, Y_LABEL
                    )
                    chi_df = df

                anova_data.extend(data_list)

                # Chi-square on either raw or melted df
                self.analysis.chi_square_test(chi_df, key)

        # Perform ANOVA analysis across all collected groups
        self.analysis.anova_test(anova_data)

    def plot(self, df: pd.DataFrame, key: str = "single", chip_index: int = 0, run_stats: bool = True) -> None:
        """
        Plot from a single dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to plot.
        key : str, optional
            Used to build the output path (e.g. self.graph_str + f"{key}/...").
            Defaults to "single".
        chip_index : int, optional
            Index used in the filename (graph_{chip_index}-{j}.png).
            Defaults to 0.
        run_stats : bool, optional
            If True, run chi-square and ANOVA on this dataframe's data.
        """
        X_LABEL = "Pressure In (mb)"
        Y_LABEL = "Flow Rate Out (µL/min)"

        data_list = self._plot_single_df(df, key, chip_index, X_LABEL, Y_LABEL)

        if run_stats:
            # Same stats pipeline as before, but only for this dataframe
            self.analysis.chi_square_test(df, key)
            self.analysis.anova_test(data_list)

    def _plot_single_df(self, df: pd.DataFrame, key: str, chip_index: int, X_LABEL: str, Y_LABEL: str):
        """
        Internal helper to plot all series for a single dataframe (one chip)
        and return the list of y-data series for ANOVA.
        """
        data_list = []

        for j in self.i_range:
            x_col = self.COL_NAMES[f"X_COL_{j}"]
            y_col = self.COL_NAMES[f"Y_COL_{j}"]

            # Compute axis limits for this series
            x_min = 0
            y_min = 0 if df[y_col].min() >= 0 else df[y_col].min()
            x_max = df[x_col].max()
            y_max = df[y_col].max()

            # Make the plot
            plot = (
                ggplot(df, aes(x_col, y_col)) +  # NOTE: kept your original aes usage
                geom_point() +
                ggtitle(f"{Y_LABEL} vs {X_LABEL}") +
                labs(
                    x=X_LABEL,
                    y=Y_LABEL
                ) +
                xlim(x_min, x_max) +
                ylim(y_min, y_max)
            )

            # Save the plot
            plot.save(
                self.graph_str + f"{key}/graph_{chip_index}-{j}.png",
                width=16,
                height=9,
                dpi=600
            )

            # Add data to the analysis list
            data_list.append(df[y_col].tolist())

        return data_list
    
    def plot_with_regression(
        self,
        df: pd.DataFrame,
        key: str = "single",
        chip_index: int = 0,
        run_stats: bool = True
    ) -> None:
        """
        Plot from a single dataframe with regression lines + annotations.

        Uses the same layout as plot_all(with_regression=True) but only for
        the provided dataframe.
        """
        X_LABEL = "Pressure In (mb)"
        Y_LABEL = "Flow Rate Out (µL/min)"

        data_list, df_melt = self._plot_single_df_with_regression(
            df, key, chip_index, X_LABEL, Y_LABEL
        )

        if run_stats:
            self.analysis.chi_square_test(df_melt, key)
            self.analysis.anova_test(data_list)
    
    def _plot_single_df_with_regression(
        self,
        df: pd.DataFrame,
        key: str,
        chip_index: int,
        X_LABEL: str,
        Y_LABEL: str
    ):
        """
        Internal helper: plot scatter(s) + regression fits for one dataframe.
        Returns (data_list, df_melt) for ANOVA and chi-square.
        """
        # Find regression equations for this dataframe
        lin_eqs = self.find_lin_fit(df)          # assumed available in module scope
        logistic_eqs = self.find_logistic_fit(df)

        # Melt regression-fit columns for legend + plotting
        df_melt = self._melt_data(df)

        data_list = []

        for j in self.i_range:
            x_col = self.COL_NAMES[f"X_COL_{j}"]
            y_col = self.COL_NAMES[f"Y_COL_{j}"]

            x_max = df_melt[x_col].max()
            y_max = df_melt[y_col].max()
            y_min = 0 if df_melt[y_col].min() >= 0 else df_melt[y_col].min()

            # Step for text annotations
            annotation_step = ((y_max - y_min) / 4) / 3 if (y_max - y_min) != 0 else 1

            # Filter melted data to just the fits for this j
            df_melt_j = df_melt[df_melt["Fit Type"].str.endswith(f"{j}")]

            plot = (
                ggplot(df_melt, aes(x_col, y_col)) +
                geom_point() +
                ggtitle(f"{Y_LABEL} vs {X_LABEL}") +
                geom_line(
                    aes(
                        y="Fit Value",
                        color="Fit Type"
                    ),
                    data=df_melt_j
                ) +
                scale_color_manual(
                    values={
                        f"Linear Fit {j}": "red",
                        f"Logistic Fit {j}": "green"
                    },
                    labels={
                        f"Linear Fit {j}": "Linear Fit",
                        f"Logistic Fit {j}": "Logistic Fit"
                    }
                ) +
                labs(
                    x=X_LABEL,
                    y=Y_LABEL,
                    color="Model Fit"
                ) +
                xlim(0, x_max) +
                ylim(0, y_max) +
                annotate(
                    "text",
                    x=0,
                    y=y_max - annotation_step * 1,
                    label=logistic_eqs[j - 1],
                    ha="left",
                    va="top",
                    size=8,
                    color="green"
                ) +
                annotate(
                    "text",
                    x=0,
                    y=y_max - annotation_step * 0,
                    label=lin_eqs[j - 1],
                    ha="left",
                    va="top",
                    size=8,
                    color="red"
                )
            )

            plot.save(
                self.graph_str + f"{key}/graph_{chip_index}-{j}.png",
                width=16,
                height=9,
                dpi=600
            )

            data_list.append(df_melt[y_col].tolist())

        return data_list, df_melt

    # GETTERS AND SETTERS
    def get_data_dict(self):
        return self.filtered_dict if self.filtered_dict is not None else self.data_dict
