# Import libraries
import os
import math
import numpy as np
import pandas as pd
import scipy.stats as stats


class Analysis:
    def __init__(self, analysis_str: str, col_names: dict):
        self.analysis_str = analysis_str
        self.COL_NAMES = col_names

        if os.path.isdir(self.analysis_str) == False:
            os.makedirs(self.analysis_str)

        num_graphs = math.floor(self.COL_NAMES.keys().__len__() / 2)
        self.i_range = list(range(1, num_graphs + 1))

    def chi_square_test(self, df_melt: pd.DataFrame, chip: str, alpha: float = 0.05):
        """
        Perform Chi-Square test
        """
        def calculate_bins(data: list) -> int:
            """
            Calculate the number of histogram bins using the Freedman-Diaconis rule.
            """
            # Pre-process
            data = np.asarray(data)
            data = data[~np.isnan(data)]

            q75, q25 = np.percentile(data, [75 ,25])
            iqr = q75 - q25
            n = len(data)

            # Fallback: just one bin
            if iqr == 0 or n <= 1:
                return 1

            bin_width = 2 * iqr / (n ** (1 / 3))

            # Avoid division by zero
            if bin_width == 0:
                return 1

            data_range = data.max() - data.min()
            bins = int(np.ceil(data_range / bin_width))

            # Always at least one bin
            return max(bins, 1)

        for i in self.i_range:
            x_col = self.COL_NAMES[f"X_COL_{i}"]
            y_col = self.COL_NAMES[f"Y_COL_{i}"]

            # Replace with your DataFrame and column name
            df = df_melt.copy()
            x_data = df[x_col]
            y_data = df[y_col]

            # Calculate the number of bins using Freedman-Diaconis rule
            x_bins = calculate_bins(x_data)
            y_bins = calculate_bins(y_data)

            # Bin the x and y data
            df['x_binned'] = pd.cut(x_data, bins=x_bins)
            df['y_binned'] = pd.cut(y_data, bins=y_bins)

            # Create contingency table (cross-tab of binned values)
            contingency_table = pd.crosstab(df['x_binned'], df['y_binned'])

            # Perform chi-squared test
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

            # Write data to file
            with open(self.analysis_str + f"{chip}_chi_square_results_{i}.txt", "w") as f:
                # Log results to file
                f.write(f"Chi-squared statistic: {chi2}\n")
                f.write(f"Degrees of freedom: {dof}\n")
                f.write(f"Optimal bin count: x = {x_bins}, y = {y_bins}\n")
                f.write(f"Expected frequencies:\n{expected}\n")
                f.write(f"P-value: {p}\n")

                # Interpret the result
                if p < alpha:
                    f.write("Result: Reject the null hypothesis — association exists between binned x and y.")
                else:
                    f.write("Result: Fail to reject the null — no significant association.")

    def anova_test(self, y_vals: list, alpha: float = 0.05) -> None:
        """
        Perform one-way ANOVA test
        """
        # Number of groups and total number of observations
        k = len(y_vals)
        n = 0

        # Return if there is only one value
        if (k <= 1):
            return

        for group in y_vals:
            n += len(group)

        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*y_vals)

        # Degrees of freedom
        df_between = k - 1
        df_within = n - k

        # Calculate critical F value
        f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

        # Write data to file
        with open(self.analysis_str + "anova_results.txt", "w") as f:
            # Log results to file
            f.write(f"F-statistic: {f_statistic:.5f}\n")
            f.write(f"p-value: {p_value:.5f}\n")
            f.write(f"F-critical (alpha = {alpha}): {f_critical:.5f}\n")

            # Interpret the result
            if f_statistic > f_critical:
                f.write(f"Reject the null hypothesis: At least one group mean is different.")
            else:
                f.write(f"Fail to reject the null hypothesis: No significant difference between group means.")

    def estimate_burst_pressure(
        self,
        df: pd.DataFrame,
        chip: str,
        press_key: str = "PRESS",
        time_key: str = "TIME",
        window: int = 5,
        drop_fraction: float = 0.1,
        run_label: str | None = None,
    ) -> float:
        """
        Estimate the burst pressure of a diode from a pressure-vs-time trace.

        Heuristic:
        - Smooth the pressure with a rolling median (window points).
        - Compute the first large negative drop in the smoothed trace.
        - Define burst pressure as the maximum pressure just before that drop.
        - If no strong drop is found, fall back to the global max pressure.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time and pressure columns (already cleaned).
        chip : str
            Chip identifier, used in the output filename.
        press_key : str
            Key in self.COL_NAMES for the pressure column (default "PRESS").
        time_key : str
            Key in self.COL_NAMES for the time column (default "TIME").
        window : int
            Rolling median window size for smoothing (in points).
        drop_fraction : float
            Minimum drop magnitude as a fraction of the total pressure range
            required to treat a drop as a burst event. Default = 0.1 (10%).
        run_label : str | None
            Optional additional label (e.g., run index) for the output file.

        Returns
        -------
        float
            Estimated burst pressure (same units as the pressure column).
            Returns np.nan if no data is available.
        """
        # Resolve actual column names from keys
        p_col = self.COL_NAMES.get(press_key, press_key)
        t_col = self.COL_NAMES.get(time_key, time_key)

        # Extract and clean relevant columns
        cols = [p_col]
        if t_col in df.columns:
            cols.append(t_col)

        data = df[cols].copy()
        data[p_col] = pd.to_numeric(data[p_col], errors="coerce")
        data = data.dropna(subset=[p_col])

        if data.empty:
            burst_pressure = np.nan
        else:
            # Sort by time if available
            if t_col in data.columns:
                data = data.sort_values(by=t_col)

            p = data[p_col].to_numpy(dtype=float)

            if p.size < 3:
                burst_pressure = float(np.nan)
            else:
                # Rolling-median smoothing
                # (simple Python implementation to avoid extra dependencies)
                from collections import deque
                import statistics

                smooth = []
                dq = deque()
                for val in p:
                    dq.append(val)
                    if len(dq) > window:
                        dq.popleft()
                    smooth.append(statistics.median(dq))
                smooth = np.asarray(smooth, dtype=float)

                # First differences of smoothed pressure
                dp = np.diff(smooth)
                min_dp_idx = int(np.argmin(dp))
                min_dp = dp[min_dp_idx]          # most negative change
                drop_mag = -min_dp               # positive magnitude of the drop
                p_range = float(smooth.max() - smooth.min())

                # Default: global max
                burst_idx = int(np.argmax(smooth))
                burst_pressure = float(smooth[burst_idx])

                # Check if we have a "strong enough" drop to call it a burst
                if p_range > 0 and drop_mag >= drop_fraction * p_range:
                    # Look for the max pressure prior to the drop
                    pre_drop_end = min(min_dp_idx + 1, smooth.size)
                    if pre_drop_end > 0:
                        burst_idx = int(np.argmax(smooth[:pre_drop_end]))
                        burst_pressure = float(smooth[burst_idx])

        # Write a small report file
        suffix = f"_{run_label}" if run_label is not None else ""
        out_path = os.path.join(self.analysis_str, f"{chip}_burst_pressure{suffix}.txt")

        with open(out_path, "w") as f:
            f.write("Burst Pressure Estimation\n")
            f.write("-------------------------\n")
            f.write(f"Chip: {chip}\n")
            f.write(f"Pressure column: {p_col}\n")
            f.write(f"Window (points): {window}\n")
            f.write(f"Drop fraction threshold: {drop_fraction:.3f}\n\n")

            if np.isnan(burst_pressure):
                f.write("Result: No clear burst event detected (insufficient or invalid data).\n")
            else:
                f.write(f"Estimated burst pressure: {burst_pressure:.3f}\n")

        return burst_pressure

    # GETTERS AND SETTER
    def get_i_range(self):
        return self.i_range
