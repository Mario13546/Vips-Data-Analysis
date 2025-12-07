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

    # GETTERS AND SETTER
    def get_i_range(self):
        return self.i_range
