import re
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from statsmodels.stats.multitest import multipletests


class Base:
    def __init__(self, results=None, from_file=None):
        self.results = []
        if results is not None:
            self.results = results
        elif from_file is not None:
            self.results = self.read_file(from_file)

        if not self.results:
            raise ValueError(
                "No results loaded. Check the file format or input data.")

    def read_file(self, filename):
        """
        Read results from file and return in format expected by build_dataframe.

        File format:
        Results for experimentA:
        experimentA_mice_zscore_mask0: {'test_loss_mean': np.float64(0.397...), ...}
        ...

        Returns:
        List of tuples: [(tag, metrics_dict), ...]
        """
        import ast
        import numpy as np

        results = []

        with open(filename, 'r') as f:
            lines = f.readlines()

        # Skip the header line
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            # Split on first colon to separate tag from metrics
            if ':' not in line:
                continue

            tag, metrics_str = line.split(':', 1)
            tag = tag.strip()
            metrics_str = metrics_str.strip()

            try:
                # Replace np.float64(...) with just the float value
                # This handles the numpy wrapper in the string representation
                cleaned_metrics_str = re.sub(
                    r'np\.float64\((.*?)\)', r'\1', metrics_str)

                # Parse the dictionary string
                metrics_dict = ast.literal_eval(cleaned_metrics_str)

                results.append((tag, metrics_dict))

            except (ValueError, SyntaxError) as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")
                continue

        return results


class ExpA(Base):
    def run(self):
        df = self.build_dataframe(self.results)
        anova = self.run_anova(df)
        tukey = self.run_tukey(df)
        self.report(anova, tukey)

    def report(self, anova: pd.DataFrame, tukey):
        try:
            anova.to_latex("tables/expA_anova.tex", float_format="%.3f")
        except Exception as e:
            print(f"Error saving ANOVA table: {e}")
            anova.to_csv("tables/expA_anova.csv", float_format="%.3f")

        with open("tables/expA_tukey.txt", "w") as f:
            try:
                f.write(tukey.summary().as_latex())
            except Exception as e:
                print(f"Error saving Tukey summary: {e}")
                f.write(tukey.summary().as_text())

    def parse_tag(self, tag: str):
        """
        Convert 'experimentA_mice_robust_mask10'  -> ('mice', 'robust', 0.10)
        """
        m = re.match(r"experimentA_(\w+)_(\w+)_mask(\d+)", tag)
        if m is None:
            raise ValueError(f"Unrecognised tag format: {tag}")
        imp, scl, pct_str = m.groups()
        return imp, scl, int(pct_str) / 100.0

    def build_dataframe(self, results):
        """
        results : list[(tag:str, metrics_dict)]
            e.g. ('experimentA_mice_robust_mask10', {'test_mae_mean': 0.042, 'seed_test_mae': [0.041, 0.043, ...]})
        -> tidy DataFrame with columns:
            imputer, scaler, mask_pct, mae
        """
        rows = []
        for tag, metrics in results:
            try:
                imp, scl, mask_pct = self.parse_tag(tag)

                # ── pull per-seed MAE if available ───────────────────────────────
                if "seed_test_mae" in metrics:  # preferred key name
                    maes = metrics["seed_test_mae"]
                elif "seed_mae" in metrics:  # alternate key name
                    maes = metrics["seed_mae"]
                elif "per_seed" in metrics:  # another alternate key name
                    maes = metrics["per_seed"]
                else:  # fall back to average
                    maes = [metrics["test_mae_mean"]]

                for mae in maes:
                    rows.append(
                        dict(
                            imputer=imp,
                            scaler=scl,
                            # categorical for ANOVA
                            mask_pct=f"{mask_pct:.0%}",
                            mae=mae,
                        )
                    )
            except Exception as e:
                print(f"Skipping {tag}: {e}")

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No valid rows parsed — check tag formats.")
        return df

    def run_anova(self, df):
        """
        Three-way ANOVA with all interactions.
        """

        model = ols(
            "mae ~ C(imputer) * C(scaler) * C(mask_pct)",
            data=df
        ).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        return anova_tbl

    def run_tukey(self, df):
        """
        Tukey HSD on every unique (imputer, scaler, mask_pct) triple.
        """

        df["combo"] = (
            df["imputer"] + "_" + df["scaler"] + "_mask" + df["mask_pct"]
        )
        tukey = pairwise_tukeyhsd(
            endog=df["mae"],
            groups=df["combo"],
            alpha=0.05,
        )
        return tukey


class ExpB(Base):
    def run(self):
        df = self.build_dataframe(self.results)
        anova = self.run_anova(df)
        tukey = self.run_tukey(df)
        self.report(anova, tukey)

    def report(self, anova: pd.DataFrame, tukey):
        try:
            anova.to_latex("tables/expB_anova.tex", float_format="%.3f")
        except Exception as e:
            print(f"Error saving ANOVA table: {e}")
            anova.to_csv("tables/expB_anova.csv", float_format="%.3f")

        with open("tables/expB_tukey.txt", "w") as f:
            try:
                f.write(tukey.summary().as_latex())
            except Exception as e:
                print(f"Error saving Tukey summary: {e}")
                f.write(tukey.summary().as_text())

    def parse_tag(self, tag: str):
        """
        Convert 'experimentB_70p'  -> '0.7'
        """
        m = re.match(r"experimentB_(\d+)p", tag)
        if m is None:
            raise ValueError(f"Unrecognised tag format: {tag}")
        return f"{int(m.group(1)) / 100:.1f}"   # e.g. '0.7', '0.8', '0.9'

    def build_dataframe(self, results):
        """
        Expected structure of `results` (same as Experiment A):
            results = [
                ('experimentB_70p', {'test_mae_mean': 0.041, 'seed_test_mae': [ ... ]}),
                ...
            ]

        The code prefers a per-seed list (`seed_test_mae`, `seed_mae` or `per_seed`) so that the
        ANOVA has replicate observations.  If such a list is missing it falls
        back to the single aggregated MAE, which still works but yields only
        one datum per threshold.
        """
        rows = []
        for tag, metrics in results:
            # '0.7', '0.8', '0.9'
            thr = self.parse_tag(tag)
            # ── pull per-seed MAE if available ───────────────────────────────
            if "seed_test_mae" in metrics:  # preferred key name
                maes = metrics["seed_test_mae"]
            elif "seed_mae" in metrics:  # alternate key name
                maes = metrics["seed_mae"]
            elif "per_seed" in metrics:  # another alternate key name
                maes = metrics["per_seed"]
            else:  # fall back to average
                maes = [metrics.get(
                    "test_mae_mean", metrics.get("test_mae", 0.0))]

            for mae in maes:
                rows.append({"threshold": thr, "mae": mae})

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No valid rows parsed — check tag formats.")
        return df

    def run_anova(self, df):
        """One-way ANOVA on threshold."""
        model = ols("mae ~ C(threshold)", data=df).fit()
        return sm.stats.anova_lm(model, typ=2)

    def run_tukey(self, df):
        """Tukey HSD for all pair-wise threshold comparisons."""
        return pairwise_tukeyhsd(
            endog=df["mae"],
            groups=df["threshold"],
            alpha=0.05,
        )


class ExpC(Base):
    def run(self):
        df = self.build_dataframe(self.results)
        ci95 = self.summary_ci95(df)
        wilcoxon = self.wilcoxon_vs_baseline(df, baseline="LSTM_256")
        self.report(ci95, wilcoxon)

    def report(self, ci95: pd.Series, wilcoxon: pd.DataFrame):
        """
        Save the results of the ci95 and Wilcoxon tests.
        """
        try:
            ci95.to_latex("tables/expC_ci95.tex", float_format="%.3f")
        except Exception as e:
            print(f"Error saving CI95 table: {e}")
            ci95.to_csv("tables/expC_ci95.csv", float_format="%.3f")

        try:
            wilcoxon.to_latex("tables/expC_wilcoxon.tex", float_format="%.3f")
        except Exception as e:
            print(f"Error saving Wilcoxon table: {e}")
            wilcoxon.to_csv("tables/expC_wilcoxon.csv", index=False)

    def _parse_tag(self, tag: str) -> str:
        """
        'experimentC_LSTM_256' -> 'LSTM_256'
        """
        m = re.match(r"experimentC_(.+)", tag)
        if m is None:
            raise ValueError(f"Bad tag: {tag}")
        return m.group(1)

    def build_dataframe(self, results):
        """
        Parameters
        ----------
        results : list[(tag:str, metrics:dict)]
            where metrics must contain either
            • 'seed_mae'  - list[float]  OR
            • 'per_seed'  - list[float]  OR
            • 'test_mae'  - scalar       (fallback)

        Returns
        -------
        tidy DataFrame with columns: model, seed, mae
        """
        rows = []
        for tag, metrics in results:
            model = self._parse_tag(tag)

            # ── pull per-seed MAE if available ───────────────────────────────
            if "seed_test_mae" in metrics:                  # preferred key name
                maes = metrics["seed_test_mae"]
            elif "seed_mae" in metrics:                     # alternate key name
                maes = metrics["seed_mae"]
            elif "per_seed" in metrics:                     # another alternate key name
                maes = metrics["per_seed"]
            else:                                           # fall back to average
                maes = [metrics.get(
                    "test_mae_mean", metrics.get("test_mae", 0.0))]

            for seed_idx, mae in enumerate(maes):
                rows.append({"model": model, "seed": seed_idx, "mae": mae})

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No MAE values parsed.")
        return df

    def summary_ci95(self, df: pd.DataFrame):
        """
        Compute mean ±95% CI for each model.

        Returns
        -------
        DataFrame with index=model, columns=[mean, ci95]
        """
        def _ci(series):
            n = series.size
            se = series.std(ddof=1) / n**0.5
            t = stats.t.ppf(0.975, n - 1)  # two‑sided 95 %
            return se * t

        out = df.groupby("model")["mae"].agg(mean="mean", ci95=_ci)
        return out

    def wilcoxon_vs_baseline(self, df, baseline="LSTM_256"):
        """
        Paired Wilcoxon signed‑rank tests between baseline and every other model.

        Returns
        -------
        DataFrame with columns: model, p_raw, p_holm, reject_H0
        """
        seeds = sorted(df["seed"].unique())
        base_vec = df.query(
            "model == @baseline").set_index("seed").loc[seeds, "mae"]

        pvals, models = [], []
        for mdl in df["model"].unique():
            if mdl == baseline:
                continue
            vec = df.query("model == @mdl").set_index("seed").loc[seeds, "mae"]
            stat, p = stats.wilcoxon(base_vec, vec, alternative="two-sided")
            models.append(mdl)
            pvals.append(p)

        # Holm correction
        reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")

        return pd.DataFrame(
            dict(model=models, p_raw=pvals, p_holm=p_holm, reject_H0=reject)
        ).sort_values("p_holm")
