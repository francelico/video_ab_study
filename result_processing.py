import argparse
import numpy as np
import pandas as pd
import math

from app import METRICS as METRICS_DEF

METHODS_DEF = {
    "persist_novox": "PERSIST-Base",
    "persist_xl_novox": "PERSIST-XL",
    "persist_xl_vox": "PERSIST-XL$+\vw_0$",
    "oasis": "Oasis",
    "world_mem_eval": "WorldMem",
}

METRICS = [m["key"] for m in METRICS_DEF]
METRIC_NAME = {m["key"]: m["name"] for m in METRICS_DEF}


# -----------------------------
# Helper: parse method base name
# -----------------------------
def base_method(method_full: str, set_name: str) -> str:
    if pd.isna(method_full):
        return method_full
    suffix = f"_{set_name}"
    return method_full[: -len(suffix)] if method_full.endswith(suffix) else method_full

def count_unique_participants(long: pd.DataFrame) -> int:
    """
    Number of unique participants contributing ratings.
    """
    return long["participant_id"].nunique()

def count_total_ratings(long: pd.DataFrame) -> int:
    """
    Total number of ratings provided overall.
    Each row in `long` corresponds to one rating.
    """
    return len(long)

def mean_and_se(g: pd.DataFrame) -> pd.Series:
    """
    Compute mean and standard error (SE = std / sqrt(n)) for each metric.
    Uses sample std (ddof=1). If n<=1, SE is NaN.
    Returns a flat Series with columns like metric_a_mean, metric_a_se, metric_a_n, ...
    """
    out = {}
    for m in METRICS:
        x = g[m].dropna()
        n = int(x.shape[0])
        out[f"{m}_mean"] = float(x.mean()) if n > 0 else np.nan
        out[f"{m}_se"] = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        out[f"{m}_n"] = n
    return pd.Series(out)


def mean_and_se_over_sets(g: pd.DataFrame) -> pd.Series:
    """
    Same as above, but `g[m]` is already per-set means (one row per set),
    so N is number of sets contributing.
    """
    out = {}
    for m in METRICS:
        x = g[m].dropna()
        n = int(x.shape[0])
        out[f"{m}_mean"] = float(x.mean()) if n > 0 else np.nan
        out[f"{m}_se"] = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        out[f"{m}_n_sets"] = n
    return pd.Series(out)


def format_table_methods_x_metrics(stats_df: pd.DataFrame, *, verbose: bool, n_kind: str) -> pd.DataFrame:
    """
    Convert a stats dataframe with columns:
      method, metric_*_mean, metric_*_se, metric_*_n or metric_*_n_sets
    into a "methods as rows, metrics as columns" table.
    """
    if stats_df is None or len(stats_df) == 0:
        return pd.DataFrame(columns=["method"] + [METRIC_NAME[m] for m in METRICS])

    # Start from a real column slice (keeps index aligned, avoids scalar-construction issues)
    out = stats_df[["method"]].copy()

    for m in METRICS:
        col_base = METRIC_NAME[m]
        out[col_base] = stats_df[f"{m}_mean"].values

        if verbose:
            out[f"{col_base} (SE)"] = stats_df[f"{m}_se"].values
            if n_kind == "n":
                out[f"{col_base} (N)"] = stats_df[f"{m}_n"].values
            elif n_kind == "n_sets":
                out[f"{col_base} (N_sets)"] = stats_df[f"{m}_n_sets"].values
            else:
                raise ValueError(f"Unknown n_kind: {n_kind}")

    return out.sort_values("method").reset_index(drop=True)

def filter_all_zero_metric_rows(long: pd.DataFrame, metrics: list[str]):
    """
    Remove rows where all metric values are zero (after left/right merge).

    Parameters
    ----------
    long : pd.DataFrame
        Long-form dataframe containing one row per (trial × side × method).
    metrics : list[str]
        List of metric column names (e.g. ["metric_a", "metric_b", ...]).

    Returns
    -------
    cleaned : pd.DataFrame
        DataFrame with all-zero metric rows removed.
    removed : pd.DataFrame
        Audit table of removed rows.
    """
    metric_vals = long[metrics].fillna(0)

    all_zero_mask = (metric_vals == 0).all(axis=1)

    removed = long.loc[all_zero_mask].copy()
    cleaned = long.loc[~all_zero_mask].copy()

    return cleaned, removed

def _round_sig(x: float, sig: int) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    x = float(x)
    if x == 0.0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def _format_sig(x: float, sig: int) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    r = _round_sig(float(x), sig)
    if np.isnan(r):
        return "--"
    if r == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(r))))
    decimals = max(0, sig - 1 - exp)
    s = f"{r:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def df_to_latex_methods_metrics(
    stats_df: pd.DataFrame,
    *,
    metrics_def: list[dict],
    methods_def: dict[str, str],
    caption: str | None = None,
    label: str | None = None,
    table_env: bool = True,
    booktabs: bool = True,
) -> str:
    """
    Convert a dataframe with columns:
      - method
      - <metric_key>_mean
      - <metric_key>_se
      - (optionally) <metric_key>_n or <metric_key>_n_sets (ignored)
    into a LaTeX table with:
      - rows = methods (filtered + labeled by METHODS_DEF)
      - columns = metrics
      - cell = mean (2 sig figs) ± ste (2 sig figs)
      - bold highest mean in each metric column
    """
    metric_keys = [m["key"] for m in metrics_def]
    metric_names = [m["name"] for m in metrics_def]

    allowed_methods = list(methods_def.keys())
    df = stats_df.copy()
    df = df[df["method"].isin(allowed_methods)].copy()
    df["method"] = pd.Categorical(df["method"], categories=allowed_methods, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)

    best_method_by_metric: dict[str, str | None] = {}
    for k in metric_keys:
        col = f"{k}_mean"
        if col not in df.columns:
            best_method_by_metric[k] = None
            continue
        means = df[["method", col]].dropna()
        if len(means) == 0:
            best_method_by_metric[k] = None
            continue
        idx = means[col].astype(float).idxmax()
        best_method_by_metric[k] = str(df.loc[idx, "method"])

    col_spec = "l" + "c" * len(metric_keys)
    lines = []

    if table_env:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")

    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    if booktabs:
        lines.append(r"\toprule")

    lines.append(" & ".join(["Method"] + metric_names) + r" \\")
    lines.append(r"\midrule" if booktabs else r"\hline")

    for _, row in df.iterrows():
        method_key = str(row["method"])
        method_label = methods_def[method_key]

        cells = [method_label]
        for k in metric_keys:
            mean = row.get(f"{k}_mean", np.nan)
            se = row.get(f"{k}_se", np.nan)

            mean_s = _format_sig(mean, 2)  # UPDATED: 2 significant figures
            se_s = _format_sig(se, 2)      # 2 significant figures

            if mean_s == "--" or se_s == "--":
                cell = "--"
            else:
                cell = rf"{mean_s} $\pm$ {se_s}"

            if best_method_by_metric.get(k) == method_key and cell != "--":
                cell = rf"\textbf{{{cell}}}"

            cells.append(cell)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule" if booktabs else r"\hline")
    lines.append(r"\end{tabular}")

    if caption is not None:
        lines.append(rf"\caption{{{caption}}}")
    if label is not None:
        lines.append(rf"\label{{{label}}}")

    if table_env:
        lines.append(r"\end{table}")

    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./results.csv", help="Path to results.csv")
    ap.add_argument("--verbose", action="store_true", help="Include SE and N columns per metric")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # -----------------------------
    # Build long-form (left + right)
    # -----------------------------
    left = (
        df[
            [
                "participant_id",
                "trial_index",
                "set_name",
                "method_left",
                "metric_a_left",
                "metric_b_left",
                "metric_c_left",
                "metric_d_left",
            ]
        ]
        .rename(
            columns={
                "method_left": "method_full",
                "metric_a_left": "metric_a",
                "metric_b_left": "metric_b",
                "metric_c_left": "metric_c",
                "metric_d_left": "metric_d",
            }
        )
        .assign(side="left")
    )

    right = (
        df[
            [
                "participant_id",
                "trial_index",
                "set_name",
                "method_right",
                "metric_a_right",
                "metric_b_right",
                "metric_c_right",
                "metric_d_right",
            ]
        ]
        .rename(
            columns={
                "method_right": "method_full",
                "metric_a_right": "metric_a",
                "metric_b_right": "metric_b",
                "metric_c_right": "metric_c",
                "metric_d_right": "metric_d",
            }
        )
        .assign(side="right")
    )

    long = pd.concat([left, right], ignore_index=True)
    long["method"] = long.apply(lambda r: base_method(r["method_full"], r["set_name"]), axis=1)
    long[METRICS] = long[METRICS].apply(pd.to_numeric, errors="coerce")
    long, zero_rows = filter_all_zero_metric_rows(long, METRICS)
    n_participants = count_unique_participants(long)
    n_ratings = count_total_ratings(long)

    # -----------------------------
    # (1) By set × method (mean ± SE)
    # -----------------------------
    by_set_method = (
        long.groupby(["set_name", "method"], dropna=False, sort=True)
        .apply(mean_and_se)
        .reset_index()
        .sort_values(["set_name", "method"])
    )

    # Print one table per set, with methods as rows and metrics as columns
    print("\n=== (1) Mean scores by set (methods as rows) ===")
    for set_name, g in by_set_method.groupby("set_name", sort=True):
        g = g.sort_values("method").reset_index(drop=True)
        table = format_table_methods_x_metrics(
            g.drop(columns=["set_name"]),
            verbose=args.verbose,
            n_kind="n",
        )
        print(f"\n--- set: {set_name} ---")
        print(table.to_string(index=False))

    # -----------------------------
    # (2a) Micro-average across all trials by method (mean ± SE)
    # -----------------------------
    by_method_overall = (
        long.groupby("method", dropna=False, sort=True)
        .apply(mean_and_se)
        .reset_index()
        .sort_values("method")
    )

    table_overall = format_table_methods_x_metrics(by_method_overall, verbose=args.verbose, n_kind="n")
    print("\n=== (2a) Scores averaged across all trials by method (micro-average) ===")
    print(table_overall.to_string(index=False))

    # -----------------------------
    # (2b) Macro-average across sets by method (mean ± SE across sets)
    # -----------------------------
    set_means_only = (
        long.groupby(["set_name", "method"], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )

    by_method_across_sets = (
        set_means_only.groupby("method", dropna=False, sort=True)
        .apply(mean_and_se_over_sets)
        .reset_index()
        .sort_values("method")
    )

    table_macro = format_table_methods_x_metrics(by_method_across_sets, verbose=args.verbose, n_kind="n_sets")
    print("\n=== (2b) Scores averaged across sets by method (macro-average across sets) ===")
    print(table_macro.to_string(index=False))

    latex = df_to_latex_methods_metrics(
        by_method_overall,   # or by_method_overall
        metrics_def=METRICS_DEF,
        methods_def=METHODS_DEF,
        caption="User study scores (mean $\\pm$ s.e.).",
        label="tab:user_study_scores",
    )
    print(latex)


    print("\n=== Dataset statistics ===")
    print(f"Unique participants: {n_participants}")
    print(f"Total ratings: {n_ratings}")

if __name__ == "__main__":
    main()
