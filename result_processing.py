import argparse
import os
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from app import METRICS as METRICS_DEF

METHODS_DEF = {
    "oasis": "Oasis",
    "world_mem_eval": "WorldMem",
    "persist_novox": "PERSIST-Base",
    "persist_xl_novox": "PERSIST-XL",
    "persist_xl_vox": "PERSIST-XL$+w_0$",
}

METHODS_GT_SET = {
    "oasis": "minedojo_gt",
    "world_mem_eval": "minedojo_gt",
    "persist_novox": "minetest_gt",
    "persist_xl_novox": "minetest_gt",
    "persist_xl_vox": "minetest_gt",
}

GT_KEYS = {"minedojo_gt", "minetest_gt"}
GT_CANON = "ground_truth"  # internal key for the collapsed GT node

WELCH_TEST_A = {
    "persist_novox",
    "persist_xl_novox",
    "persist_xl_vox"
}

WELCH_TEST_B = {
    "oasis",
    "world_mem_eval",
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
      - cell = mean (2 sig figs) ± ste (1 sig figs)
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

            mean_s = _format_sig(mean, 2)
            se_s = _format_sig(se, 1)

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

def make_latex_label(s: str) -> str:
    """
    Make a LaTeX-safe-ish identifier chunk from an arbitrary set name.
    Keeps [A-Za-z0-9]+, turns others into underscores, collapses repeats.
    """
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() if s else "set"

def head_to_head_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an unordered head-to-head comparison count matrix over ALL methods
    present in the data (no METHODS_DEF filtering).

    Counts each trial once (method_left vs method_right), aggregating A/B and B/A.
    Method suffixes like _<set_name> are stripped via base_method().
    """
    pairs = df[["set_name", "method_left", "method_right"]].copy()

    # Strip per-set suffixes if present (matches how you normalize methods elsewhere)
    pairs["m_left"] = pairs.apply(lambda r: base_method(r["method_left"], r["set_name"]), axis=1)
    pairs["m_right"] = pairs.apply(lambda r: base_method(r["method_right"], r["set_name"]), axis=1)

    # Drop missing / self-comparisons
    pairs = pairs.dropna(subset=["m_left", "m_right"])
    pairs = pairs[pairs["m_left"] != pairs["m_right"]].copy()

    # Unordered pair key: sort the two method names so A/B and B/A collapse
    pair_min = pairs[["m_left", "m_right"]].min(axis=1)
    pair_max = pairs[["m_left", "m_right"]].max(axis=1)
    pairs["a"] = pair_min
    pairs["b"] = pair_max

    # Count occurrences of each unordered pair
    counts = pairs.groupby(["a", "b"]).size().reset_index(name="count")

    # All methods present in comparisons
    methods = sorted(set(counts["a"]).union(set(counts["b"])))

    # Build symmetric matrix
    mat = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    for _, row in counts.iterrows():
        a, b, c = row["a"], row["b"], int(row["count"])
        mat.loc[a, b] = c
        mat.loc[b, a] = c

    return mat

def head_to_head_mean_diffs(df: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """
    For each metric, compute an anti-symmetric matrix M such that:
      M[A,B] = mean(score(A) - score(B))
    Also compute SE of the difference.

    Only counts trials where both sides provided a score.
    """

    pairs = df[["set_name", "method_left", "method_right"] +
               [f"{m}_left" for m in metrics] +
               [f"{m}_right" for m in metrics]].copy()

    pairs["m_left"] = pairs.apply(lambda r: base_method(r["method_left"], r["set_name"]), axis=1)
    pairs["m_right"] = pairs.apply(lambda r: base_method(r["method_right"], r["set_name"]), axis=1)

    pairs = pairs.dropna(subset=["m_left", "m_right"])
    pairs = pairs[pairs["m_left"] != pairs["m_right"]]

    methods = sorted(set(pairs["m_left"]).union(set(pairs["m_right"])))

    out = {}

    for m in metrics:
        lcol, rcol = f"{m}_left", f"{m}_right"

        xL = pd.to_numeric(pairs[lcol], errors="coerce")
        xR = pd.to_numeric(pairs[rcol], errors="coerce")

        provided = xL.notna() & xR.notna() & ~((xL == 0) & (xR == 0))
        sub = pairs.loc[provided, ["m_left", "m_right"]].copy()

        if len(sub) == 0:
            mat = pd.DataFrame(np.nan, index=methods, columns=methods)
            np.fill_diagonal(mat.values, 0.0)
            out[m] = mat
            continue

        d_lr = (xL.loc[provided] - xR.loc[provided]).astype(float).values
        sub["d_lr"] = d_lr

        sub["a"] = sub[["m_left", "m_right"]].min(axis=1)
        sub["b"] = sub[["m_left", "m_right"]].max(axis=1)

        sub["d_ab"] = np.where(
            sub["m_left"] == sub["a"],
            sub["d_lr"],
            -sub["d_lr"]
        )

        agg = (
            sub.groupby(["a", "b"])["d_ab"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        agg["se"] = agg["std"] / np.sqrt(agg["count"])

        mat = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=object)
        np.fill_diagonal(mat.values, "0")

        for _, row in agg.iterrows():
            a, b = row["a"], row["b"]
            mu = float(row["mean"])
            se = float(row["se"]) if row["count"] > 1 else np.nan

            mat.loc[a, b] = (mu, se)
            mat.loc[b, a] = (-mu, se)

        out[m] = mat

    return out

def _format_diff_cell(x, show_se: bool) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"

    if isinstance(x, str):  # diagonal "0"
        return x

    mu, se = x
    if np.isnan(mu):
        return "--"

    if not show_se or se is None or np.isnan(se):
        return f"{mu:+.2f}"

    return f"{mu:+.2f} ({se:.2f})"

def _extract_mean_matrix(mat_obj: pd.DataFrame) -> pd.DataFrame:
    """
    Convert your dtype=object h2h matrix to float mu matrix.
    Diagonal is forced to 0. Missing stays NaN.
    """
    def get_mu(x):
        if x is None:
            return np.nan
        if isinstance(x, str):  # diagonal "0"
            try:
                return float(x)
            except Exception:
                return 0.0
        if isinstance(x, tuple) and len(x) >= 1:
            return float(x[0]) if x[0] is not None else np.nan
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        return np.nan

    out = mat_obj.applymap(get_mu).astype(float)
    np.fill_diagonal(out.values, 0.0)
    return out

def collapse_ground_truth_and_mask(
    mat_obj: pd.DataFrame,
    *,
    methods_def: dict[str, str],
    methods_gt_set: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Collapse minedojo_gt/minetest_gt into a single 'Ground Truth' node.
    Enforce row/column ordering:
        - Follow METHODS_DEF order
        - Only include methods present in the matrix
        - Ground Truth always last (if present)

    Returns:
        collapsed float delta matrix
        display labels (same order)
    """

    mu = _extract_mean_matrix(mat_obj)

    present = set(mu.index)
    gt_present = any(k in present for k in GT_KEYS)

    # ---- enforce ordering from METHODS_DEF ----
    ordered_methods = [
        k for k in methods_def.keys()
        if k in present
    ]

    # Add any extra methods not in METHODS_DEF (unlikely but safe)
    extras = [
        k for k in mu.index
        if k not in ordered_methods and k not in GT_KEYS
    ]
    ordered_methods.extend(sorted(extras))

    # Append Ground Truth last if needed
    keys = ordered_methods.copy()
    if gt_present:
        keys.append(GT_CANON)

    # ---- build collapsed matrix ----
    collapsed = pd.DataFrame(np.nan, index=keys, columns=keys, dtype=float)
    np.fill_diagonal(collapsed.values, 0.0)

    # Fill method vs method
    for a in ordered_methods:
        for b in ordered_methods:
            if a == b:
                continue
            if a in mu.index and b in mu.columns:
                collapsed.loc[a, b] = mu.loc[a, b]

    # Fill method vs Ground Truth
    if gt_present:
        for m in ordered_methods:
            gt_key = methods_gt_set.get(m)
            if gt_key is None:
                continue
            if gt_key not in mu.columns:
                continue

            val = mu.loc[m, gt_key] if (m in mu.index and gt_key in mu.columns) else np.nan
            if not np.isnan(val):
                collapsed.loc[m, GT_CANON] = val
                collapsed.loc[GT_CANON, m] = -val

    # ---- display labels (math-safe) ----
    display_labels = []
    for k in keys:
        if k == GT_CANON:
            display_labels.append("Ground Truth")
        else:
            display_labels.append(methods_def.get(k, k))

    return collapsed, display_labels

def save_h2h_delta_matrix_png(
    mat_float: pd.DataFrame,
    *,
    display_labels: list[str],
    metric_key: str,
    metric_name: str | None = None,
    out_path: str = "h2h.png",
    vmin: float = -5.0,
    vmax: float = 5.0,
    dpi: int = 200,
    annotate: bool = True,
    fmt: str = "{:+.2f}",
    title: str | None = None,
    fontsize: int = 7,
):
    """
    Render head-to-head delta matrix (float) as a confusion-matrix-like PNG.

    - Square cells (aspect='equal')
    - Diverging colormap over fixed [-5,5]
    - Annotations are delta only
    """
    methods = list(mat_float.index)
    n = len(methods)

    if len(display_labels) != n:
        raise ValueError("display_labels must match matrix size/order")

    data = mat_float.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)

    cell_in = 0.45
    fig_w = max(6.0, cell_in * n + 2.5)
    fig_h = max(5.5, cell_in * n + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("PiYG").copy()
    cmap.set_bad(color=(0.92, 0.92, 0.92, 1.0))  # missing

    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(display_labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(display_labels, fontsize=fontsize)

    # grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=fontsize)

    if title is None:
        pretty = metric_name if metric_name is not None else metric_key
        title = f"Head-to-head score deltas: {pretty}\nrow A, col B = score(A) - score(B)"
    ax.set_title(title, fontsize=fontsize + 2, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ score", rotation=90)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./results.csv", help="Path to results.csv")
    ap.add_argument("--verbose", action="store_true", help="Include SE and N columns per metric")
    ap.add_argument("--remove_first_n_ratings", type=int, default=0, help="Remove the first N ratings per participant (e.g. to exclude initial learning period)")
    ap.add_argument(
        "--h2h-show-se",
        action="store_true",
        help="Display standard error (SE) in head-to-head difference tables",
    )
    args = ap.parse_args()

    tables_path = "tables.txt"
    tables_f = open(tables_path, "w", encoding="utf-8")

    df = pd.read_csv(args.csv)
    # -----------------------------
    # FIRST THING: remove first N trials/rows per participant by created_at_utc
    # This affects *all* downstream analyses (head-to-head, long-form, etc.)
    # -----------------------------
    if args.remove_first_n_ratings and args.remove_first_n_ratings > 0:
        if "created_at_utc" not in df.columns:
            raise ValueError("created_at_utc column not found in CSV, required for --remove_first_n_ratings")

        # Parse timestamp; keep NaT if parsing fails
        df["_created_at_utc_ts"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=True)

        # Stable sort for deterministic selection
        df = df.sort_values(
            ["participant_id", "_created_at_utc_ts", "trial_index"],
            na_position="last",
            kind="mergesort",
        )

        # Rank trials per participant (each df row is one head-to-head comparison/trial)
        df["_trial_rank"] = df.groupby("participant_id").cumcount()

        before = len(df)
        df = df[df["_trial_rank"] >= args.remove_first_n_ratings].copy()
        after = len(df)

        # Clean temporary cols
        df = df.drop(columns=["_created_at_utc_ts", "_trial_rank"])

        print(
            f"\n=== Removed first {args.remove_first_n_ratings} trials per participant "
            f"(by created_at_utc): {before - after} rows removed ==="
        )

    # ---- print head-to-head comparison count matrix ----
    h2h = head_to_head_counts(df)
    print("\n=== Head-to-head comparison counts (unordered; A vs B aggregated with B vs A) ===")
    print(h2h.to_string())

    # ---- print mean head-to-head diffs per metric ----
    diffs = head_to_head_mean_diffs(df, METRICS)
    out_dir = "h2h_png"
    os.makedirs(out_dir, exist_ok=True)

    for m in METRICS:
        # Collapse GT + mask irrelevant GT comparisons, and produce pretty labels
        collapsed_mat, display_labels = collapse_ground_truth_and_mask(
            diffs[m],
            methods_def=METHODS_DEF,
            methods_gt_set=METHODS_GT_SET,
        )

        out_path = os.path.join(out_dir, f"h2h_{m}.png")
        save_h2h_delta_matrix_png(
            collapsed_mat,
            display_labels=display_labels,
            metric_key=m,
            metric_name=METRIC_NAME.get(m, m),
            out_path=out_path,
            vmin=-5,
            vmax=5,
            annotate=True,
            fmt="{:+.2f}",
        )
        print(f"Saved {out_path}")

    print("\n=== Head-to-head average score differences by metric ===")
    print("Convention: row A, col B = mean(score(A) - score(B))\n")

    for m in METRICS:
        print(f"--- metric: {m} ({METRIC_NAME.get(m, m)}) ---")

        disp = diffs[m].copy()
        disp = disp.applymap(lambda x: _format_diff_cell(x, args.h2h_show_se))

        print(disp.to_string())
        print()

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

    print("\n=== (1) Mean scores by set (methods as rows) ===")
    for set_name, g in by_set_method.groupby("set_name", sort=True):
        g = g.sort_values("method").reset_index(drop=True)

        # existing console table
        table = format_table_methods_x_metrics(
            g.drop(columns=["set_name"]),
            verbose=args.verbose,
            n_kind="n",
        )
        print(f"\n--- set: {set_name} ---")
        print(table.to_string(index=False))

        # NEW: LaTeX table per set (same formatting/features as section 2b’s LaTeX)
        latex_set = df_to_latex_methods_metrics(
            g.drop(columns=["set_name"]),
            metrics_def=METRICS_DEF,
            methods_def=METHODS_DEF,
            caption=rf"User study scores on set \texttt{{{set_name}}} (mean $\\pm$ s.e.).",
            label=f"tab:user_study_scores_{make_latex_label(set_name)}",
        )
        print("\n" + latex_set)

        tables_f.write(latex_set)
        tables_f.write("\n\n")

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

    # OPTIONAL (recommended): LaTeX for (2b) as well, using the same function/format.
    # latex_macro = df_to_latex_methods_metrics(
    #     by_method_across_sets,
    #     metrics_def=METRICS_DEF,
    #     methods_def=METHODS_DEF,
    #     caption="User study scores averaged across sets (macro-average; mean $\\pm$ s.e. across sets).",
    #     label="tab:user_study_scores_macro",
    # )
    # print("\n" + latex_macro)

    # Existing LaTeX table (was overall / micro-average)
    latex = df_to_latex_methods_metrics(
        by_method_overall,
        metrics_def=METRICS_DEF,
        methods_def=METHODS_DEF,
        caption="User study scores (micro-average; mean $\\pm$ s.e.).",
        label="tab:user_study_scores_micro",
    )
    print("\n" + latex)

    tables_f.write(latex)
    tables_f.write("\n\n")

    print("\n=== Dataset statistics ===")
    print(f"Unique participants: {n_participants}")
    print(f"Total ratings: {n_ratings}")

    tables_f.close()
    print(f"\nSaved LaTeX tables to {tables_path}")

if __name__ == "__main__":
    main()