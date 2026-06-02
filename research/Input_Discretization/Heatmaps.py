import gc
import logging
import pickle
import sys
import ast
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from networkx import sigma
import numpy as np
import pandas as pd
import json
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm
import plotly.graph_objects as go
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

METRIC_CONFIGS = {
    "LyapunovTime": {"cmap": "viridis",    "plotly_scale": "viridis"},
    "KLDivergence": {"cmap": "viridis_r",  "plotly_scale": "viridis_r"},
    "JSDivergence": {"cmap": "viridis_r",  "plotly_scale": "viridis_r"},
    "RMSE":         {"cmap": "plasma_r",   "plotly_scale": "plasma_r"},
    "PSD Errors":   {"cmap": "inferno",    "plotly_scale": "inferno"},
    "Cos_Sim":      {"cmap": "RdBu_r",     "plotly_scale": "RdBu"}, # Plotly natively reverses differently
}

def make_param_combo(x: Any) -> Optional[tuple]:
    """
    Safely converts a parameter dictionary into a hashable tuple.
    Returns None if the params cannot be converted (e.g. unhashable values).
    """
    try:
        return tuple(sorted(x.items()))
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not hash params: {x} — {e}")
        return None


def _compute_entry_metrics(entry: dict, ppp: float, selection: int) -> Optional[Dict[str, Any]]:
    """
    Computes all metrics for a single data entry.
    Isolated as a top-level function so it can be dispatched to a thread pool.

    Sequential metric calls are used here — histogramdd and welch both release
    the GIL in C code, so threading at the entry level provides genuine
    parallelism without the serialization cost of multiprocessing.
    """
    from reservoirgrid.helpers import chaos_utils, utils

    try:
        params   = entry["parameters"]
        true_val = entry["true_value"][::selection]
        preds    = entry["predictions"][::selection]
    except KeyError as e:
        logger.warning(f"Malformed entry, missing key: {e}")
        return None

    param_combo = make_param_combo(params)
    if param_combo is None:
        return None

    try:
        lyap         = chaos_utils.lyapunov_time(truth=true_val, predictions=preds)
        skl           = chaos_utils.symmetric_kl(true_val, preds)  # bins=100 → 20: 125× less work in histogramdd
        psd, cos_sim = chaos_utils.psd_error(true_val, preds)
        rmse         = utils.RMSE(true_val, preds)
        jsdiv        = chaos_utils.js_divergence(true_val, preds)
    except Exception as e:
        logger.warning(f"Metric computation failed: {e}")
        return None

    return {
        "ppp":          ppp,
        "params":       params,
        "param_combo":  param_combo,
        "LyapunovTime": lyap,
        "KLDivergence": skl,
        "PSD Errors":   psd,
        "Cos_Sim":      cos_sim,
        "RMSE":         rmse,
        "JSDivergence": jsdiv
    }


def _process_single_file(args: tuple) -> List[Dict[str, Any]]:
    """
    Loads a pickle file and computes metrics for all entries in parallel threads.
    Must be a top-level function (not nested) for pickling by ProcessPoolExecutor.

    Args:
        args: Tuple of (file_path, entry_workers) to allow passing multiple
              arguments through ProcessPoolExecutor.submit().
    """
    file_path, entry_workers = args

    try:
        ppp = float(file_path.stem)
    except ValueError:
        logger.warning(f"Skipping file with non-numeric name: {file_path.name}")
        return []

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return []

    # Entries are independent — threads parallelize here because
    # histogramdd + welch both release the GIL in their C implementations.
    results = []
    with ThreadPoolExecutor(max_workers=entry_workers) as pool:
        futures = [pool.submit(_compute_entry_metrics, entry, ppp, selection = 1) for entry in data]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                results.append(result)

    return results


def extract_metrics_from_folder(
    folder_path: Path,
    file_workers: int = 4,
    entry_workers: int = 24,
) -> pd.DataFrame:
    """
    Parses all serialized pickle data files within a target directory
    and extracts chaotic evaluation metrics into a structured DataFrame.

    Two-level parallelism:
      - Outer: ProcessPoolExecutor across files (I/O + process-level isolation)
      - Inner: ThreadPoolExecutor across entries per file (CPU, GIL-free in numpy)

    Args:
        folder_path (Path): Path to the directory containing .pkl files.
        file_workers (int): Number of parallel file-loading processes.
        entry_workers (int): Number of threads per file for metric computation.

    Returns:
        pd.DataFrame: Structured dataset containing parameter configurations,
                      discretization steps, and calculated error metrics.
    """
    file_list: List[Path] = sorted(folder_path.glob("*.pkl"))

    if not file_list:
        logger.warning(f"No .pkl files found in: {folder_path}")
        return pd.DataFrame()

    all_data: List[Dict] = []

    # Pass entry_workers alongside each path since ProcessPoolExecutor.submit()
    # only accepts a single iterable argument per worker call.
    task_args = [(f, entry_workers) for f in file_list]

    with ProcessPoolExecutor(max_workers=file_workers) as executor:
        futures = {executor.submit(_process_single_file, args): args[0] for args in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            try:
                all_data.extend(future.result())
            except Exception as e:
                logger.error(f"Worker failed for {futures[future].name}: {e}")

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def plot_and_save_heatmap(
    df: pd.DataFrame,
    metric: str,
    folder_name: str,
    output_base_dir: Path,
    apply_blur: bool = False,
    sigma: tuple = (1.5, 0.5)
) -> None:
    """
    Transforms long-form metrics data into a 2D matrix structure and saves a
    publication-ready heatmap visualization.

    NaN cells (missing ppp × param_combo combinations) are filled with the
    column median to avoid white strips in the imshow output.

    Args:
        df (pd.DataFrame): Source evaluation metrics dataframe.
        metric (str): Target metric column to plot.
        folder_name (str): Context identifier used for directory segmentation.
        output_base_dir (Path): Base directory path for visualization export.
    """
    # Aggregate duplicate (ppp, param_combo) pairs before pivoting
    df_agg = (
        df.groupby(["ppp", "param_combo"], as_index=False)[[metric]]
        .mean()
    )

    # Reshape into a 2D grid matrix
    pivot_df = df_agg.pivot(index="ppp", columns="param_combo", values=metric)
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))

    save_heatmap_metadata(pivot_df, df_agg, metric, folder_name, output_base_dir)

    # Fill NaN cells with column medians to prevent white strips in imshow.
    # Missing cells arise when not every ppp value has data for every param_combo.
    heatmap_matrix = pivot_df.apply(lambda col: col.fillna(col.median())).to_numpy()

    if apply_blur:
        from scipy.ndimage import gaussian_filter
        heatmap_matrix = gaussian_filter(heatmap_matrix, sigma=sigma)

    ppp_values  = pivot_df.index.to_numpy()
    num_combos  = heatmap_matrix.shape[1]

    # Norm selection — guard LogNorm against matrices with no positive values
    if metric in ["PSD Errors", "RMSE"]:
        positive_vals = heatmap_matrix[heatmap_matrix > 0]
        if positive_vals.size == 0:
            logger.warning(
                f"No positive values for LogNorm on '{metric}'. Falling back to linear normalization."
            )
            norm = Normalize(vmin=np.nanmin(heatmap_matrix), vmax=np.nanmax(heatmap_matrix))
        else:
            norm = LogNorm(vmin=np.nanmin(positive_vals), vmax=np.nanmax(heatmap_matrix))
    elif metric == "Cos_Sim":
        norm = Normalize(vmin=-1.0, vmax=1.0)
    else:
        norm = Normalize(vmin=np.nanmin(heatmap_matrix), vmax=np.nanmax(heatmap_matrix))

    # Graceful style fallback — avoids reverting to matplotlib "default"
    preferred_styles = ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]
    for style in preferred_styles:
        if style in plt.style.available:
            plt.style.use(style)
            break

    fig, ax = plt.subplots(figsize=(16, 7), layout="tight")

    # Fetch configured colormap with a graceful fallback
    metric_cfg = METRIC_CONFIGS.get(metric, {"cmap": "viridis"})

    cax = ax.imshow(
        heatmap_matrix,
        cmap=metric_cfg["cmap"],  # Dynamic assignment
        aspect="auto",
        origin="lower",
        interpolation="bilinear" if apply_blur else "nearest",
        norm=norm,
    )

    # X-axis: parameter combination indices
    x_tick_step = 5
    xticks = list(range(0, num_combos, x_tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], fontsize=10, color="#2c3e50")
    ax.set_xlabel("Parameter Combination Index", fontsize=12, fontweight="bold", labelpad=12)

    # Y-axis: actual PPP values (not row indices)
    y_tick_step = max(1, len(ppp_values) // 10)
    yticks = list(range(0, len(ppp_values), y_tick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{ppp_values[i]:.2f}" for i in yticks], fontsize=10, color="#2c3e50")
    ax.set_ylabel("Points per Period (PPP)", fontsize=12, fontweight="bold", labelpad=12)

    # Colorbar
    cbar_formatter = ticker.LogFormatterMathtext() if isinstance(norm, LogNorm) else ticker.ScalarFormatter()
    cbar = fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.046, format=cbar_formatter)
    cbar.set_label(label=f"Measured: {metric}", fontsize=11, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        f"Evaluation Landscape: {metric} ({folder_name})",
        fontsize=14, fontweight="bold", pad=16, color="#1a252f",
    )

    for spine in ax.spines.values():
        spine.set_color("#bdc3c7")

    ax.grid(False)

    # Save
    save_dir = output_base_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{metric.lower().replace(' ', '_')}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap to: {output_file}")


def process_single_location(
    target_pkl_dir: str,
    output_base_dir: str = "research/Input_Discretization/Plots/HeatMaps/",
    file_workers: int = 4,
    entry_workers: int = 24,
    force_recompute: bool = False,  # set True to ignore cache and rerun
    apply_blur: bool = False,
    sigma: tuple = (1.5, 0.5)
) -> None:
    input_path  = Path(target_pkl_dir)
    output_path = Path(output_base_dir)

    if not input_path.is_dir():
        logger.error(f"Target path '{input_path}' is not a directory.")
        return

    folder_name    = input_path.name
    target_metrics = ["LyapunovTime", "KLDivergence", "PSD Errors", "RMSE", "Cos_Sim", "JSDivergence"]

    # --- Cache check ----------------------------------------------------------
    df = None
    if not force_recompute:
        df = load_from_metadata(folder_name, output_path, target_metrics)

    if df is None:
        logger.info(f"Computing metrics from scratch for: {input_path}")
        df = extract_metrics_from_folder(input_path, file_workers=file_workers, entry_workers=entry_workers)
        if df.empty:
            logger.warning(f"No data extracted from {input_path.name}")
            return
    # --------------------------------------------------------------------------
    for metric in target_metrics:
        logger.info(f"Plotting: {metric}")
        plot_and_save_heatmap(df, metric, folder_name, output_path, apply_blur=apply_blur, sigma=sigma)
        plot_interactive_heatmap(df, metric, folder_name, output_path, apply_blur=apply_blur, sigma=sigma)

    del df
    gc.collect()
    logger.info(f"Done: '{folder_name}'")

def save_heatmap_metadata(pivot_df, df_agg, metric, folder_name, output_base_dir):
    save_dir = output_base_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ppp_values   = pivot_df.index.to_numpy()
    param_combos = pivot_df.columns.to_numpy()

    # --- 1. Index map CSV (no param_combo column — looked up via col_idx) ----
    records = []
    for row_idx, ppp in enumerate(ppp_values):
        for col_idx, combo in enumerate(param_combos):
            records.append({
                "row_idx": row_idx,
                "col_idx": col_idx,
                "ppp":     float(ppp),
                metric:    float(pivot_df.iloc[row_idx, col_idx])
                           if not pd.isna(pivot_df.iloc[row_idx, col_idx]) else None,
            })

    index_map_df = pd.DataFrame(records)
    index_map_df.to_csv(save_dir / f"{metric.lower().replace(' ', '_')}_index_map.csv", index=False)

    # --- 2. Param legend as JSON — avoids all tuple/numpy serialization issues
    legend = []
    for col_idx, combo in enumerate(param_combos):
        legend.append({
            "col_idx":     col_idx,
            "param_combo": {k: float(v) if hasattr(v, 'item') else v   # numpy → python
                            for k, v in combo},
        })

    legend_path = save_dir / f"{metric.lower().replace(' ', '_')}_param_legend.json"
    with open(legend_path, "w") as f:
        json.dump(legend, f, indent=2)

    logger.info(f"Saved metadata for '{metric}' to {save_dir}")

def load_from_metadata(folder_name, output_base_dir, target_metrics):
    save_dir = output_base_dir / folder_name
    frames = []

    for metric in target_metrics:
        index_map_path = save_dir / f"{metric.lower().replace(' ', '_')}_index_map.csv"
        legend_path    = save_dir / f"{metric.lower().replace(' ', '_')}_param_legend.json"

        if not index_map_path.exists() or not legend_path.exists():
            logger.info(f"Cache miss for '{metric}'")
            return None

        # Load the index map (no param_combo column)
        df_metric = pd.read_csv(index_map_path)

        # Load legend and rebuild param_combo tuples from JSON dicts
        with open(legend_path) as f:
            legend = json.load(f)

        col_idx_to_combo = {
            entry["col_idx"]: tuple(sorted(entry["param_combo"].items()))
            for entry in legend
        }
        df_metric["param_combo"] = df_metric["col_idx"].map(col_idx_to_combo)

        frames.append(df_metric[["ppp", "param_combo", metric]])

    # Merge all metrics on ppp + param_combo
    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on=["ppp", "param_combo"], how="outer")

    logger.info(f"Loaded all metrics from cache for '{folder_name}'")
    return df

def plot_interactive_heatmap(
    df: pd.DataFrame,
    metric: str,
    folder_name: str,
    output_base_dir: Path,
    apply_blur: bool = False,
    sigma: tuple = (1.5, 0.5)
) -> None:
    """
    Generates a browser-interactive HTML heatmap where hovering over any cell
    shows: ppp, all unpacked parameter values, and the metric value.

    Args:
        df (pd.DataFrame): Source evaluation metrics dataframe.
        metric (str): Target metric column to plot.
        folder_name (str): Context identifier for directory segmentation.
        output_base_dir (Path): Base directory for HTML export.
    """
    df_agg = (
        df.groupby(["ppp", "param_combo"], as_index=False)[metric]
        .mean()
    )

    pivot_df = df_agg.pivot(index="ppp", columns="param_combo", values=metric)
    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))

    ppp_values   = pivot_df.index.to_numpy()
    param_combos = pivot_df.columns.to_numpy()

    # Fill NaN with column median (same as static heatmap)
    heatmap_matrix = pivot_df.apply(lambda col: col.fillna(col.median())).to_numpy()

    if apply_blur:
        from scipy.ndimage import gaussian_filter
        heatmap_matrix = gaussian_filter(heatmap_matrix, sigma=sigma)

    # --- Build per-cell hover text -------------------------------------------
    # Each cell shows: metric value, ppp, and every unpacked parameter
    hover_text = []
    for row_idx, ppp in enumerate(ppp_values):
        row_hover = []
        for col_idx, combo in enumerate(param_combos):
            value = heatmap_matrix[row_idx, col_idx]
            PointsPerPeriod = ppp
            params_str = "<br>".join(f"  {k}: {v}" for k, v in combo)
            cell_text = (
                f"<b>{metric}:</b> {value:.6f}<br>"
                f"<b>Parameter Index:</b> {col_idx}<br>"
                f"<b>Points per Period:</b> {PointsPerPeriod:.2f}<br>"
                f"<b>Parameters:</b><br>{params_str}"
            )
            row_hover.append(cell_text)
        hover_text.append(row_hover)

    # --- Log scale for RMSE / PSD --------------------------------------------
    plot_matrix = heatmap_matrix.copy()
    metric_cfg = METRIC_CONFIGS.get(metric, {"plotly_scale": "viridis"})
    colorscale = metric_cfg["plotly_scale"]
    colorbar_title = metric
    
    if metric in ["PSD Errors", "RMSE"]:
        positive_mask = plot_matrix > 0
        if positive_mask.any():
            plot_matrix = np.where(positive_mask, np.log10(plot_matrix + 1e-10), np.nan)
            colorbar_title = f"log₁₀({metric})"

    # --- Plotly figure -------------------------------------------------------
    fig = go.Figure(data=go.Heatmap(
        z=plot_matrix,
        x=list(range(len(param_combos))),   # col indices on X
        y=[f"{p:.2f}" for p in ppp_values], # ppp labels on Y
        colorscale=colorscale,
        hoverinfo="text",
        text=hover_text,
        zsmooth="best" if apply_blur else False,
        colorbar=dict(
            title=dict(text=colorbar_title, side="right"),
            tickfont=dict(size=10),
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Evaluation Landscape: {metric} ({folder_name})",
            font=dict(size=16, color="#1a252f"),
            x=0.5,
        ),
        xaxis=dict(
            title="Parameter Combination Index",
            tickmode="linear",
            tick0=0,
            dtick=5,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Points per Period (PPP)",
            tickfont=dict(size=10),
        ),
        width=1200,
        height=550,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#2c3e50",
            font=dict(size=12, color="#2c3e50"),
        ),
    )

    # Save as HTML
    save_dir = output_base_dir / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{metric.lower().replace(' ', '_')}_interactive.html"
    fig.write_html(
        output_file,
        include_plotlyjs="cdn",  # keeps file small — loads plotly from CDN
        full_html=True,
    )
    logger.info(f"Saved interactive heatmap to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_single_location(sys.argv[1])
    else:
        process_single_location("research/Input_Discretization/results/Chaotic/Lorenz", force_recompute=False, apply_blur=False
                                , sigma=(1.5, 0.5))
        