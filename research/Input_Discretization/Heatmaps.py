import ast
import gc
import json
import logging
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --- 1. CONFIGURATION & LOGGING ---
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
    "Cos_Sim":      {"cmap": "RdBu_r",     "plotly_scale": "RdBu"},
}


# --- 2. LOW-LEVEL UTILITIES ---
def make_param_combo(x: Any) -> Optional[tuple]:
    """Safely converts a parameter dictionary into a hashable, sorted tuple."""
    try:
        return tuple(sorted(x.items()))
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not hash params: {x} — {e}")
        return None


def make_sort_key(param_name: str):
    """Dynamically creates a sorting key function based on a specific parameter string."""
    def key_function(combo: tuple) -> tuple:
        combo_dict = dict(combo)
        return (combo_dict.get(param_name, 0.0), combo)
    return key_function


# --- 3. DATA COMPUTATION CORE (PARALLEL PROCESSING) ---
def _compute_entry_metrics(entry: dict, ppp: float, selection: int = 1) -> Optional[Dict[str, Any]]:
    """Computes all chaotic evaluation metrics for a single simulation entry."""
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
        lyap    = chaos_utils.lyapunov_time(truth=true_val, predictions=preds)
        skl     = chaos_utils.symmetric_kl(true_val, preds)
        psd, cs = chaos_utils.psd_error(true_val, preds)
        rmse    = utils.RMSE(true_val, preds)
        jsdiv   = chaos_utils.js_divergence(true_val, preds)
    except Exception as e:
        logger.warning(f"Metric computation failed: {e}")
        return None

    return {
        "ppp": ppp, "params": params, "param_combo": param_combo,
        "LyapunovTime": lyap, "Symmetric KLDivergence": skl, "PSD Errors": psd,
        "Cos_Sim": cs, "RMSE": rmse, "JSDivergence": jsdiv
    }


def _process_single_file(args: tuple) -> List[Dict[str, Any]]:
    """Loads a single pickle file and updates entries using thread-level parallelism."""
    file_path, entry_workers = args

    try:
        ppp = float(file_path.stem)
    except ValueError:
        logger.warning(f"Skipping non-numeric file name: {file_path.name}")
        return []

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=entry_workers) as pool:
        futures = [pool.submit(_compute_entry_metrics, entry, ppp) for entry in data]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results.append(res)
    return results


def extract_metrics_from_folder(folder_path: Path, file_workers: int = 4, entry_workers: int = 24) -> pd.DataFrame:
    """Orchestrates 2-level nested parallelism to process all pickle data in a target directory."""
    file_list = sorted(folder_path.glob("*.pkl"))
    if not file_list:
        logger.warning(f"No .pkl files found in: {folder_path}")
        return pd.DataFrame()

    all_data = []
    task_args = [(f, entry_workers) for f in file_list]

    with ProcessPoolExecutor(max_workers=file_workers) as executor:
        futures = {executor.submit(_process_single_file, args): args[0] for args in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            try:
                all_data.extend(future.result())
            except Exception as e:
                logger.error(f"Worker failed for {futures[future].name}: {e}")

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


# --- 4. DATA STATE & CACHING LAYER ---
def save_heatmap_metadata(pivot_df: pd.DataFrame, metric: str, folder_name: str, output_base_dir: Path) -> None:
    """Saves structural mapping coordinates and data legends for caching and lookups."""
    save_dir = output_base_dir / folder_name / "metadata"
    save_dir.mkdir(parents=True, exist_ok=True)

    ppp_values   = pivot_df.index.to_numpy()
    param_combos = pivot_df.columns.to_numpy()

    # 1. Save CSV Map Index
    records = [
        {
            "row_idx": r, "col_idx": c, "ppp": float(ppp),
            metric: float(pivot_df.iloc[r, c]) if not pd.isna(pivot_df.iloc[r, c]) else None
        }
        for r, ppp in enumerate(ppp_values)
        for c, combo in enumerate(param_combos)
    ]
    pd.DataFrame(records).to_csv(save_dir / f"{metric.lower().replace(' ', '_')}_index_map.csv", index=False)

    # 2. Save Parameter JSON Legend
    legend = [
        {"col_idx": c, "param_combo": {k: float(v) if hasattr(v, 'item') else v for k, v in combo}}
        for c, combo in enumerate(param_combos)
    ]
    with open(save_dir / f"{metric.lower().replace(' ', '_')}_param_legend.json", "w") as f:
        json.dump(legend, f, indent=2)


def load_from_metadata(folder_name: str, output_base_dir: Path, target_metrics: List[str]) -> Optional[pd.DataFrame]:
    """Reconstructs the long-form metrics DataFrame from CSV/JSON cache files."""
    save_dir = output_base_dir / folder_name
    frames = []

    for metric in target_metrics:
        index_map_path = save_dir / f"{metric.lower().replace(' ', '_')}_index_map.csv"
        legend_path    = save_dir / f"{metric.lower().replace(' ', '_')}_param_legend.json"

        if not index_map_path.exists() or not legend_path.exists():
            return None

        df_metric = pd.read_csv(index_map_path)
        with open(legend_path) as f:
            legend = json.load(f)

        col_idx_to_combo = {item["col_idx"]: tuple(sorted(item["param_combo"].items())) for item in legend}
        df_metric["param_combo"] = df_metric["col_idx"].map(col_idx_to_combo)
        frames.append(df_metric[["ppp", "param_combo", metric]])

    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on=["ppp", "param_combo"], how="outer")
    return df


# --- 5. VISUALIZATION PREPARATION SHARED LOGIC ---
def _prepare_heatmap_data(df: pd.DataFrame, metric: str, sort_by_param: str, apply_blur: bool, sigma: tuple) -> Tuple[pd.DataFrame, np.ndarray]:
    """Shared pipeline asset: Aggregates, pivots, sorts, and fills missing matrix values."""
    df_agg = df.groupby(["ppp", "param_combo"], as_index=False)[[metric]].mean()
    
    pivot_df = df_agg.pivot(index="ppp", columns="param_combo", values=metric).sort_index(ascending=True)
    # if len(pivot_df.columns) > 0:
    #     sample_key = list(dict(pivot_df.columns[0]).keys())
    #     print(f"Sample parameter keys for sorting: {sample_key}")

    # Dynamic parameter layout execution
    sorted_columns = sorted(pivot_df.columns, key=make_sort_key(sort_by_param))
    pivot_df = pivot_df.reindex(columns=sorted_columns)

    # Impute missing values via column medians
    heatmap_matrix = pivot_df.apply(lambda col: col.fillna(col.median())).to_numpy()
    if apply_blur:
        heatmap_matrix = gaussian_filter(heatmap_matrix, sigma=sigma)
        
    return pivot_df, heatmap_matrix


# --- 6. VISUALIZATION ENGINE ---
def plot_and_save_heatmap(df: pd.DataFrame, metric: str, folder_name: str, output_base_dir: Path, apply_blur: bool, sigma: tuple, sort_by_param: str) -> None:
    """Renders and saves a publication-ready static matplotlib heatmap landscape."""
    pivot_df, heatmap_matrix = _prepare_heatmap_data(df, metric, sort_by_param, apply_blur, sigma)
    save_heatmap_metadata(pivot_df, metric, folder_name, output_base_dir)

    ppp_values = pivot_df.index.to_numpy()
    num_combos = heatmap_matrix.shape[1]

    # Handle Normalization Scalings
    if metric in ["PSD Errors", "RMSE"] and (heatmap_matrix > 0).any():
        norm = LogNorm(vmin=np.nanmin(heatmap_matrix[heatmap_matrix > 0]), vmax=np.nanmax(heatmap_matrix))
    elif metric == "Cos_Sim":
        norm = Normalize(vmin=-1.0, vmax=1.0)
    else:
        norm = Normalize(vmin=np.nanmin(heatmap_matrix), vmax=np.nanmax(heatmap_matrix))

    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        if style in plt.style.available:
            plt.style.use(style)
            break

    fig, ax = plt.subplots(figsize=(16, 7), layout="tight")
    metric_cfg = METRIC_CONFIGS.get(metric, {"cmap": "viridis"})

    cax = ax.imshow(
        heatmap_matrix, cmap=metric_cfg["cmap"], aspect="auto", origin="lower",
        interpolation="bilinear" if apply_blur else "nearest", norm=norm
    )

    # X-Axis Layout Config
    xticks = list(range(0, num_combos, 5))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks], fontsize=5, color="#2c3e50", rotation= 90, ha ="center")
    ax.set_xlabel(f"Parameter Index (Sorted by {sort_by_param})", fontsize=12, fontweight="bold", labelpad=12)

    # Y-Axis Layout Config
    y_tick_step = max(1, len(ppp_values) // 10)
    yticks = list(range(0, len(ppp_values), y_tick_step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(ppp_values[i])}" for i in yticks], fontsize=8, color="#2c3e50")
    ax.set_ylabel("Points per Period (PPP)", fontsize=12, fontweight="bold", labelpad=12)

    cbar_formatter = ticker.LogFormatterMathtext() if isinstance(norm, LogNorm) else ticker.ScalarFormatter()
    cbar = fig.colorbar(cax, ax=ax, pad=0.02, fraction=0.046, format=cbar_formatter)
    cbar.set_label(label=f"Measured: {metric}", fontsize=11, fontweight="bold", labelpad=10)

    ax.set_title(f"Evaluation Landscape: {metric} ({folder_name})", fontsize=14, fontweight="bold", pad=16, color="#1a252f")
    ax.grid(False)

    save_dir = output_base_dir / folder_name
    fig.savefig(save_dir / f"{metric.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_interactive_heatmap(df: pd.DataFrame, metric: str, folder_name: str, output_base_dir: Path, apply_blur: bool, sigma: tuple, sort_by_param: str) -> None:
    """Generates an advanced browser-interactive Plotly heatmap document."""
    pivot_df, heatmap_matrix = _prepare_heatmap_data(df, metric, sort_by_param, apply_blur, sigma)
    ppp_values = pivot_df.index.to_numpy()
    param_combos = pivot_df.columns.to_numpy()

    # Build Hover Metadata Structures
    hover_text = [
        [
            f"<b>{metric}:</b> {heatmap_matrix[r, c]:.6f}<br>"
            f"<b>Parameter Index:</b> {c}<br>"
            f"<b>Points per Period:</b> {ppp:.2f}<br>"
            f"<b>Parameters:</b><br>" + "<br>".join(f"  {k}: {v}" for k, v in combo)
            for c, combo in enumerate(param_combos)
        ]
        for r, ppp in enumerate(ppp_values)
    ]

    plot_matrix = heatmap_matrix.copy()
    metric_cfg = METRIC_CONFIGS.get(metric, {"plotly_scale": "viridis"})
    colorbar_title = metric
    
    if metric in ["PSD Errors", "RMSE"] and (plot_matrix > 0).any():
        plot_matrix = np.where(plot_matrix > 0, np.log10(plot_matrix + 1e-10), np.nan)
        colorbar_title = f"log₁₀({metric})"

    fig = go.Figure(data=go.Heatmap(
        z=plot_matrix, x=list(range(len(param_combos))), y=[f"{int(p)}" for p in ppp_values],
        colorscale=metric_cfg["plotly_scale"], hoverinfo="text", text=hover_text,
        zsmooth="best" if apply_blur else False,
        colorbar=dict(title=dict(text=colorbar_title, side="right"), tickfont=dict(size=10)),
    ))

    fig.update_layout(
        title=dict(text=f"Evaluation Landscape: {metric} ({folder_name})", font=dict(size=16, color="#1a252f"), x=0.5),
        xaxis=dict(title=f"Parameter Index (Sorted by {sort_by_param})", tickmode="linear", tick0=0, dtick=5, tickfont=dict(size=5)),
        yaxis=dict(title="Points per Period (PPP)", tickfont=dict(size=8)),
        width=1200, height=550, paper_bgcolor="white", plot_bgcolor="white"
    )

    save_dir = output_base_dir / folder_name
    fig.write_html(save_dir / f"{metric.lower().replace(' ', '_')}_interactive.html", include_plotlyjs="cdn", full_html=True)


# --- 7. MASTER PIPELINE DIRECTOR ---
def process_single_location(
    target_pkl_dir: str,
    output_base_dir: str = "research/Input_Discretization/Plots/HeatMaps/",
    file_workers: int = 4,
    entry_workers: int = 24,
    force_recompute: bool = False,
    apply_blur: bool = False,
    sigma: tuple = (1.5, 0.5),
    sort_by_param: str = "spectral_radius"
) -> None:
    """Defines a strict, non-cyclical pipeline assembly line execution loop."""
    input_path, output_path = Path(target_pkl_dir), Path(output_base_dir)
    if not input_path.is_dir():
        logger.error(f"Target path '{input_path}' is not a directory.")
        return

    folder_name    = input_path.name
    target_metrics = ["LyapunovTime", "Symmetric KLDivergence", "PSD Errors", "RMSE", "Cos_Sim", "JSDivergence"]

    # Step 1: Manage Data Lifecycles cleanly
    df = None if force_recompute else load_from_metadata(folder_name, output_path, target_metrics)

    if df is None:
        logger.info(f"Cache miss. Computing metrics from scratch for: {input_path}")
        df = extract_metrics_from_folder(input_path, file_workers=file_workers, entry_workers=entry_workers)
        
    if df.empty:
        logger.warning(f"No usable evaluation data managed for {folder_name}")
        return

    # Step 2: Execute Visualizations
    for metric in target_metrics:
        logger.info(f"Plotting Metric Landscape Component: {metric}")
        plot_and_save_heatmap(df, metric, folder_name, output_path, apply_blur, sigma, sort_by_param)
        plot_interactive_heatmap(df, metric, folder_name, output_path, apply_blur, sigma, sort_by_param)

    del df
    gc.collect()
    logger.info(f"Pipeline complete for target run: '{folder_name}'")


if __name__ == "__main__":
    process_single_location(
        "research/Input_Discretization/results/Chaotic/GuckenheimerHolmesLHS",
        force_recompute=True,
        apply_blur=True,
        sigma=(1.5, 0.5),
        sort_by_param="LeakyRate"
    )
