import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservoirgrid.helpers import chaos_utils
from reservoirgrid.helpers import utils
from reservoirgrid.helpers import viz

###--------------------Functions--------------------###

def plot_attractor_grid(indices, metric_values, metric_name, data_source, 
                        filename="output_beautiful.svg", cols=None, save_image=False,
                        system_name=None, dying_pct=None):
    """
    Plots a highly polished, beautiful 2D projection (X vs Z) of Lorenz attractors.
    Optionally renders a small system name + dying percentage heading above the grid.
    """
    n = len(indices)
    if n == 0:
        print("No data provided to plot.")
        return

    if cols is None:
        cols = n if n <= 4 else (4 if n <= 16 else 6)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2),
                             squeeze=False, facecolor="#F8FAFC")
    axes_flat = axes.flatten()

    # ── Small heading: system name + dying % ──────────────────────────────────
    if system_name is not None or dying_pct is not None:
        parts = []
        if system_name:
            parts.append(system_name)
        if dying_pct is not None:
            parts.append(f"Dying: {dying_pct:.1f}%")
        heading = "  ·  ".join(parts)

        fig.text(
            0.01, 0.99,           # top-left corner
            heading,
            fontsize=8.5,
            fontfamily="monospace",
            color="#64748B",      # muted slate — unobtrusive
            weight="semibold",
            va="top", ha="left",
            transform=fig.transFigure,
        )
    # ──────────────────────────────────────────────────────────────────────────

    for i, u in enumerate(indices):
        ax = axes_flat[i]
        true_trajectory = np.array(data_source[u]["true_value"])
        pred_trajectory = np.array(data_source[u]["predictions"])

        score = metric_values[i]
        is_dying = str(chaos_utils.trajectory_vitality(true_trajectory, pred_trajectory)["is_dying"])

        ax.plot(true_trajectory[:, 0], true_trajectory[:, 2],
                color="#4A5568", linewidth=0.8, alpha=0.25, label="True", zorder=1)
        ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 2],
                color="#FF5A36", linewidth=1.1, linestyle=':', alpha=0.95, label="Pred", zorder=2)

        title_text = f"{metric_name}: {score:.4f}\nDying: {is_dying}"
        ax.set_title(title_text, fontsize=8.0, pad=10, weight='semibold',
                     fontfamily='monospace', color="#1E293B", loc='left')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#E2E8F0")
            spine.set_linewidth(1.0)
        ax.set_facecolor("#FFFFFF")

    plt.tight_layout(pad=3.0)
    # Leave a little room at the top for the heading
    if system_name is not None or dying_pct is not None:
        plt.subplots_adjust(top=0.90)

    if save_image:
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Saving beautiful vector graphic to {filename}...")
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=300,
                    facecolor=fig.get_facecolor())
        plt.close()
    else:
        plt.show()


def visualize_metric(metric_name, top_n: int, top_n_percentage: int, **kwargs):
    best_subset = matrices[metric_name].nsmallest(top_n)
    idx = best_subset.index.values
    values = best_subset.values

    # Compute dying % for the same top_n to show in the heading
    dying_pct = compute_dying_percentage(metric_name, top_n=top_n_percentage)

    path = os.path.join(save_path, system_name, f"{metric_name}.png")
    plot_attractor_grid(
        idx, values, metric_name, data_10,
        system_name=system_name,
        filename=path,
        dying_pct=dying_pct,
        **kwargs     
    )
    
def compute_dying_percentage(metric_name, top_n=10):
    """
    Calculates the percentage of trajectories classified as 'is_dying'
    within the best performing subset for a given metric.
    """
    
    best_subset = matrices[metric_name].nsmallest(top_n)
    indices = best_subset.index.values
    
    dying_count = 0
    total_items = len(indices)
    
    if total_items == 0:
        print(f"No records found for metric: {metric_name}")
        return 0.0

    # Match the data retrieval loop from your grid visualization function
    for u in indices:
        true_val = data_10[u]["true_value"]
        pred_val = data_10[u]["predictions"]
        
        # Run the same vitality utility check
        vitality_status = chaos_utils.trajectory_vitality(true_val, pred_val)
        
        # Check if the boolean or string flag is marked True
        if str(vitality_status.get("is_dying", "")).lower() == "true":
            dying_count += 1
            
    # Calculate the percentage
    percentage = (dying_count / total_items) * 100
    
    print(f"[{metric_name} - Best {top_n}] Dying Rate: {percentage:.2f}% ({dying_count}/{total_items})")
    return percentage


###------------- Computation -------------###

path = "research/Input_Discretization/results/Chaotic/"
save_path = "research/Input_Discretization/Plots/"
system_name = "ThomasLHS"
system_path = os.path.join(path, system_name)

file = os.path.join(system_path, "80.0.pkl")
with open(file, "rb") as f:
    data_10 = pickle.load(f)

rows = []
for data in data_10:
    # Extract values
    true, pred = data["true_value"], data["predictions"]
    
    # Build a dictionary for this iteration
    row = {
        #"Lyapunov Exponent": chaos_utils.Lyapunov_exponent(true, pred, fit_end=30),
        "KL Divergence": chaos_utils.kl_divergence(true, pred),
        "JS Divergence": chaos_utils.js_divergence(true, pred),
        "Symmetric KL": chaos_utils.symmetric_kl(true, pred),
        #"Correlation Dimension": chaos_utils.correlation_dimension(true, pred),
        "PSD Error": chaos_utils.psd_error(true, pred)[0],
        "RMSE": utils.RMSE(true, pred),
        "True": true,
        "Pred": pred
        #"params": data["parameters"]
    }
    rows.append(row)


# Create the DataFrame
matrices = pd.DataFrame(rows)


visualize_metric("JS Divergence", top_n=8, top_n_percentage=100, save_image=True)
visualize_metric("RMSE", top_n=16, top_n_percentage=100, save_image=True)
visualize_metric("KL Divergence", top_n=8, top_n_percentage=100, save_image=True)
visualize_metric("Symmetric KL", top_n=8, top_n_percentage=100, save_image=True)
visualize_metric("PSD Error", top_n=8, top_n_percentage=100, save_image=True)


















































