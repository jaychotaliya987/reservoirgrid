from timeit import default_timer as timer
import gc
import tracemalloc

import os
import torch
torch._dynamo.config.dynamic_shapes = True
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"C:\Users\jaych\ReservoirGrid\inductor_cache"

import numpy as np
import pickle

from reservoirgrid.helpers import utils
from scipy.stats import qmc

TARGET_PP = [100] 
system_list = ["MultiChua"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameter Sweep Setup
d = 3
n = 1000

sampler = qmc.LatinHypercube(d=d, rng = 42)
sample_01 = sampler.random(n=n)

l_bounds = [0.5, 0.1, 0.2]  
u_bounds = [1.5, 0.95, 0.9] 

scaled_sample = np.round(qmc.scale(sample_01, l_bounds, u_bounds), 2)

parameter_dict = {
    "SpectralRadius": scaled_sample[:, 0],
    "LeakyRate":      scaled_sample[:, 1],
    "InputScaling":   scaled_sample[:, 2]
}

print(f"Total parameter combinations: {len(parameter_dict['SpectralRadius'])}")
print(f"Device: {device}\n")

for system in system_list:
    system_name = "/Chaotic/" + system
    print(f"Selected system: {system_name}")
    path = "../../src/reservoirgrid/datasets" + system_name + ".npy"

    if not os.path.exists(path):
        print("System does not exist. Generate first.")
        exit()
    else:
        print("Loading system from datasets...")
        system_data = np.load(path, allow_pickle=True)
        print("System loaded.\n")

    T_system = utils.truncate(system_data)
    
    selected_indices = []
    for i in range(len(T_system['pp'])):
        if T_system[i][0] in TARGET_PP:
            selected_indices.append(i)
        elif TARGET_PP == 'all':
            selected_indices.append(i)

    if not selected_indices:
        print(f"Warning: TARGET_PP {TARGET_PP} not found in system {system_name}. Skipping.")
        continue
    
    for pp_select in selected_indices:
        pp_value = T_system[pp_select][0]
        print(f"\n{'='*70}")
        print(f"Processing PP (points per period): {pp_value}")
        print(f"{'='*70}")
        
        overall_start = timer()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Data preparation
        input_data = T_system[pp_select][1]
        input_data = utils.normalize_data(input_data).astype(np.float32)
        
        r_dim = input_data.shape[1]
        
        print(f"Input shape: {input_data.shape}")
        print(f"Input/Output dim: {r_dim}")
        
        # IMPROVED: Call batched parameter_sweep with batch_size tuned to GPU
        results = utils.parameter_sweep(
            inputs = input_data,
            parameter_dict = parameter_dict,
            reservoir_dim = 1300,
            input_dim = r_dim,
            output_dim = r_dim,
            sparsity = 0.9,
            return_targets = True,
            batch_size = 32
        )

        # Save results
        pp_num = str(pp_value)
        result_folder = "results" + system_name + "LHS"
        result_path = os.path.join(result_folder, f"{pp_num}.pkl")
        
        os.makedirs(result_folder, exist_ok=True)
        
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {result_path}")

        del results
        gc.collect()

        overall_end = timer()
        snapshot2 = tracemalloc.take_snapshot()
        stats = snapshot2.compare_to(snapshot1, 'lineno')

        print(f"Memory released: {stats[0].size_diff / 10**6:.2f} MB")
        print(f"Total time for PP {pp_value}: {overall_end - overall_start:.2f} seconds")
        print(f"{'='*70}\n")
