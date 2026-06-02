'''
Main Reservoir class for the ReservoirGrid project.
This class implements a reservoir computing model with the following features:
- Reservoir state update with leaky integration
- Readout layer for prediction
- Training of the readout layer with ridge regression
- Optional training of the reservoir with backpropagation
- Prediction with optional teacher forcing
- Saving and loading of the model
- Echo state property checks
- Reservoir state and weights visualization and control
- Device management (CPU/GPU)

Unified Batched Architecture:
    All operations are treated as batched (B >= 1) utilizing torch.bmm and torch.einsum.
    Pass 1D array/tensor for spectral_radius, leak_rate, input_scaling
    to run B reservoir configs in parallel (e.g. for hyperparameter sweeps).
    For a single configuration, pass scalars, and the model will default to B=1.

Prebuilt weights mode (sweep optimisation):
    Pass prebuilt_weights={"W_in": tensor, "W": tensor} to skip random
    generation and eigval computation entirely. Used by parameter_sweep
    to build all N reservoir matrices once before the batch loop.
'''

import torch
from torch import nn
from torch import optim
import optuna

from typing import Optional, Callable, Type, Union, Dict
import numpy as np

# Default device (can be overridden)
_DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DEFAULT_DTYPE = torch.float32

class Reservoir(nn.Module):
    """
    A unified batched Echo State Network (ESN) Reservoir Computing model.

    This class supports executing B parallel reservoir configurations simultaneously 
    on the GPU/CPU to facilitate high-throughput hyperparameter sweeps and optimizations.
    All math blocks and internal operations default to batch tensor operations.
    """
    def __init__(self,
                 input_dim: int,
                 reservoir_dim: int,
                 output_dim: int,
                 spectral_radius: Union[float, np.ndarray, torch.Tensor] = 0.9,
                 leak_rate: Union[float, np.ndarray, torch.Tensor] = 0.3,
                 sparsity: float = 0.9,
                 input_scaling: Union[float, np.ndarray, torch.Tensor] = 1.0,
                 noise_level: float = 0.01,
                 activation: Callable = torch.tanh,
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = _DEFAULT_DTYPE,
                 prebuilt_weights: Optional[Dict[str, torch.Tensor]] = None):
        """
        Initializes the Reservoir model and sets up configuration parameters.

        Args:
            input_dim (int): Dimensionality of the input timeseries data.
            reservoir_dim (int): Number of internal hidden recurrent neurons.
            output_dim (int): Dimensionality of the target/output timeseries data.
            spectral_radius (Union[float, np.ndarray, torch.Tensor]): Target spectral radius 
                for the reservoir weights. Scalar or 1D array/tensor of shape (B,). Defaults to 0.9.
            leak_rate (Union[float, np.ndarray, torch.Tensor]): Leaky integration parameter 
                governing state retention. Scalar or 1D array/tensor of shape (B,). Defaults to 0.3.
            sparsity (float): Proportion of zero-valued weights in the reservoir matrix. Defaults to 0.9.
            input_scaling (Union[float, np.ndarray, torch.Tensor]): Uniform scaling multiplier 
                for input weights. Scalar or 1D array/tensor of shape (B,). Defaults to 1.0.
            noise_level (float): Standard deviation of Gaussian noise injected into state updates. Defaults to 0.01.
            activation (Callable): Non-linear element-wise activation function. Defaults to torch.tanh.
            device (Optional[Union[str, torch.device]]): Target torch device context. Defaults to CPU/GPU auto-select.
            dtype (torch.dtype): Tensor precision data type. Defaults to torch.float32.
            prebuilt_weights (Optional[Dict[str, torch.Tensor]]): External matrices dictionary 
                containing pre-initialized "W_in" and "W" tensors to skip creation steps. Defaults to None.
        """
        super(Reservoir, self).__init__()

        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.noise_level = noise_level
        self.activation = activation
        self.device = torch.device(device) if device else _DEFAULT_DEVICE
        self.dtype = dtype

        def _to_tensor(val):
            if isinstance(val, torch.Tensor):
                return val.to(self.device, self.dtype)
            if isinstance(val, np.ndarray):
                return torch.tensor(val, device=self.device, dtype=self.dtype)
            return torch.tensor([val], device=self.device, dtype=self.dtype)

        sr_t = _to_tensor(spectral_radius)
        lr_t = _to_tensor(leak_rate)
        is_t = _to_tensor(input_scaling)

        self.B = sr_t.numel()

        self.register_buffer("spectral_radii", sr_t)
        self.register_buffer("leak_rates",     lr_t)
        self.register_buffer("input_scalings", is_t)

        # --- Parameter Validation ---
        assert torch.all(lr_t >= 0) and torch.all(lr_t <= 1), "Leak rate must be in [0, 1]"
        assert 0.0 <= sparsity <= 1.0,                        "Sparsity must be in [0, 1]"
        assert torch.all(sr_t >= 0),                          "Spectral radius must be non-negative"
        assert reservoir_dim > 0,                             "Reservoir dimension must be positive"

        # --- Initialize Weights ---
        if prebuilt_weights is not None:
            W_in = prebuilt_weights["W_in"].to(self.device, self.dtype)
            W    = prebuilt_weights["W"].to(self.device, self.dtype)
            assert W_in.shape == (self.B, reservoir_dim, input_dim), \
                f"prebuilt W_in shape mismatch: expected {(self.B, reservoir_dim, input_dim)}, got {W_in.shape}"
            assert W.shape == (self.B, reservoir_dim, reservoir_dim), \
                f"prebuilt W shape mismatch: expected {(self.B, reservoir_dim, reservoir_dim)}, got {W.shape}"
        else:
            W_in = (torch.rand(self.B, reservoir_dim, input_dim, device=self.device, dtype=dtype) * 2 - 1)
            W_in = W_in * is_t[:, None, None]

            W = torch.rand(self.B, reservoir_dim, reservoir_dim, device=self.device, dtype=dtype) * 2 - 1
            mask = (torch.rand_like(W) > sparsity).to(dtype)
            W = W * mask

            try:
                eigs = torch.linalg.eigvals(W)
                current_sr = torch.max(eigs.abs(), dim=-1).values
                current_sr = current_sr.clamp(min=1e-9)
                scale = sr_t / current_sr
                W = W * scale[:, None, None]
            except torch.linalg.LinAlgError:
                print("Warning: Eigenvalue computation failed. Using unscaled reservoir weights.")

        self.W_in = nn.Parameter(W_in, requires_grad=False)   # (B, R, I)
        self.W    = nn.Parameter(W,    requires_grad=False)   # (B, R, R)

        # --- Readout layer buffers (Replaces nn.Linear) ---
        self.register_buffer(
            "W_out", torch.zeros(self.B, output_dim, reservoir_dim, device=self.device, dtype=dtype)
        )
        self.register_buffer(
            "b_out", torch.zeros(self.B, output_dim, device=self.device, dtype=dtype)
        )

        # --- Initial reservoir state (B, R) ---
        self.register_buffer(
            'reservoir_states_buf',
            torch.zeros(self.B, reservoir_dim, device=self.device, dtype=dtype)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _readout(self, states: torch.Tensor) -> torch.Tensor:
        """
        Maps collected hidden reservoir internal states to output prediction space.

        Args:
            states (torch.Tensor): Collected internal activations of shape (T, B, R).

        Returns:
            torch.Tensor: Readout model predictions of shape (T, B, O).
        """
        return torch.einsum("tbr,bor->tbo", states, self.W_out) + self.b_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, u: torch.Tensor, reset_state: bool = True) -> torch.Tensor:
        """
        Executes a sequential step-by-step forward pass through the hidden reservoir.

        Loops across the temporal length, projects external signals via W_in, computes
        recurrent updates using W, accumulates with leak parameters, and maps results to W_out.

        Args:
            u (torch.Tensor): External driving sequential input. Shape (T, I) or (T, 1, I).
            reset_state (bool): If True, clears past hidden state memory vectors to zero. Defaults to True.

        Returns:
            torch.Tensor: Evaluated outputs matching all B parallel setups. Shape (T, B, O).
        """
        u = u.to(self.device, self.dtype)

        if u.ndim == 3:
            u = u.squeeze(1)
        T = u.shape[0]

        if reset_state:
            self.reservoir_states_buf = torch.zeros(
                self.B, self.reservoir_dim, device=self.device, dtype=self.dtype
            )

        # Preallocate (T, B, R)
        self.reservoir_states = torch.empty(
            T, self.B, self.reservoir_dim, device=self.device, dtype=self.dtype
        )

        # Preallocate full noise tensor
        noise_all = torch.randn(
            T, self.B, self.reservoir_dim, device=self.device, dtype=self.dtype
        ) * self.noise_level

        leak = self.leak_rates[:, None]   # (B, 1)

        for t in range(T):
            ut = u[t]   # (I,)
            input_term     = torch.einsum("bri,i->br", self.W_in, ut)
            recurrent_term = torch.bmm(self.W, self.reservoir_states_buf.unsqueeze(-1)).squeeze(-1)
            activated = self.activation(input_term + recurrent_term + noise_all[t])
            self.reservoir_states_buf = (1.0 - leak) * self.reservoir_states_buf + leak * activated
            self.reservoir_states[t] = self.reservoir_states_buf

        return self._readout(self.reservoir_states)
    

    # ------------------------------------------------------------------
    # Train readout
    # -----------------------------------------------------------------
    def train_readout(self,
                      inputs: torch.Tensor,
                      targets: torch.Tensor,
                      warmup: int = 0,
                      alpha: float = 1e-6,
                      chunk_size: int = 5000):
        """
        Trains the linear readout layers using Chunked (Streaming) Ridge Regression.

        Processes long inputs iteratively in window chunks to maintain a safe VRAM 
        footprint. Collects and updates the global analytical Tikhonov system covariance 
        matrices ($X^TX$, $X^TY$) before optimizing with structural linear solver calls.

        Args:
            inputs (torch.Tensor): Training sequential input sequence. Shape (T, I) or (T, 1, I).
            targets (torch.Tensor): Corresponding ground-truth target sequence. Shape (T, O).
            warmup (int): Transient initial timesteps to discard before parameter fitting. Defaults to 0.
            alpha (float): Ridge (Tikhonov) L2 regularization strength factor. Defaults to 1e-6.
            chunk_size (int): Temporal size of windows split for sequential execution. Defaults to 5000.

        Returns:
            None: Modifies the internal object state buffers `W_out` and `b_out` in-place.
        """
        inputs  = inputs.to(self.device, self.dtype)
        targets = targets.to(self.device, self.dtype)

        if inputs.ndim == 3:
            inputs = inputs.squeeze(1)
        
        T = inputs.shape[0]
        
        # 1. Preallocate running accumulators directly on the GPU
        XtX_total = torch.zeros(self.B, self.reservoir_dim, self.reservoir_dim, 
                                device=self.device, dtype=self.dtype)
        XtY_total = torch.zeros(self.B, self.reservoir_dim, self.output_dim, 
                                device=self.device, dtype=self.dtype)
        
        reset_state = True

        # 2. Stream through the time dimension in chunks
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            
            if end <= warmup:
                with torch.no_grad():
                    self.forward(inputs[start:end], reset_state=reset_state)
                reset_state = False
                continue
            
            with torch.no_grad():
                self.forward(inputs[start:end], reset_state=reset_state)
            reset_state = False 
            
            chunk_warmup = max(0, warmup - start)
            
            X_chunk = self.reservoir_states[chunk_warmup:]  # (T_chunk_eff, B, R)
            Y_chunk = targets[start + chunk_warmup:end]     # (T_chunk_eff, O)
            
            T_chunk_eff = X_chunk.shape[0]
            if T_chunk_eff == 0:
                continue
                
            Y_chunk_exp = Y_chunk.unsqueeze(1).expand(T_chunk_eff, self.B, self.output_dim)
            X_b = X_chunk.permute(1, 0, 2)       # (B, T_chunk_eff, R)
            Y_b = Y_chunk_exp.permute(1, 0, 2)   # (B, T_chunk_eff, O)
            
            # 3. Incrementally accumulate into the total correlation buffers
            XtX_total.add_(torch.bmm(X_b.transpose(1, 2), X_b))
            XtY_total.add_(torch.bmm(X_b.transpose(1, 2), Y_b))
            
        # 4. Apply Tikhonov regularization and solve the accumulated global system
        I = torch.eye(self.reservoir_dim, device=self.device, dtype=self.dtype).unsqueeze(0)
        A = XtX_total + alpha * I

        try:
            solution = torch.linalg.solve(A, XtY_total)   # (B, R, O)
        except torch.linalg.LinAlgError:
            print("Warning: solve failed, falling back to lstsq.")
            solution = torch.linalg.lstsq(A, XtY_total).solution

        # 5. Overwrite readout layer parameters
        with torch.no_grad():
            self.W_out.copy_(solution.transpose(1, 2))
            self.b_out.zero_()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self,
                initial_input: torch.Tensor,
                steps: int,
                teacher_forcing_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates multi-step ahead autonomous sequence forecasts.

        Warms up internal states with the `initial_input` sequence, and then rolls 
        forward iteratively. The output at step $t$ is fed back as the input for step $t+1$, 
        with optional override overrides if `teacher_forcing_targets` are provided.

        Args:
            initial_input (torch.Tensor): Prompting sequence for initial state warmup. Shape (T, I).
            steps (int): Total sequence length iterations to run predictions autonomously.
            teacher_forcing_targets (Optional[torch.Tensor]): Optional array containing ground truth
                override targets for each loop step. Shape (Steps, O). Defaults to None.

        Returns:
            torch.Tensor: Compiled forecast output sequence array. Shape (Steps, B, O).
        """
        initial_input = initial_input.to(self.device, self.dtype)
        if initial_input.ndim == 3:
            initial_input = initial_input.squeeze(1)

        if self.output_dim != self.input_dim and teacher_forcing_targets is None:
            raise ValueError("output_dim must match input_dim for autonomous prediction.")

        self.eval()

        with torch.no_grad():
            self.forward(initial_input, reset_state=True)

            current = (torch.einsum("br,bor->bo", self.reservoir_states_buf, self.W_out) + self.b_out)
            leak = self.leak_rates[:, None]
            predictions = []

            for step in range(steps):
                input_term     = torch.einsum("bri,bi->br", self.W_in, current)
                recurrent_term = torch.bmm(self.W, self.reservoir_states_buf.unsqueeze(-1)).squeeze(-1)
                activated      = self.activation(input_term + recurrent_term)
                self.reservoir_states_buf = (1.0 - leak) * self.reservoir_states_buf + leak * activated

                pred = (torch.einsum("br,bor->bo", self.reservoir_states_buf, self.W_out) + self.b_out)
                predictions.append(pred)

                if teacher_forcing_targets is not None and step < teacher_forcing_targets.size(0):
                    current = teacher_forcing_targets[step].to(self.device, self.dtype)
                    if current.ndim == 1:
                        current = current.unsqueeze(0).expand(self.B, -1)
                else:
                    current = pred

        return torch.stack(predictions, dim=0)

    # ------------------------------------------------------------------
    # Reservoir control
    # ------------------------------------------------------------------

    def update_reservoir(self, u: torch.Tensor):
        """
        Manually forces/overwrites internal state buffer arrays with explicit user tensors.

        Args:
            u (torch.Tensor): Targeted hidden states replacement array. Shape (B, R).

        Returns:
            None
        """
        print("Warning: `update_reservoir` will update the reservoir states manually.")
        self.reservoir_states_buf = u
        self.reservoir_states = torch.cat(
            (self.reservoir_states, self.reservoir_states_buf.unsqueeze(0)), dim=0
        )

    def freeze_reservoir(self):
        """
        Disables gradient computing tracks for input and recurrent weights.

        Returns:
            None
        """
        self.W_in.requires_grad = False
        self.W.requires_grad    = False
        print("Reservoir weights (W_in, W) frozen.")

    def unfreeze_reservoir(self):
        """
        Enables gradient computing tracks for input and recurrent weights.

        Returns:
            None
        """
        self.W_in.requires_grad = True
        self.W.requires_grad    = True
        print("Reservoir weights (W_in, W) unfrozen.")

    def reset_state(self, batch_size: int = 1):
        """
        Resets and clears active tracking internal hidden memory buffers back to zero.

        Args:
            batch_size (int): Size of structural dimension mapping configurations.
                 If zero or negative, defaults back to the model's setup size `self.B`. Defaults to 1.

        Returns:
            None
        """
        b = batch_size if batch_size > 0 else self.B
        self.reservoir_states_buf = torch.zeros(
            b, self.reservoir_dim, device=self.device, dtype=self.dtype
        )
        print(f"Reservoir state reset for batch size {b}.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        """
        Serializes and exports model parameters and buffer states to disk.

        Args:
            path (str): File path string locating output target destination.

        Returns:
            None
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        """
        Imports and overwrites model weights/buffers using saved disk state snapshots.

        Args:
            path (str): File path string locating input target resource.
            map_location (Optional[Union[str, torch.device]]): Execution device target mapping. 
                Defaults to active model configuration device destination.

        Returns:
            None
        """
        if map_location is None:
            map_location = self.device
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(self.device)
        print(f"Model loaded from {path} to device {self.device}")

    # ------------------------------------------------------------------
    # Finetune
    # ------------------------------------------------------------------

    def finetune(self,
                 inputs: torch.Tensor,
                 targets: torch.Tensor,
                 epochs: int,
                 lr: float,
                 criterion_class: Type[nn.Module] = nn.MSELoss,
                 optimizer_class: Type[optim.Optimizer] = optim.Adam,
                 print_every: int = 10):
        """
        Fine-tunes internal parameters and readout buffers via optimization backpropagation.

        Args:
            inputs (torch.Tensor): Sequential input signals tensor.
            targets (torch.Tensor): Target sequential truth alignment tensor.
            epochs (int): Total validation gradient iteration cycles.
            lr (float): Step multiplier value optimization rate.
            criterion_class (Type[nn.Module]): Torch loss module implementation. Defaults to nn.MSELoss.
            optimizer_class (Type[optim.Optimizer]): Torch gradient algorithm constructor. Defaults to optim.Adam.
            print_every (int): Metric monitoring log frequency interval. Defaults to 10.

        Returns:
            None
        """
        pass

    # ------------------------------------------------------------------
    # In-Class Hyperparameter Optimization Routine
    # -----------------------------------------------------------------
    def optimize(self,
                 X_train: torch.Tensor,
                 Y_train: torch.Tensor,
                 X_val: torch.Tensor,
                 Y_val: torch.Tensor,
                 sampler: optuna.samplers.BaseSampler,
                 metric_fn: Callable,
                 n_trials: int = 100,
                 batch_size: int = 10,
                 direction: str = "minimize",
                 warmup: int = 100,
                 alpha: float = 1e-5
                ) -> dict:
        """
        Executes an in-class optimization search to identify elite performance parameters.

        Uses parallel batched trial evaluation tasks powered by Optuna TPE samplers. 
        Upon evaluation completion, automatically re-shapes and fits the instance into a 
        single-configuration optimized model layout ($B=1$) using the discovered parameters.

        Args:
            X_train (torch.Tensor): Training series input tensor data. Shape (T_train, I).
            Y_train (torch.Tensor): Training series labels/target alignment data. Shape (T_train, O).
            X_val (torch.Tensor): Verification sequence input data used for scores. Shape (T_val, I).
            Y_val (torch.Tensor): Verification validation target reference comparison data. Shape (T_val, O).
            metric_fn (Callable): Quantitative valuation scorer algorithm injected to rate prediction paths.
            n_trials (int): Global scale limit setting maximum parameter attempts. Defaults to 100.
            batch_size (int): Total parallel trial variations executed concurrently. Defaults to 10.
            direction (str): Optimization objective logic selection. "minimize" or "maximize". Defaults to "minimize".
            warmup (int): Transient initial sequences skipped over during analytical training steps. Defaults to 100.
            alpha (float): Ridge (Tikhonov) scaling regularization constant. Defaults to 1e-5.
            sampler (optuna.samplers.BaseSampler): Optuna sampler instance. Defaults to None.
        Returns:
            dict: Key-value attributes mapping the discovered optimal parameter configuration.
        """
        sampler = sampler or optuna.samplers.CmaEsSampler()

        X_train = X_train.to(self.device, self.dtype)
        Y_train = Y_train.to(self.device, self.dtype)
        X_val = X_val.to(self.device, self.dtype)
        Y_val = Y_val.to(self.device, self.dtype)

        study = optuna.create_study(direction=direction, sampler=sampler)
        num_batches = int(np.ceil(n_trials / batch_size))

        print(f"[{type(self).__name__}] Starting in-class optimization using '{metric_fn.__name__}' ({direction} mode)...")

        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_trials, desc="Optimizing Reservoir", unit="trial")
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # --- Track the absolute best matrices explicitly ---
        best_score_overall = float("inf") if direction == "minimize" else float("-inf")
        best_matrices = {}

        for b_idx in range(num_batches):
            trials = [study.ask() for _ in range(min(batch_size, n_trials - len(study.trials)))]
            if not trials:
                break

            sr_list, lr_list, is_list = [], [], []
            for trial in trials:
                sr_list.append(trial.suggest_float("spectral_radius", 0.1, 1.5, step=0.01))
                lr_list.append(trial.suggest_float("leak_rate", 0.05, 1.0, step=0.05))
                is_list.append(trial.suggest_float("input_scaling", 0.1, 1.0, step=0.01))

            search_batch = Reservoir(
                input_dim=self.input_dim,
                reservoir_dim=self.reservoir_dim,
                output_dim=self.output_dim,
                spectral_radius=np.array(sr_list),
                leak_rate=np.array(lr_list),
                input_scaling=np.array(is_list),
                sparsity=self.sparsity,
                noise_level=self.noise_level,
                activation=self.activation,
                device=self.device,
                dtype=self.dtype
            )

            search_batch.train_readout(inputs=X_train, targets=Y_train, warmup=warmup, alpha=alpha)

            search_batch.eval()
            with torch.no_grad():
                val_predictions = search_batch(X_val, reset_state=True)

            for i, trial in enumerate(trials):
                pred_slice = val_predictions[:, i, :]
                try:
                    score = metric_fn(Y_val, pred_slice)
                except TypeError:
                    score = metric_fn(Y_val.cpu().numpy(), pred_slice.cpu().numpy())

                study.tell(trial, score)

                # --- Capture the exact matrix slices if this is a new best ---
                is_best = (score < best_score_overall) if direction == "minimize" else (score > best_score_overall)
                if is_best:
                    best_score_overall = score
                    best_matrices["W_in"] = search_batch.W_in[i].clone()
                    best_matrices["W"] = search_batch.W[i].clone()
                    best_matrices["W_out"] = search_batch.W_out[i].clone()
                    best_matrices["b_out"] = search_batch.b_out[i].clone()

            if use_tqdm:
                try:
                    pbar.set_postfix({"best_score": f"{study.best_value:.5f}"})
                except ValueError:
                    pbar.set_postfix({"best_score": "NaN"})
                pbar.update(len(trials))
            else:
                current_best = study.best_value if len(study.trials) > 0 else "N/A"
                print(f" -> Batch {b_idx + 1}/{num_batches} processed. Current Best Score: {current_best}")

        if use_tqdm:
            pbar.close()

        print(f"Optimization complete! Best Parameters: {study.best_params}")

        # --- UPDATE THE CURRENT INSTANCE STATE ---
        print(f"Injecting exact winning matrices into the model...")
        
        self.B = 1
        
        self.register_buffer("spectral_radii", torch.tensor([study.best_params["spectral_radius"]], device=self.device, dtype=self.dtype))
        self.register_buffer("leak_rates", torch.tensor([study.best_params["leak_rate"]], device=self.device, dtype=self.dtype))
        self.register_buffer("input_scalings", torch.tensor([study.best_params["input_scaling"]], device=self.device, dtype=self.dtype))

        # 1. Inject the perfectly randomized and scaled tensors from the winning trial
        self.W_in = nn.Parameter(best_matrices["W_in"].unsqueeze(0), requires_grad=False)
        self.W = nn.Parameter(best_matrices["W"].unsqueeze(0), requires_grad=False)       
        self.W_out = best_matrices["W_out"].unsqueeze(0)
        self.b_out = best_matrices["b_out"].unsqueeze(0)
        
        return study.best_params