import numpy as np
import torch

from sklearn.model_selection import train_test_split 
from dysts.flows import Lorenz
from reservoirgrid.datasets import LorenzAttractor
from reservoirgrid.models import Reservoir
from reservoirgrid.helpers import utils
from reservoirgrid.helpers import viz

print("Imports Done!\n")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

#load dataset and split into train and test sets
dataset = np.load("src/reservoirgrid/datasets/Chaotic/Lorenz.npy", allow_pickle=True)
data = dataset[15][1]
data = utils.normalize_data(data)
train_inputs, test_inputs, train_targets, test_targets = utils.split(
    data, test_size=0.2
)

ResLorenz = Reservoir(
    input_dim=3,
    reservoir_dim=1300,
    output_dim=3,
    spectral_radius=1,
    leak_rate=0.5,
    sparsity=0.9,
    input_scaling=0.5,
    noise_level = 0.01
)

ResLorenz.train_readout(train_inputs, train_targets, warmup=1000)
time_steps = np.arange(len(test_targets))

# Generate predictions using test inputs
with torch.no_grad():
    predictions = ResLorenz.predict(train_inputs, steps=len(test_targets))

predictions_np = predictions.cpu().numpy().squeeze(1)
test_targets_np = test_targets.cpu().numpy()

error = utils.RMSE(y_true=test_targets_np[:],y_pred=predictions_np[:])
print(f"RMSE: {error:.4f}")

viz.compare_plot([test_targets_np, predictions_np], labels=["True", "Predicted"], title="Lorenz Attractor Prediction").show()