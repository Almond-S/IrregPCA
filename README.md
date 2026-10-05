# IrregPCA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Neural-network functional PCA for irregularly observed data.

IrregPCA fits a mean function and a sequence of orthogonal principal component
functions to scalar functional data observed at **irregular, possibly
subject-specific** locations. All functions are represented by neural network
models trained sequentially via gradient descent, based on a risk minimisation
methodology derived from the Eckart–Young–Mirsky theorem for the
Hilbert–Schmidt norm.

---

## Installation

This package is not yet published to PyPI. Install directly from GitHub:

```bash
pip install git+https://github.com/kgwstat/IrregPCA.git
```

For development:

```bash
git clone https://github.com/kgwstat/IrregPCA
cd IrregPCA
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy ≥ 1.22.

---

## Quick start — config API (recommended)

`IrregPCAConfig` is the canonical way to configure IrregPCA. All options are
in one place and validated at construction time.

```python
import torch
from irregpca import IrregPCA, IrregPCAConfig

# sample_ids: (N,)   — integer sample/curve identifier per observation
# locations:  (N, d) — observation location in R^d  (d = 1 most commonly)
# values:     (N,)   — scalar observation value

cfg = IrregPCAConfig(
    n_components=3,
    epochs=600,
    lr=1e-3,
    patience=300,
    valid_split=0.2,
    random_state=42,
)

est = IrregPCA(config=cfg)
result = est.fit(sample_ids=sample_ids, locations=locations, values=values)

grid = torch.linspace(0, 1, 500).unsqueeze(-1)   # shape (500, 1)
mu        = result.mean(grid)           # (500,)
phi1      = result.component(0, grid)   # (500,)
all_comps = result.components(grid)     # (500, 3)
```

### Architecture customization

Control the model size via `hidden_width` and `hidden_depth`:

```python
cfg = IrregPCAConfig(
    n_components=3,
    hidden_width=128,   # units per hidden layer
    hidden_depth=4,     # number of hidden layers
)
```

Use the built-in residual architecture:

```python
cfg = IrregPCAConfig(
    n_components=3,
    model_kind="resnet",   # "mlp" (default) or "resnet"
    hidden_width=128,
    hidden_depth=4,
)
```

Inject a fully custom architecture:

```python
import torch.nn as nn

def my_factory(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, output_dim),
    )

cfg = IrregPCAConfig(
    n_components=3,
    model_factory=my_factory,
)
```

---

## Quick start — estimator API

`IrregPCA` follows scikit-learn conventions. Pass a config or individual
keyword arguments:

```python
from irregpca import IrregPCA

# Keyword arguments — no config object needed
est = IrregPCA(
    n_components=3,
    epochs=600,
    hidden_width=64,
    hidden_depth=2,
    random_state=42,
)
result = est.fit(sample_ids=sample_ids, locations=locations, values=values)
```

Packed input (one tensor of shape `(N, d+2)` with columns
`[sample_id, loc_1, …, loc_d, value]`):

```python
result = est.fit(data=data)
```

---

## Functional API

`fit_irreg_pca` is a convenience wrapper that creates an `IrregPCA` instance
internally and returns the result directly. All model and training arguments
that `IrregPCAConfig` accepts are forwarded here.

```python
from irregpca import fit_irreg_pca

result = fit_irreg_pca(
    sample_ids=sample_ids,
    locations=locations,
    values=values,
    n_components=3,
    epochs=600,
    lr=1e-3,
    patience=300,
    hidden_width=64,
    hidden_depth=2,
    model_kind="mlp",   # or "resnet"
    random_state=42,
)
```

Equivalent to:

```python
cfg = IrregPCAConfig(n_components=3, epochs=600, hidden_width=64, ...)
result = IrregPCA(config=cfg).fit(sample_ids=..., locations=..., values=...)
```

---

## Argument reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `n_components` | `int` | — | Number of principal component functions to fit. |
| `epochs` | `int` | `600` | Maximum training epochs per component. Increase if `best_epoch == epochs`. |
| `lr` | `float` | `1e-3` | Adam optimizer learning rate. Lower values train more slowly but may generalize better. |
| `patience` | `int` | `300` | Early-stopping patience in epochs. `0` disables early stopping. |
| `valid_split` | `float` | `0.2` | Fraction of samples (not observations) to hold out for validation. Must be in `(0, 1)`. |
| `random_state` | `int` or `None` | `None` | Seed for the train/validation split and weight initialization. `None` means no fixed seed. |
| `training_mode` | `str` | `"full_batch"` | One of `"full_batch"`, `"mini_batch"`, `"streaming"`. See [Training modes](#training-modes). |
| `batch_size` | `int` or `None` | `None` | Samples per mini-batch. Only used for `mini_batch` / `streaming`. |
| `integration_mode` | `str` | `"grid"` | One of `"grid"`, `"monte_carlo"`, `"weighted_discrete"`. See [Integration modes](#integration-modes). |
| `quadrature_points` | `int` | `4096` | Quadrature nodes per dimension for `"grid"`, or draws for `"monte_carlo"`. |
| `model_kind` | `str` | `"mlp"` | Built-in architecture: `"mlp"` (default MLP) or `"resnet"` (residual MLP). Ignored when `model_factory` is set. |
| `hidden_width` | `int` | `64` | Hidden layer width. Increase for more expressive models; decrease to regularize. |
| `hidden_depth` | `int` | `2` | Number of hidden layers. Depth ≥ 2 is generally needed to capture nonlinear structure. |
| `activation` | `str` | `"tanh"` | Activation function: `"tanh"`, `"relu"`, `"gelu"`, or `"silu"`. |
| `model_factory` | `callable` or `None` | `None` | User-supplied factory `factory(input_dim, output_dim) -> nn.Module`. Overrides `model_kind`. |
| `model_kwargs` | `dict` or `None` | `None` | Extra keyword arguments forwarded to `model_factory`. |
| `measure` | `InnerProductMeasure` or `None` | `None` | Custom integration measure. Overrides `integration_mode`. |
| `verbose` | `bool` | `False` | Print per-epoch progress. |
| `callbacks` | `list` | `[]` | List of callback objects (e.g. `LiveLossPlotCallback`). |
| `validation_frequency` | `int` | `1` | Validate every N epochs. Increase to speed up training on large datasets. |
| `track_joint_loss` | `bool` | `True` | Record the joint loss over all models in `history.joint_train` / `joint_valid`. Diagnostic only; set to `False` to skip it, which can cut run time by several times for `n_components >= 2` without changing the fit. |
| `device` | `str` or `None` | `None` | Device for training. `None` auto-selects `cuda > mps > cpu`. |
| `num_workers` | `int` | `0` | DataLoader worker processes. |

---

## Training modes

| Mode | Description |
|------|-------------|
| `"full_batch"` | Default. All training samples used each epoch. Exact empirical objective. |
| `"mini_batch"` | Grouped mini-batches by sample ID. Scalable to large datasets. |
| `"streaming"` | For memory-mapped / disk-backed datasets. |

```python
# Config API
cfg = IrregPCAConfig(n_components=3, training_mode="mini_batch", batch_size=32)
result = IrregPCA(config=cfg).fit(sample_ids=..., locations=..., values=...)

# Functional API
result = fit_irreg_pca(..., training_mode="mini_batch", batch_size=32)
```

---

## Integration modes

Inner products between functions (for regularisation and orthogonality) are
computed using a configurable integration measure.

| Mode | Description | d support |
|------|-------------|-----------|
| `"grid"` | Tensor-product uniform quadrature on `[0,1]^d`. | Any `d` |
| `"monte_carlo"` | Monte Carlo draws from a sampler. Unbiased stochastic. | Any `d` |
| `"weighted_discrete"` | User-supplied nodes + weights. Exact. | Any `d` |

### Grid mode and dimensionality

For `d = 1`, `"grid"` places `quadrature_points` equally-spaced nodes on
`[0, 1]`. For `d > 1`, the grid is extended via **tensor product**: each
dimension gets `quadrature_points` nodes, giving `quadrature_points ** d`
nodes total with uniform weights.

**Scaling warning:** the total node count grows exponentially with `d`. For
`quadrature_points=4096` and `d=2` that is ~16 million nodes, which may be
intractably large. A built-in guard raises an error when the count exceeds
`10_000_000`. To work in `d > 1`, either reduce `quadrature_points` or switch
to `"monte_carlo"`:

```python
# d=2, small grid (25 nodes)
cfg = IrregPCAConfig(n_components=2, integration_mode="grid", quadrature_points=5)

# d=2, Monte Carlo (recommended for higher dimensions)
cfg = IrregPCAConfig(n_components=2, integration_mode="monte_carlo", quadrature_points=4096)
```

### Custom weighted discrete measure

```python
from irregpca.objectives.quadrature import WeightedDiscreteMeasure

pts = torch.tensor([[0.25], [0.5], [0.75]])  # (3, 1)
wts = torch.tensor([1.0, 2.0, 1.0])          # (3,)
measure = WeightedDiscreteMeasure(pts, wts)

cfg = IrregPCAConfig(n_components=3, measure=measure)
result = IrregPCA(config=cfg).fit(sample_ids=..., locations=..., values=...)
```

---

## Result summaries

```python
# Norms of component functions under the integration measure
result.component_norms()            # (n_components,)

# Proxy for principal values (squared L²-norms of fitted component functions)
result.principal_values()           # (n_components,)

# Fraction of total variance explained per component (proxy — see note below)
result.explained_variance_proxy()   # (n_components,), sums to 1

# Gram matrix of inner products between component functions
result.orthogonality_matrix()       # (n_components, n_components)

# Best epoch per model (mean + components)
result.history.best_epochs          # list of ints
```

**On `explained_variance_proxy`:** this substitutes ‖φ̂ₖ‖² for the true
eigenvalue. Because component functions are constrained to be orthonormal
during training, a well-converged fit has ‖φ̂ₖ‖ ≈ 1 and `principal_values()`
≈ (1, 1, …). Interpret with care: this is not the true fraction of L²-process
variance and has no direct connection to eigenvalues of the covariance
operator.

### Inspecting training history

```python
import matplotlib.pyplot as plt

history  = result.history
n_models = 1 + result.n_components   # mean + components

fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3))

for i, ax in enumerate(axes):
    label = "mean" if i == 0 else f"component {i}"
    ax.plot(history.train_losses[i], label="train")
    ax.plot(history.valid_losses[i], label="valid")
    ax.axvline(history.best_epochs[i], color="red", linestyle="--",
               label=f"best={history.best_epochs[i]}")
    ax.set_title(label)
    ax.set_xlabel("epoch")
    ax.legend()

plt.tight_layout()
plt.show()
```

**Convergence diagnostics:**
- `best_epoch == epochs` → training was cut short; increase `epochs` or lower `lr`.
- `best_epoch < 10% of epochs` → `patience` may be too aggressive; increase it.
- Large train/valid gap → overfitting; try smaller `hidden_width` / `hidden_depth`.
- Off-diagonal entries of `orthogonality_matrix()` near zero → components are orthogonal.

### Live loss plot during training

```python
from irregpca import fit_irreg_pca, LiveLossPlotCallback

callback = LiveLossPlotCallback(
    update_every=10,        # redraw every 10 epochs
    save_path="loss.png",   # optional: save figure when training ends
)

result = fit_irreg_pca(
    sample_ids=sample_ids,
    locations=locations,
    values=values,
    n_components=3,
    epochs=600,
    callbacks=[callback],
)
```

> **Note:** live plotting requires `matplotlib` (`pip install irregpca[viz]`).

---

## Serialization

```python
result.save("my_result.pt")
result = IrregPCAResult.load("my_result.pt")

# Move all models to a different device
result.to("cpu")
```

---

## Large datasets

```python
from irregpca.data import (
    GroupedObservationDataset, GroupedMapDataset,
    grouped_collate_fn, GroupedBatchSampler,
)
from torch.utils.data import DataLoader

ds      = GroupedObservationDataset.from_split(sample_ids, locations, values)
map_ds  = GroupedMapDataset(ds)
sampler = GroupedBatchSampler(n_samples=len(map_ds), batch_size=32)
loader  = DataLoader(map_ds, batch_sampler=sampler, collate_fn=grouped_collate_fn)
```

For on-disk memory-mapped data:

```python
from irregpca.data.memmap import load_memmap_dataset

ds = load_memmap_dataset(
    locations_path="locs.bin",
    values_path="vals.bin",
    sample_ids_path="ids.bin",
    input_dim=1,
    n_obs=1_000_000,
    n_samples=10_000,
)
```

---

## Notes and limitations

- **Grid integration scaling:** `"grid"` with `d > 1` constructs a full
  tensor-product grid. Node count is `quadrature_points ** d`, growing
  exponentially. Use `"monte_carlo"` for `d ≥ 3` or whenever memory is a
  concern.
- **Stochastic objectives:** mini-batch and streaming modes use stochastic
  approximations to the full-batch objective; results may differ from full-batch
  training on identical data.
- **Variance proxy:** `explained_variance_proxy()` is not a true fraction of
  process variance. See the note in [Result summaries](#result-summaries).
- **GPU reproducibility:** results may vary across hardware and PyTorch versions
  even under a fixed `random_state`.

---

## Citation

```bibtex
@software{irregpca,
  author  = {Waghmare, Kartik G. and Stoecker, Almond and Panaretos, Victor M.},
  title   = {{IrregPCA}: Neural-network functional PCA for irregularly observed data},
  url     = {https://github.com/kgwstat/IrregPCA},
  year    = {2026},
}
```

## License

MIT. See [LICENSE](LICENSE).
