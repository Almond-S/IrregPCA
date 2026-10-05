# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

* `track_joint_loss` option on `IrregPCAConfig`, `IrregPCA`, `fit_irreg_pca()`
  and `fit_sequential()`. The joint loss over all models
  (`LossHistory.joint_train` / `joint_valid`) is a diagnostic that nothing in
  the fit reads -- early stopping uses each model's own validation loss -- yet
  recording it evaluates every model and every orthogonality pair twice per
  epoch, which was measured at ~80% of the run time for `n_components = 3`.
  `track_joint_loss=False` skips it, leaving the two history lists empty and
  the fitted models bit-identical under `integration_mode="grid"` (under
  `"monte_carlo"` the skipped evaluations no longer consume quadrature draws,
  so the random stream differs). The default stays `True`, so existing
  behaviour is unchanged.
* `LiveLossPlotCallback` in `irregpca.training.callbacks`: live-updating loss
  plot during training, one subplot per model, with best-epoch marker.
  Exported from the top-level `irregpca` namespace.
* `IrregPCAResult.plot_loss()`: post-training loss curve visualisation.
  Requires `matplotlib` (`pip install irregpca[viz]`).
* `IrregPCAConfig.callbacks` field: pass a list of callback objects to
  `fit_irreg_pca()` or `IrregPCA.fit()`.
* `examples/synthetic_irregular_2d.py`: end-to-end example for d=2 domains
  using `integration_mode="monte_carlo"`.
* `scripts/` directory with reproducible simulation and benchmark scripts.

### Fixed

* `IrregPCAResult.plot_loss()`: corrected field names (`train_losses` /
  `valid_losses` instead of `component_train` / `component_valid`).
* `illustration.png` path in `README.md` corrected (removed spurious
  `assets/` prefix).
* Repository URLs in `pyproject.toml` and `CONTRIBUTING.md` corrected from
  `kgwaghmare/irregpca` to `kgwstat/IrregPCA`.

### Changed

* `requires-python` bumped to `>=3.10` to match CI-tested range.
* Duplicate ruff and pytest configuration removed from `pyproject.toml`.
* Ruff lint errors resolved across `src/` and `tests/`.

---

## [0.2.0] — 2026-04-11

Major research-grade refactor. All changes relative to the 0.1.x prototype.

### Added
- `IrregPCAConfig` dataclass for structured, validated configuration.
- `data/` subpackage: `GroupedObservationDataset`, `GroupedMapDataset`,
  `GroupedBatchSampler`, `grouped_collate_fn`, `split_by_sample_id`,
  and a memory-mapped dataset loader (`data.memmap`).
- `objectives/` subpackage: `mean.py`, `covariance.py`, `penalties.py`,
  `quadrature.py` (including `UnitIntervalGridMeasure`,
  `WeightedDiscreteMeasure`, `MonteCarloMeasure`), and `losses.py`.
- `training/` subpackage: `engine.py` (sequential component training loop),
  `callbacks.py` (typed event callbacks), `checkpointing.py`, `devices.py`,
  and `metrics.py` (epoch timing and step counts).
- `models/` subpackage: `base.py`, `mlp.py` (default MLP), `factory.py`.
- `utils/` subpackage: `random.py` (seed control), `validation.py`
  (centralized input and hyperparameter validation), `typing.py`.
- `IrregPCAResult` methods: `component_norms()`, `principal_values()`,
  `explained_variance_proxy()`, `orthogonality_matrix()`, `save()`, `load()`,
  `to(device)`.
- `LossHistory` fields: `best_epochs`, `best_valid_losses`.
- Training modes: `"full_batch"`, `"mini_batch"`, `"streaming"`.
- Integration modes: `"grid"` (d=1 quadrature), `"monte_carlo"`,
  `"weighted_discrete"`.
- Automated test suite: 76 tests across 9 modules.
- GitHub Actions CI: lint, type-check, test (Python 3.10–3.12), build.
- `examples/` directory: `basic_small_dataset.py`,
  `synthetic_irregular_1d.py`, `large_dataset_memmap.py`,
  `custom_measure.py`.
- `CONTRIBUTING.md`, `.editorconfig`, `pytest.ini`, `ruff.toml`.
- `legacy/` directory preserving the pre-refactor implementation.

### Fixed
- **Early-stopping restoration bug**: best model state is now always restored
  after the epoch loop, not only when patience is exhausted mid-run.
- Domain/integration assumptions centralised; `[0, 1]` grid no longer
  hard-coded outside `UnitIntervalGridMeasure`.
- Hyperparameter validation raises descriptive `ValueError` before training.
- Centralized device handling; `device_` exposed as `torch.device`.

### Changed
- Package restructured from a single-file prototype to a `src/` layout with
  five subpackages (`data`, `objectives`, `training`, `models`, `utils`).
- Training loop refactored to consume `GroupedObservationDataset` rather than
  raw tensors.
- Loss functions accept grouped batch dicts instead of the full training tensor.
- Public API exports: `IrregPCA`, `IrregPCAConfig`, `IrregPCAResult`,
  `fit_irreg_pca`.
- `pyproject.toml` updated with complete metadata, classifiers, URLs, and
  optional dependency groups (`dev`, `docs`).

### Deprecated
- `core.py` legacy fitting path (kept in `legacy/` for reference; will be
  removed in 0.3.0).

---

## [0.1.0] — 2026-02-26

Initial research prototype.

### Added
- Basic estimator: `IrregPCA`, `fit_irreg_pca`.
- Default MLP model.
- Sequential component fitting with early stopping.
- Packed-tensor input format `(N, d+2)`.
- MIT License.
