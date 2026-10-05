from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .utils.validation import validate_hyperparams


@dataclass
class IrregPCAConfig:
    """Configuration for :class:`IrregPCA`.

    All fields that control training, data loading, and integration are
    consolidated here so the estimator signature stays stable.

    Parameters
    ----------
    n_components : int
        Number of principal component functions to fit.
    epochs : int
        Maximum training epochs per component (default 600).
    lr : float
        Adam optimiser learning rate (default 1e-3).
    patience : int
        Early-stopping patience in epochs (default 300). Set to 0 to
        disable early stopping.
    valid_split : float
        Fraction of *samples* (not observations) to hold out for
        validation. Must be in ``(0, 1)`` (default 0.2).
    random_state : int or None
        Seed for the train/validation split and weight initialisation.
        ``None`` means no fixed seed.
    device : str, torch.device, or None
        Device to run training on. ``None`` auto-selects
        ``cuda > mps > cpu``.
    batch_size : int or None
        Number of *samples* (curves) per mini-batch. ``None`` uses the
        full training set (full-batch mode). Only relevant for
        ``training_mode="mini_batch"`` or ``"streaming"``.
    num_workers : int
        Number of DataLoader worker processes (default 0).
    training_mode : str
        One of ``"full_batch"``, ``"mini_batch"``, or ``"streaming"``
        (default ``"full_batch"``).
    integration_mode : str
        One of ``"grid"``, ``"monte_carlo"``, or ``"weighted_discrete"``
        (default ``"grid"``).
    quadrature_points : int
        Number of quadrature nodes used by grid integration
        (default 4096).
    validation_frequency : int
        Evaluate validation loss every this many epochs (default 1).
    track_joint_loss : bool
        Also record the joint loss over *all* models
        (``LossHistory.joint_train`` / ``joint_valid``) at every validation
        step (default ``True``). This is a diagnostic only -- early stopping
        reads each model's own validation loss -- but it evaluates every
        model and every orthogonality pair twice per epoch, which dominates
        the run time once there is more than one component. ``False`` leaves
        those two lists empty and the fit itself unchanged.
    measure : Any or None
        Custom :class:`InnerProductMeasure` instance. If provided,
        overrides ``integration_mode``.
    verbose : bool
        Print training progress (default ``False``).
    model_kind : str
        Built-in model architecture: ``"mlp"`` (default) or ``"resnet"``.
        Ignored if ``model_factory`` is provided.
    hidden_width : int
        Number of units per hidden layer (default 64).
    hidden_depth : int
        Number of hidden layers (default 2).
    activation : str
        Activation function name: ``"tanh"`` (default) or ``"relu"``.
    model_factory : callable or None
        User-supplied factory ``factory(input_dim, output_dim, **model_kwargs)
        -> nn.Module``. Overrides ``model_kind`` when provided.
    model_kwargs : dict or None
        Extra keyword arguments forwarded to ``model_factory``.

    Examples
    --------
    ::

        cfg = IrregPCAConfig(n_components=3, epochs=200, lr=5e-4)
        est = IrregPCA(cfg)
    """

    n_components: int
    epochs: int = 600
    lr: float = 1e-3
    patience: int = 300
    valid_split: float = 0.2
    random_state: int | None = None
    device: str | torch.device | None = None
    batch_size: int | None = None
    num_workers: int = 0
    training_mode: str = "full_batch"
    integration_mode: str = "grid"
    quadrature_points: int = 4096
    validation_frequency: int = 1
    track_joint_loss: bool = True
    measure: Any | None = None
    verbose: bool = False
    callbacks: list = field(default_factory=list)
    # Model architecture
    model_kind: str = "mlp"
    hidden_width: int = 64
    hidden_depth: int = 2
    activation: str = "tanh"
    model_factory: Callable[..., torch.nn.Module] | None = field(default=None, repr=False)
    model_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_hyperparams(
            n_components=self.n_components,
            epochs=self.epochs,
            lr=self.lr,
            patience=self.patience,
            valid_split=self.valid_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            quadrature_points=self.quadrature_points,
            validation_frequency=self.validation_frequency,
            training_mode=self.training_mode,
            integration_mode=self.integration_mode,
            model_kind=self.model_kind,
            hidden_width=self.hidden_width,
            hidden_depth=self.hidden_depth,
            model_factory=self.model_factory,
        )
