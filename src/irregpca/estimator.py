from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .config import IrregPCAConfig
from .models.factory import build_model
from .objectives.quadrature import make_measure
from .result import IrregPCAResult, LossHistory
from .training.engine import fit_sequential
from .utils.random import seed_everything


def _parse_input(
    data: torch.Tensor | None = None,
    *,
    sample_ids: torch.Tensor | None = None,
    locations: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
) -> tuple:
    """Normalize packed or split input into (sample_ids, locations, values)."""
    if data is not None:
        if data.ndim != 2:
            raise ValueError(
                f"Expected `data` to be a 2-D tensor, got {data.ndim}-D. "
                "Expected shape (N, d+2) with columns [sample_id, loc..., value]."
            )
        if data.shape[1] < 3:
            raise ValueError(
                f"Expected `data` to have >= 3 columns (got {data.shape[1]}). "
                "Expected shape (N, d+2) with columns [sample_id, loc..., value]."
            )
        return data[:, 0], data[:, 1:-1], data[:, -1]
    if sample_ids is None or locations is None or values is None:
        raise ValueError(
            "Provide either `data` (packed mode) or all three of "
            "`sample_ids`, `locations`, and `values` (split mode)."
        )
    if sample_ids.ndim != 1:
        raise ValueError(f"Expected `sample_ids` to be 1-D, got shape {tuple(sample_ids.shape)}.")
    if locations.ndim != 2:
        raise ValueError(
            f"Expected `locations` to be 2-D with shape (N, d), got shape {tuple(locations.shape)}."
        )
    if values.ndim != 1:
        raise ValueError(f"Expected `values` to be 1-D, got shape {tuple(values.shape)}.")
    n = sample_ids.shape[0]
    if not (locations.shape[0] == n and values.shape[0] == n):
        raise ValueError(
            f"Length mismatch: `sample_ids` has {n} rows, "
            f"`locations` has {locations.shape[0]}, "
            f"`values` has {values.shape[0]}. All must match."
        )
    return sample_ids, locations, values


class IrregPCA:
    """Estimator for irregular functional PCA.

    Fits a mean function and a set of principal component functions to
    irregularly observed functional data using neural network models.

    Parameters
    ----------
    n_components : int
        Number of principal component functions to fit.
    config : IrregPCAConfig or None
        Full configuration object. If provided, all other keyword arguments
        are ignored.
    epochs : int
    lr : float
    patience : int
    valid_split : float
    random_state : int or None
    device : str, torch.device, or None
    model_factory : callable or None
    verbose : bool
    callbacks : list of callable or None
    training_mode : str
        ``"full_batch"`` (default), ``"mini_batch"``, or ``"streaming"``.
    integration_mode : str
        ``"grid"`` (default), ``"monte_carlo"``, or ``"weighted_discrete"``.
    quadrature_points : int
    measure : Any or None
        Custom :class:`InnerProductMeasure`. Overrides ``integration_mode``.

    Examples
    --------
    Split-input usage::

        est = IrregPCA(n_components=3, epochs=600)
        result = est.fit(sample_ids=idx, locations=U, values=Y)
        mu = result.mean(grid)

    Packed-input usage::

        est = IrregPCA(n_components=3)
        result = est.fit(data=data)

    Using a config object::

        cfg = IrregPCAConfig(n_components=3, epochs=200, lr=5e-4)
        est = IrregPCA(config=cfg)
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        config: IrregPCAConfig | None = None,
        epochs: int = 600,
        lr: float = 1e-3,
        patience: int = 300,
        valid_split: float = 0.2,
        random_state: int | None = None,
        device: str | torch.device | None = None,
        model_factory: Callable[..., torch.nn.Module] | None = None,
        verbose: bool = False,
        callbacks: list[Callable[[dict], None]] | None = None,
        training_mode: str = "full_batch",
        integration_mode: str = "grid",
        quadrature_points: int = 4096,
        measure: Any | None = None,
        model_kind: str = "mlp",
        hidden_width: int = 64,
        hidden_depth: int = 2,
        activation: str = "tanh",
        model_kwargs: dict[str, Any] | None = None,
        track_joint_loss: bool = True,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            if n_components is None:
                raise ValueError("Provide either `n_components` or a `config` object.")
            self.config = IrregPCAConfig(
                n_components=n_components,
                epochs=epochs,
                lr=lr,
                patience=patience,
                valid_split=valid_split,
                random_state=random_state,
                device=device,
                verbose=verbose,
                training_mode=training_mode,
                integration_mode=integration_mode,
                quadrature_points=quadrature_points,
                measure=measure,
                model_kind=model_kind,
                hidden_width=hidden_width,
                hidden_depth=hidden_depth,
                activation=activation,
                model_factory=model_factory,
                model_kwargs=model_kwargs,
                track_joint_loss=track_joint_loss,
            )

        self._callbacks = callbacks
        if callbacks:
            self.config.callbacks = list(callbacks)

        # fitted attributes
        self.mean_model_: torch.nn.Module | None = None
        self.component_models_: list[torch.nn.Module] | None = None
        self.history_: LossHistory | None = None
        self.result_: IrregPCAResult | None = None
        self.input_dim_: int | None = None
        self.device_: str | None = None
        self.is_fitted_: bool = False

    # Convenience property aliases
    @property
    def n_components(self) -> int:
        return self.config.n_components

    def fit(
        self,
        data: torch.Tensor | None = None,
        *,
        sample_ids: torch.Tensor | None = None,
        locations: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> IrregPCAResult:
        """Fit the model to irregularly observed functional data.

        Accepts either packed or split input format.

        **Packed input** (``data`` of shape ``(N, d+2)``)::

            result = est.fit(data=data)

        **Split input**::

            result = est.fit(sample_ids=idx, locations=U, values=Y)

        Parameters
        ----------
        data : torch.Tensor, optional
            Packed tensor, shape ``(N, d+2)``.
        sample_ids, locations, values : torch.Tensor, optional
            Split input tensors.

        Returns
        -------
        IrregPCAResult
        """
        sample_ids_t, locations_t, values_t = _parse_input(
            data,
            sample_ids=sample_ids,
            locations=locations,
            values=values,
        )

        cfg = self.config

        if cfg.random_state is not None:
            seed_everything(cfg.random_state)

        d = locations_t.shape[1]
        measure = make_measure(
            integration_mode=cfg.integration_mode,
            quadrature_points=cfg.quadrature_points,
            custom_measure=cfg.measure,
            input_dim=d,
        )

        # Build a single-argument factory closure from config model fields.
        # fit_sequential calls factory(input_dim) to create each model.
        _mf = cfg.model_factory
        _mk = cfg.model_kind
        _w = cfg.hidden_width
        _dep = cfg.hidden_depth
        _act = cfg.activation
        _mkw = cfg.model_kwargs

        def _model_factory_from_config(input_dim: int) -> torch.nn.Module:
            return build_model(
                input_dim=input_dim,
                model_kind=_mk,
                width=_w,
                depth=_dep,
                activation=_act,
                model_factory=_mf,
                model_kwargs=_mkw,
            )

        result = fit_sequential(
            sample_ids=sample_ids_t,
            locations=locations_t,
            values=values_t,
            n_components=cfg.n_components,
            epochs=cfg.epochs,
            lr=cfg.lr,
            patience=cfg.patience,
            valid_split=cfg.valid_split,
            random_state=cfg.random_state,
            device=cfg.device,
            model_factory=_model_factory_from_config,
            verbose=cfg.verbose,
            callbacks=self._callbacks,
            measure=measure,
            validation_frequency=cfg.validation_frequency,
            track_joint_loss=cfg.track_joint_loss,
        )

        self.result_ = result
        self.mean_model_ = result.mean_model
        self.component_models_ = result.component_models
        self.history_ = result.history
        self.input_dim_ = result.input_dim
        self.device_ = result.device
        self.is_fitted_ = True

        return result

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "This IrregPCA instance is not fitted yet. "
                "Call `.fit(...)` before using inference methods."
            )

    def mean(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted mean function."""
        self._check_fitted()
        assert self.result_ is not None
        return self.result_.mean(locations)

    def component(self, index: int, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate a single principal component function (0-based)."""
        self._check_fitted()
        assert self.result_ is not None
        return self.result_.component(index, locations)

    def components(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate all component functions, shape ``(N, n_components)``."""
        self._check_fitted()
        assert self.result_ is not None
        return self.result_.components(locations)


def fit_irreg_pca(
    data: torch.Tensor | None = None,
    *,
    sample_ids: torch.Tensor | None = None,
    locations: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    n_components: int,
    epochs: int = 600,
    lr: float = 1e-3,
    patience: int = 300,
    valid_split: float = 0.2,
    random_state: int | None = None,
    device: str | torch.device | None = None,
    model_factory: Callable[..., torch.nn.Module] | None = None,
    verbose: bool = False,
    callbacks: list[Callable[[dict], None]] | None = None,
    training_mode: str = "full_batch",
    integration_mode: str = "grid",
    quadrature_points: int = 4096,
    measure: Any | None = None,
    model_kind: str = "mlp",
    hidden_width: int = 64,
    hidden_depth: int = 2,
    activation: str = "tanh",
    model_kwargs: dict[str, Any] | None = None,
    track_joint_loss: bool = True,
) -> IrregPCAResult:
    """Fit IrregPCA and return the result in a single call.

    Parameters
    ----------
    data, sample_ids, locations, values : torch.Tensor, optional
        Input data in packed or split format.
    n_components : int
    epochs, lr, patience, valid_split, random_state, device,
    model_factory, verbose, callbacks, training_mode, integration_mode,
    quadrature_points, measure, track_joint_loss
        See :class:`IrregPCA` and :class:`IrregPCAConfig`.

    Returns
    -------
    IrregPCAResult

    Examples
    --------
    ::

        result = fit_irreg_pca(sample_ids=idx, locations=U, values=Y, n_components=3)
        mu = result.mean(grid)
        phi1 = result.component(0, grid)
    """
    est = IrregPCA(
        n_components=n_components,
        epochs=epochs,
        lr=lr,
        patience=patience,
        valid_split=valid_split,
        random_state=random_state,
        device=device,
        model_factory=model_factory,
        verbose=verbose,
        callbacks=callbacks,
        training_mode=training_mode,
        integration_mode=integration_mode,
        quadrature_points=quadrature_points,
        measure=measure,
        model_kind=model_kind,
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        activation=activation,
        model_kwargs=model_kwargs,
        track_joint_loss=track_joint_loss,
    )
    return est.fit(
        data,
        sample_ids=sample_ids,
        locations=locations,
        values=values,
    )
