from __future__ import annotations

from collections.abc import Callable

import torch
from torch import optim

from ..objectives.losses import build_loss_fns
from ..objectives.quadrature import InnerProductMeasure
from ..result import IrregPCAResult, LossHistory
from .callbacks import CallbackList, TrainingEvent
from .checkpointing import Checkpoint
from .devices import resolve_device
from .metrics import EpochMetrics


def _set_grad(models: list[torch.nn.Module], active_idx: int) -> None:
    """Enable gradients only for model at ``active_idx``, freeze all others."""
    for i, m in enumerate(models):
        requires = i == active_idx
        for p in m.parameters():
            p.requires_grad_(requires)


def fit_sequential(
    *,
    sample_ids: torch.Tensor,
    locations: torch.Tensor,
    values: torch.Tensor,
    n_components: int,
    epochs: int,
    lr: float,
    patience: int,
    valid_split: float,
    random_state: int | None,
    device: str | torch.device | None,
    model_factory: Callable[[int], torch.nn.Module] | None,
    verbose: bool,
    callbacks: list[Callable] | None,
    measure: InnerProductMeasure,
    validation_frequency: int = 1,
    track_joint_loss: bool = True,
) -> IrregPCAResult:
    """Fit IrregPCA sequentially, fixing the early-stopping restoration bug.

    The best validation checkpoint is **always** restored at the end of each
    component's training loop, whether the loop ended by early stopping or
    by exhausting all epochs.

    Parameters
    ----------
    sample_ids, locations, values : torch.Tensor
        Input data tensors.
    n_components : int
    epochs : int
    lr : float
    patience : int
    valid_split : float
    random_state : int or None
    device : str, torch.device, or None
    model_factory : callable or None
    verbose : bool
    callbacks : list of callable or None
    measure : InnerProductMeasure
        Integration measure for inner products.
    validation_frequency : int
        Validate every this many epochs (default 1).
    track_joint_loss : bool
        Also record the joint loss over *all* models
        (``history.joint_train`` / ``history.joint_valid``) at every
        validation step (default True). Nothing in the fit reads it -- early
        stopping uses the active model's own validation loss -- but it
        integrates every model and every orthogonality pair twice per epoch,
        which dominates the run time for more than one component. With
        ``False`` the two history lists stay empty and the corresponding
        ``TrainingEvent`` fields are ``None``. The fitted models are
        unchanged: bit-identical under ``integration_mode="grid"``; under
        ``"monte_carlo"`` the skipped evaluations no longer draw quadrature
        points, so the random stream (and hence the fit) differs.

    Returns
    -------
    IrregPCAResult
    """
    from ..data.schemas import GroupedObservationDataset
    from ..data.split import split_by_sample_id
    from ..models.factory import make_model

    device = resolve_device(device)
    device_str = str(device)
    cb_list = CallbackList(callbacks)

    # ------------------------------------------------------------------ #
    # Build grouped dataset and split                                       #
    # ------------------------------------------------------------------ #
    grouped = GroupedObservationDataset.from_split(sample_ids, locations, values)
    train_ds, valid_ds = split_by_sample_id(
        grouped,
        train_ratio=1.0 - valid_split,
        random_state=random_state,
    )

    # pack for legacy full-batch loss functions
    data_train = train_ds.to_device(device).to_packed()
    data_valid = valid_ds.to_device(device).to_packed()

    d = locations.shape[1]
    k = n_components

    # ------------------------------------------------------------------ #
    # Create models                                                         #
    # ------------------------------------------------------------------ #
    models: list[torch.nn.Module] = [make_model(d, model_factory).to(device) for _ in range(k + 1)]

    # ------------------------------------------------------------------ #
    # Build loss functions                                                  #
    # ------------------------------------------------------------------ #
    lossfns = build_loss_fns(k=k, measure=measure, device=device)

    def loss_joint(models: list[torch.nn.Module], data: torch.Tensor) -> torch.Tensor:
        parts = [lossfns[j](models, data) for j in range(k + 1)]
        return torch.stack(parts).sum()

    # ------------------------------------------------------------------ #
    # History accumulators                                                  #
    # ------------------------------------------------------------------ #
    history = LossHistory(
        joint_train=[],
        joint_valid=[],
        train_losses=[[] for _ in range(k + 1)],
        valid_losses=[[] for _ in range(k + 1)],
        best_epochs=[],
        best_valid_losses=[],
    )

    cb_list.fire(TrainingEvent(stage="train_start", device=device_str))
    cb_list.fire_lifecycle("on_train_begin", n_components=n_components)

    # ------------------------------------------------------------------ #
    # Sequential training loop                                              #
    # ------------------------------------------------------------------ #
    for j, model in enumerate(models):
        is_mean = j == 0
        component_index = None if is_mean else (j - 1)
        label = "mean" if is_mean else f"component {component_index}"

        if verbose:
            print(f"\n===> Training {label}")

        cb_list.fire(
            TrainingEvent(
                stage="component_start",
                component_index=component_index,
                is_mean=is_mean,
                device=device_str,
            )
        )
        cb_list.fire_lifecycle("on_component_begin", component_index=j)

        _set_grad(models, j)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        checkpoint = Checkpoint(model, patience=patience)

        for epoch in range(epochs):
            # train step
            model.train()
            optimizer.zero_grad()
            loss_train = lossfns[j](models, data_train)
            loss_train.backward()
            optimizer.step()

            # validation (at configured frequency)
            if (epoch + 1) % validation_frequency == 0 or epoch == 0:
                # The joint loss is a diagnostic only; see `track_joint_loss`.
                joint_train_val: float | None = None
                joint_valid_val: float | None = None
                model.eval()
                with torch.no_grad():
                    loss_valid = lossfns[j](models, data_valid)
                    if track_joint_loss:
                        joint_train_val = loss_joint(models, data_train).item()
                        joint_valid_val = loss_joint(models, data_valid).item()

                train_loss_val = loss_train.item()
                valid_loss_val = loss_valid.item()

                history.train_losses[j].append(train_loss_val)
                history.valid_losses[j].append(valid_loss_val)
                if joint_train_val is not None and joint_valid_val is not None:
                    history.joint_train.append(joint_train_val)
                    history.joint_valid.append(joint_valid_val)

                # ----- key fix: always restore best state after loop -----
                checkpoint.maybe_update(valid_loss_val, epoch)

                if verbose:
                    print(
                        f"  epoch {epoch:4d} | train {train_loss_val:.6f} "
                        f"| valid {valid_loss_val:.6f}",
                        end="\r",
                        flush=True,
                    )

                if cb_list:
                    event = TrainingEvent(
                        stage="epoch_end",
                        component_index=component_index,
                        epoch=epoch,
                        train_loss=train_loss_val,
                        valid_loss=valid_loss_val,
                        joint_train_loss=joint_train_val,
                        joint_valid_loss=joint_valid_val,
                        is_mean=is_mean,
                        device=device_str,
                        best_epoch=checkpoint.best_epoch,
                        best_valid_loss=checkpoint.best_valid_loss,
                    )
                    cb_list.fire(event)
                    cb_list.fire_lifecycle(
                        "on_epoch_end",
                        metrics=EpochMetrics(
                            epoch=epoch,
                            train_loss=train_loss_val,
                            valid_loss=valid_loss_val,
                        ),
                    )

                if checkpoint.should_stop:
                    if verbose:
                        print(
                            f"  Early stop at epoch {epoch}. Best epoch: {checkpoint.best_epoch}."
                        )
                    cb_list.fire(
                        TrainingEvent(
                            stage="early_stop",
                            component_index=component_index,
                            epoch=epoch,
                            is_mean=is_mean,
                            device=device_str,
                            best_epoch=checkpoint.best_epoch,
                            best_valid_loss=checkpoint.best_valid_loss,
                        )
                    )
                    break

        if verbose:
            print()  # end the \r line
        # ----- ALWAYS restore best state (fixes the early-stopping bug) -----
        checkpoint.restore()
        history.best_epochs.append(checkpoint.best_epoch)
        history.best_valid_losses.append(checkpoint.best_valid_loss)
        cb_list.fire_lifecycle("on_component_end", best_epoch=checkpoint.best_epoch)

    # freeze all models for inference
    _set_grad(models, -1)

    cb_list.fire(TrainingEvent(stage="fit_end", device=device_str))
    cb_list.fire_lifecycle("on_train_end")

    return IrregPCAResult(
        mean_model=models[0],
        component_models=models[1:],
        history=history,
        n_components=n_components,
        input_dim=d,
        device=device_str,
    )
