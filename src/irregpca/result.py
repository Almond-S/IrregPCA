from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import matplotlib.figure


@dataclass
class LossHistory:
    """Container for training and validation loss history.

    Attributes
    ----------
    joint_train : list of float
        Joint loss over all models on the training set, one entry per
        validation step across the whole sequential fit. Empty when the fit
        was run with ``track_joint_loss=False``.
    joint_valid : list of float
        As ``joint_train``, on the validation set.
    train_losses : list of list of float
    valid_losses : list of list of float
    best_epochs : list of int
        Best epoch index per fitted model (mean model first, then components).
    best_valid_losses : list of float
        Best validation loss per fitted model.
    """

    joint_train: list[float] = field(default_factory=list)
    joint_valid: list[float] = field(default_factory=list)
    train_losses: list[list[float]] = field(default_factory=list)
    valid_losses: list[list[float | None]] = field(default_factory=list)
    best_epochs: list[int] = field(default_factory=list)
    best_valid_losses: list[float] = field(default_factory=list)


@dataclass
class IrregPCAResult:
    """Fitted result returned by :class:`IrregPCA` and :func:`fit_irreg_pca`.

    Attributes
    ----------
    mean_model : torch.nn.Module
    component_models : list of torch.nn.Module
    history : LossHistory
    n_components : int
    input_dim : int
    device : str
    """

    mean_model: torch.nn.Module
    component_models: list[torch.nn.Module]
    history: LossHistory
    n_components: int
    input_dim: int
    device: str

    # ------------------------------------------------------------------ #
    # Core evaluation methods (preserved from original API)                #
    # ------------------------------------------------------------------ #

    def mean(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate the fitted mean function.

        Parameters
        ----------
        locations : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        locs = locations.to(torch.device(self.device))
        self.mean_model.eval()
        with torch.no_grad():
            out: torch.Tensor = self.mean_model(locs).squeeze(-1)
        return out

    def component(self, index: int, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate a single principal component function (0-based).

        Parameters
        ----------
        index : int
        locations : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.

        Raises
        ------
        ValueError
            If ``index`` is out of range.
        """
        if index < 0 or index >= self.n_components:
            raise ValueError(
                f"Component index {index} is out of range. "
                f"Valid indices are 0 to {self.n_components - 1}."
            )
        locs = locations.to(torch.device(self.device))
        model = self.component_models[index]
        model.eval()
        with torch.no_grad():
            comp: torch.Tensor = model(locs).squeeze(-1)
        return comp

    def components(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate all component functions stacked along the last dimension.

        Parameters
        ----------
        locations : torch.Tensor
            Shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, n_components)``.
        """
        locs = locations.to(torch.device(self.device))
        parts = []
        for model in self.component_models:
            model.eval()
            with torch.no_grad():
                parts.append(model(locs).squeeze(-1))
        if not parts:
            return torch.zeros(locs.shape[0], 0, device=locs.device)
        return torch.stack(parts, dim=-1)

    def all_functions(self, locations: torch.Tensor) -> torch.Tensor:
        """Evaluate mean and all components, shape ``(N, 1 + n_components)``."""
        locs = locations.to(torch.device(self.device))
        self.mean_model.eval()
        with torch.no_grad():
            mean_vals = self.mean_model(locs).squeeze(-1)
        comp_vals = []
        for model in self.component_models:
            model.eval()
            with torch.no_grad():
                comp_vals.append(model(locs).squeeze(-1))
        return torch.stack([mean_vals] + comp_vals, dim=-1)

    # ------------------------------------------------------------------ #
    # New PCA-style summaries                                               #
    # ------------------------------------------------------------------ #

    def component_norms(
        self,
        num_points: int = 4096,
    ) -> torch.Tensor:
        """Approximate L2 norm of each component function on [0,1]^d.

        Uses a uniform grid on [0,1] (1-D only).

        Parameters
        ----------
        num_points : int

        Returns
        -------
        torch.Tensor
            Shape ``(n_components,)``.
        """
        dev = torch.device(self.device)
        grid = torch.linspace(0.0, 1.0, num_points, device=dev).unsqueeze(-1)
        norms = []
        for model in self.component_models:
            model.eval()
            with torch.no_grad():
                vals = model(grid).squeeze(-1)
            norms.append((vals.pow(2).mean()).sqrt())
        if not norms:
            return torch.zeros(0, device=dev)
        return torch.stack(norms)

    def principal_values(self, num_points: int = 4096) -> torch.Tensor:
        """Return approximate squared norms (proxy for principal values).

        Parameters
        ----------
        num_points : int

        Returns
        -------
        torch.Tensor
            Shape ``(n_components,)``.
        """
        return self.component_norms(num_points=num_points).pow(2)

    def explained_variance_proxy(self, num_points: int = 4096) -> torch.Tensor:
        """Return per-component explained variance as a fraction of total.

        Parameters
        ----------
        num_points : int

        Returns
        -------
        torch.Tensor
            Shape ``(n_components,)``, sums to 1.
        """
        pv = self.principal_values(num_points=num_points)
        total = pv.sum()
        if total == 0:
            return torch.zeros_like(pv)
        return pv / total

    def orthogonality_matrix(self, num_points: int = 4096) -> torch.Tensor:
        """Compute the Gram matrix of inner products between component functions.

        Entry ``(i, j)`` is ⟨e_i, e_j⟩ approximated on a uniform [0,1] grid.

        Parameters
        ----------
        num_points : int

        Returns
        -------
        torch.Tensor
            Shape ``(n_components, n_components)``.
        """
        dev = torch.device(self.device)
        grid = torch.linspace(0.0, 1.0, num_points, device=dev).unsqueeze(-1)
        evals = []
        for model in self.component_models:
            model.eval()
            with torch.no_grad():
                evals.append(model(grid).squeeze(-1))  # (num_points,)
        gram = torch.zeros(self.n_components, self.n_components, device=dev)
        for i in range(self.n_components):
            for j in range(self.n_components):
                gram[i, j] = (evals[i] * evals[j]).mean()
        return gram

    # ------------------------------------------------------------------ #
    # Visualisation                                                         #
    # ------------------------------------------------------------------ #

    def plot_loss(
        self,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot training and validation loss curves for each fitted model.

        One subplot is drawn per model (mean first, then each component).
        Each subplot shows train and validation loss vs. epoch, with a
        vertical dashed line marking the best-validation-loss epoch.

        Parameters
        ----------
        figsize : (width, height) or None
            Passed to ``matplotlib.pyplot.subplots``.  Defaults to
            ``(5 * (1 + n_components), 4)``.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ImportError
            If ``matplotlib`` is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plot_loss. "
                "Install it with: pip install matplotlib"
            ) from exc

        n_models = 1 + self.n_components
        if figsize is None:
            figsize = (5 * n_models, 4)

        fig, axes = plt.subplots(1, n_models, figsize=figsize, squeeze=False)
        axes = axes[0]

        labels = ["mean"] + [f"component {i}" for i in range(self.n_components)]

        for j, (ax, label) in enumerate(zip(axes, labels)):  # noqa: B905
            train_losses = self.history.train_losses[j]
            valid_losses = self.history.valid_losses[j]  # may be None
            epochs = list(range(len(train_losses)))

            ax.plot(epochs, train_losses, label="train")
            if valid_losses is not None:
                ax.plot(epochs, valid_losses, label="valid")

            if j < len(self.history.best_epochs):
                best_ep = self.history.best_epochs[j]
                ax.axvline(
                    best_ep,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    label=f"best (ep {best_ep})",
                )

            ax.set_title(label)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend(fontsize=8)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                 #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save the result to a file using ``torch.save``.

        Parameters
        ----------
        path : str or Path
        """
        torch.save(self, str(path))

    @staticmethod
    def load(path: str | Path) -> IrregPCAResult:
        """Load a previously saved result.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        IrregPCAResult
        """
        loaded: IrregPCAResult = torch.load(str(path), weights_only=False)
        return loaded

    def to(self, device: str | torch.device) -> IrregPCAResult:
        """Move all models to ``device`` and return self.

        Parameters
        ----------
        device : str or torch.device

        Returns
        -------
        IrregPCAResult
            Same object (modified in-place).
        """
        dev = torch.device(device) if isinstance(device, str) else device
        self.mean_model = self.mean_model.to(dev)
        self.component_models = [m.to(dev) for m in self.component_models]
        self.device = str(dev)
        return self
