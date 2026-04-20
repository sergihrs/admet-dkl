"""Stochastic Variational Deep Kernel Learning (SV-DKL) model.

Architecture (Wilson et al., 2016):
    Pre-computed MoLFormer embeddings [B, 768]
        -> TaskHead (identical to MLP baseline's feature extractor)
        -> deep kernel k(phi(x), phi(x'))
        -> ApproximateGP with M inducing points
        -> GaussianLikelihood (regression) | BernoulliLikelihood (classification)

The TaskHead here is the *same class* as in baseline_mlp.py, enabling a clean
ablation: the GP wraps the MLP body with a learned kernel, replacing only the
final linear + MSE loss with a variational GP + PredictiveLogLikelihood.
"""

from __future__ import annotations

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from torch import Tensor

from src.models.baseline_mlp import TaskHead


class DeepKernelGP(ApproximateGP):
    """Variational GP whose kernel operates in the TaskHead feature space.

    Inducing points live in the *latent* (post-TaskHead) space, not in the
    raw embedding space, giving the GP access to learned representations.

    Args:
        inducing_points: Initial inducing point locations [M, feat_dim].
        kernel_type: "rbf" (Wilson 2016 baseline) or "matern".
        matern_nu: Matern smoothness; only used when kernel_type=="matern".
    """

    def __init__(
        self,
        inducing_points: Tensor,
        kernel_type: str = "rbf",
        matern_nu: float = 2.5,
    ) -> None:
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[0]
        )
        # WhitenedVariationalStrategy stabilises training vs raw strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_dist,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        ard = inducing_points.shape[-1]
        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard)
        elif kernel_type == "matern":
            if matern_nu not in (0.5, 1.5, 2.5):
                raise ValueError(f"matern_nu must be 0.5, 1.5, or 2.5 (got {matern_nu})")
            base_kernel = MaternKernel(nu=matern_nu, ard_num_dims=ard)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type!r}")
        self.covar_module = ScaleKernel(base_kernel)

        # In high dimensions, tiny initial ARD lengthscales can make K(x, Z)
        # underflow to exact zeros and block gradients to the deep feature map.
        # Set a broader prior scale tied to inducing-point spread.
        avg_std = inducing_points.detach().std(dim=0).mean().clamp_min(1e-3)
        init_lengthscale = float(avg_std.item() * (inducing_points.shape[-1] ** 0.5))
        init_lengthscale = max(init_lengthscale, 5.0)
        self.covar_module.base_kernel.lengthscale = torch.full(
            (1, 1, inducing_points.shape[-1]),
            init_lengthscale,
            dtype=inducing_points.dtype,
            device=inducing_points.device,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Compute the GP prior at feature locations x.

        Args:
            x: Feature tensor of shape [B, feat_dim].

        Returns:
            MultivariateNormal distribution over function values.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class DKLModel(torch.nn.Module):
    """End-to-end SV-DKL: feature extractor + variational GP.

    Supports both regression and binary classification by switching the
    likelihood and adjusting inference accordingly.

    Args:
        input_dim: MoLFormer embedding dimension (768).
        hidden_dims: Hidden layer sizes for TaskHead feature extractor.
        n_inducing: Number of GP inducing points M (default 512).
        use_batchnorm: Passed to TaskHead.
        dropout_rate: Passed to TaskHead (also used in active training mode).
        use_residual: Passed to TaskHead.
        use_spectral_norm: Apply spectral norm to TaskHead linear layers.
        kernel_type: Deep-kernel base kernel ("rbf" or "matern").
        matern_nu: Matern smoothness, used only when kernel_type=="matern".
        task: "regression" or "classification".
        inducing_init_inputs: Optional raw embeddings used to initialise
            inducing points in latent feature space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_inducing: int = 512,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.15,
        use_residual: bool = False,
        use_spectral_norm: bool = False,
        kernel_type: str = "rbf",
        matern_nu: float = 2.5,
        task: str = "regression",
        inducing_init_inputs: Tensor | None = None,
    ) -> None:
        super().__init__()
        assert task in {"regression", "classification"}, f"Unknown task: {task}"
        self.task = task

        self.feature_extractor = TaskHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_spectral_norm=use_spectral_norm,
        )

        feat_dim = self.feature_extractor.out_dim

        # Initialise inducing points in the same latent scale as TaskHead outputs
        # to prevent K(x, Z) from collapsing to exact zeros at startup.
        inducing_points = self._init_inducing_points(
            n_inducing=n_inducing,
            feat_dim=feat_dim,
            inducing_init_inputs=inducing_init_inputs,
        )
        self.gp = DeepKernelGP(
            inducing_points,
            kernel_type=kernel_type,
            matern_nu=matern_nu,
        )

        if task == "regression":
            self.likelihood: GaussianLikelihood | BernoulliLikelihood = GaussianLikelihood()
        else:
            self.likelihood = BernoulliLikelihood()

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Map raw embeddings through TaskHead and query the GP.

        Args:
            x: MoLFormer embedding tensor of shape [B, 768].

        Returns:
            GPyTorch MultivariateNormal over the output function.
        """
        features = self.feature_extractor(x)  # [B, feat_dim]
        return self.gp(features)

    @torch.no_grad()
    def _init_inducing_points(
        self,
        n_inducing: int,
        feat_dim: int,
        inducing_init_inputs: Tensor | None,
    ) -> Tensor:
        """Build initial inducing locations in latent feature space.

        If raw input embeddings are provided, map them through the initial
        TaskHead and sample from that cloud. Otherwise, fall back to a narrow
        Gaussian in latent space.
        """
        if inducing_init_inputs is None or inducing_init_inputs.numel() == 0:
            return 0.1 * torch.randn(n_inducing, feat_dim)

        if inducing_init_inputs.ndim != 2:
            raise ValueError("inducing_init_inputs must be a 2D tensor [N, input_dim].")

        # Keep current mode (train by default) so BN/Dropout statistics match
        # the distribution seen in the first optimization steps.
        features = self.feature_extractor(inducing_init_inputs.float()).detach()

        n_available = int(features.shape[0])
        if n_available == 0:
            return 0.1 * torch.randn(n_inducing, feat_dim)

        if n_available >= n_inducing:
            idx = torch.randperm(n_available)[:n_inducing]
            return features[idx].clone()

        repeats = (n_inducing + n_available - 1) // n_available
        tiled = features.repeat((repeats, 1))[:n_inducing].clone()
        tiled += 1e-3 * torch.randn_like(tiled)
        return tiled

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior predictive mean and variance.

        Wraps gpytorch.settings.fast_pred_var for efficient inference.

        Args:
            x: Embedding tensor of shape [B, 768].

        Returns:
            mean: Predictive mean of shape [B].
            var:  Predictive variance of shape [B] (epistemic + aleatoric for GP).
        """
        self.eval()
        self.likelihood.eval()

        with gpytorch.settings.fast_pred_var():
            features = self.feature_extractor(x)  # [B, feat_dim]
            pred = self.likelihood(self.gp(features))

        return pred.mean, pred.variance


def make_elbo(model: DKLModel, n_data: int) -> gpytorch.mlls.PredictiveLogLikelihood:
    """Construct the PredictiveLogLikelihood (ELBO) objective.

    PredictiveLogLikelihood (rather than VariationalELBO) is recommended for
    DKL as it avoids the double-counting issue with deep feature extractors.

    Args:
        model: Instantiated DKLModel.
        n_data: Total number of training examples (for KL scaling).

    Returns:
        GPyTorch PredictiveLogLikelihood loss module.
    """
    return gpytorch.mlls.PredictiveLogLikelihood(
        model.likelihood,
        model.gp,
        num_data=n_data,
    )
