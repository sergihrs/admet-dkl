"""MLP baseline with Monte Carlo Dropout for uncertainty quantification.

Architecture:
    TaskHead  ->  shared MLP body (used as-is by baseline; re-used by DKL as
                  feature extractor, so it deliberately has NO final output layer).
    MLPPredictor  ->  TaskHead + single Linear head + MC Dropout inference.

MC Dropout protocol (Gal & Ghahramani, 2016):
    1. Call self.eval() to freeze BatchNorm running stats.
    2. Loop through modules and force Dropout layers back to train() mode.
    3. Run T stochastic forward passes; return empirical mean & variance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm


class TaskHead(nn.Module):
    """Configurable MLP body shared between the MLP baseline and SV-DKL.

    Builds a stack of Linear -> [BatchNorm] -> ReLU -> Dropout blocks.
    The final block has no activation/norm so it can serve as a feature
    extractor for a downstream GP or a linear prediction head.

    Args:
        input_dim: Dimension of input embeddings (768 from MoLFormer).
        hidden_dims: List of hidden layer widths, e.g. [512, 256].
        use_batchnorm: Whether to insert BatchNorm1d after each linear layer.
        dropout_rate: Dropout probability applied after each hidden block.
        use_residual: Add residual (skip) connections where dimensions match.
        use_spectral_norm: Wrap each Linear with spectral normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        use_batchnorm: bool = True,
        dropout_rate: float = 0.15,
        use_residual: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.blocks = nn.ModuleList()

        dims = [input_dim, *hidden_dims]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(in_d, out_d)
            if use_spectral_norm:
                spectral_norm(linear)
            block: list[nn.Module] = [linear]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(out_d))
            block.append(nn.ReLU())
            block.append(nn.Dropout(p=dropout_rate))
            self.blocks.append(nn.Sequential(*block))

        self.out_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all hidden blocks.

        Args:
            x: Input tensor of shape [B, input_dim].

        Returns:
            Feature tensor of shape [B, hidden_dims[-1]].
        """
        h = x
        for block in self.blocks:
            out = block(h)
            # Residual only when dimensions match (no projection needed)
            if self.use_residual and h.shape == out.shape:
                out = out + h
            h = out
        return h  # [B, hidden_dims[-1]]


class MLPPredictor(nn.Module):
    """MLP baseline predictor with MC Dropout uncertainty estimation.

    Wraps TaskHead with a final Linear output layer. Supports both
    regression (output_dim=1) and binary classification (output_dim=1 + BCE).

    Args:
        input_dim: MoLFormer embedding dimension (768).
        hidden_dims: Hidden layer sizes for TaskHead.
        output_dim: Number of output units (1 for regression/binary).
        use_batchnorm: Passed to TaskHead.
        dropout_rate: Dropout probability (also used for MC inference).
        use_residual: Passed to TaskHead.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        use_batchnorm: bool = True,
        dropout_rate: float = 0.15,
        use_residual: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.feature_extractor = TaskHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_spectral_norm=use_spectral_norm,
        )
        self.head = nn.Linear(self.feature_extractor.out_dim, output_dim)
        if use_spectral_norm:
            spectral_norm(self.head)

    def forward(self, x: Tensor) -> Tensor:
        """Single deterministic forward pass.

        Args:
            x: Embedding tensor of shape [B, 768].

        Returns:
            Raw logits/predictions of shape [B, output_dim].
        """
        return self.head(self.feature_extractor(x))

    @torch.no_grad()
    def predict_uncertainty(self, x: Tensor, T: int = 30) -> tuple[Tensor, Tensor]:
        """MC Dropout inference: T stochastic forward passes.

        Protocol:
            1. self.eval() freezes BatchNorm running statistics.
            2. Explicitly re-enable only Dropout layers for stochasticity.
            3. Collect T samples; return empirical mean and variance.

        Args:
            x: Embedding tensor of shape [B, 768].
            T: Number of Monte Carlo samples.

        Returns:
            mean: Predictive mean of shape [B, output_dim].
            var:  Predictive variance of shape [B, output_dim] (epistemic proxy).
        """
        self.eval()

        # Step 2: selectively re-enable Dropout for MC sampling
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        samples = torch.stack(
            [self.head(self.feature_extractor(x)) for _ in range(T)],
            dim=0,
        )  # [T, B, output_dim]

        mean = samples.mean(dim=0)  # [B, output_dim]
        var = samples.var(dim=0)    # [B, output_dim]
        return mean, var
