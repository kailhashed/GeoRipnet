"""
Multi-Objective Loss Functions for GeoRipNet.

Implements composite loss function combining:
1. Huber Loss - Robust regression loss
2. Directional Loss - Sign/direction accuracy
3. Ripple Correlation Loss - Correlation structure preservation
4. Quantile Loss - Uncertainty quantification

Mathematical Formulation:
    L_total = λ₁ * L_huber + λ₂ * L_dir + λ₃ * L_corr + λ₄ * L_quantile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HuberLoss(nn.Module):
    """
    Huber loss - robust to outliers, smooth L1 loss.
    
    L_huber(y, ŷ) = {
        0.5 * (y - ŷ)² if |y - ŷ| ≤ δ
        δ * (|y - ŷ| - 0.5 * δ) otherwise
    }
    
    Args:
        delta: Threshold for switching between L2 and L1 (default: 1.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_countries) or (batch,)
            targets: (batch, num_countries) or (batch,)
        
        Returns:
            loss: Scalar or tensor depending on reduction
        """
        diff = torch.abs(predictions - targets)
        
        # Huber formula
        loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DirectionalLoss(nn.Module):
    """
    Directional accuracy loss - penalizes incorrect sign predictions.
    
    Encourages model to predict correct direction of price movement.
    
    L_dir = 1 - accuracy(sign(ŷ_t - ŷ_{t-1}), sign(y_t - y_{t-1}))
    
    Args:
        smoothing: Temperature for smooth sign approximation (default: 0.1)
        weight_magnitude: Whether to weight by magnitude of change (default: True)
    """
    
    def __init__(self, smoothing: float = 0.1, weight_magnitude: bool = True):
        super().__init__()
        self.smoothing = smoothing
        self.weight_magnitude = weight_magnitude
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_countries) - Current predictions
            targets: (batch, num_countries) - Current actual values
            prev_targets: (batch, num_countries) - Previous time step actual values
        
        Returns:
            loss: Directional loss scalar
        """
        # Compute changes
        pred_change = predictions - prev_targets
        true_change = targets - prev_targets
        
        # Smooth sign function: tanh(x / smoothing)
        pred_sign = torch.tanh(pred_change / self.smoothing)
        true_sign = torch.tanh(true_change / self.smoothing)
        
        # Cosine similarity between signs (1 = same direction, -1 = opposite)
        directional_agreement = pred_sign * true_sign
        
        # Weight by magnitude of actual change (focus on significant movements)
        if self.weight_magnitude:
            magnitude_weight = torch.abs(true_change)
            magnitude_weight = magnitude_weight / (magnitude_weight.mean() + 1e-8)
            directional_agreement = directional_agreement * magnitude_weight
        
        # Loss is 1 - agreement (0 = perfect, 2 = worst)
        loss = 1.0 - directional_agreement.mean()
        
        return loss
    
    def compute_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard directional accuracy (percentage).
        
        Returns:
            accuracy: Float in [0, 1]
        """
        pred_change = predictions - prev_targets
        true_change = targets - prev_targets
        
        # Hard sign
        pred_sign = torch.sign(pred_change)
        true_sign = torch.sign(true_change)
        
        # Accuracy
        correct = (pred_sign == true_sign).float()
        accuracy = correct.mean()
        
        return accuracy


class RippleCorrelationLoss(nn.Module):
    """
    Ripple correlation loss - preserves correlation structure between countries.
    
    Ensures that predicted deltas maintain realistic correlation patterns
    with actual deltas across countries.
    
    L_corr = 1 - Pearson_Corr(Corr(Δ̂), Corr(Δ))
    
    Args:
        method: Correlation method ('pearson', 'spearman')
    """
    
    def __init__(self, method: str = 'pearson'):
        super().__init__()
        self.method = method
    
    def compute_correlation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation matrix across countries.
        
        Args:
            x: (batch, num_countries) - Delta values
        
        Returns:
            corr: (num_countries, num_countries) - Correlation matrix
        """
        # Center the data
        x_centered = x - x.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = torch.matmul(x_centered.T, x_centered) / (x.size(0) - 1)
        
        # Compute standard deviations
        std = torch.sqrt(torch.diag(cov))
        
        # Correlation = Cov / (std_i * std_j)
        corr = cov / (torch.outer(std, std) + 1e-8)
        
        return corr
    
    def forward(
        self,
        predicted_deltas: torch.Tensor,
        true_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_deltas: (batch, num_countries) - Predicted delta values
            true_deltas: (batch, num_countries) - Actual delta values
        
        Returns:
            loss: Correlation structure loss
        """
        # Compute correlation matrices
        pred_corr = self.compute_correlation_matrix(predicted_deltas)
        true_corr = self.compute_correlation_matrix(true_deltas)
        
        # Flatten correlation matrices (exclude diagonal)
        num_countries = predicted_deltas.size(1)
        mask = ~torch.eye(num_countries, dtype=torch.bool, device=predicted_deltas.device)
        
        pred_corr_flat = pred_corr[mask]
        true_corr_flat = true_corr[mask]
        
        # Compute correlation between correlation vectors
        pred_centered = pred_corr_flat - pred_corr_flat.mean()
        true_centered = true_corr_flat - true_corr_flat.mean()
        
        correlation = (pred_centered * true_centered).sum() / (
            torch.sqrt((pred_centered ** 2).sum()) *
            torch.sqrt((true_centered ** 2).sum()) + 1e-8
        )
        
        # Loss is 1 - correlation
        loss = 1.0 - correlation
        
        return loss


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    Enables uncertainty quantification by predicting multiple quantiles.
    
    L_q(y, ŷ_q) = max(q * (y - ŷ_q), (q - 1) * (y - ŷ_q))
    
    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
    """
    
    def __init__(self, quantiles: list = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = torch.tensor(quantiles)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_countries, num_quantiles) - Quantile predictions
            targets: (batch, num_countries) - Actual values
        
        Returns:
            loss: Average quantile loss across all quantiles
        """
        quantiles = self.quantiles.to(predictions.device)
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)
        
        errors = targets_expanded - predictions
        
        # Pinball loss
        loss = torch.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        return loss.mean()


class MultiObjectiveLoss(nn.Module):
    """
    Combined multi-objective loss for GeoRipNet training.
    
    L_total = λ₁ * L_huber + λ₂ * L_dir + λ₃ * L_corr + λ₄ * L_quantile
    
    Args:
        lambda_huber: Weight for Huber loss (default: 1.0)
        lambda_directional: Weight for directional loss (default: 0.3)
        lambda_correlation: Weight for correlation loss (default: 0.2)
        lambda_quantile: Weight for quantile loss (default: 0.0, disabled by default)
        huber_delta: Delta parameter for Huber loss (default: 1.0)
        directional_smoothing: Smoothing for directional loss (default: 0.1)
        use_quantile: Whether to use quantile loss (default: False)
    """
    
    def __init__(
        self,
        lambda_huber: float = 1.0,
        lambda_directional: float = 0.3,
        lambda_correlation: float = 0.2,
        lambda_quantile: float = 0.0,
        huber_delta: float = 1.0,
        directional_smoothing: float = 0.1,
        use_quantile: bool = False
    ):
        super().__init__()
        
        self.lambda_huber = lambda_huber
        self.lambda_directional = lambda_directional
        self.lambda_correlation = lambda_correlation
        self.lambda_quantile = lambda_quantile
        self.use_quantile = use_quantile
        
        # Initialize component losses
        self.huber_loss = HuberLoss(delta=huber_delta)
        self.directional_loss = DirectionalLoss(smoothing=directional_smoothing)
        self.correlation_loss = RippleCorrelationLoss()
        
        if use_quantile:
            self.quantile_loss = QuantileLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prev_targets: torch.Tensor,
        predicted_deltas: torch.Tensor = None,
        true_deltas: torch.Tensor = None,
        quantile_predictions: torch.Tensor = None
    ) -> dict:
        """
        Compute multi-objective loss.
        
        Args:
            predictions: (batch, num_countries) - Price predictions
            targets: (batch, num_countries) - Actual prices
            prev_targets: (batch, num_countries) - Previous time step prices
            predicted_deltas: (batch, num_countries) - Optional predicted deltas
            true_deltas: (batch, num_countries) - Optional true deltas
            quantile_predictions: (batch, num_countries, num_quantiles) - Optional
        
        Returns:
            Dictionary containing:
                - total_loss: Combined loss
                - huber_loss: Huber component
                - directional_loss: Directional component
                - correlation_loss: Correlation component (if deltas provided)
                - quantile_loss: Quantile component (if enabled)
                - directional_accuracy: Hard directional accuracy
        """
        losses = {}
        
        # 1. Huber Loss
        l_huber = self.huber_loss(predictions, targets)
        losses['huber_loss'] = l_huber
        
        # 2. Directional Loss
        l_directional = self.directional_loss(predictions, targets, prev_targets)
        losses['directional_loss'] = l_directional
        
        # Compute directional accuracy (for monitoring)
        with torch.no_grad():
            dir_acc = self.directional_loss.compute_accuracy(
                predictions, targets, prev_targets
            )
            losses['directional_accuracy'] = dir_acc
        
        # 3. Ripple Correlation Loss (if deltas are provided)
        if predicted_deltas is not None and true_deltas is not None:
            l_correlation = self.correlation_loss(predicted_deltas, true_deltas)
            losses['correlation_loss'] = l_correlation
        else:
            l_correlation = torch.tensor(0.0, device=predictions.device)
            losses['correlation_loss'] = l_correlation
        
        # 4. Quantile Loss (if enabled and predictions provided)
        if self.use_quantile and quantile_predictions is not None:
            l_quantile = self.quantile_loss(quantile_predictions, targets)
            losses['quantile_loss'] = l_quantile
        else:
            l_quantile = torch.tensor(0.0, device=predictions.device)
            if self.use_quantile:
                losses['quantile_loss'] = l_quantile
        
        # Total weighted loss
        total_loss = (
            self.lambda_huber * l_huber +
            self.lambda_directional * l_directional +
            self.lambda_correlation * l_correlation +
            self.lambda_quantile * l_quantile
        )
        
        losses['total_loss'] = total_loss
        
        return losses


class AdaptiveLossWeights(nn.Module):
    """
    Learnable adaptive loss weights using uncertainty weighting (Kendall et al., 2018).
    
    Instead of fixed λ values, learns optimal weights during training.
    
    L_total = Σ_i (1 / (2 * σ_i²)) * L_i + log(σ_i)
    
    where σ_i are learnable uncertainty parameters.
    """
    
    def __init__(self, num_losses: int = 4):
        super().__init__()
        # Learnable log-variance (more stable than variance directly)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses: list) -> torch.Tensor:
        """
        Args:
            losses: List of loss tensors [L_1, L_2, ..., L_n]
        
        Returns:
            weighted_loss: Adaptively weighted total loss
        """
        weighted_loss = torch.tensor(0.0, device=losses[0].device)
        
        for i, loss in enumerate(losses):
            # Precision = 1 / variance = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            
            # Weighted loss + regularization
            weighted_loss += precision * loss + self.log_vars[i]
        
        return weighted_loss
    
    def get_weights(self) -> torch.Tensor:
        """Get current loss weights (for logging)."""
        return torch.exp(-self.log_vars)


if __name__ == "__main__":
    # Test all loss functions
    print("Testing Multi-Objective Loss Functions...")
    
    batch_size = 16
    num_countries = 20
    
    # Random data
    predictions = torch.randn(batch_size, num_countries)
    targets = torch.randn(batch_size, num_countries)
    prev_targets = torch.randn(batch_size, num_countries)
    
    predicted_deltas = torch.randn(batch_size, num_countries)
    true_deltas = torch.randn(batch_size, num_countries)
    
    quantile_predictions = torch.randn(batch_size, num_countries, 3)
    
    # Test individual losses
    print("\n1. Testing HuberLoss...")
    huber = HuberLoss(delta=1.0)
    l_huber = huber(predictions, targets)
    print(f"   Huber loss: {l_huber.item():.4f}")
    
    print("\n2. Testing DirectionalLoss...")
    directional = DirectionalLoss()
    l_dir = directional(predictions, targets, prev_targets)
    dir_acc = directional.compute_accuracy(predictions, targets, prev_targets)
    print(f"   Directional loss: {l_dir.item():.4f}")
    print(f"   Directional accuracy: {dir_acc.item():.4f}")
    
    print("\n3. Testing RippleCorrelationLoss...")
    correlation = RippleCorrelationLoss()
    l_corr = correlation(predicted_deltas, true_deltas)
    print(f"   Correlation loss: {l_corr.item():.4f}")
    
    print("\n4. Testing QuantileLoss...")
    quantile = QuantileLoss()
    l_quantile = quantile(quantile_predictions, targets)
    print(f"   Quantile loss: {l_quantile.item():.4f}")
    
    # Test combined loss
    print("\n5. Testing MultiObjectiveLoss...")
    multi_loss = MultiObjectiveLoss(
        lambda_huber=1.0,
        lambda_directional=0.3,
        lambda_correlation=0.2,
        lambda_quantile=0.1,
        use_quantile=True
    )
    
    losses = multi_loss(
        predictions, targets, prev_targets,
        predicted_deltas, true_deltas,
        quantile_predictions
    )
    
    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    print(f"   - Huber: {losses['huber_loss'].item():.4f}")
    print(f"   - Directional: {losses['directional_loss'].item():.4f}")
    print(f"   - Correlation: {losses['correlation_loss'].item():.4f}")
    print(f"   - Quantile: {losses['quantile_loss'].item():.4f}")
    print(f"   - Dir. Accuracy: {losses['directional_accuracy'].item():.4f}")
    
    # Test adaptive weights
    print("\n6. Testing AdaptiveLossWeights...")
    adaptive = AdaptiveLossWeights(num_losses=4)
    loss_list = [l_huber, l_dir, l_corr, l_quantile]
    adaptive_loss = adaptive(loss_list)
    weights = adaptive.get_weights()
    print(f"   Adaptive total loss: {adaptive_loss.item():.4f}")
    print(f"   Current weights: {weights.detach().cpu().numpy()}")
    
    print("\n✓ All loss function tests passed!")

