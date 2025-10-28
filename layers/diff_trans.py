import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for time series.
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function for time series.
    """

    def __init__(self, d_model):
        super().__init__()
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        g = F.silu(self.WG(x))
        z = self.W1(x)
        return self.W2(g * z)


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention for time series.
    Removes causal mask for bidirectional processing of time series.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))

        self.lambda_init = lambda_init
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X):
        batch, N, d_model = X.shape

        # Project inputs
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # Reshape for multi-head attention
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        # Split into components
        Q1, Q2 = Q.chunk(2, dim=-1)
        K1, K2 = K.chunk(2, dim=-1)

        # Compute lambda values
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_val = (
            torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init
        )
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Compute attention scores (no causal mask for time series)
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)
        attention2 = F.softmax(A2, dim=-1)
        attention = attention1 - lambda_val * attention2

        # Apply attention to values
        O = torch.matmul(attention, V)

        # Normalize each head
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale output
        O_normalized = O_normalized * (1 - self.lambda_init)

        # Concatenate heads
        O_concat = (
            O_normalized.transpose(1, 2)
            .contiguous()
            .view(batch, N, self.num_heads * 2 * self.d_head)
        )

        # Final projection
        out = self.W_o(O_concat)
        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer for time series.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x):
        y = self.attn(self.norm1(x)) + x
        z = self.ff(self.norm2(y)) + y
        return z


class TimeSeriesDiffTransformer(nn.Module):
    """
    DiffTransformer model adapted for time series data.
    Supports both univariate and multivariate time series.
    """

    def __init__(
        self,
        input_dim,
        d_model,
        num_heads,
        num_layers,
        output_dim=None,
        max_seq_length=512,
        use_positional_encoding=True,
        dropout=0.1,
    ):
        """
        Args:
            input_dim: Number of input features (channels)
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Number of output features (if None, same as input_dim)
            max_seq_length: Maximum sequence length
            use_positional_encoding: Whether to use positional encoding
            dropout: Dropout rate
        """

        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                DiffTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    lambda_init=0.8 - 0.6 * math.exp(-0.3 * (l - 1)),
                )
                for l in range(1, num_layers + 1)
            ]
        )

        # Output layers
        self.norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, self.output_dim)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, x):
        """
        Forward pass for time series data.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            Output tensor of shape (batch_size, seq_length, output_dim)
        """
        batch_size, seq_length, _ = x.shape

        # Project input to d_model
        x = self.input_proj(x)  # (batch_size, seq_length, d_model)

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization and projection
        x = self.norm(x)
        output = self.output_proj(x)  # (batch_size, seq_length, output_dim)

        return output


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for time series transformer.
    """

    def __init__(self, d_model, max_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_length, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    """
    Encoder-only version for time series representation learning.
    """

    def __init__(
        self,
        input_dim,
        d_model,
        num_heads,
        num_layers,
        hidden_dim=None,
        max_seq_length=512,
    ):
        super().__init__()

        self.transformer = TimeSeriesDiffTransformer(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=d_model,  # Keep same dimension for representation
            max_seq_length=max_seq_length,
        )

        # Optional projection head for downstream tasks
        hidden_dim = hidden_dim or d_model
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, return_representation=True):
        """
        Args:
            x: Input time series (batch_size, seq_length, input_dim)
            return_representation: If True, return representation; else return projection
        """
        # Get transformer output
        transformer_out = self.transformer(x)  # (batch_size, seq_length, d_model)

        # Global average pooling over time dimension
        representation = transformer_out.mean(dim=1)  # (batch_size, d_model)

        if return_representation:
            return representation
        else:
            # Apply projection head
            projection = self.projection_head(representation)
            return projection


# Example usage for different time series tasks
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example 1: Time series forecasting
    print("=== Time Series Forecasting ===")
    batch_size, seq_length, input_dim = 32, 100, 5  # 5 features
    forecast_horizon = 10

    model_forecast = TimeSeriesDiffTransformer(
        input_dim=input_dim,
        d_model=128,
        num_heads=8,
        num_layers=6,
        output_dim=input_dim,  # Forecast same number of features
        max_seq_length=seq_length + forecast_horizon,
    ).to(device)

    # Input: historical data
    x_forecast = torch.randn(batch_size, seq_length, input_dim).to(device)
    output_forecast = model_forecast(x_forecast)
    print(f"Forecasting input: {x_forecast.shape}")
    print(f"Forecasting output: {output_forecast.shape}")

    # Example 2: Time series classification
    print("\n=== Time Series Classification ===")
    num_classes = 3

    model_classifier = TimeSeriesDiffTransformer(
        input_dim=input_dim,
        d_model=128,
        num_heads=8,
        num_layers=6,
        output_dim=num_classes,  # Output class probabilities
        max_seq_length=seq_length,
    ).to(device)

    x_classify = torch.randn(batch_size, seq_length, input_dim).to(device)
    output_classify = model_classifier(x_classify)
    # Take last time step for classification
    logits = output_classify[:, -1, :]  # (batch_size, num_classes)
    print(f"Classification input: {x_classify.shape}")
    print(f"Classification logits: {logits.shape}")

    # Example 3: Representation learning
    print("\n=== Representation Learning ===")
    model_encoder = TimeSeriesEncoder(
        input_dim=input_dim,
        d_model=128,
        num_heads=8,
        num_layers=6,
        max_seq_length=seq_length,
    ).to(device)

    x_representation = torch.randn(batch_size, seq_length, input_dim).to(device)
    representation = model_encoder(x_representation)
    projection = model_encoder(x_representation, return_representation=False)
    print(f"Representation input: {x_representation.shape}")
    print(f"Representation output: {representation.shape}")
    print(f"Projection output: {projection.shape}")
