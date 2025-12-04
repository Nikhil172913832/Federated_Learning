"""
Adapter and LoRA modules for personalized federated learning.

Implements parameter-efficient fine-tuning (PEFT) techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import math


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    
    Adds trainable low-rank matrices A and B to a frozen pretrained weight matrix W.
    Forward: h = Wx + (BA)x, where only A and B are trained.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of decomposition (typically 4-8)
        alpha: Scaling factor (typically 16-32)
        dropout: Dropout probability for LoRA path
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize like Kaiming
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            LoRA output [..., out_features]
        """
        # x @ A^T @ B^T with dropout
        result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Combines a frozen linear layer with a trainable LoRA layer.
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_base: bool = True
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        if freeze_base:
            for param in self.linear.parameters():
                param.requires_grad = False
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through base linear + LoRA adaptation."""
        return self.linear(x) + self.lora(x)
        
    def merge_weights(self) -> None:
        """Merge LoRA weights into base linear (for inference efficiency)."""
        if self.linear.weight.requires_grad:
            return  # Don't merge if base is trainable
            
        with torch.no_grad():
            # W' = W + BA * scaling
            delta_w = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            self.linear.weight.add_(delta_w)
            # Zero out LoRA params after merge
            self.lora.lora_A.zero_()
            self.lora.lora_B.zero_()


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter module.
    
    Standard adapter architecture with down-projection, activation, up-projection.
    Residual connection allows easy integration.
    
    Args:
        input_dim: Input/output dimension
        bottleneck_dim: Hidden dimension (typically 64-256)
        activation: Activation function name
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
            
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize near-identity
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with residual connection.
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            x + adapter(x)
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class AdapterLayer(nn.Module):
    """
    Adapter layer wrapper for any module.
    
    Adds adapter after the base module with optional layer norm.
    """
    
    def __init__(
        self,
        base_module: nn.Module,
        adapter: nn.Module,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.base_module = base_module
        self.adapter = adapter
        self.layer_norm = nn.LayerNorm(adapter.input_dim) if use_layernorm else nn.Identity()
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward through base module then adapter."""
        x = self.base_module(*args, **kwargs)
        x = self.layer_norm(x)
        x = self.adapter(x)
        return x


class PrefixTuningAdapter(nn.Module):
    """
    Prefix tuning adapter.
    
    Prepends learnable prefix tokens to the input sequence.
    Useful for transformer-based models.
    
    Args:
        prefix_length: Number of prefix tokens
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers to apply prefix to
    """
    
    def __init__(
        self,
        prefix_length: int = 10,
        hidden_dim: int = 768,
        num_layers: int = 12
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Learnable prefix tokens
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, prefix_length, hidden_dim)
        )
        
        # Initialize
        nn.init.normal_(self.prefix_tokens, mean=0.0, std=0.02)
        
    def forward(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Prepend prefix tokens to hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            layer_idx: Which transformer layer (0-indexed)
            
        Returns:
            [batch, prefix_length + seq_len, hidden_dim]
        """
        batch_size = hidden_states.size(0)
        prefix = self.prefix_tokens[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prefix, hidden_states], dim=1)


class ModelWithAdapters(nn.Module):
    """
    Wrapper to add adapters to an existing model.
    
    Supports multiple adapter types and flexible placement.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: Dict,
        target_modules: Optional[List[str]] = None
    ):
        """
        Args:
            base_model: Base model to adapt
            adapter_config: Configuration dict with keys:
                - type: "lora", "bottleneck", or "prefix"
                - rank: For LoRA (default 4)
                - alpha: For LoRA (default 16)
                - bottleneck_dim: For bottleneck (default 64)
                - dropout: Dropout probability
            target_modules: List of module names to adapt (e.g., ["layer1", "layer2"])
        """
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config
        self.target_modules = target_modules or []
        
        self._add_adapters()
        
    def _add_adapters(self):
        """Add adapters to target modules."""
        adapter_type = self.adapter_config.get("type", "lora")
        
        for name, module in self.base_model.named_modules():
            # Check if this module should get an adapter
            if self.target_modules and name not in self.target_modules:
                continue
                
            if isinstance(module, nn.Linear) and adapter_type == "lora":
                # Replace with LoRA linear
                rank = self.adapter_config.get("rank", 4)
                alpha = self.adapter_config.get("alpha", 16.0)
                dropout = self.adapter_config.get("dropout", 0.0)
                
                lora_linear = LoRALinear(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_base=True
                )
                
                # Replace in parent module
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                if parent_name:
                    parent = dict(self.base_model.named_modules())[parent_name]
                else:
                    parent = self.base_model
                setattr(parent, child_name, lora_linear)
                
            elif adapter_type == "bottleneck":
                # Add bottleneck adapter after module
                if hasattr(module, "out_features"):
                    bottleneck_dim = self.adapter_config.get("bottleneck_dim", 64)
                    dropout = self.adapter_config.get("dropout", 0.1)
                    
                    adapter = BottleneckAdapter(
                        input_dim=module.out_features,
                        bottleneck_dim=bottleneck_dim,
                        dropout=dropout
                    )
                    
                    # Wrap module with adapter
                    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                    if parent_name:
                        parent = dict(self.base_model.named_modules())[parent_name]
                    else:
                        parent = self.base_model
                        
                    wrapped = AdapterLayer(module, adapter)
                    setattr(parent, child_name, wrapped)
                    
    def forward(self, *args, **kwargs):
        """Forward through adapted model."""
        return self.base_model(*args, **kwargs)
        
    def get_adapter_params(self) -> Dict[str, torch.Tensor]:
        """Get only adapter parameters (for personalization storage)."""
        adapter_params = {}
        for name, param in self.named_parameters():
            if "lora" in name or "adapter" in name or "prefix" in name:
                adapter_params[name] = param.detach().cpu()
        return adapter_params
        
    def load_adapter_params(self, adapter_params: Dict[str, torch.Tensor]):
        """Load adapter parameters."""
        for name, param in self.named_parameters():
            if name in adapter_params:
                param.data.copy_(adapter_params[name].to(param.device))
                
    def freeze_base_model(self):
        """Freeze all non-adapter parameters."""
        for name, param in self.named_parameters():
            if "lora" not in name and "adapter" not in name and "prefix" not in name:
                param.requires_grad = False
                
    def unfreeze_base_model(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_adapter_summary(model: nn.Module):
    """Print summary of adapter configuration."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    adapter_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and ("lora" in name or "adapter" in name or "prefix" in name)
    )
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Adapter parameters: {adapter_params:,} ({100 * adapter_params / total_params:.2f}%)")
