"""Gradient compression for communication efficiency."""

import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics for compression operation."""

    original_size: int
    compressed_size: int
    compression_ratio: float
    num_params: int


class QuantizationCompressor:
    """Adaptive quantization compressor."""

    def __init__(self, num_bits: int = 8):
        """Initialize quantization compressor.

        Args:
            num_bits: Number of bits for quantization (1-32)
        """
        if not (1 <= num_bits <= 32):
            raise ValueError(f"num_bits must be in [1, 32], got {num_bits}")
        self.num_bits = num_bits
        self.num_levels = 2**num_bits

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compress tensor using quantization.

        Args:
            tensor: Input tensor to compress

        Returns:
            Tuple of (quantized_tensor, metadata)
        """
        min_val = tensor.min()
        max_val = tensor.max()

        scale = (max_val - min_val) / (self.num_levels - 1)
        if scale == 0:
            scale = 1.0

        quantized = torch.round((tensor - min_val) / scale)
        quantized = quantized.clamp(0, self.num_levels - 1)

        if self.num_bits <= 8:
            quantized = quantized.to(torch.uint8)
        elif self.num_bits <= 16:
            quantized = quantized.to(torch.int16)
        else:
            quantized = quantized.to(torch.int32)

        metadata = {
            "min_val": min_val.item(),
            "max_val": max_val.item(),
            "scale": scale.item(),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
        }

        return quantized, metadata

    def decompress(self, quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decompress quantized tensor.

        Args:
            quantized: Quantized tensor
            metadata: Compression metadata

        Returns:
            Decompressed tensor
        """
        quantized = quantized.float()
        tensor = quantized * metadata["scale"] + metadata["min_val"]
        return tensor.reshape(metadata["shape"])


class TopKSparsifier:
    """Top-K sparsification compressor."""

    def __init__(self, sparsity: float = 0.9):
        """Initialize Top-K sparsifier.

        Args:
            sparsity: Fraction of values to zero out (0-1)
        """
        if not (0 <= sparsity < 1):
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
        self.sparsity = sparsity

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compress tensor using top-k sparsification.

        Args:
            tensor: Input tensor to compress

        Returns:
            Tuple of (sparse_values, metadata)
        """
        flat = tensor.flatten()
        k = max(1, int(len(flat) * (1 - self.sparsity)))

        values, indices = torch.topk(flat.abs(), k)
        signs = torch.sign(flat[indices])
        sparse_values = values * signs

        metadata = {"indices": indices, "shape": tensor.shape, "k": k}

        return sparse_values, metadata

    def decompress(self, sparse_values: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """Decompress sparse tensor.

        Args:
            sparse_values: Sparse values
            metadata: Compression metadata

        Returns:
            Decompressed tensor
        """
        flat = torch.zeros(torch.prod(torch.tensor(metadata["shape"])).item())
        flat[metadata["indices"]] = sparse_values
        return flat.reshape(metadata["shape"])


class HybridCompressor:
    """Hybrid compressor combining quantization and sparsification."""

    def __init__(
        self,
        quantization_bits: int = 8,
        sparsity: float = 0.9,
        use_error_feedback: bool = True,
    ):
        """Initialize hybrid compressor.

        Args:
            quantization_bits: Bits for quantization
            sparsity: Sparsity level
            use_error_feedback: Whether to use error feedback
        """
        self.quantizer = QuantizationCompressor(quantization_bits)
        self.sparsifier = TopKSparsifier(sparsity)
        self.use_error_feedback = use_error_feedback
        self.error_buffer: Dict[str, torch.Tensor] = {}

    def compress_gradients(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], CompressionStats]:
        """Compress gradients in state dict.

        Args:
            state_dict: Model state dictionary

        Returns:
            Tuple of (compressed_dict, metadata_dict, stats)
        """
        compressed = {}
        metadata = {}
        original_size = 0
        compressed_size = 0
        num_params = 0

        for name, param in state_dict.items():
            if param.numel() == 0:
                continue

            tensor = param.clone()

            if self.use_error_feedback and name in self.error_buffer:
                tensor += self.error_buffer[name]

            sparse_vals, sparse_meta = self.sparsifier.compress(tensor)
            quant_vals, quant_meta = self.quantizer.compress(sparse_vals)

            compressed[name] = quant_vals
            metadata[name] = {"sparse": sparse_meta, "quant": quant_meta}

            if self.use_error_feedback:
                decompressed = self.decompress_gradients(
                    {name: quant_vals}, {name: metadata[name]}
                )[name]
                self.error_buffer[name] = tensor - decompressed

            original_size += param.numel() * param.element_size()
            compressed_size += quant_vals.numel() * quant_vals.element_size()
            num_params += param.numel()

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=original_size / max(compressed_size, 1),
            num_params=num_params,
        )

        return compressed, metadata, stats

    def decompress_gradients(
        self, compressed: Dict[str, torch.Tensor], metadata: Dict[str, Dict]
    ) -> Dict[str, torch.Tensor]:
        """Decompress gradients.

        Args:
            compressed: Compressed gradients
            metadata: Compression metadata

        Returns:
            Decompressed state dict
        """
        decompressed = {}

        for name, quant_vals in compressed.items():
            meta = metadata[name]
            sparse_vals = self.quantizer.decompress(quant_vals, meta["quant"])
            tensor = self.sparsifier.decompress(sparse_vals, meta["sparse"])
            decompressed[name] = tensor

        return decompressed
