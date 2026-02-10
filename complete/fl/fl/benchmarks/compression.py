"""Compression benchmarking and metrics tracking."""

import time
import torch
from typing import Dict, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from fl.compression import HybridCompressor, CompressionStats
from fl.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CompressionBenchmark:
    """Benchmark results for compression."""

    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: float
    original_size_mb: float
    compressed_size_mb: float
    accuracy_loss: float
    method: str
    config: dict


class CompressionBenchmarker:
    """Benchmark compression algorithms."""

    def __init__(self, output_dir: str = "./benchmarks"):
        """Initialize benchmarker.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[CompressionBenchmark] = []

    def benchmark_compression(
        self,
        state_dict: Dict[str, torch.Tensor],
        quantization_bits: int = 8,
        sparsity: float = 0.9,
        use_error_feedback: bool = True,
    ) -> CompressionBenchmark:
        """Benchmark compression on state dict.

        Args:
            state_dict: Model state dictionary
            quantization_bits: Bits for quantization
            sparsity: Sparsity level
            use_error_feedback: Whether to use error feedback

        Returns:
            Benchmark results
        """
        compressor = HybridCompressor(
            quantization_bits=quantization_bits,
            sparsity=sparsity,
            use_error_feedback=use_error_feedback,
        )

        start = time.perf_counter()
        compressed, metadata, stats = compressor.compress_gradients(state_dict)
        compression_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        decompressed = compressor.decompress_gradients(compressed, metadata)
        decompression_time = (time.perf_counter() - start) * 1000

        accuracy_loss = self._compute_accuracy_loss(state_dict, decompressed)

        result = CompressionBenchmark(
            compression_ratio=stats.compression_ratio,
            compression_time_ms=compression_time,
            decompression_time_ms=decompression_time,
            original_size_mb=stats.original_size / (1024 * 1024),
            compressed_size_mb=stats.compressed_size / (1024 * 1024),
            accuracy_loss=accuracy_loss,
            method="hybrid",
            config={
                "quantization_bits": quantization_bits,
                "sparsity": sparsity,
                "use_error_feedback": use_error_feedback,
            },
        )

        self.results.append(result)
        logger.info(
            f"Compression: {result.compression_ratio:.2f}x, "
            f"Time: {result.compression_time_ms:.2f}ms, "
            f"Loss: {result.accuracy_loss:.4f}"
        )

        return result

    def _compute_accuracy_loss(
        self, original: Dict[str, torch.Tensor], decompressed: Dict[str, torch.Tensor]
    ) -> float:
        """Compute relative error between original and decompressed.

        Args:
            original: Original state dict
            decompressed: Decompressed state dict

        Returns:
            Relative error
        """
        total_error = 0.0
        total_norm = 0.0

        for name in original.keys():
            if name in decompressed:
                error = torch.norm(original[name] - decompressed[name])
                norm = torch.norm(original[name])
                total_error += error.item()
                total_norm += norm.item()

        return total_error / max(total_norm, 1e-10)

    def run_sweep(
        self,
        state_dict: Dict[str, torch.Tensor],
        quantization_bits_list: List[int] = [4, 8, 16],
        sparsity_list: List[float] = [0.5, 0.7, 0.9, 0.95],
    ) -> List[CompressionBenchmark]:
        """Run parameter sweep.

        Args:
            state_dict: Model state dictionary
            quantization_bits_list: List of bit widths to test
            sparsity_list: List of sparsity levels to test

        Returns:
            List of benchmark results
        """
        logger.info("Running compression parameter sweep...")

        for bits in quantization_bits_list:
            for sparsity in sparsity_list:
                self.benchmark_compression(
                    state_dict,
                    quantization_bits=bits,
                    sparsity=sparsity,
                    use_error_feedback=True,
                )

        return self.results

    def save_results(self, filename: str = "compression_benchmarks.json"):
        """Save benchmark results to file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        logger.info(f"Saved {len(self.results)} benchmark results to {output_path}")

    def get_best_config(self, target_compression: float = 20.0) -> CompressionBenchmark:
        """Get best configuration for target compression ratio.

        Args:
            target_compression: Target compression ratio

        Returns:
            Best benchmark result
        """
        valid_results = [
            r for r in self.results if r.compression_ratio >= target_compression
        ]

        if not valid_results:
            logger.warning(f"No results meet target compression {target_compression}x")
            return max(self.results, key=lambda r: r.compression_ratio)

        return min(valid_results, key=lambda r: r.accuracy_loss)


def benchmark_model_compression(model: torch.nn.Module) -> CompressionBenchmark:
    """Benchmark compression on a model.

    Args:
        model: PyTorch model

    Returns:
        Benchmark result
    """
    benchmarker = CompressionBenchmarker()
    state_dict = model.state_dict()

    result = benchmarker.benchmark_compression(
        state_dict, quantization_bits=8, sparsity=0.9, use_error_feedback=True
    )

    benchmarker.save_results()
    return result
