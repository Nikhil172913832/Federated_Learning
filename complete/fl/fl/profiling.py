"""Performance profiling and monitoring.

This module provides tools for:
- Training performance profiling
- CPU and GPU profiling
- Memory profiling
- Bottleneck identification
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any

import torch

try:
    from torch.profiler import profile, ProfilerActivity, schedule
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiling for FL training."""

    def __init__(self, output_dir: Path):
        """Initialize profiler.
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def profile_training(
        self,
        profile_memory: bool = True,
        record_shapes: bool = True,
        with_stack: bool = False,
    ):
        """Profile training performance.
        
        Args:
            profile_memory: Whether to profile memory usage
            record_shapes: Whether to record tensor shapes
            with_stack: Whether to record stack traces
            
        Yields:
            Profiler instance
        """
        if not PROFILER_AVAILABLE:
            logger.warning("PyTorch profiler not available, skipping profiling")
            yield None
            return

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            on_trace_ready=self._trace_handler,
        ) as prof:
            yield prof

        # Log top operations
        self._log_top_operations(prof)

    def _trace_handler(self, prof):
        """Handle profiling trace.
        
        Args:
            prof: Profiler instance
        """
        # Export Chrome trace
        trace_path = self.output_dir / f"trace_{int(time.time())}.json"
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"Saved profiling trace to {trace_path}")

    def _log_top_operations(self, prof, top_k: int = 10):
        """Log top operations by time.
        
        Args:
            prof: Profiler instance
            top_k: Number of top operations to log
        """
        if prof is None:
            return

        stats = prof.key_averages()
        
        logger.info("=" * 60)
        logger.info(f"Top {top_k} operations by CPU time:")
        logger.info("=" * 60)
        
        for i, stat in enumerate(stats[:top_k], 1):
            logger.info(
                f"{i}. {stat.key:50s} {stat.cpu_time_total/1000:10.2f}ms "
                f"({stat.cpu_time_total/sum(s.cpu_time_total for s in stats)*100:5.1f}%)"
            )

        if torch.cuda.is_available():
            logger.info("")
            logger.info(f"Top {top_k} operations by CUDA time:")
            logger.info("=" * 60)
            
            cuda_stats = sorted(stats, key=lambda x: x.cuda_time_total, reverse=True)
            for i, stat in enumerate(cuda_stats[:top_k], 1):
                logger.info(
                    f"{i}. {stat.key:50s} {stat.cuda_time_total/1000:10.2f}ms "
                    f"({stat.cuda_time_total/sum(s.cuda_time_total for s in cuda_stats)*100:5.1f}%)"
                )

    @contextmanager
    def measure_time(self, operation_name: str):
        """Measure execution time of an operation.
        
        Args:
            operation_name: Name of the operation
            
        Yields:
            None
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            logger.info(f"{operation_name}: {elapsed:.3f}s")

    def profile_memory(self) -> Dict[str, float]:
        """Profile current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        if torch.cuda.is_available():
            stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            stats["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        else:
            stats["cuda_allocated_mb"] = 0.0
            stats["cuda_reserved_mb"] = 0.0
            stats["cuda_max_allocated_mb"] = 0.0

        return stats

    def log_memory_stats(self):
        """Log current memory statistics."""
        stats = self.profile_memory()
        
        logger.info("Memory Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f} MB")

    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("Reset peak memory statistics")
