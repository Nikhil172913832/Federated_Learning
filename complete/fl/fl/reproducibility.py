"""Experiment reproducibility infrastructure.

This module provides tools to ensure ML experiments are fully reproducible
by capturing all inputs, environment state, and data fingerprints.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import torch
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


logger = logging.getLogger(__name__)


@dataclass
class ExperimentManifest:
    """Complete record of experiment inputs for reproducibility.
    
    Attributes:
        config: Experiment configuration dictionary
        environment: Environment details (Python, PyTorch, CUDA versions)
        data_fingerprint: Hash of dataset for integrity verification
        dependencies_hash: Hash of requirements for dependency tracking
        git_commit: Git commit SHA if available
        seed: Random seed used
    """
    config: Dict[str, Any]
    environment: Dict[str, str]
    data_fingerprint: str
    dependencies_hash: str
    git_commit: Optional[str]
    seed: int
    
    @classmethod
    def create(cls, config: Dict[str, Any], dataset_path: Optional[Path] = None):
        """Create experiment manifest from config.
        
        Args:
            config: Experiment configuration
            dataset_path: Optional path to dataset for fingerprinting
            
        Returns:
            ExperimentManifest instance
        """
        return cls(
            config=config,
            environment=cls._capture_environment(),
            data_fingerprint=cls._hash_dataset(dataset_path) if dataset_path else "N/A",
            dependencies_hash=cls._hash_requirements(),
            git_commit=cls._get_git_commit(),
            seed=config.get("seed", 42),
        )
    
    @staticmethod
    def _capture_environment() -> Dict[str, str]:
        """Capture environment details.
        
        Returns:
            Dictionary with Python, PyTorch, CUDA versions
        """
        env = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        
        # Add CUDA version if available
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "N/A"
            env["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
            env["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            env["cuda_version"] = "N/A"
            env["cudnn_version"] = "N/A"
            env["gpu_name"] = "N/A"
        
        return env
    
    @staticmethod
    def _hash_dataset(dataset_path: Optional[Path]) -> str:
        """Compute hash of dataset for integrity verification.
        
        Args:
            dataset_path: Path to dataset directory or file
            
        Returns:
            SHA256 hash of dataset
        """
        if dataset_path is None or not dataset_path.exists():
            return "N/A"
        
        hasher = hashlib.sha256()
        
        if dataset_path.is_file():
            # Hash single file
            with open(dataset_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
        elif dataset_path.is_dir():
            # Hash all files in directory (sorted for consistency)
            for file_path in sorted(dataset_path.rglob('*')):
                if file_path.is_file():
                    hasher.update(str(file_path.relative_to(dataset_path)).encode())
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    @staticmethod
    def _hash_requirements() -> str:
        """Hash requirements.txt or installed packages.
        
        Returns:
            SHA256 hash of dependencies
        """
        # Try to find requirements.txt
        req_files = [
            Path("requirements.txt"),
            Path("complete/fl/requirements.txt"),
            Path("requirements-dev.txt"),
        ]
        
        for req_file in req_files:
            if req_file.exists():
                with open(req_file, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
        
        # Fallback: hash pip freeze output
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            return hashlib.sha256(result.stdout.encode()).hexdigest()
        except subprocess.CalledProcessError:
            logger.warning("Could not hash dependencies")
            return "N/A"
    
    @staticmethod
    def _get_git_commit() -> Optional[str]:
        """Get current git commit SHA.
        
        Returns:
            Git commit SHA or None if not in git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file.
        
        Args:
            path: Path to save manifest
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Saved experiment manifest to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentManifest":
        """Load manifest from JSON file.
        
        Args:
            path: Path to manifest file
            
        Returns:
            ExperimentManifest instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def verify(self, tolerance: float = 1e-6) -> Dict[str, Any]:
        """Verify current environment matches manifest.
        
        Args:
            tolerance: Tolerance for floating point comparisons
            
        Returns:
            Dictionary with verification results:
            {
                "passed": bool,
                "mismatches": List[str],
                "warnings": List[str]
            }
        """
        current_env = self._capture_environment()
        mismatches = []
        warnings = []
        
        # Check Python version (major.minor must match)
        current_py = current_env["python_version"].split()[0]
        manifest_py = self.environment["python_version"].split()[0]
        current_py_major_minor = ".".join(current_py.split(".")[:2])
        manifest_py_major_minor = ".".join(manifest_py.split(".")[:2])
        
        if current_py_major_minor != manifest_py_major_minor:
            mismatches.append(
                f"Python version mismatch: {current_py_major_minor} vs {manifest_py_major_minor}"
            )
        
        # Check PyTorch version (major.minor must match)
        current_torch = current_env["torch_version"].split("+")[0]  # Remove +cu118 suffix
        manifest_torch = self.environment["torch_version"].split("+")[0]
        current_torch_major_minor = ".".join(current_torch.split(".")[:2])
        manifest_torch_major_minor = ".".join(manifest_torch.split(".")[:2])
        
        if current_torch_major_minor != manifest_torch_major_minor:
            mismatches.append(
                f"PyTorch version mismatch: {current_torch_major_minor} vs {manifest_torch_major_minor}"
            )
        
        # Check CUDA availability (warn if different)
        if current_env["cuda_version"] != self.environment["cuda_version"]:
            warnings.append(
                f"CUDA version changed: {current_env['cuda_version']} vs {self.environment['cuda_version']}"
            )
        
        # Check dependencies hash
        current_deps = self._hash_requirements()
        if current_deps != self.dependencies_hash and current_deps != "N/A":
            warnings.append(
                "Dependencies changed (requirements hash mismatch)"
            )
        
        # Check git commit
        current_commit = self._get_git_commit()
        if current_commit and self.git_commit and current_commit != self.git_commit:
            warnings.append(
                f"Git commit changed: {current_commit[:8]} vs {self.git_commit[:8]}"
            )
        
        return {
            "passed": len(mismatches) == 0,
            "mismatches": mismatches,
            "warnings": warnings,
        }


def ensure_reproducibility(config: Dict[str, Any], output_dir: Path) -> ExperimentManifest:
    """Ensure experiment reproducibility by creating and saving manifest.
    
    Args:
        config: Experiment configuration
        output_dir: Directory to save manifest
        
    Returns:
        ExperimentManifest instance
    """
    manifest = ExperimentManifest.create(config)
    manifest_path = output_dir / "experiment_manifest.json"
    manifest.save(manifest_path)
    
    return manifest


def verify_reproducibility(manifest_path: Path, tolerance: float = 1e-6) -> bool:
    """Verify current environment matches saved manifest.
    
    Args:
        manifest_path: Path to saved manifest
        tolerance: Tolerance for comparisons
        
    Returns:
        True if environment matches, False otherwise
    """
    manifest = ExperimentManifest.load(manifest_path)
    result = manifest.verify(tolerance)
    
    if result["passed"]:
        logger.info("✅ Environment matches manifest")
    else:
        logger.error("❌ Environment does not match manifest")
        for mismatch in result["mismatches"]:
            logger.error(f"  - {mismatch}")
    
    if result["warnings"]:
        logger.warning("⚠️  Warnings:")
        for warning in result["warnings"]:
            logger.warning(f"  - {warning}")
    
    return result["passed"]
