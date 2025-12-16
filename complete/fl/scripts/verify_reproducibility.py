#!/usr/bin/env python3
"""Verify experiment reproducibility.

This script loads an experiment manifest and verifies that the current
environment matches the recorded environment. It can optionally re-run
the experiment and compare results.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.reproducibility import ExperimentManifest, verify_reproducibility


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Verify experiment reproducibility"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to experiment manifest JSON file"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for floating point comparisons (default: 1e-6)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check manifest exists
    if not args.manifest.exists():
        logger.error(f"Manifest file not found: {args.manifest}")
        return 1
    
    logger.info(f"Loading manifest from {args.manifest}")
    manifest = ExperimentManifest.load(args.manifest)
    
    # Print manifest details
    logger.info("=" * 60)
    logger.info("EXPERIMENT MANIFEST")
    logger.info("=" * 60)
    logger.info(f"Seed: {manifest.seed}")
    logger.info(f"Git commit: {manifest.git_commit or 'N/A'}")
    logger.info(f"Data fingerprint: {manifest.data_fingerprint[:16]}...")
    logger.info(f"Dependencies hash: {manifest.dependencies_hash[:16]}...")
    logger.info("")
    logger.info("Environment:")
    for key, value in manifest.environment.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    logger.info("")
    
    # Verify environment
    logger.info("Verifying current environment matches manifest...")
    result = manifest.verify(tolerance=args.tolerance)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFICATION RESULT")
    logger.info("=" * 60)
    
    if result["passed"]:
        logger.info("✅ PASSED - Environment matches manifest")
        exit_code = 0
    else:
        logger.error("❌ FAILED - Environment does not match manifest")
        logger.error("")
        logger.error("Mismatches:")
        for mismatch in result["mismatches"]:
            logger.error(f"  - {mismatch}")
        exit_code = 1
    
    if result["warnings"]:
        logger.warning("")
        logger.warning("⚠️  Warnings:")
        for warning in result["warnings"]:
            logger.warning(f"  - {warning}")
    
    logger.info("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
