#!/usr/bin/env python3
"""
Quick Demo: Federated Learning with MNIST

This demonstrates the FL platform in action using the existing complete/fl/ code.
Run this to see federated learning work in under 2 minutes.

Requirements:
- Docker and Docker Compose V2
- Or: Python 3.9+ with dependencies from complete/fl/pyproject.toml
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("FEDERATED LEARNING QUICK DEMO")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent
    complete_dir = project_root / "complete"
    
    if not complete_dir.exists():
        print("❌ Error: complete/ directory not found")
        print("   Run this script from the project root")
        return 1
    
    print("This demo will:")
    print("  1. Start the FL platform with Docker Compose")
    print("  2. Run 10 federated learning rounds")
    print("  3. Show accuracy improving over time")
    print()
    print("Expected runtime: ~2 minutes")
    print()
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Demo cancelled")
        return 0
    
    print()
    print("Starting FL platform...")
    print("-" * 60)
    
    try:
        # Launch platform
        subprocess.run(
            ["./launch-platform.sh"],
            cwd=project_root,
            check=True,
        )
        
        print()
        print("=" * 60)
        print("✓ FL Platform Running!")
        print("=" * 60)
        print()
        print("Access points:")
        print("  - Dashboard: http://localhost:8050")
        print("  - MLflow: http://localhost:5000")
        print()
        print("The platform is training a model across multiple clients.")
        print("Check the dashboard to see real-time progress!")
        print()
        print("To stop: docker compose -f complete/compose-with-ui.yml down")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching platform: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
