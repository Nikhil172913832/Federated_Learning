import argparse
from pathlib import Path

import torch

from fl.task import Net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    model = Net()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    print("Loaded checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()


