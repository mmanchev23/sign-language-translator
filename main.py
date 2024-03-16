#!/usr/bin/env python

import sys
import torch
import torchvision

def main() -> None:
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

if __name__ == "__main__":
    main()
