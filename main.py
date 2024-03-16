#!/usr/bin/env python

from dataset import ASLDataset
from torchvision import transforms

def main() -> None:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ASLDataset(root="dataset/", transform=transform)

if __name__ == "__main__":
    main()
