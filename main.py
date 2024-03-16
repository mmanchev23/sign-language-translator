#!/usr/bin/env python

from dataset import ASLDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def main() -> None:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ASLDataset(root="dataset/", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    main()
