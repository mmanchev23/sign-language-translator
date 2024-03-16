#!/usr/bin/env python

from model import ASLModel
from dataset import ASLDataset

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ASLDataset(root="dataset/", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    model = ASLModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    main()
