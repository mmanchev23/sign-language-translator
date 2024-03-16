from model import ASLModel
from dataset import ASLDataset

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader


def train() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ASLDataset(root="dataset/train", transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    model = ASLModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
    
    with open("model.pth", "wb") as f:
        torch.save(model, f)

    print("Finished Training")