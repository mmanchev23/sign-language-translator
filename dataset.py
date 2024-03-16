from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ASLDataset(Dataset):
    def __init__(self, root, transform=None) -> None:
        self.data = ImageFolder(root=root, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        return self.data[idx]

    @property
    def classes(self) -> list[str]:
        return self.data.classes
