#!/usr/bin/env python

from train import train
from test import test

def main() -> None:
    choice = str(input("Enter 'train' to train the model or 'test' to test the model: ")).lower()

    if choice == "train":
        train()
    elif choice == "test":
        test()
    else:
        print("Invalid choice. Please enter 'train' or 'test'.")

if __name__ == "__main__":
    main()
