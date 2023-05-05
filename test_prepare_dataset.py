#! /usr/bin/env python3

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms

transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
training_dataset = datasets.CIFAR10(root="./training_data", train=True, download=True, transform=transform)
