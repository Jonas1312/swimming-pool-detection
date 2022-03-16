"""Train CNN on the swimming pools dataset."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

from architectures.resnet import ResNet as Model


def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    nb_samples = 0
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        nb_samples += len(data)
        data, target = data.to(device), target.to(device).float()
        output = model(data).view_as(target)
        loss = F.binary_cross_entropy_with_logits(output, target, reduction="sum")
        epoch_loss += loss.item()
        loss /= len(data)  # mean batch loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}".format(
                epoch,
                nb_samples,
                len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                loss.item(),
            ),
            end="\r",
        )

    epoch_loss /= len(train_loader.dataset)
    print(
        "Train Epoch: {} [{}/{} ({:.0f}%)], Average epoch loss: {:.6f}".format(
            epoch, nb_samples, len(train_loader.dataset), 100.0, epoch_loss
        )
    )
    return epoch_loss


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data).view_as(target)
            test_loss += F.binary_cross_entropy_with_logits(
                output, target, reduction="sum"
            ).item()
            pred = output.view_as(target) >= 0.5
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "Test set: Average loss: {:.6f}, Correct: {}/{}, Accuracy: ({:.2f}%)".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy


def checkpoint(model, test_loss, test_acc, optimizer, batch_size, epoch, weight_decay):
    file_name = "{}_acc={:.2f}_loss={:.5f}_{}_bs={}_ep={}_wd={}.pth".format(
        Model.__name__,
        test_acc,
        test_loss,
        optimizer.__class__.__name__,
        batch_size,
        epoch,
        weight_decay,
    )
    path = os.path.join("../../models/", file_name)
    if not os.path.isfile(path):
        torch.save(model.state_dict(), path)
        print("Saved: ", file_name)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset_path = "../../data/processed/dataset_pools/"

    # Hyperparams
    batch_size = 32
    epochs = 40
    # input_size = (50,) * 2
    weight_decay = 1e-4

    # Load indices
    train_indices = np.load("../../data/processed/train_indices.npy")
    test_indices = np.load("../../data/processed/test_indices.npy")
    # valid_indices = np.load("../../data/processed/valid_indices.npy")

    # Transforms
    train_transforms = transforms.Compose(
        [
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose([transforms.ToTensor()])

    # Load datasets
    train_set = torch.utils.data.Subset(
        datasets.ImageFolder(dataset_path, train_transforms), train_indices
    )
    test_set = torch.utils.data.Subset(
        datasets.ImageFolder(dataset_path, test_transforms), test_indices
    )
    print("Training set size: ", len(train_set))
    print("Test set size : ", len(test_set))
    print("Total: ", len(train_set) + len(test_set))

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    model = Model().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=1e-1, momentum=0.9, weight_decay=weight_decay
    )
    # optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])

    best_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"####################### EPOCH {epoch}/{epochs} #######################")

        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group["lr"])

        train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = validate(model, device, test_loader)
        scheduler.step()

        if test_acc > best_acc and test_acc > 98:
            best_acc = test_acc
            checkpoint(
                model, test_loss, test_acc, optimizer, batch_size, epoch, weight_decay
            )


if __name__ == "__main__":
    main()
