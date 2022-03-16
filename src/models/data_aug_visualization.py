"""Data augmentation hyperparams tests and visualization."""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def main():
    batch_size = 16
    dataset_path = "../../data/processed/dataset_pools/"
    transforms_ = transforms.Compose(
        [
            # transforms.RandomAffine(
            #     degrees=0,
            #     # translate=(0.05, 0.05),
            #     scale=(0.95, 1.05),
            #     # shear=10,
            #     resample=2,
            #     # fillcolor=0,
            # ),
            # transforms.ColorJitter(
            #     brightness=0.20, contrast=0.15, saturation=0.15, hue=0.04
            # ),
            # transforms.RandomCrop(input_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(dataset_path, transform=transforms_)
    print("Dataset size: ", len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    for data, target in loader:
        print(data.size(), target.size())
        print("target: ", target)
        data = data.permute(0, 2, 3, 1)

        # Show all batch images
        # for i, class_ in enumerate(target):
        #     plt.imshow(data[i])
        #     plt.title("class: {}".format("pool" if class_.item() else "no pool"))
        #     plt.show()

        # Show first image only
        plt.imshow(data[0])
        plt.title("class: {}".format("pool" if target[0].item() else "no pool"))
        plt.show()


if __name__ == "__main__":
    main()
