"""Predict if there's a pool in an image and visualize CNN attention with CAMs."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

import cv2
from architectures.resnet import ResNet as Model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    weights_name = "ResNet_acc=98.48_loss=0.04981_SGD_bs=64_ep=32_wd=0.0001.pth"
    dataset_path = "../../data/processed/dataset_pools/"

    # Load indices
    test_indices = np.load("../../data/processed/test_indices.npy")

    # Transforms
    test_transforms = transforms.ToTensor()

    # Load dataset
    test_set = torch.utils.data.Subset(
        datasets.ImageFolder(dataset_path, test_transforms), test_indices
    )
    print("Test set size : ", len(test_set))

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=True, pin_memory=True
    )

    model = Model().to(device)
    model.load_state_dict(torch.load(os.path.join("../../models/", weights_name)))
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, apply_avgpool=False)

            # Predicted class
            probability = (
                model.fc(torch.flatten(model.avgpool(output), 1)).sigmoid().item()
            )
            predicted_class = probability >= 0.5
            good_pred = target.item() == predicted_class

            # Generate heatmap from class activation maps
            heatmap = output * model.fc.weight.view((1, -1, 1, 1)).flip(1)
            heatmap = heatmap.mean(dim=1, keepdim=True)
            heatmap = torch.nn.functional.interpolate(
                heatmap, size=(data.size()[2:]), mode="bicubic", align_corners=True
            )
            heatmap = heatmap[0, 0]
            heatmap = heatmap.cpu().numpy()
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Plot
            data = data.permute(0, 2, 3, 1)  # BCHW to BHWC
            data = (data.cpu().numpy()[0] * 255).astype(np.uint8)
            merged = cv2.addWeighted(data, 0.6, heatmap, 0.4, 0)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
            ax1.imshow(data)
            ax2.imshow(heatmap)
            ax3.imshow(merged)

            fig.suptitle(
                'Predicted class "{}" with {:.1f}% confidence\n({} prediction)'.format(
                    "pool" if predicted_class else "no pool",
                    probability * 100
                    if probability >= 0.5
                    else (1 - probability) * 100,
                    "right" if good_pred else "WRONG",
                )
            )
            plt.show()


if __name__ == "__main__":
    main()
