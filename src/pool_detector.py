import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models.architectures.resnet import ResNet as Model


class PoolDetector:
    """Swimming pool detector.

    Args:
        weights_path (str): Path of CNN weights.
        device (str, optional): device for torch computations.

    Example::

        pool_detector = PoolDetector("weights.pth")
        pool_coords = pool_detector.detect(img_path)
    """

    def __init__(self, weights_path, device=None):
        self.device = torch.device(device) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(weights_path)

    def load_model(self, weights_path):
        model = Model().to(self.device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        return model

    def detect(self, img_path, heatmap_thresh=170):
        """Main method for swimming pool detection.

        Args:
            img_path (str): Path of image to process.
            heatmap_thresh (int, optional): Heatmap threshold. Defaults to 170.

        Returns:
            list: List that contains swimming pool coordinates.
        """
        img = Image.open(img_path)
        img = np.array(img)

        heatmap = self.generate_heatmap(img)
        pools_dict = self.find_pools(heatmap, heatmap_thresh)
        return pools_dict

    def generate_heatmap(self, img):
        """Method to generate a heatmap from CNN class activation maps (CAMs).

        Args:
            img (np.array): Input image.
            window_size (int): Tile size to feed into the CNN.

        Returns:
            np.array: Heatmap
        """

        # convert numpy to tensor,  scale, todevice
        batch = transforms.ToTensor()(img)
        batch = torch.unsqueeze(batch, 0)
        batch = batch.to(self.device)

        # inference
        with torch.no_grad():
            cam = self.model(batch, apply_avgpool=False)  # (1, 256, 100, 200)
            cam = cam * self.model.fc.weight.view((1, -1, 1, 1)).flip(1)  # class activation map weights
            cam = cam.mean(dim=1, keepdim=True)  # (1, 1, 100, 200)
            cam = torch.nn.functional.interpolate(
                cam,
                size=(800, 1600),
                mode="bicubic",
                align_corners=True,
            )

        # Post process
        cam = cam.cpu().numpy()
        heatmap = cam[0, 0, ...]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap = 1 - heatmap
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    @staticmethod
    def find_pools(heatmap, threshold, min_contour_area=2):
        """Find swimming pool coordinates from heatmap.

        Algorithm:
        1) Blur.
        2) Binarize.
        3) Detect contours.
        4) Compute contour centers.

        Args:
            heatmap (np.array): Heatmap.
            threshold (int): Threshold keep only high activations.
            min_contour_area (int, optional): Filter contours by area. Defaults to 2.

        Returns:
            list: List that contains swimming pool coordinates.
        """
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        _, heatmap = cv2.threshold(heatmap, threshold, 255, 0)
        contours, _ = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [x for x in contours if cv2.contourArea(x) >= min_contour_area]
        pools = []
        for cnt in contours:
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pools.append((cy, cx))
        return pools
