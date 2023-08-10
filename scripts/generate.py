import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, "..")
sys.path.append(root_dir)

from fastsam import FastSAM, FastSAMPrompt

model = FastSAM("./weights/FastSAM.pt")
DEVICE = "cpu"


def create_images():
    if not os.path.exists("output"):
        os.makedirs("output")

    for image in os.listdir("./archive"):
        # get full path
        image_path = os.path.join("./archive", image)

        everything_results = model(
            image_path,
            device=DEVICE,
            retina_masks=False,
            imgsz=640,
            conf=0.2,
            iou=0.7,
        )
        prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)

        # everything prompt
        ann = prompt_process.everything_prompt()

        # Normalize and convert to numpy
        normalized_layers = [ann[i] / ann[i].max() for i in range(ann.shape[0])]
        numpy_layers = [layer.numpy() for layer in normalized_layers]

        # Assign a unique random color to each layer
        colors = np.random.rand(len(numpy_layers), 3)

        # Start with a blank image
        x, y = numpy_layers[0].shape
        final_image = np.zeros((x, y, 3))

        # Replace the final image's color where each layer has non-zero values
        for layer, color in zip(numpy_layers, colors):
            mask = layer > 0  # Non-zero pixels
            final_image[mask] = color  # Replace with the corresponding color

        # Save as png
        plt.imsave(f"./output/{image}", final_image)


if __name__ == "__main__":
    create_images()
