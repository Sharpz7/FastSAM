import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

MIN_PIXEL_COUNT = 10000
MAX_PIXEL_COUNT = 30000


def filter_region(image):
    """Count the number of pixels in the segmentation and return True if
    the count is within the specified minimum and maximum bounds."""
    pixel_count = np.sum(image)
    if MIN_PIXEL_COUNT <= pixel_count <= MAX_PIXEL_COUNT:
        return True
    return False


def a3_function(points):
    """Calculates the angle between 3 points in a point cloud."""
    n = len(points)
    indices = np.random.default_rng(42).choice(n, size=(5000, 3))

    v1 = points[indices[:, 0]]
    v2 = points[indices[:, 1]]
    v3 = points[indices[:, 2]]
    vectors1 = v1 - v2
    vectors2 = v3 - v2

    dot_products = np.sum(vectors1 * vectors2, axis=-1)
    norms1 = np.linalg.norm(vectors1, axis=-1)
    norms2 = np.linalg.norm(vectors2, axis=-1)

    with np.errstate(invalid="ignore"):
        angles = np.arccos(dot_products / (norms1 * norms2))

    angles = np.rad2deg(angles)
    angles = angles[~np.isnan(angles)]

    return angles


def d2_function(points):
    """Calculates the distances between pairs of points in a point cloud."""
    n = len(points)
    indices = np.random.default_rng(42).choice(n, size=(5000, 2))

    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]

    vectors = p1 - p2
    distances = np.linalg.norm(vectors, axis=-1)

    return distances


def process_image(image, unique_colors, func):
    """Process the image for either a3 or d2 functions based on the provided func."""
    results_list = []
    colors_list = []
    segmentation_number = 0

    for i, color in enumerate(unique_colors):
        region = np.all(image == color.astype(int), axis=-1).astype(int)

        if filter_region(region):
            points = np.column_stack(np.where(region > 0))
            results = func(points)
            results_list.append(results)
            colors_list.append((segmentation_number, color))
            segmentation_number += 1

    return results_list, colors_list


def plot_histogram(results_list, colors_list, filename, func_name, num_bins=20):
    """Plot histograms for the results."""
    for (segmentation_number, color), results in zip(colors_list, results_list):
        hist, bins = np.histogram(results, bins=num_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        color = color / 255
        plt.plot(
            bin_centers, hist, label=f"Segmentation {segmentation_number}", color=color
        )

    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title(f"{func_name} distribution for different segmentations")
    plt.legend()

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{func_name}-{filename}.png")
    print(f"Saved plot for {func_name} function for {filename}")
    plt.close()


def get_unique_colors(image):
    """Get unique colors from the image."""
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    return unique_colors


def main():
    """Main processing function."""
    for filename in os.listdir("output"):
        image_path = os.path.join("output", filename)
        image = io.imread(image_path)
        unique_colors = get_unique_colors(image)

        print(f"Found {len(unique_colors)} unique colors in {filename}")

        # Process for d2 function
        results_list_d2, colors_list_d2 = process_image(
            image, unique_colors, d2_function
        )
        plot_histogram(results_list_d2, colors_list_d2, filename, "D2")

        # Process for a3 function
        results_list_a3, colors_list_a3 = process_image(
            image, unique_colors, a3_function
        )
        plot_histogram(results_list_a3, colors_list_a3, filename, "A3")


if __name__ == "__main__":
    main()
