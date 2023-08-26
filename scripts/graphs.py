import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


MIN_PIXEL_COUNT = 5000
MAX_PIXEL_COUNT = 100000


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


def bhattacharyya_distance(P, Q):
    """Compute the Bhattacharyya distance between histograms P and Q."""
    # B = np.sum(np.sqrt(P * Q))
    # return -np.log(B)

    return np.sum(abs(P - Q))


def plot_histogram(results_list, colors_list, filename, func_name, num_bins=20):
    """Plot histograms for the results."""

    if func_name == "A3":
        baseline = "data/baseline-A3-1.txt"
    elif func_name == "D2":
        baseline = "data/baseline-D2-1.txt"

    with open(baseline, "r", encoding="utf-8") as f:
        baseline = [float(line.strip()) for line in f.readlines()]
        results_list.append(baseline)
        colors_list.append([99, [0, 0, 0, 255]])  # Adding black color for baselines

    baseline_hist, _ = np.histogram(baseline, bins=num_bins, density=True)

    # List to store segmentation distances
    distances = []

    for (segmentation_number, color), results in zip(colors_list, results_list):
        hist, bins = np.histogram(results, bins=num_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        color_tuple = tuple(x / 255 for x in color)
        label = (
            f"Segmentation {segmentation_number}"
            if segmentation_number != 99
            else "Baselines"
        )
        plt.plot(bin_centers, hist, label=label, color=color_tuple)

        # Compute Bhattacharyya distance if this is not the baseline
        if baseline is not None and segmentation_number != 99:
            distance = bhattacharyya_distance(hist, baseline_hist)
            distances.append((distance, segmentation_number, color))

    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title(f"{func_name} distribution for different segmentations")
    plt.legend()

    if not os.path.exists("plots"):
        os.makedirs("plots")

    plt.savefig(f"plots/{func_name}-{filename}")
    print(f"Saved plot for {func_name} function for {filename}")

    plt.close()

    return distances


def get_unique_colors(image):
    """Get unique colors from the image."""
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    return unique_colors


def create_final_image(image, top_two_colors):
    """
    Processes the image and keeps only the pixels with the top two colors.
    All other pixels are set to white.
    """
    # Make a copy of the image to modify
    final_image = np.copy(image)

    # Loop through all pixels and set those not in top two colors to white
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if not any(np.array_equal(image[i, j], color) for color in top_two_colors):
                final_image[i, j] = [255, 255, 255, 255]  # set to white color

    return final_image


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
        distances_D2 = plot_histogram(results_list_d2, colors_list_d2, filename, "D2")

        # Process for a3 function
        results_list_a3, colors_list_a3 = process_image(
            image, unique_colors, a3_function
        )
        distances_A3 = plot_histogram(results_list_a3, colors_list_a3, filename, "A3")

        # Order both distances by segmentation number
        distances_D2 = sorted(distances_D2, key=lambda x: x[1])
        distances_A3 = sorted(distances_A3, key=lambda x: x[1])

        # merge together. Assume that number and colour are the same for both
        distances = [
            ((d2[0] + a3[0]) / 2, d2[1], d2[2])
            for d2, a3 in zip(distances_D2, distances_A3)
        ]

        # find two smallest distances
        distances = sorted(distances, key=lambda x: x[0])

        top_two = distances[:2]

        print(
            f"\n* Top two segmentations for final/{filename} ||"
            f" output/{filename} || plots/A3-{filename} || plots/D2-{filename}:"
        )
        top_colors = []
        for distance, segmentation_number, color in top_two:
            print(f"Segmentation {segmentation_number} with distance {distance}")
            top_colors.append(color)
        print()

        # Create final image with only top two colors
        final_img = create_final_image(image, top_colors)

        # Save the final image to the "final" folder
        if not os.path.exists("final"):
            os.makedirs("final")
        final_image_path = os.path.join("final", filename)
        io.imsave(final_image_path, final_img)


if __name__ == "__main__":
    main()
