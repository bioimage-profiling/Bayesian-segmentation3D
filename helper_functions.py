import numpy as np
import inspect
import json

from skimage import exposure
from skimage.measure import label, regionprops_table

import os

import pandas as pd

def print_details(skip):
    # Get the current frame and the caller frame
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    function_name = caller_frame.f_code.co_name

    # Get function arguments
    args_info = inspect.getargvalues(caller_frame)

    # Print positional arguments and their values
    print("#############", function_name, "###############")
    print("Positional arguments:", args_info.args[skip:])
    print("Values of positional arguments:", [args_info.locals[arg] for arg in args_info.args[skip:]])


def adjust_brightness(image1, image2, factor):
    """Adjust the brightness of image2 and merge it with image1."""
    print_details(2)
        # If factor is 0 or 1, return the original image2 without adjustment or merging
    if factor in [0, 1]:
        #return image2.copy()
        image_2_adjusted = image2.copy()
    #if factor == 0:
        # If factor is 0, return the original image2 without adjustment
        #image_2_adjusted = image2.copy()  # Ensure a copy is returned to prevent any unintended modifications
    else:
        image_2_adjusted = np.clip(image2 * factor, 0, image2.max())
    if image_2_adjusted.shape != image1.shape:  # Check if both images have the same dimensions
        raise ValueError("The dimensions of the two images do not match.")
    if factor != 0 and factor != 1:
        image_2_adjusted = np.mean([image_2_adjusted, image1], axis=0).astype(image1.dtype)  # Merge the images by averaging their values
    return image_2_adjusted


# Adjusted CLAHE function
# PARAMETER TO TUNE WITH: clip_limit: 0.1 - 0.5, nbins: 128, 156, 512,
# ALSO NO CLAHE POSSIBLE! Visual inspection suggest 0.04 or lower is irrelevant
def apply_clahe(image, clip_limit=0.10, nbins=256):
    print_details(1)
    if (clip_limit < 0.04):
        return image
    else:
        image = image.astype(np.float32) / 65535
        clahe = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
        clahe = (clahe * 255).astype(np.uint8)
        return clahe


# Contrast stretching function
def linear_scale(z, z_max, start_value, end_value):
    # print_details(0)
    return start_value + (end_value - start_value) * (z / z_max)


## PARAMETER TO TUNE: percentiles= 5-99 (good one so far 10-97, 10-99),
def contrast_stretching(image, lower_percentile=10, upper_percentile=99, scale_factor=[0, 0]):
    print_details(1)
    print(image.shape)
    if scale_factor[0] == 0:
        return image
    else:
        image = image.astype(np.float32) / np.max(image)  # normalize 0-1
        z_max = image.shape[0] - 1  # Get number of z planes

        image_slices = []
        for z in range(image.shape[0]):
            slice_image = image[z]
            # Calculate scale_factor based on the z-plane index
            sf = linear_scale(z, z_max, start_value=scale_factor[0], end_value=scale_factor[1])

            # Apply contrast stretching    #PARAMETER TO TUNE TOO: percentiles= 5-99
            p_low, p_high = np.percentile(slice_image, [lower_percentile, upper_percentile])
            image_stretched = np.clip((slice_image - p_low) / (p_high - p_low), 0, 1)
            image_stretched = np.clip(image_stretched * sf, 0, 1)

            image_slices.append(image_stretched)
        image = np.stack(image_slices, axis=0)
        image = (image * 255).astype(np.uint16)  # or uint8?
        print("Contrast enhancement completed.")
        return image

       ### Objective functions ###


def calculate_features(final_mask, image):
    # properties = ['label', 'area'] # Specify the properties (features) that you want to extract from the final mask
    properties = ['label', 'area', 'extent']
    label_final_mask = label(final_mask)  # Get region properties as a table, including pixel intensity-based properties
    features = regionprops_table(label_final_mask, intensity_image=image, properties=properties)
    features = pd.DataFrame(features)
    return features

# (old)
def calculate_freq_outside(features):
    voxel_size_x, voxel_size_y, voxel_size_z = 0.298988, 0.298988, 2.5
    voxel_volume_um3 = voxel_size_x * voxel_size_y * voxel_size_z
    features['REAL_area'] = features['area'] * voxel_volume_um3

    features['within_range'] = features['REAL_area'].apply(lambda x: True if 905 <= x <= 8180 else False)

    count_within = features['within_range'].sum()
    total = len(features['within_range'])
    freq_within = count_within / total if total != 0 else 0
    freq_outside = 1 - freq_within
    print("Current frequency of objects outside range is", freq_outside)
    print("Current number of cells within range is", count_within)
    return freq_outside


# (old:)
def cell_count_penalty(features):
    voxel_size_x, voxel_size_y, voxel_size_z = 0.298988, 0.298988, 2.5
    voxel_volume_um3 = voxel_size_x * voxel_size_y * voxel_size_z
    features['REAL_area'] = features['area'] * voxel_volume_um3
    features['within_range'] = features['REAL_area'].apply(lambda x: True if 905 <= x <= 8180 else False)
    count_within = features['within_range'].sum()

    if count_within <= 0:
        penalty = 0.5  # Maximum penalty for zero cells
    elif count_within >= 150:
        penalty = 0.0  # No penalty for 150 or more cells
    else:
        penalty = 1.0 - (count_within / 150.0)  # Linear decrease in penalty

    return penalty


def pixel_information_lost(img, mask):
    """Calculates how much pixel information lies outside mask boundaries."""
    # Ensure the mask has only binary values (0 and 1)
    mask = mask > 0

    # Convert the image to float32 for consistent arithmetic operations
    img = img.astype(np.float32)

    total_info_image = np.sum(img)
    print(f"Total information in image: {total_info_image}")

    if total_info_image == 0:
        return 0
    else:
        img_copy = img.copy()
        pixel_index_with_info = mask
        img_copy[pixel_index_with_info] = 0
        info_outside_mask = np.sum(img_copy)
        print(f"Information outside mask: {info_outside_mask}")

        percentage_lost = info_outside_mask / total_info_image
        print(f"Percentage lost: {percentage_lost}")

    return percentage_lost


def solidity_score(features):
    if 'extent' in features.columns and not features.empty:
        indexes_proper_cells = features['within_range']
        if indexes_proper_cells.any():  # Check if there are any proper cells
            ss = 1 - features[indexes_proper_cells]['extent'].mean()
            print("Real solidity score", ss)
        else:
            ss = 1
    else:
        ss = 1
    return ss

# Function to save the best result images (relative)

def save_best_result_images(best_params, current_directory, images_first_mask, images_second_mask,
                            mask_files, model, model2, results_df, segmentation_pipeline):
    (tile_xy, tile_z, halo_xy, halo_z, foreground_threshold, brightness_factor, clip_limit_image1, clip_limit_image2,
     nbins_image1, nbins_image2, lower_percentile_image_1, lower_percentile_image_2, upper_percentile_image1,
     upper_percentile_image_2, scale_factor_image_1, scale_factor_image_2) = best_params

    # Define a relative path for the output directory where images will be saved
    best_experiment = 'best_result_images_OPTIMISED'
    output_directory = os.path.join(current_directory, best_experiment)

    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Run segmentation pipeline with save_images=True
    segmentation_pipeline(
        images_first_mask, images_second_mask, mask_files,
        model, model2, results_df,
        tile_xy, tile_z, halo_xy, halo_z,
        foreground_threshold, brightness_factor, clip_limit_image1, clip_limit_image2,
        nbins_image1, nbins_image2, lower_percentile_image_1, lower_percentile_image_2,
        upper_percentile_image1, upper_percentile_image_2, scale_factor_image_1,
        scale_factor_image_2, save_images=True, output_directory=output_directory, upldate_df=False)


# Function to save optimization results to a JSON file (RELATIVE)

def save_optimization_results(result, current_directory, file_name="optimisation_result.json"):
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

    result_data = {
        'params': [convert_to_serializable(param) for param in result.x],
        'objective_value': convert_to_serializable(result.fun)}

    # Define the output path relative to the current directory
    output_path = os.path.join(current_directory, file_name)

    with open(output_path, 'w') as f:
        json.dump(result_data, f)

    print(f"Optimization results saved to {output_path}")

