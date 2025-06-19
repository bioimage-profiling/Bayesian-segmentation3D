import os

import numpy as np
from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.segmentation import watershed

from tifffile import imread
from csbdeep.utils import normalize

from xarray import DataArray

import bioimageio
import bioimageio.core

import pandas as pd
import logging

from helper_functions import (adjust_brightness, apply_clahe, contrast_stretching, 
                              calculate_features, calculate_freq_outside, cell_count_penalty, pixel_information_lost, 
                              solidity_score)
from parameters import (tile_z_possibilities, tile_xy_possibilities, halo_z_possibilities, halo_xy_possibilities,
                        stretching_scales, clahe_nbins, n_tiles, axis_norm)

def segmentation_pipeline(images_first_mask, images_second_mask, mask_files,
                          model, model2,
                          results_df,
                          # Tiling
                          tile_xy: str, tile_z: str,
                          halo_xy: str, halo_z: str,
                          foreground_threshold: float,
                          # Brightness
                          brightness_factor: int,
                          # Clahe
                          clip_limit_image1: float,
                          clip_limit_image2: float,
                          nbins_image1: str,
                          nbins_image2: str,
                          # Contrast stretching
                          lower_percentile_image_1: int,
                          lower_percentile_image_2: int,
                          upper_percentile_image1: int,
                          upper_percentile_image_2: int,
                          scale_factor_image_1: str,
                          scale_factor_image_2: str,
                          # Save progress parameters
                          save_images=False, output_directory="output",
                          upldate_df=True):
    # Log the parameters used for this run
    logging.info(f"Running segmentation with params: tile_xy={tile_xy}, tile_z={tile_z}, "
                 f"halo_xy={halo_xy}, halo_z={halo_z}, foreground_threshold={foreground_threshold}, "
                 f"brightness_factor={brightness_factor}, clip_limit_image1={clip_limit_image1}, "
                 f"clip_limit_image2={clip_limit_image2}, nbins_image1={nbins_image1}, "
                 f"nbins_image2={nbins_image2}, lower_percentile_image_1={lower_percentile_image_1}, "
                 f"lower_percentile_image_2={lower_percentile_image_2}, upper_percentile_image1={upper_percentile_image1}, "
                 f"upper_percentile_image_2={upper_percentile_image_2}, scale_factor_image_1={scale_factor_image_1}, "
                 f"scale_factor_image_2={scale_factor_image_2}")

    tile_xy = tile_xy_possibilities[tile_xy]
    tile_z = tile_z_possibilities[tile_z]
    halo_xy = halo_xy_possibilities[halo_xy]
    halo_z = halo_z_possibilities[halo_z]
    lp_img_1 = lower_percentile_image_1
    lp_img_2 = lower_percentile_image_2
    up_img_1 = upper_percentile_image1
    up_img_2 = upper_percentile_image_2
    sf_img_1 = stretching_scales[scale_factor_image_1]
    sf_img_2 = stretching_scales[scale_factor_image_2]

    # Verify if same number of images per channel
    assert len(images_first_mask) == len(images_second_mask)
    freq_overall = []
    info_lost_list = []
    solidity_score_list = []
    cell_count_penalty_list = []

    # Creating all masks
    for item in tqdm(range(len(images_first_mask)), desc="Processing images"):
        ### Reading images
        # image1
        name_first_mask = os.path.basename(images_first_mask[item])
        print(f"Reading image {name_first_mask}")
        image1 = imread(images_first_mask[item])

        # image2
        name_second_mask = os.path.basename(images_second_mask[item])
        print(f"Reading image {name_second_mask}")
        image2 = imread(images_second_mask[item])
        image2_initial = image2  # Copy of the original image

        # whole-sphere mask
        mask_path = mask_files[item]
        whole_sphere_mask = imread(mask_path).astype(bool)
        print(f"Loaded whole-sphere mask: {os.path.basename(mask_path)}")

        # Ensure the mask has the same shape as the image
        if whole_sphere_mask.shape != image1.shape:
            raise ValueError(f"Shape mismatch: mask ({whole_sphere_mask.shape}) and image ({image1.shape})")

        # Apply the mask to image2
        image2[~whole_sphere_mask] = 0

        ### Processing starts
        # Brightness and merging for new image2
        image2 = adjust_brightness(image1, image2, brightness_factor)
        print("Done with brightness")
        # Apply adjusted CLAHE
        image1 = apply_clahe(image1, clip_limit=clip_limit_image1, nbins=clahe_nbins[nbins_image1])
        image2 = apply_clahe(image2, clip_limit=clip_limit_image2, nbins=clahe_nbins[nbins_image2])
        print("Done with CLAHE")
        # Contrast stretching
        image1 = contrast_stretching(image1, lp_img_1, up_img_1, scale_factor=sf_img_1)
        image2 = contrast_stretching(image2, lp_img_2, up_img_2, scale_factor=sf_img_2)
        print('Done with Contrast stretching')
        ### Making first mask
        image1 = normalize(image1, 1, 99.8, axis=axis_norm)
        print('Done with first mask: image1 normalization')
        first_mask, _ = model.predict_instances(image1, n_tiles=n_tiles, verbose=True)
        print('Done with first mask generation')
        ### Making second mask
        # Reshaping
        image2 = np.expand_dims(image2, axis=0)  # Add a channel dimension (assuming single-channel).
        image2 = np.expand_dims(image2, axis=0)  # Add a batch dimension.
        print('Done with second mask: expanding dimensions of image2')
        # Try to infer with tiles, if doesn't work return zero tensor.
        try:
            with bioimageio.core.create_prediction_pipeline(model2) as pp:
                input_ = DataArray(image2, dims=("b", "c", "z", "y", "x"))
                tiling = {"tile": {"x": tile_xy, "y": tile_xy, "z": tile_z},
                          "halo": {"x": halo_xy, "y": halo_xy, "z": halo_z}}
                print("Processing second masks")
                second_mask = bioimageio.core.predict_with_tiling(pp, input_, tiling)[0].squeeze().values
                # Log shape information
                print(f"Shape of input (image2): {image2.shape}")
                print(f"Shape of second_mask: {second_mask.shape}")
        except Exception as e:
            # Log tile and halo values
            print("Error for the following tile values")
            print(f"Tile XY: {tile_xy}, Tile Z: {tile_z}")
            print(f"Halo XY: {halo_xy}, Halo Z: {halo_z}")
            print(f"Error: {e}")
            # Create a zero-filled second mask
            second_mask = np.zeros((2, 45, 2160, 2160), dtype=image2.dtype)
            print(f"Returning zero-filled second mask with shape: {second_mask.shape}")
        image2 = image2.reshape(45, 2160, 2160)  # removing batch and channel dimensions
        ### Watershed mask
        foreground = second_mask[0]
        boundaries = second_mask[1]

        ### Ensure the shapes are consistent
        print(f"Shape of first_mask: {first_mask.shape}")
        print(f"Shape of boundaries: {boundaries.shape}")
        print(f"Shape of foreground: {foreground.shape}")
        if first_mask.shape[0] != boundaries.shape[0]:
            first_mask = first_mask[0]
            print(f"Adjusted shape of first_mask: {first_mask.shape}")
        ### Final mask
        final_mask = watershed(boundaries, markers=first_mask, mask=foreground > foreground_threshold)
        ### Extract basic features
        features = calculate_features(final_mask, image2_initial)
        f_out = calculate_freq_outside(features)
        freq_overall.append(f_out)

        ### Save final mask if required
        if save_images:
            output_path = os.path.join(output_directory, f"final_mask_{item}.tif")
            imsave(output_path, final_mask.astype(np.uint16))
            print(f"Final mask saved to {output_path}")

            intermediate_dir = os.path.join(output_directory, f"intermediate_{item}")
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            imsave(os.path.join(intermediate_dir, "image1.tif"), image1.astype(image1.dtype))
            imsave(os.path.join(intermediate_dir, "image2.tif"), image2.astype(image1.dtype))
            imsave(os.path.join(intermediate_dir, "image2_initial.tif"), image2_initial.astype(image1.dtype))
            imsave(os.path.join(intermediate_dir, "first_mask.tif"), first_mask.astype(first_mask.dtype))
            #imsave(os.path.join(intermediate_dir, "second_mask.tif"), second_mask.astype(second_mask.dtype))
            imsave(os.path.join(intermediate_dir, "final_mask.tif"), final_mask.astype(first_mask.dtype))

        # Extract solidity values within cells within the proper size range
        if not features.empty:
            solidity_s = solidity_score(features)
        else:
            solidity_s = 1
        solidity_score_list.append(solidity_s)

        info_lost = pixel_information_lost(image2_initial, final_mask)
        info_lost_list.append(info_lost)
        print(f"Current % pixels outside final masks for image {item} is {info_lost}")

        # Cell count penalty
        count_penalty = cell_count_penalty(features)
        cell_count_penalty_list.append(count_penalty)

        ov0 = (f_out + info_lost + solidity_s) / 3
        of_init = ov0 + count_penalty
        print("O-value for this run is:", of_init)

    ### Calculate overall objective function
    freq_overall = np.array(freq_overall).mean()
    info_lost_list = np.array(info_lost_list).mean()
    solidity_score_list = np.array(solidity_score_list).mean()
    cell_count_penalty_list = np.array(cell_count_penalty_list).mean()
    print("OF calculation:", (freq_overall + info_lost_list + solidity_score_list) / 3 + cell_count_penalty_list)

    of = (freq_overall + info_lost_list + solidity_score_list) / 3
    objective_value = of + cell_count_penalty_list

    print("objective value for all images is:", objective_value)

    # Log the objective value
    logging.info(f"Objective value for current run: {objective_value}")

    # Append to DataFrame
    run_results = {
        'tile_xy': tile_xy, 'tile_z': tile_z, 'halo_xy': halo_xy, 'halo_z': halo_z,
        'foreground_threshold': foreground_threshold, 'brightness_factor': brightness_factor,
        'clip_limit_image1': clip_limit_image1, 'clip_limit_image2': clip_limit_image2,
        'nbins_image1': nbins_image1, 'nbins_image2': nbins_image2,
        'lower_percentile_image_1': lower_percentile_image_1,
        'lower_percentile_image_2': lower_percentile_image_2,
        'upper_percentile_image1': upper_percentile_image1,
        'upper_percentile_image_2': upper_percentile_image_2,
        'scale_factor_image_1': scale_factor_image_1,
        'scale_factor_image_2': scale_factor_image_2,
        'objective_value': objective_value
    }
    results_df = pd.concat([results_df, pd.DataFrame([run_results])], ignore_index=True)

    # Check if the output directory exists, and create it if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if upldate_df:
        # Periodically save results to CSV
        results_csv_path = os.path.join(output_directory, "optimization_results.csv")
    else:
        # Save the best results to CSV
        results_csv_path = os.path.join(output_directory, "best_optimization_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    return objective_value
