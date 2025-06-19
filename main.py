import glob
from functools import partial

from skopt import gp_minimize

import pandas as pd

import bioimageio.core

from stardist.models import StarDist3D

import torch
import logging

import sys

from parameters import params
from helper_functions import *

from segmentation_pipeline import segmentation_pipeline

import re


# Redirect print statements to logging
def print(*args, **kwargs):
    logging.info(" ".join(map(str, args)))
    

def main():
    '''
    Load file paths with specific patterns
    '''
    current_directory = os.getcwd()
    input_directory = os.path.join(current_directory)
    # Relative path to masks ### NOTE: IF MASKS NOT USED, JUST USE THE ORIGINAL IMAGE
    mask_directory = os.path.join(current_directory, "whole_sphere_masks")

    nuclear_ch = '*_c4*'  # Pattern for image1
    celltracker_ch = '*_c3*'  # Pattern for image2

    def extract_key(filepath):
        match = re.search(r'p1_w\w\d+', filepath)
        return match.group(0) if match else filepath  # Return match or full path as fallback

    # Get file lists and sort using the extracted key
    images_nuclear_mask = sorted(glob.glob(os.path.join(input_directory, nuclear_ch)), key=extract_key)
    images_cell_mask = sorted(glob.glob(os.path.join(input_directory, celltracker_ch)), key=extract_key)

    # Ensure masks are correctly aligned with images
    mask_files = sorted(glob.glob(os.path.join(mask_directory, '*')), key=extract_key)
    assert len(mask_files) == len(images_cell_mask), "Number of masks does not match number of images!"

    # Validate alignment
    # To do: automatic validation
    for img1, img2 in zip(images_nuclear_mask, images_cell_mask):
        print(f"Nuclear mask: {img1}, Cell mask: {img2}")

    '''
    Load segmentation model
    '''

    # Load models (assuming models are located in the current directory under 'models')
    model = StarDist3D(None, name='stardist', basedir=os.path.join(current_directory, 'models'))
    model2 = bioimageio.core.load_resource_description("10.5281/zenodo.7701632")

    '''
    Logging
    '''
    # Define log file
    log_file = os.path.join(current_directory, "optimization_log.txt")

    # Configure logging to write to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )


    '''
    Placeholder for results
    '''

    # DataFrame to track parameters and results
    results_df = pd.DataFrame(columns=[
        'tile_xy', 'tile_z', 'halo_xy', 'halo_z', 'foreground_threshold',
        'brightness_factor', 'clip_limit_image1', 'clip_limit_image2',
        'nbins_image1', 'nbins_image2', 'lower_percentile_image_1',
        'lower_percentile_image_2', 'upper_percentile_image1',
        'upper_percentile_image_2', 'scale_factor_image_1', 'scale_factor_image_2',
        'objective_value'
    ])

    '''
    Running the pipeline
    '''


    def objective_function(params):
        print("PARAMETERS FOR THIS CALL:", params)
        (tile_xy, tile_z, halo_xy, halo_z, foreground_threshold, brightness_factor, clip_limit_image1, clip_limit_image2,
        nbins_image1, nbins_image2, lower_percentile_image_1, lower_percentile_image_2, upper_percentile_image1,
        upper_percentile_image_2, scale_factor_image_1, scale_factor_image_2) = params

        return segmentation_pipeline(images_nuclear_mask, images_cell_mask, mask_files, model, model2, results_df,
                                    tile_xy, tile_z, halo_xy, halo_z, foreground_threshold, brightness_factor,
                                    clip_limit_image1, clip_limit_image2, nbins_image1, nbins_image2,
                                    lower_percentile_image_1, lower_percentile_image_2, upper_percentile_image1,
                                    upper_percentile_image_2, scale_factor_image_1, scale_factor_image_2)
    
    result_new = gp_minimize(objective_function, params, n_calls=80, random_state=3, verbose=True)
    
    # result_new = gp_minimize(objective_function, params, n_calls=80, random_state=3, verbose=True,
    #                           args=(images_nuclear_mask, images_cell_mask, mask_files, model, model2,
    #                                 results_df)
    #                           )
    
    # objective_wrapped = partial(
    #     objective_function,
    #     images_nuclear_mask,
    #     images_cell_mask,
    #     mask_files,
    #     model,
    #     model2,
    #     results_df
    #     )

    # result_new = gp_minimize(
    #     objective_wrapped,
    #     params,  # list of search space dimensions
    #     n_calls=80,
    #     random_state=3,
    #     verbose=True
    # )

    print(f"Best parameters: {result_new.x}")
    print(f"Best objective value: {result_new.fun}")
    print("Optimized result: ", result_new)

    # Save results
    save_best_result_images(result_new.x, images_nuclear_mask, images_cell_mask, mask_files, model, model2, results_df, segmentation_pipeline)
    save_optimization_results(result_new, current_directory)

if __name__ == "__main__":
    main()
