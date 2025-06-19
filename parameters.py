
from skopt.space import Real, Categorical, Integer

n_tiles = [2,5,5]
axis_norm = (0,1,2)

# Definition of categorical parameters
# Tiling
tile_xy_possibilities = {'256': 256}
tile_z_possibilities = {'10': 10}

halo_xy_possibilities = {'32': 32}
halo_z_possibilities = {'2': 2}

# Contrast stretching parameters

stretching_scales = {'range0': [0, 0], 'range1': [1, 1.2], 'range2': [0.5, 1.2],
                     'range3': [0.8, 1.3], 'range4': [1.2, 1.4]}

# CLAHE parameters
clahe_clip_limits = {
    'no_clahe': 0.0,  # Represents no CLAHE
    'clip_0.1': 0.1,
    'clip_0.2': 0.2,
    'clip_0.3': 0.3,
    'clip_0.4': 0.4}

clahe_nbins = {
    'nbins_128': 128,
    'nbins_444': 444,
    'nbins_256': 256,
    'nbins_512': 512}

# Brightness adjustment factor
brightness_factors = {
    'adjustment_0': 0,
    'adjustment_1': 1,
    'adjustment_10': 10,
    'adjustment_15': 15,
    'adjustment_20': 20}

# Define the search space
params = [
    # Basic parameters
    Categorical(tile_xy_possibilities.keys(), name='tile_xy'),
    Categorical(tile_z_possibilities.keys(), name='tile_z'),
    Categorical(halo_xy_possibilities.keys(), name='halo_xy'),
    Categorical(halo_z_possibilities.keys(), name='halo_z'),
    Real(0.2, 0.7, name='foreground_threshold'),

    # Brightness
    Integer(0, 1, name='brightness_factor'),

    # CLAHE
    Real(0.0, 0.5, name='clip_limit_image1'),
    Real(0.0, 0.5, name='clip_limit_image2'),
    Categorical(clahe_nbins.keys(), name='nbins_image1'),
    Categorical(clahe_nbins.keys(), name='nbins_image2'),

    # Contrast stretching
    Integer(1, 20, name='lower_percentile_image_1'),
    Integer(1, 20, name='lower_percentile_image_2'),
    Integer(90, 99, name='upper_percentile_image1'),
    Integer(90, 99, name='upper_percentile_image_2'),
    Categorical(stretching_scales.keys(), name='scale_factor_image_1'),
    Categorical(stretching_scales.keys(), name='scale_factor_image_2')
    ]