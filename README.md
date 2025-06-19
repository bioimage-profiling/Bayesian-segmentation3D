# Bayesian-segmentation3D
Bayesian optimisation for 3D segmentation of single-cells

# Achieving High-Resolution Single-Cell Segmentation in Convoluted Cancer Spheroids via Bayesian Optimization and Deep Learning

This project uses Bayesian optimization and deep learning to achieve precise single-cell segmentation in complex cancer spheroids.

### Instructions

- Modify optimization parameter ranges in `parameters.py`.  
- Configure image paths and patterns in `main.py`.  
- The best optimization results are saved to `output/best_optimization_results.csv`.
- Each channel is being optimised separately (for whole-cell segmentations) along with the nuclear channel
- The 3D images to be optimised should be in .tiff format, with dimensions: `1 channel x Number of z slices x Height x Width`
- The images should be located in the same directory as the scripts
- There is a folder called 'whole_sphere_masks' which should have the masks from the whole structure/spheres (if there are such masks), if not, just add same images from the channel being optimised during that round. 


### Run

```bash
python main.py
```

### Installation

Install with `virtualenv`.

```bash
pip install -r requirements.txt
```