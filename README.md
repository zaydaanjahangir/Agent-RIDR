## Quick Start

The easiest way to run the full Agent-RIDR pipeline is to open and execute:

**`ridr_combined_pipeline_and_evals_patched (1).ipynb`**

in **Google Colab**.

This notebook is the main end-to-end entry point and includes:
- dataset loading
- Channel 1 evaluation
- Channel 2 evaluation
- joint OR / AND policy evaluation
- metric generation and analysis outputs

If you are just trying to reproduce the project results, you do **not** need to run the other files individually. The notebook already contains the combined pipeline used for evaluation.

## Recommended Usage

1. Upload or open `ridr_combined_pipeline_and_evals_patched (1).ipynb` in Google Colab
2. Add all the files from the data folder into a `/mnt/data/` folder
3. Run the notebook cells from top to bottom  
4. Review the generated metrics, plots, and evaluation tables  

## Repository Notes

Several additional files in this repository reflect intermediate development, standalone experiments, or supporting implementations for Channel 1. These are useful for reference, but they are **not** the recommended starting point for reproduction.

For most users, the correct file to run is:

**`ridr_combined_pipeline_and_evals_patched (1).ipynb`**
