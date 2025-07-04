# Adversarial Disentanglement by Backpropagation with Physics-Informed Variational Autoencoder

This repository contains the code accompanying the paper: "Adversarial Disentanglement by Backpropagation with Physics-Informed Variational Autoencoder".
The pre-print version can be found here: https://arxiv.org/abs/2506.13658.

## Project Structure

- **cases/**: Contains subdirectories for each case study (e.g., `bridge`, `damped_oscillator`, `simple_beam`). Each subdirectory includes the necessary models, data files, and an `__init__.py` file that defines the case study and its presets.
- **models/**: Neural network modules and model components used by the case studies.
- **utils/**: Utility functions, including parameter defaults, data handling, loss functions, metrics, and visualization tools. The `utils/__init__.py` file contains global default parameters for all case studies.
- **figures/**, **output/**: Output directories for generated figures and results.
- **dpivae.py**: Core functions for setting up, training, and evaluating models. This module is called by the main scripts.
- **requirements.txt**: List of required Python packages.

## Requirements
The code in this repository has been tested with Python 3.11 but will likely work with other versions of Python and the packages.
A detailed list of the packages and versions used can be found in `requirements.txt`. A brief list is provided below:

- **numpy**
- **pytorch**
- **pytorch_lightning**
- **torchrl**
- **sklearn**
- **scipy**
- **pandas**
- **matplotlib**
- **seaborn**

## Case Studies
Three case studies are presented in the paper:

- **Beam**
- **Oscillator**
- **Population of bridges**

Each case study is located in its own subdirectory under `cases/`. These subdirectories contain:
- All necessary pretrained neural networks, physics-based models, and data files for running the case study.
- Simulator data used to train the Neural Networks, also needed to initialize some of the scalers used in the model.
- An `__init__.py` file that defines the case study, including different model parameter presets.

## Usage Notes

- The main entry points are the scripts in the root directory:
  - `0_single_run.py`: Run a single experiment for a selected case study
  - `1_disentanglement_metric.py`: Evaluate disentanglement metrics
  - `2_regression_comparison.py`: Compare regression performance across cases
- Run the scripts with the working directory set to the root of the folder `./DPI-VAE/`.
- Each script contains a `case` variable to select the case study.
- The `0_single_run` and `1_disentanglement_metric` scripts additionally contain a `preset` variable to specify the model preset.
- Multiple model presets can be specified per case study in its `__init__.py` file.
- The `utils/__init__.py` file contains global default parameters, which are partially overwritten by the presets defined in each case study.
- The `dpivae.py` module provides functions for model setup, training, and evaluation, which are used by the main scripts.
- Regularization methods not used in the paper (e.g. adjusting $\alpha$ and $\beta$, constraining the neural network output magnitude and gradient clipping) have not been tested.

## References
The simulators used to generate data for the case studies are as follows:

- **Beam**: We use the `beef` Python package: https://knutankv.github.io/beef/beef.html.
- **Oscillator**: Implementation based on the code accompanying the publication:  N. Takeishi and A. Kalousis (2021) "Physics-Integrated Variational Autoencoders for Robust and Interpretable Generative Modeling". In Advances in Neural Information Processing Systems 34 (NeurIPS), 2021. Available at the link: https://github.com/n-takeishi/phys-vae.
- **Population of bridges**: We use the FE model presented in: K. Tatsis and E. Chatzi (2019) - "A numerical benchmark for system identification under operational and environmental variability". 8th IOMAC - International Operational Modal Analysis Conference, Proceedings. Available at the link: https://github.com/ETH-WindMil/benchmarktu1402.