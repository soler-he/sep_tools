[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15058294.svg)](https://doi.org/10.5281/zenodo.15058293)
[![Python versions](https://img.shields.io/badge/python-3.10_--_3.13-blue)]()
[![pytest](https://github.com/soler-he/sep_tools/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/soler-he/sep_tools/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/soler-he/sep_tools/graph/badge.svg?token=YW5I35VUIC)](https://codecov.io/gh/soler-he/sep_tools)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

# SOLER SEP Tools

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)

## About

This repository contains multiple Python tools for the analysis of Solar Energetic Particles (SEP). They are provided as Jupyter Notebooks that act as an interface to the actual software and also provide the necessary documentation.

## Installation

1. These tools require a recent Python (>=3.10) installation. [Following SunPy's approach, we recommend installing Python via miniforge (click for instructions).](https://docs.sunpy.org/en/stable/tutorial/installation.html#installing-python)
2. [Download this file](https://github.com/soler-he/sep_tools/archive/refs/heads/main.zip) and extract to a folder of your choice (or clone the repository [https://github.com/soler-he/sep_tools](https://github.com/soler-he/sep_tools) if you know how to use `git`).
3. Open a terminal or the miniforge prompt and move to the directory where the code is.
4. Create a new virtual environment (e.g., `conda create --name sep_tools python=3.12`, or `python -m venv venv_sep_tools` if you don't use miniforge/conda).
5. Activate the just created virtual environment (e.g., `conda activate sep_tools`, or `source venv_sep_tools/bin/activate` if you don't use miniforge/conda).
6. Install the Python dependencies from the *requirements.txt* file with `pip install -r requirements.txt`

## Usage

Activate the created virtual environment in the terminal (step 5. of [Installation](#installation)), go to the folder where the tools have been extracted to, and run `jupyter-lab`. This will open the default web-browser. There, open the *.ipynb* file of the specific tool:

- `SEP_Fluence-Spectra.ipynb`<br>Determines and visualizes SEP energy spectra accumulated over a whole event duration
- `SEP_Multi-Instrument-Plot.ipynb`<br>Makes multi-panel time-series plots of various different in-situ measurements, including also selected remote-sensing observations
- `SEP_Multi-Spacecraft-Plot.ipynb`<br>Makes a plot of SEP intensity-time profiles combining observations by different spacecraft
- `SEP_PADs-and-Anisotropy.ipynb`<br>Determines and visualizes SEP Pitch-Angle Distributions (PADs) and first-order anisotropies, including methods for background subtraction
- `SEP_Regression-Onset.ipynb`<br>Determines SEP onset times based on a regression method

If you are new to Jupyter Notebooks, the official documentation will give you more info about [What is the Jupyter Notebook?](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.html) and [Running Code](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html) with it.

## Contributing

Contributions to this tool are very much welcome and encouraged! Contributions can take the form of [issues](https://github.com/soler-he/sep_tools/issues) to report bugs and request new features or [pull requests](https://github.com/soler-he/sep_tools/pulls) to submit new code.

If you don't have a GitHub account, you can [sign-up for free here](https://github.com/signup), or you can also reach out to us with feedback by sending an email to jan.gieseler@utu.fi.

## Acknowledgement

<img align="right" height="80px" src="https://github.com/user-attachments/assets/28c60e00-85b4-4cf3-a422-6f0524c42234" alt="EU flag">
<img align="right" height="80px" src="https://github.com/user-attachments/assets/5bec543a-5d80-4083-9357-f11bc4b339bd" alt="SOLER logo">

These tools are developed within the SOLER (*Energetic Solar Eruptions: Data and Analysis Tools*) project. SOLER has received funding from the European Union’s Horizon Europe programme under grant agreement No 101134999.

The tool reflects only the authors’ view and the European Commission is not responsible for any use that may be made of the information it contains.
