[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15058294.svg)](https://doi.org/10.5281/zenodo.15058293)
[![Python versions](https://img.shields.io/badge/python-3.10_--_3.13-blue)]()
[![pytest](https://github.com/soler-he/sep_tools/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/soler-he/sep_tools/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/soler-he/sep_tools/graph/badge.svg?token=YW5I35VUIC)](https://codecov.io/gh/soler-he/sep_tools)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![website](https://img.shields.io/badge/Project%20Website-blue)](https://soler-horizon.eu)

# SOLER SEP Tools

- [About](#about)
- [Installation](#installation)
    - [Update](#update)
- [Usage](#usage)
- [Access online (JupyterHub)](#access-online-jupyterhub)
- [Contributing](#contributing)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)

## About

This repository contains multiple Python tools for the analysis of Solar Energetic Particles (SEP). They are provided as Jupyter Notebooks that act as an interface to the actual software and also provide the necessary documentation. All tools can be either [installed locally](#installation) or [accessed online without installation](#access-online-jupyterhub) on [SOLER's JupyterHub](https://soler-horizon.eu/hub/)!

## Installation

1. These tools require a recent Python (>=3.10) installation. [Following SunPy's approach, we recommend installing Python via miniforge (click for instructions).](https://docs.sunpy.org/en/stable/tutorial/installation.html#installing-python)
2. [Download this file](https://github.com/soler-he/sep_tools/archive/refs/heads/main.zip) and extract to a folder of your choice (or clone the repository [https://github.com/soler-he/sep_tools](https://github.com/soler-he/sep_tools) if you know how to use `git`).
3. Open a terminal or the miniforge prompt and move to the directory where the code is.
4. Create a new virtual environment (e.g., `conda create --name sep_tools python=3.12`, or `python -m venv venv_sep_tools` if you don't use miniforge/conda).
5. Activate the just created virtual environment (e.g., `conda activate sep_tools`, or `source venv_sep_tools/bin/activate` if you don't use miniforge/conda).
6. If you **don't** have `git` installed (try executing it), install it with `conda install conda-forge::git`.
7. Install the Python dependencies from the *requirements.txt* file with `pip install -r requirements.txt`

### Update

To update the tools to the latest version, [download this file](https://github.com/soler-he/sep_tools/archive/refs/heads/main.zip) and extract it. Note that changes that you made to the Notebooks will be overwritten if you extract into the same directory as used in the initial installation; in this case we recommend you make a copy of your edited version. 

Afterwards, upgrade the required Python packages by first activating the virtual environment that you have created for these tools (i.e., execute step 5 of Installation) and then running `pip install --upgrade --upgrade-strategy eager -r requirements.txt` within the extracted folder.

The tools are continuously updated with small changes. After bigger updates, release versions are tagged and indexed at [Zenodo](https://doi.org/10.5281/zenodo.15058293). You can see the latest release version in the right sidebar of the repository (above Contributors) or get a full list at [soler-he/sep_tools/releases](https://github.com/soler-he/sep_tools/releases).

## Usage

Activate the created virtual environment in the terminal (step 5. of [Installation](#installation)), go to the folder where the tools have been extracted to, and run `jupyter-lab`. This will open the default web-browser. There, open the *.ipynb* file of the specific tool:

- `SEP_Fluence-Spectra.ipynb`<br>Has been merged with [`SEP_Spectra.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_Spectra.ipynb), please use that one instead.
- [`SEP_Multi-Instrument-Plot.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_Multi-Instrument-Plot.ipynb)<br>Makes multi-panel time-series plots of various different in-situ measurements, including also selected remote-sensing observations
- [`SEP_Multi-Spacecraft-Plot.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_Multi-Spacecraft-Plot.ipynb)<br>Makes a plot of SEP intensity-time profiles combining observations by different spacecraft
- [`SEP_PADs-and-Anisotropy.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_PADs-and-Anisotropy.ipynb)<br>Determines and visualizes SEP Pitch-Angle Distributions (PADs) and first-order anisotropies, including methods for background subtraction
- [`SEP_PyOnset.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_PyOnset.ipynb)<br>Determines SEP onset times and their uncertainties from in-situ intensity measurements using a hybrid Poisson-CUSUM-bootstrapping approach (see [PyOnset](https://github.com/Christian-Palmroos/PyOnset) for more details)
- [`SEP_Regression-Onset.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_Regression-Onset.ipynb)<br>Determines SEP onset times based on a regression method
- [`SEP_Spectra.ipynb`](https://github.com/soler-he/sep_tools/blob/main/SEP_Spectra.ipynb)<br>Determines and visualizes SEP energy spectra for peak values or accumulated over whole event duration


If you are new to Jupyter Notebooks, the official documentation will give you more info about [What is the Jupyter Notebook?](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.html) and [Running Code](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html) with it.

## Access online (JupyterHub)

All tools can be accessed online without installation on [SOLER's JupyterHub](https://soler-horizon.eu/hub/). You only need a (free) [GitHub account](https://github.com/signup) for verification. Click on the Notebook file below to open the specific tool:

- `SEP_Fluence-Spectra.ipynb`<br>Has been merged with [`SEP_Spectra.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_Spectra.ipynb), please use that one instead.
- [`SEP_Multi-Instrument-Plot.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_Multi-Instrument-Plot.ipynb)<br>Makes multi-panel time-series plots of various different in-situ measurements, including also selected remote-sensing observations
- [`SEP_Multi-Spacecraft-Plot.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_Multi-Spacecraft-Plot.ipynb)<br>Makes a plot of SEP intensity-time profiles combining observations by different spacecraft
- [`SEP_PADs-and-Anisotropy.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_PADs-and-Anisotropy.ipynb)<br>Determines and visualizes SEP Pitch-Angle Distributions (PADs) and first-order anisotropies, including methods for background subtraction
- [`SEP_PyOnset.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_PyOnset.ipynb)<br>Determines SEP onset times and their uncertainties from in-situ intensity measurements using a hybrid Poisson-CUSUM-bootstrapping approach (see [PyOnset](https://github.com/Christian-Palmroos/PyOnset) for more details)
- [`SEP_Regression-Onset.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_Regression-Onset.ipynb)<br>Determines SEP onset times based on a regression method
- [`SEP_Spectra.ipynb`](https://hub-route-serpentine-soler.2.rahtiapp.fi/hub/user-redirect/lab/workspaces/auto-8/tree/soler/sep_tools/SEP_Spectra.ipynb)<br>Determines and visualizes SEP energy spectra for peak values or accumulated over whole event duration

## Contributing

Contributions to this tool are very much welcome and encouraged! Contributions can take the form of [issues](https://github.com/soler-he/sep_tools/issues) to report bugs and request new features or [pull requests](https://github.com/soler-he/sep_tools/pulls) to submit new code.

If you don't have a GitHub account, you can [sign-up for free here](https://github.com/signup), or you can also reach out to us with feedback by sending an email to jan.gieseler@utu.fi.

## Citing

To cite these tools, please obtain the citation from the corresponding [Zenodo entry](https://doi.org/10.5281/zenodo.15058293), where you can generate different citation styles on the bottom right of the page.

## Acknowledgement

<img align="right" height="80px" src="https://github.com/user-attachments/assets/28c60e00-85b4-4cf3-a422-6f0524c42234" alt="EU flag">
<img align="right" height="80px" src="https://github.com/user-attachments/assets/5bec543a-5d80-4083-9357-f11bc4b339bd" alt="SOLER logo">

These tools are developed within the SOLER (*Energetic Solar Eruptions: Data and Analysis Tools*) project. SOLER has received funding from the European Union’s Horizon Europe programme under grant agreement No 101134999.

The tool reflects only the authors’ view and the European Commission is not responsible for any use that may be made of the information it contains.
