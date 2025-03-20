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
4. Create a new virtual environment (e.g., `conda create --name sep_tools`, or `python -m venv venv_sep_tools` if you don't use miniforge/conda).
5. Activate the just created virtual environment (e.g., `conda activate sep_tools`, or `source venv_sep_tools/bin/activate` if you don't use miniforge/conda).
6. Install the Python dependencies from the *requirements.txt* file with `pip install -r requirements.txt`

## Usage

Activate the created virtual environment in the terminal (step 5. of [Installation](#installation)), go to the folder where the tools have been extracted to, and run `jupyter-lab`. This will open the default web-browser. There, open the *.ipynb* file of the specific tool:

- `SEP_Multi-Spacecraft-Plot.ipynb`: SEP Multi-Spacecraft-Plot tool
- `SEP_PADs-and-Anisotropy.ipynb`: SEP PADs-and-Anisotropy tool

## Contributing

Contributions to this tool are very much welcome and encouraged! Contributions can take the form of [issues](https://github.com/soler-he/sep_tools/issues) to report bugs and request new features or [pull requests](https://github.com/soler-he/sep_tools/pulls) to submit new code. 

If you don't have a GitHub account, you can [sign-up for free here](https://github.com/signup), or you can also reach out to us with feedback by sending an email to jan.gieseler@utu.fi.

## Acknowledgement

<img align="right" height="80px" src="https://github.com/user-attachments/assets/28c60e00-85b4-4cf3-a422-6f0524c42234"> 
<img align="right" height="80px" src="https://github.com/user-attachments/assets/5bec543a-5d80-4083-9357-f11bc4b339bd"> 

This tool is developed within the SOLER (*Energetic Solar Eruptions: Data and Analysis Tools*) project. SOLER has received funding from the European Union’s Horizon Europe programme under grant agreement No 101134999. 

The tool reflects only the authors’ view and the European Commission is not responsible for any use that may be made of the information it contains.

