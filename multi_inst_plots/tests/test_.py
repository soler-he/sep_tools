import os
from IPython.display import display
from multi_inst_plots import multi_inst_plot as m
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=multi_inst_plots/tests/baseline multi_inst_plots/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -rP --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_Multi_Instrument_Plot_PSP():
    m.options.path = f"{os.getcwd()}{os.sep}data"
    display(m.plot_range(m.options.startdate.value, m.options.enddate.value))
    m.options.spacecraft.value = 'PSP'
    # deactivate STIX for now as it crashes on GitHub
    m.options.stix.value = False
    m.load_data()
    fig, axs = m.make_plot()
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_Multi_Instrument_Plot_SolO():
    m.options.path = f"{os.getcwd()}{os.sep}data"
    display(m.plot_range(m.options.startdate.value, m.options.enddate.value))
    m.options.spacecraft.value = 'SolO'
    # deactivate STIX for now as it crashes on GitHub
    m.options.stix.value = False
    m.load_data()
    fig, axs = m.make_plot()
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_Multi_Instrument_Plot_STEREO():
    m.options.path = f"{os.getcwd()}{os.sep}data"
    display(m.plot_range(m.options.startdate.value, m.options.enddate.value))
    m.options.spacecraft.value = 'STEREO'
    m.load_data()
    fig, axs = m.make_plot()
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_Multi_Instrument_Plot_L1():
    m.options.path = f"{os.getcwd()}{os.sep}data"
    display(m.plot_range(m.options.startdate.value, m.options.enddate.value))
    m.options.spacecraft.value = 'L1 (Wind/SOHO)'
    m.load_data()
    fig, axs = m.make_plot()
    return fig
