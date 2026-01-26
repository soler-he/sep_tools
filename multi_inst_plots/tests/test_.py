import os
# from IPython.display import display
import multi_inst_plots as m
from seppy.util import jupyterhub_data_path
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=multi_inst_plots/tests/baseline multi_inst_plots/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Multi_Instrument_Plot_PSP(monkeypatch):
    m.options.path = f"{os.getcwd()}{os.sep}data"
    m.options.path = jupyterhub_data_path(m.options.path)
    m.options.spacecraft.value = 'Parker Solar Probe'
    # deactivate STIX for now as it crashes on GitHub
    m.options.stix.value = False
    # manually select GOES satellite 16 bc. automatic detection sometimes give different results (17 not always shown)
    m.options.goes_man_select.value = True
    monkeypatch.setattr('builtins.input', lambda _: "16")
    data, metadata = m.load_data()
    m.energy_channel_selection()
    m.range_selection()
    m.options.plot_start = None
    m.options.plot_end = None
    fig, axs = m.make_plot()
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Multi_Instrument_Plot_SolO(monkeypatch):
    m.options.path = f"{os.getcwd()}{os.sep}data"
    m.options.spacecraft.value = 'Solar Orbiter'
    # deactivate STIX for now as it crashes on GitHub
    m.options.stix.value = False
    # manually select GOES satellite 16 bc. automatic detection sometimes give different results (17 not always shown)
    m.options.goes_man_select.value = True
    monkeypatch.setattr('builtins.input', lambda _: "16")
    data, metadata = m.load_data()
    m.energy_channel_selection()
    m.range_selection()
    m.options.plot_start = None
    m.options.plot_end = None
    fig, axs = m.make_plot()
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Multi_Instrument_Plot_STEREO(monkeypatch):
    m.options.path = f"{os.getcwd()}{os.sep}data"
    m.options.spacecraft.value = 'STEREO'
    m.options.stix.value = False
    # manually select GOES satellite 16 bc. automatic detection sometimes give different results (17 not always shown)
    m.options.goes_man_select.value = True
    monkeypatch.setattr('builtins.input', lambda _: "16")
    data, metadata = m.load_data()
    m.energy_channel_selection()
    m.range_selection()
    m.options.plot_start = None
    m.options.plot_end = None
    fig, axs = m.make_plot()
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
def test_SEP_Multi_Instrument_Plot_L1(monkeypatch):
    m.options.path = f"{os.getcwd()}{os.sep}data"
    m.options.spacecraft.value = 'L1 (Wind/SOHO)'
    m.options.stix.value = False
    # manually select GOES satellite 16 bc. automatic detection sometimes give different results (17 not always shown)
    m.options.goes_man_select.value = True
    monkeypatch.setattr('builtins.input', lambda _: "16")
    data, metadata = m.load_data()
    m.energy_channel_selection()
    m.range_selection()
    m.options.plot_start = None
    m.options.plot_end = None
    fig, axs = m.make_plot()
    return fig
