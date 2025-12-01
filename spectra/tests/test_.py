import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest
from IPython.display import display
from seppy.util import jupyterhub_data_path

import spectra.widgets as w
from spectra import Event

# ignore divide by zero warnings in numpy
np.seterr(divide='ignore', invalid='ignore')

"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=spectra/tests/baseline spectra/tests/test_.py

To run these specific tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html spectra/tests/test_.py

To run alls tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.parametrize("spectral_type, species", [('integral', 'protons'), ('peak', 'protons')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_PSP_ISOIS_EPIHI(spectral_type, species):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'PSP'
    w.sensor_drop.value = 'EPIHI-HET'
    w.view_drop.value = 'A'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    return fig


@pytest.mark.parametrize("spectral_type, species", [('peak', 'protons')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_SOHO_ERNE_HED(spectral_type, species):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'SOHO'
    w.sensor_drop.value = 'ERNE-HED'
    # w.view_drop.value = 'asun'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_SOHO_ERNE_HED_None():
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'SOHO'
    w.sensor_drop.value = 'ERNE-HED'
    # w.view_drop.value = 'asun'
    w.species_drop.value = 'protons'
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end)
    #
    fig, ax = E.plot_spectrum(savefig=False, resample=resample)
    return fig


@pytest.mark.parametrize("spectral_type, species", [('integral', 'electrons'), ('peak', 'ions')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:solo_epd_loader")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_Solar_Orbiter_EPT(spectral_type, species):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'Solar Orbiter'
    w.sensor_drop.value = 'EPT'
    w.view_drop.value = 'asun'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    return fig


@pytest.mark.parametrize("spectral_type, species", [('integral', 'electrons'), ('peak', 'protons')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:solo_epd_loader")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_Solar_Orbiter_HET(spectral_type, species):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'Solar Orbiter'
    w.sensor_drop.value = 'HET'
    w.view_drop.value = 'asun'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    return fig


@pytest.mark.parametrize("spectral_type, species", [('integral', 'electrons'), ('peak', 'ions')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_STEREO_A_SEPT(spectral_type, species):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'STEREO-A'
    w.sensor_drop.value = 'SEPT'
    w.view_drop.value = 'asun'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    #
    # Temporal evolution of the spectra
    interval_start = spec_start
    interval_end = spec_end
    duration = pd.Timedelta(hours=1)
    #
    E.get_spec_slices(interval_start, interval_end, duration, subtract_background=subtract_background, background_start=background_start, background_end=background_end)
    assert os.path.isfile(E.gif_filename)
    return fig


@pytest.mark.parametrize("spectral_type, species", [('integral', 'electrons'), ('peak', 'protons')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_STEREO_A_HET(spectral_type, species):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'STEREO-A'
    w.sensor_drop.value = 'HET'
    # w.view_drop.value = 'asun'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spec_start = dt.datetime(2021, 10, 28, 16, 0)
    spec_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spec_start, spec_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    return fig


@pytest.mark.parametrize("spectral_type, species, viewing", [('integral', 'electrons', 'omnidirectional'), ('peak', 'protons', 'sector 7')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_SEP_Spectra_Wind_3DP(spectral_type, species, viewing):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'Wind'
    w.sensor_drop.value = '3DP'
    w.view_drop.value = viewing  # 'omnidirectional'
    w.species_drop.value = species
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    spectral_type_start = dt.datetime(2021, 10, 28, 16, 0)
    spectral_type_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E = Event()
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, data_path)
    #
    fig, ax = E.plot_flux(spectral_type_start, spectral_type_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          savefig=False, spec_type=spectral_type, resample=resample)
    #
    E.get_spec(spectral_type_start, spectral_type_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end,
                          spec_type=spectral_type, resample=resample)
    #
    fig, ax = E.plot_spectrum(savefig=False)
    return fig
