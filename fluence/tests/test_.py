import os
import datetime as dt
# import pandas as pd
from fluence import Event
import fluence.widgets as w
from IPython.display import display
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=fluence/tests/baseline fluence/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -rP --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_Fluence_Spectra():
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    w.spacecraft_drop.value = 'STEREO-A'
    w.sensor_drop.value = 'SEPT'
    w.view_drop.value = 'asun'
    w.species_drop.value = 'electrons'
    #
    # spectral integration interval:
    startdate = dt.datetime(2021, 10, 28)
    enddate = dt.datetime(2021, 11, 2)
    #
    subtract_background = True
    background_start = dt.datetime(2021, 10, 28, 2, 0)
    background_end = dt.datetime(2021, 10, 28, 14, 0)
    #
    integration_start = dt.datetime(2021, 10, 28, 16, 0)
    integration_end = dt.datetime(2021, 11, 1)
    #
    resample = '30min'  # '60s'
    #
    # set your local path where you want to save the data files:
    data_path = f"{os.getcwd()}/data/"
    #
    print('init')
    E = Event()
    print('load')
    E.load_data(w.spacecraft_drop.value, w.sensor_drop.value, w.species_drop.value, startdate, enddate, w.view_drop.value, resample, data_path)
    #
    print('plot 1')
    fig, ax = E.plot_flux(integration_start, integration_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end, savefig=False)
    #
    print('spec')
    E.get_integrated_spec(integration_start, integration_end, subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end)
    #
    print('plot 2')
    fig, ax = E.plot_spectrum(savefig=False)
    return fig
