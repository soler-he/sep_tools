import os
import pytest
import datetime as dt
import pandas as pd
import numpy as np
from spectra import Event
import spectra.widgets as w
from IPython.display import Image, display
from seppy.util import jupyterhub_data_path


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=spectra/tests/baseline spectra/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html --durations=0
"""


@pytest.mark.parametrize("spacecraft, sensor, view, species, level, spectral_type, resample",
                         [('PSP', 'EPIHI-HET', 'A', 'protons', 'L2', 'peak', '2min'),
                          ('PSP', 'EPIHI-HET', 'B', 'protons', 'L2', 'integral', None),
                          ('SOHO', 'ERNE-HED', None, 'protons', 'L2', 'integral', '5min'),
                          ('SOHO', 'ERNE-HED', None, 'protons', 'L2', 'peak', '2min'),
                          ('Solar Orbiter', 'HET', 'sun', 'electrons', 'L2', 'integral', '1min'),
                          ('Solar Orbiter', 'HET', 'sun', 'protons', 'L2', 'peak', '1min'),
                          ('Solar Orbiter', 'EPT', 'south', 'electrons', 'L2', 'integral', None),
                          ('Solar Orbiter', 'EPT', 'north', 'ions', 'L2', 'peak', None),
                          ('STEREO-A', 'HET', None, 'electrons', 'L2', 'integral', '5min'),
                          ('STEREO-A', 'HET', None, 'protons', 'L2', 'peak', None),
                          ('STEREO-A', 'SEPT', 'asun', 'electrons', 'L2', 'peak', '5min'),
                          ('STEREO-A', 'SEPT', 'asun', 'protons', 'L2', 'integral', None),
                          ('Wind', '3DP', 'omnidirectional', 'electrons', 'L2', 'integral', '5min'),
                          ('Wind', '3DP', 'sector 1', 'protons', 'L2', 'peak', None)
                          ])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
# @pytest.mark.filterwarnings("ignore::UserWarning:seppy")
# @pytest.mark.filterwarnings("ignore::UserWarning:solo_epd_loader")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Multi_Spacecraft_Plot(spacecraft, sensor, view, species, level, spectral_type, resample):
    display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop, w.level_drop)

    startdate = dt.datetime(2021, 10, 28, 8)
    enddate = dt.datetime(2021, 10, 29, 20)

    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)

    # loading the data
    E = Event()
    E.load_data(spacecraft=spacecraft, instrument=sensor,
                species=species, startdate=startdate, enddate=enddate,
                viewing=view, data_level=level,
                data_path=data_path)

    # spectral_type = 'integral'  # 'integral' or 'peak'

    spec_start = dt.datetime(2021, 10, 28, 16)
    spec_end = dt.datetime(2021, 10, 29, 0)

    subtract_background = True

    background_start = dt.datetime(2021, 10, 28, 12)
    background_end = dt.datetime(2021, 10, 28, 15)

    # resample = '5min'

    save_quicklook_plot = False

    fig_ts, ax_ts = E.plot_flux(spec_start, spec_end, subtract_background=subtract_background,
                                background_start=background_start, background_end=background_end,
                                savefig=save_quicklook_plot, spec_type=spectral_type, resample=resample)

    E.get_spec(spec_start, spec_end, spec_type=spectral_type, subtract_background=subtract_background,
               background_start=background_start, background_end=background_end, resample=resample)
    fig, ax = E.plot_spectrum(savefig=True)

    # 6. Spectral temporal evolution
    duration = pd.Timedelta(hours=1)
    interval_start = spec_start
    interval_end = spec_end
    num_steps = int((interval_end-interval_start) / duration)

    for i in np.arange(1, num_steps, 1):
        time = interval_start + i * duration
        ax_ts.axvline(time, color='k')

    E.get_spec_slices(interval_start, interval_end, duration, subtract_background=subtract_background, background_start=background_start, background_end=background_end)
    gif_path = E.gif_filename
    _ = Image(filename=gif_path)

    return fig
