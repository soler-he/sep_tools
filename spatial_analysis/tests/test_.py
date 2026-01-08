import os
import pytest
import datetime as dt
import matplotlib.pyplot as plt
from spatial_analysis import SpatialEvent as sp_event
from seppy.util import jupyterhub_data_path


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=spatial_analysis/tests/baseline spatial_analysis/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html --durations=0 spatial_analysis/tests/test_.py
"""


@pytest.mark.parametrize("obs_vsw, resampling, process_background_subtraction, process_intercalibration, process_radial_scaling",
                         [(400, '15min', True, True, True),
                          # (None, None, False, False, False)
                          ])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
# @pytest.mark.filterwarnings("ignore::UserWarning:seppy")
# @pytest.mark.filterwarnings("ignore::UserWarning:solo_epd_loader")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Spatial_Distribution(obs_vsw, resampling, process_background_subtraction, process_intercalibration, process_radial_scaling):
    # Event date and location
    startdate = dt.datetime(2021, 5, 28, 22, 19)
    enddate = startdate + dt.timedelta(days=1, hours=12)

    # Eruption location on the Sun in Stonyhurst coords
    source_location = [67, 18]  # [Longitude, Latitude]

    # Set your local path where you want to save the raw data files.
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)

    # Set your folder where you want to save the output files and figures.
    out_path = f"{os.getcwd()}{os.sep}output_spatial_analysis"

    # Set the solar wind speeds (n km/s) or use 'None' to find the speed for each observer.
    # obs_vsw = 400  # None

    # Initialise the object
    solar_event = sp_event(dates=[startdate, enddate],
                           filepaths=[out_path, data_path],
                           flare_loc=source_location,
                           V_sw=obs_vsw)

    # Define the energy channels for each observer
    # Don't change the spacecraft names, just the channels
    spacecraft_channels = {'PSP': [3, 4],  # spacecraft: [channels]
                           'SOHO': [0],
                           'STEREO A': [0],
                           'Solar Orbiter': [10, 12]}

    # Set the time averaging
    # resampling = '15min'

    # Load the observed intensity data
    solar_event.load_spacecraft_data(channels=spacecraft_channels, resampling=resampling)

    # Plot a quickview of the observed intensities
    solar_event.plot_intensities()

    # process_background_subtraction = True

    if process_background_subtraction:
        # User input on background
        background_window = [startdate - dt.timedelta(hours=2),
                             startdate + dt.timedelta(hours=1)]

        # User can check the window
        solar_event.plot_intensities(background_window=background_window)
    else:
        background_window = []

    # Perform the Background subtraction
    solar_event.background_subtract(background_window=background_window,
                                    perform_process=process_background_subtraction)

    # process_intercalibration = True

    # Intercalibration factors
    ic_factors = {'PSP': 1,
                  'SOHO': 0.67,
                  'STEREO A': 1,
                  'Solar Orbiter': 1}

    solar_event.intercalibrate(intercalibration_factors=ic_factors,
                               perform_process=process_intercalibration)

    # Set to False to skip this process
    # Note that other processes also occur during this step, so even if the radial scaling is skipped, it could still take a while.
    # process_radial_scaling = True

    radscaling_values = [2.14, 0.26]  # Values must be provided

    solar_event.radial_scale(radial_scaling_factors=radscaling_values,
                             perform_process=process_radial_scaling)

    solar_event.plot_peak_fits(window_length=10)

    solar_event.calc_Gaussian_fit()

    solar_event.plot_Gauss_results()
    fig = plt.gcf()

    solar_event.plot_simple_curve_at_timestep(dt.datetime(2021, 5, 29, 5, 15))

    return fig
