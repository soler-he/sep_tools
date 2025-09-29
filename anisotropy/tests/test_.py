from anisotropy import run_SEPevent, select_sc_inst
from seppy.util import jupyterhub_data_path
import os
import pandas as pd
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=anisotropy/tests/baseline anisotropy/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -rP --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_PADs_and_Anisotropy_Wind_3DP_full():
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    spacecraft_instrument = select_sc_inst()
    spacecraft_instrument.value = 'Wind 3DP'
    #
    species = "e"
    channels = 3  # no channel averaging implemented for 3DP!
    averaging = "2min"  # data averaging
    #
    start_time = pd.to_datetime('2021-10-31 16:00:00')
    end_time = pd.to_datetime('2021-11-01 18:00:00')
    event = run_SEPevent(data_path, spacecraft_instrument.value, start_time, end_time, species=species, channels=channels, averaging=averaging)
    fig, axes = event.overview_plot()

    # chose a background window. Setting these to None will set default window [start_time, start_time + 5 hours]
    bg_start = pd.to_datetime('2021-10-31 23:00:00')  # None
    bg_end = pd.to_datetime('2021-11-01 01:30:00')  # None
    #
    # chose an end time for the background subtraction:
    # if None then background subraction stops 3 hours after the end of the background window (bg_end)
    corr_window_end = None  # pd.to_datetime('2021-11-01 18:00:00')
    #
    # resets background window and event.corr_window_end which is used to check that the background is not above the observations near the event start
    event.set_background_window(bg_start, bg_end, corr_window_end)
    #
    # averaging (in minutes) used for the background analysis
    bg_av_min = 10
    #
    event.background_analysis_all(minutes=bg_av_min)
    fig, axes = event.overview_plot_bgsub()
    #
    ani_method = 'weighted_sum'  # 'weighted_sum', 'weighted_sum_bootstrap', or 'fit'; 'weighted_sum_bootstrap' not available for Wind
    #
    event.calculate_anisotropy(ani_method=ani_method)
    fig, axes = event.anisotropy_plot(ani_method=ani_method)

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_PADs_and_Anisotropy_Wind_3DP_full_fit():
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    spacecraft_instrument = select_sc_inst()
    spacecraft_instrument.value = 'Wind 3DP'
    #
    species = "e"
    channels = 3  # no channel averaging implemented for 3DP!
    averaging = "2min"  # data averaging
    #
    start_time = pd.to_datetime('2021-10-31 16:00:00')
    end_time = pd.to_datetime('2021-11-01 18:00:00')
    event = run_SEPevent(data_path, spacecraft_instrument.value, start_time, end_time, species=species, channels=channels, averaging=averaging)
    fig, axes = event.overview_plot()

    # chose a background window. Setting these to None will set default window [start_time, start_time + 5 hours]
    bg_start = pd.to_datetime('2021-10-31 23:00:00')  # None
    bg_end = pd.to_datetime('2021-11-01 01:30:00')  # None
    #
    # chose an end time for the background subtraction:
    # if None then background subraction stops 3 hours after the end of the background window (bg_end)
    corr_window_end = None  # pd.to_datetime('2021-11-01 18:00:00')
    #
    # resets background window and event.corr_window_end which is used to check that the background is not above the observations near the event start
    event.set_background_window(bg_start, bg_end, corr_window_end)
    #
    # averaging (in minutes) used for the background analysis
    bg_av_min = 10
    #
    event.background_analysis_all(minutes=bg_av_min)
    fig, axes = event.overview_plot_bgsub()
    #
    ani_method = 'fit'  # 'weighted_sum', 'weighted_sum_bootstrap', or 'fit'; 'weighted_sum_bootstrap' not available for Wind
    #
    event.calculate_anisotropy(ani_method=ani_method)
    fig, axes = event.anisotropy_plot(ani_method=ani_method)

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_PADs_and_Anisotropy_Solar_Orbiter_EPT_loading_only():
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    spacecraft_instrument = select_sc_inst()
    spacecraft_instrument.value = 'Solar Orbiter EPT'
    #
    species = "e"
    channels = [3, 6]  # no channel averaging implemented for 3DP!
    averaging = "2min"  # data averaging
    #
    start_time = pd.to_datetime('2021-10-31 16:00:00')
    end_time = pd.to_datetime('2021-11-01 18:00:00')
    event = run_SEPevent(data_path, spacecraft_instrument.value, start_time, end_time, species=species, channels=channels, averaging=averaging)
    fig, axes = event.overview_plot()

    # # chose a background window. Setting these to None will set default window [start_time, start_time + 5 hours]
    # bg_start = pd.to_datetime('2021-10-31 23:00:00')  # None
    # bg_end = pd.to_datetime('2021-11-01 01:30:00')  # None
    # #
    # # chose an end time for the background subtraction:
    # # if None then background subraction stops 3 hours after the end of the background window (bg_end)
    # corr_window_end = None  # pd.to_datetime('2021-11-01 18:00:00')
    # #
    # # resets background window and event.corr_window_end which is used to check that the background is not above the observations near the event start
    # event.set_background_window(bg_start, bg_end, corr_window_end)
    # #
    # # averaging (in minutes) used for the background analysis
    # bg_av_min = 10
    # #
    # event.background_analysis_all(minutes=bg_av_min)
    # fig, axes = event.overview_plot_bgsub()
    # #
    # ani_method = 'weighted_sum_bootstrap'  # 'weighted_sum', 'weighted_sum_bootstrap', or 'fit'; 'weighted_sum_bootstrap' not available for Wind
    # #
    # event.calculate_anisotropy(ani_method=ani_method)
    # fig, axes = event.anisotropy_plot(ani_method=ani_method)

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_PADs_and_Anisotropy_Solar_Orbiter_HET_full_weighted_sum_bootstrap():
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    spacecraft_instrument = select_sc_inst()
    spacecraft_instrument.value = 'Solar Orbiter HET'
    #
    species = "e"
    channels = [1, 3]  # no channel averaging implemented for 3DP!
    averaging = "2min"  # data averaging
    #
    start_time = pd.to_datetime('2021-10-31 16:00:00')
    end_time = pd.to_datetime('2021-11-01 18:00:00')
    event = run_SEPevent(data_path, spacecraft_instrument.value, start_time, end_time, species=species, channels=channels, averaging=averaging)
    fig, axes = event.overview_plot()

    # chose a background window. Setting these to None will set default window [start_time, start_time + 5 hours]
    bg_start = pd.to_datetime('2021-10-31 23:00:00')  # None
    bg_end = pd.to_datetime('2021-11-01 01:30:00')  # None
    #
    # chose an end time for the background subtraction:
    # if None then background subraction stops 3 hours after the end of the background window (bg_end)
    corr_window_end = None  # pd.to_datetime('2021-11-01 18:00:00')
    #
    # resets background window and event.corr_window_end which is used to check that the background is not above the observations near the event start
    event.set_background_window(bg_start, bg_end, corr_window_end)
    #
    # averaging (in minutes) used for the background analysis
    bg_av_min = 10
    #
    event.background_analysis_all(minutes=bg_av_min)
    fig, axes = event.overview_plot_bgsub()
    #
    ani_method = 'weighted_sum_bootstrap'  # 'weighted_sum', 'weighted_sum_bootstrap', or 'fit'; 'weighted_sum_bootstrap' not available for Wind
    #
    event.calculate_anisotropy(ani_method=ani_method)
    fig, axes = event.anisotropy_plot(ani_method=ani_method)

    return fig


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_PADs_and_Anisotropy_STEREO_A_SEPT_loading_only():
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    spacecraft_instrument = select_sc_inst()
    spacecraft_instrument.value = 'STEREO-A SEPT'
    #
    species = "e"
    channels = [3, 6]  # no channel averaging implemented for 3DP!
    averaging = "2min"  # data averaging
    #
    start_time = pd.to_datetime('2021-10-31 16:00:00')
    end_time = pd.to_datetime('2021-11-01 18:00:00')
    event = run_SEPevent(data_path, spacecraft_instrument.value, start_time, end_time, species=species, channels=channels, averaging=averaging)
    fig, axes = event.overview_plot()

    # # chose a background window. Setting these to None will set default window [start_time, start_time + 5 hours]
    # bg_start = pd.to_datetime('2021-10-31 23:00:00')  # None
    # bg_end = pd.to_datetime('2021-11-01 01:30:00')  # None
    # #
    # # chose an end time for the background subtraction:
    # # if None then background subraction stops 3 hours after the end of the background window (bg_end)
    # corr_window_end = None  # pd.to_datetime('2021-11-01 18:00:00')
    # #
    # # resets background window and event.corr_window_end which is used to check that the background is not above the observations near the event start
    # event.set_background_window(bg_start, bg_end, corr_window_end)
    # #
    # # averaging (in minutes) used for the background analysis
    # bg_av_min = 10
    # #
    # event.background_analysis_all(minutes=bg_av_min)
    # fig, axes = event.overview_plot_bgsub()
    # #
    # ani_method = 'weighted_sum_bootstrap'  # 'weighted_sum', 'weighted_sum_bootstrap', or 'fit'; 'weighted_sum_bootstrap' not available for Wind
    # #
    # event.calculate_anisotropy(ani_method=ani_method)
    # fig, axes = event.anisotropy_plot(ani_method=ani_method)

    return fig
