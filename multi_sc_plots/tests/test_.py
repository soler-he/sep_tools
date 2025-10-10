import os
import datetime as dt
import pytest
from multi_sc_plots import Event
from seppy.util import jupyterhub_data_path


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=multi_sc_plots/tests/baseline multi_sc_plots/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=True, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:seppy")
@pytest.mark.filterwarnings("ignore::UserWarning:solo_epd_loader")
@pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Multi_Spacecraft_Plot():
    E = Event()
    instruments = E.instrument_selection()
    #
    startdate = dt.datetime(2023, 5, 9, 12, 0)
    enddate = "2023/05/10 22:00:00"
    #
    _ = E.viewing
    #
    E.viewing['Parker Solar Probe/EPI-Hi HET'] = 'A'  # 'A'='sun', 'B'='asun'
    E.viewing['Parker Solar Probe/EPI-Lo PE'] = 3  # 3='sun', 7='asun'
    E.viewing['Parker Solar Probe/EPI-Lo IC'] = 35  # 3x='sun', 7x='asun'
    E.viewing['Solar Orbiter/EPT'] = 'sun'  # 'asun', 'sun', 'north', 'south'
    E.viewing['Solar Orbiter/HET'] = 'sun'  # 'asun', 'sun', 'north', 'south'
    E.viewing['STEREO-A/SEPT'] = 'sun'  # 'asun', 'sun', 'north', 'south'
    #
    data_path = f"{os.getcwd()}{os.sep}data"
    data_path = jupyterhub_data_path(data_path)
    #
    E.load_data(startdate, enddate, instruments, data_path=data_path)
    #
    E.print_energies()
    #
    E.channels_e['BepiColombo/SIXS e'] = 5  # 2 for 100 keV  # channel combination not supported!
    E.channels_e['Parker Solar Probe/EPI-Hi HET e'] = [3, 10]
    E.channels_e['Parker Solar Probe/EPI-Lo PE e'] = [4, 5]
    E.channels_e['SOHO/EPHIN e'] = 2  # channel combination not supported!
    E.channels_e['Solar Orbiter/EPT e'] = [6, 7]  # L2: [14, 18]
    E.channels_e['Solar Orbiter/HET e'] = [0, 1]
    E.channels_e['STEREO-A/HET e'] = [0, 1]
    E.channels_e['STEREO-A/SEPT e'] = [6, 7]
    E.channels_e['WIND/3DP e'] = 3  # channel combination not supported!
    #
    E.channels_p['BepiColombo/SIXS p'] = 8  # channel combination not supported!
    E.channels_p['Parker Solar Probe/EPI-Hi HET p'] = [8, 9]
    E.channels_p['Parker Solar Probe/EPI-Lo IC p'] = [10, 16]
    E.channels_p['SOHO/ERNE-HED p'] = [3, 4]
    E.channels_p['Solar Orbiter/EPT p'] = [20, 21]  # L2: [50, 56]
    E.channels_p['Solar Orbiter/HET p'] = [19, 24]
    E.channels_p['STEREO-A/HET p'] = [5, 8]
    E.channels_p['STEREO-A/SEPT p'] = [25, 30]
    E.channels_p['WIND/3DP p'] = 6  # channel combination not supported!
    #
    _ = E.plot_colors
    #
    E.plot_colors['Parker Solar Probe/EPI-Hi HET'] = 'blueviolet'
    #
    fig, axes = E.plot(averaging='5min', plot_range=[dt.datetime(2023, 5, 9, 18, 0), dt.datetime(2023, 5, 10, 2, 0)], dict_plot_instruments=instruments)
    #
    return fig
