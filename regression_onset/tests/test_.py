import os
import pandas as pd
from seppy.tools import Event
import regression_onset as reg
from regression_onset import select_data
from IPython.display import display
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=regression_onset/tests/baseline regression_onset/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -rP --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
def test_SEP_Regression_Analysis():
    display(select_data.data_file)
    #
    # manual select SEPpy as input:
    # select_data.data_file.value = 'SEPpy'
    #
    # This is the path to your data directory
    path = f"{os.getcwd()}{os.sep}data"
    #
    # The name of your data file, if you're loading in your own data.
    filename = "solo_ept_sun_e.csv"
    #
    # To download (or load if files are locally present) SEPpy data, one needs to provide a time span.
    # If you're not using SEPpy, this can be ignored.
    start_date = "2022-01-20"
    end_date = "2022-01-21"
    #
    if select_data._seppy_selected(select_data.data_file):
        import seppy.tools.widgets as w
        display(w.spacecraft_drop, w.sensor_drop, w.view_drop, w.species_drop)
    #
    # manual select SEPpy selection:
    # w.spacecraft_drop.value = 'STEREO-A'
    # w.sensor_drop.value = 'SEPT'
    # w.view_drop.value = 'asun'
    # w.species_drop.value = 'electrons'
    #
    if select_data._seppy_selected(select_data.data_file):
        # Initializes the SEPpy Event object
        seppy_data = Event(spacecraft=w.spacecraft_drop.value, sensor=w.sensor_drop.value, species=w.species_drop.value,
                           start_date=start_date, end_date=end_date, data_level="l2",
                           data_path=path, viewing=w.view_drop.value)
        # Exports the data to a pandas dataframe
        df = reg.externals.export_seppy_data(event=seppy_data)
    else:
        # Uses pandas to_csv() to load in a local data file:
        df = pd.read_csv(f"{path}{os.sep}{filename}", parse_dates=True, index_col=0)
    #
    display(df)
    #
    # Initializing the tool with input data
    event = reg.Reg(data=df)
    #
    # Choose the channel(s) to examine in quicklook plot
    channel = "E4"
    #
    # Leaving selection to None
    selection = None
    selection = ["2022-01-20 02:00", "2022-01-20 12:00"]
    #
    # Display a quicklook plot of the input data (df).
    # Apply the selection of data for the tool by 'selection' parameter or by clicking
    # on the plot.
    # The line magic 'ipympl' enables interactive mode
    # %matplotlib ipympl
    event.quicklook(channel=channel, resample="5 min", selection=selection)
    #
    # Title for the figure (optional)
    title = r"Solar Orbiter / EPT$^{\mathrm{sun}}$ ($0.0439 - 0.0467$) MeV electrons, 1 min data"
    #
    # The channel to consider
    # channel = "E5"
    #
    # Time-averages the data to given cadence
    resample = "1 min"
    #
    # The number of breakpoints to seek from the data selection
    num_of_breaks = 3
    #
    # Boundaries of the time axis
    xlim = ["2022-01-20 00:00", "2022-01-21 06:00"]
    #
    # Fills zero counts with a filler falue f
    fill_zeroes = True
    #
    # %matplotlib inline
    results = event.find_breakpoints(channel=channel, breaks=num_of_breaks, fill_zeroes=fill_zeroes,
                                     xlim=xlim, title=title, resample=resample)
    return results['fig']
