import datetime as dt
import os
import pandas as pd
from seppy.tools import Event
import regression_onset as reg
from regression_onset import select_data
from seppy.util import jupyterhub_data_path
from IPython.display import display
import pytest


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=regression_onset/tests/baseline regression_onset/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html
"""


@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
def test_SEP_Regression_Analysis():
    display(select_data.data_file)
    #
    # manual select SEPpy as input:
    # select_data.data_file.value = 'SEPpy'
    #
    # This is the path to your data directory
    data_path = f"{os.getcwd()}{os.sep}data"
    # The name of your data file, if you're loading in your own data.
    filename = "solo_ept_sun_e.csv"
    # To download (or load if files are locally present) SEPpy data, one needs to provide a time span.
    # If you're not using SEPpy, this can be ignored.
    # The format is (year, month, day)
    start_date = dt.datetime(2022, 1, 20)
    end_date = dt.datetime(2022, 1, 21)
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
        data_path = jupyterhub_data_path(data_path)
        # Initializes the SEPpy Event object
        seppy_data = Event(spacecraft=w.spacecraft_drop.value, sensor=w.sensor_drop.value, species=w.species_drop.value,
                           start_date=start_date, end_date=end_date, data_level="l2",
                           data_path=data_path, viewing=w.view_drop.value)
        # Exports the data to a pandas dataframe
        df = reg.externals.export_seppy_data(event=seppy_data)
        meta_df, meta_dict = reg.externals.parse_seppy_metadata(event=seppy_data)
    else:
        # Uses pandas to_csv() to load in a local data file:
        df = pd.read_csv(f"{data_path}{os.sep}{filename}", parse_dates=True, index_col=0)
        meta_df, meta_dict = None, None
    #
    display(df)
    #
    # Initializing the tool with input data
    event = reg.Reg(data=df, data_source=select_data.data_file.value, meta_df=meta_df, meta_dict=meta_dict)
    # Choose the channel (column name, see display(df) above)
    channel = "E4"
    selection = ["2022-01-20 02:00", "2022-01-20 12:00"]
    # selection = None
    # Display a quicklook plot of the input data (df).
    # Apply the selection of data for the tool by 'selection' parameter or by clicking
    # on the plot.
    # The line magic 'ipympl' enables interactive mode
    # %matplotlib ipympl
    event.quicklook(channel=channel, resample="5min", selection=selection)
    #
    # The number of breakpoints to seek from the data selection
    num_of_breaks = 3
    #
    # Fills zero counts with a filler falue f
    fill_zeroes = True
    #
    #
    # Time-averages the data to given cadence
    resample = "1min"
    #
    # Boundaries of the time axis
    xlim = ["2022-01-20 00:00", "2022-01-21 06:00"]
    #
    # Title for the figure (optional)
    if select_data._seppy_selected(select_data.data_file):
        title = r"Solar Orbiter / EPT$^{\mathrm{sun}}$ ($0.0439 - 0.0467$) MeV electrons, 1 min data"
    else:
        title = ''  # set own title for User defined file
    #
    # %matplotlib inline
    results = event.find_breakpoints(channel=channel, breaks=num_of_breaks, fill_zeroes=fill_zeroes, xlim=xlim, title=title, resample=resample)
    return results['fig']
