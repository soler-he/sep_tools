import ipywidgets as w
# import numpy as np
import pandas as pd
import sunpy
import warnings
from anisotropy.SEPevent import SEPevent
from anisotropy.solo_methods import solo_specieschannels
from anisotropy.stereo_methods import stereo_specieschannels
from anisotropy.wind_methods import wind_specieschannels
from IPython.display import display
from sunpy import log


# omit some warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', message='invalid value encountered in divide', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', message='No units provided for variable', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='astropy did not recognize units of', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
log.setLevel('WARNING')
# _ = np.seterr(divide='ignore', invalid='ignore')  # supress some numpy errors
# pd.options.mode.chained_assignment = None  # default='warn'


def select_sc_inst():
    spacecraft_instrument = w.RadioButtons(options=['Solar Orbiter EPT', 'Solar Orbiter HET', 'STEREO-A SEPT', 'STEREO-B SEPT', 'Wind 3DP'],
                                           value='Wind 3DP',
                                           layout={'width': 'max-content'},  # If the items' names are long
                                           description='Spacecraft & instrument:',
                                           disabled=False)
    display(spacecraft_instrument)
    return spacecraft_instrument


def run_SEPevent(path, spacecraft_instrument, starttime, endtime, species, bg_start=None, bg_end=None, channels=2, specieschannel=None, averaging=None, corr_window_end=None, solo_ept_ion_contamination_correction=False, plot_folder=None):
    if spacecraft_instrument in ['Solar Orbiter EPT', 'Solar Orbiter HET', 'STEREO-A SEPT', 'STEREO-B SEPT', 'Wind 3DP']:
        if spacecraft_instrument == 'Solar Orbiter EPT':
            spacecraft = 'Solar Orbiter'
            instrument = 'EPT'
        elif spacecraft_instrument == 'Solar Orbiter HET':
            spacecraft = 'Solar Orbiter'
            instrument = 'HET'
        elif spacecraft_instrument == 'STEREO-A SEPT':
            spacecraft = 'STEREO A'
            instrument = 'SEPT'
        elif spacecraft_instrument == 'STEREO-B SEPT':
            spacecraft = 'STEREO B'
            instrument = 'SEPT'
        elif spacecraft_instrument == 'Wind 3DP':
            spacecraft = 'Wind'
            instrument = '3DP'
    else:
        print(f'{spacecraft_instrument} is not a supported string for spacecraft & instrument selection!')
        raise
        return None
    # Check that the spacecraft is valid and implemented
    try:
        sc, instrument, species, channels, av_min = initial_checks(spacecraft, specieschannel, instrument, species, channels, starttime, endtime, averaging)
    except:
        raise
        return None
    print("Instrument: {}. Species: {}. Channels: {}.".format(instrument, species, channels))

    # Correction of electron measurements due to ion contamination only implemented for SOLO EPT electrons.
    if (instrument == "EPT") and (species == "e"):
        pass
    else:
        solo_ept_ion_contamination_correction = False

    event_id = f'{spacecraft}_{instrument}'  # used for the output plot filenames
    event = SEPevent(event_id, path, sc, instrument, species, channels, starttime, endtime, averaging, av_min, solo_ept_ion_contamination_correction, plot_folder=plot_folder)
    event.check_background_window(bg_start, bg_end, corr_window_end)
    event.download_and_prepare()
    if instrument != 'HET':
        event.en_channel_string_to_keV()

    return event

def print_available_channels(spacecraft_instrument):
    if spacecraft_instrument == 'Solar Orbiter EPT':
        spacecraft = 'Solar Orbiter'.replace(' ', '_')
        instrument = 'EPT'
    elif spacecraft_instrument == 'Solar Orbiter HET':
        spacecraft = 'Solar Orbiter'.replace(' ', '_')
        instrument = 'HET'
    elif spacecraft_instrument == 'STEREO-A SEPT':
        spacecraft = 'STEREO'
        instrument = 'SEPT'
    elif spacecraft_instrument == 'STEREO-B SEPT':
        spacecraft = 'STEREO'
        instrument = 'SEPT'
    elif spacecraft_instrument == 'Wind 3DP':
        spacecraft = 'Wind'
        instrument = '3DP'

    en_ch_df_p = pd.read_csv(f'anisotropy/channels_{spacecraft}_{instrument}_p.csv')
    en_ch_df_e = pd.read_csv(f'anisotropy/channels_{spacecraft}_{instrument}_e.csv')

    pd.set_option('display.max_rows', 500)
    print(spacecraft_instrument)
    print('Protons/Ions:')
    print(en_ch_df_p.to_string(index=False))
    print('')
    print('Electrons:')
    print(en_ch_df_e.to_string(index=False))
    return None

def initial_checks(spacecraft, specieschannel, instrument, species, channels, starttime, endtime, averaging):
    sc = check_spacecraft(spacecraft)
    if specieschannel is not None:
        instrument, species, channels = check_instrumentchannels(spacecraft, specieschannel)
    else:
        instrument = check_instrument(sc, instrument)
        species = check_species(species)
        channels = check_channels(channels)
    if starttime >= endtime:
        raise ValueError("Start datetime later than end datetime.")

    if averaging is not None:
        split_strings = averaging.split("min")
        if len(split_strings) == 1:
            split_strings = averaging.split("s")
            if len(split_strings) == 2:
                av_min = float(split_strings[0])/60
            else:
                raise ValueError("Invalid averaging.")
        elif len(split_strings) == 2:
            av_min = float(split_strings[0])
        else:
            raise ValueError("Invalid averaging.")
    else:
        av_min = None
    return sc, instrument, species, channels, av_min


def check_instrumentchannels(spacecraft, specieschannel):
    # Converts the SERPENTINE species+channel strings to instrument+species+channels
    # that we can use.
    # Checks whether the instrument has been implemented.
    if spacecraft == "Solar Orbiter":
        instrument, species, channels = solo_specieschannels(specieschannel)
    elif "STEREO" in spacecraft:
        instrument, species, channels = stereo_specieschannels(specieschannel)
    elif spacecraft == "Wind":
        instrument, species, channels = wind_specieschannels(specieschannel)
    else:
        raise ValueError("Spacecraft {} not implemented.".format(spacecraft))
    return instrument, species, channels


def check_channels(channels):
    # Check that the channels is a valid input assuming that channel
    # numbering starts from 0.
    # Does not check if these channels exist for the specific instrument.
    if isinstance(channels, int):
        if channels < 0:
            raise ValueError("Negative channel number: {}.".format(channels))
    elif isinstance(channels, list):
        if len(channels) > 2:
            raise ValueError("Channel list has too many elements: {}. Must have two items at most (lowest and highest channels).".format(channels))
        elif len(channels) == 0:
            raise ValueError("Channel list is empty.")
    else:
        raise TypeError("Channels should be an integer or a list.")
    return channels


def check_species(species):
    # Check that the species is valid.
    if species.lower() in ["p", "ion", "ions", "i", "protons", "proton", "h"]:
        species = "p"
    elif species.lower() in ["e", "electron", "electrons"]:
        species = "e"
    else:
        raise ValueError("Species {} not understood.".format(species))
    return species


def check_instrument(spacecraft, instrument):
    # Checks whether the instrument has been implemented.
    if spacecraft == "Solar Orbiter":
        if instrument not in ["EPT", "HET"]:
            instrument = None
    elif "STEREO" in spacecraft:
        # IMPLEMENT LET
        if instrument not in ["SEPT"]:
            instrument = None
    elif spacecraft == "Wind":
        if instrument not in ["3DP"]:
            instrument = None
    else:
        instrument = None

    if instrument is None:
        raise ValueError("Instrument {} not implemented for spacecraft {}.".format(instrument, spacecraft))
    return instrument


def check_spacecraft(spacecraft):
    # Checks whether the spacecraft has been implemented.
    if spacecraft == "Solar Orbiter":
        sc = spacecraft
    elif "STEREO" in spacecraft:
        sc = spacecraft
    elif spacecraft == "Parker Solar Probe":
        sc = None
    elif spacecraft == "BepiColombo":
        sc = None
    elif "Wind" in spacecraft:
        print("Input was {}, using Wind.".format(spacecraft))
        sc = "Wind"
    else:
        raise ValueError("Invalid spacecraft: {}.".format(spacecraft))

    if sc is None:
        raise ValueError("Spacecraft {} not implemented.".format(spacecraft))
    else:
        print("Spacecraft: {}".format(sc))
    return sc
