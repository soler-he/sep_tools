# import numpy as np
from anisotropy.SEPevent import SEPevent
from anisotropy.solo_methods import solo_specieschannels
from anisotropy.stereo_methods import stereo_specieschannels
from anisotropy.wind_methods import wind_specieschannels


def run_SEPevent(event_id, path, plot_folder, spacecraft, starttime, endtime, bg_start=None, bg_end=None, instrument=None, species=None, channels=None, specieschannel=None, averaging=None, corr_window_end=None, solo_ept_ion_contamination_correction=False):
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

    event = SEPevent(event_id, path, plot_folder, sc, instrument, species, channels, starttime, endtime, averaging, av_min, solo_ept_ion_contamination_correction)
    event.check_background_window(bg_start, bg_end, corr_window_end)
    event.download_and_prepare()
    if instrument != 'HET':
        event.en_channel_string_to_keV()

    return event


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
