
"""
A file that contains functions for the 'external' usage, e.g., functionality ran in a notebook.

"""

__author__ = "Christian Palmroos"


import pandas as pd

from seppy.tools import calc_av_en_flux_ST_HET, calc_av_en_flux_SEPT, calc_av_en_flux_PSP_EPILO, \
                            calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_ERNE, calc_av_en_flux_sixs

FIGURE_KEY = "fig"


def parse_seppy_metadata(event):
    """
    
    """

    meta_keys = ("Spacecraft", "Sensor", "Viewing", "Species")
    seppy_names = [event.spacecraft, event.sensor, event.viewing, event.species]

    # Fill in a dictionary with the proper names
    meta_dict = {}
    for i, key in enumerate(meta_keys):

        name = seppy_names[i]
        if key=="Spacecraft":
            name = _proper_sc_name(name)
        if key=="Sensor":
            name = name.upper()
        if key=="Viewing":
            name = str(name)
        if key=="Species":
            name = _proper_species_name(name)

        meta_dict[key] = name

    # Gets the energy channel values as strings
    meta_df = print_energies(event=event, return_df=True)

    return meta_df, meta_dict


def generate_column_indices(columns:list, meta_index:list):
    """
    Generates a dictionary mapping the old column names to integer
    numbers; a convention used in SEPpy.

    columns : {pd.DataFrame.columns} The columns of the dataframe that contains the intensity data.
    meta_index : {pd.DataFrame.index} The index of the meta dataframe (contains energy channel strings).
    """
    new_columns = {}
    for i, idx in enumerate(meta_index):
        new_columns[columns[i]] = idx
    return new_columns


def export_seppy_data(event, viewing=None, species:str=None) -> pd.DataFrame:
    """
    The data is contained inside SEPpy Event objects. This function
    exports the dataframe from the object.

    Parameters:
    -----------
    event : {seppy.Event}

    viewing : {str|int} optional.

    species : {str} optional.

    Returns:
    ---------
    df : {pd.DataFrame}
    """

    # Viewing is necessary information relating to WHICH dataframe is
    # exported from Event.
    if viewing is None:
        viewing = event.viewing

    if species is None:
        species = event.species

    # Running choose_data() asserts the correct dataframes as
    event.choose_data(viewing=viewing)

    # PSP data files
    if event.spacecraft == "psp":
        if species == 'e':
            return event.df_e
        else:
            return event.df_i

    # Solar Orbiter data files have a MultiIndex-structure:
    elif event.spacecraft == "solo":
        if species == 'e':
            return event.current_df_e.copy(deep=True)["Electron_Flux"]
        else:
            if event.sensor == "ept":
                return event.current_df_i.copy(deep=True)["Ion_Flux"]
            if event.sensor == "het":
                return event.current_df_i.copy(deep=True)["H_Flux"]

    # Wind/3DP intensities are per eV, not per MeV as the other datasets => multiply with 1e6
    elif event.spacecraft == "wind":
        if species == 'e':
            return event.current_df_e.copy(deep=True)*1e6
        else:
            return event.current_df_i.copy(deep=True)*1e6

    # Now there are a maximum of two dataframes, one for positive and one
    # for negative particle charges
    else:
        if species == 'e':
            return event.current_df_e.copy(deep=True)
        else:
            return event.current_df_i.copy(deep=True)


def save_figure(results:dict, name:str, facecolor:str="white", transparent:bool=False) -> None:
    """
    Saves a figure to local directory with name.
    """

    try:
        _ = results[FIGURE_KEY]
    except KeyError:
        print("There is no figure in results to save!")
        return None

    results[FIGURE_KEY].savefig(name, facecolor=facecolor, transparent=transparent, bbox_inches="tight")


def combine_energy_channels(event, channels:list) -> tuple:
    """
    A function that returns the average intensity over given channels.

    Parameters:
    -----------
    event : {seppy.Event}
    channels : {list[int]}

    Returns:
    -----------
    flux_series : {pd.Series}
    en_channel_string : {str}
    """

    if channels is None:
        return None, None

    viewing = event.viewing
    event.choose_data(viewing)
    
    if (event.spacecraft[:2].lower() == 'st' and event.sensor == 'sept') \
            or (event.spacecraft.lower() == 'psp' and event.sensor.startswith('isois')) \
            or (event.spacecraft.lower() == 'solo' and event.sensor == 'ept') \
            or (event.spacecraft.lower() == 'solo' and event.sensor == 'het') \
            or (event.spacecraft.lower() == 'wind' and event.sensor == '3dp') \
            or (event.spacecraft.lower() == 'bepi'):
        event.viewing_used = viewing
        event.choose_data(viewing)
    elif (event.spacecraft[:2].lower() == 'st' and event.sensor == 'het'):
        event.viewing_used = ''
    elif (event.spacecraft.lower() == 'soho' and event.sensor == 'erne'):
        event.viewing_used = ''
    elif (event.spacecraft.lower() == 'soho' and event.sensor == 'ephin'):
        event.viewing_used = ''

    if event.spacecraft == 'solo':

        if event.sensor == 'het':

            if event.species in ['p', 'i']:

                df_flux, en_channel_string =\
                    event.calc_av_en_flux_HET(event.current_df_i,
                                                event.current_energies,
                                                channels)
            elif event.species == 'e':

                df_flux, en_channel_string =\
                    event.calc_av_en_flux_HET(event.current_df_e,
                                                event.current_energies,
                                                channels)

        elif event.sensor == 'ept':

            if event.species in ['p', 'i']:

                df_flux, en_channel_string =\
                    event.calc_av_en_flux_EPT(event.current_df_i,
                                                event.current_energies,
                                                channels)
            elif event.species == 'e':

                df_flux, en_channel_string =\
                    event.calc_av_en_flux_EPT(event.current_df_e,
                                                event.current_energies,
                                                channels)

        else:
            invalid_sensor_msg = "Invalid sensor!"
            raise Exception(invalid_sensor_msg)

    if event.spacecraft[:2] == 'st':

        # Super ugly implementation, but easiest to just wrap both sept and het calculators
        # in try block. KeyError is caused by an invalid channel choice.
        try:

            if event.sensor == 'het':

                if event.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        calc_av_en_flux_ST_HET(event.current_df_i,
                                                event.current_energies['channels_dict_df_p'],
                                                channels,
                                                species='p')
                elif event.species == 'e':

                    df_flux, en_channel_string =\
                        calc_av_en_flux_ST_HET(event.current_df_e,
                                                event.current_energies['channels_dict_df_e'],
                                                channels,
                                                species='e')

            elif event.sensor == 'sept':

                if event.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        calc_av_en_flux_SEPT(event.current_df_i,
                                                event.current_i_energies,
                                                channels)
                elif event.species == 'e':

                    df_flux, en_channel_string =\
                        calc_av_en_flux_SEPT(event.current_df_e,
                                                event.current_e_energies,
                                                channels)

        except KeyError:
            raise Exception(f"{channels} is an invalid channel or a combination of channels!")

    if event.spacecraft == 'soho':

        # A KeyError here is caused by invalid channel
        try:

            if event.sensor == 'erne':

                if event.species in ['p', 'i']:

                    df_flux, en_channel_string =\
                        calc_av_en_flux_ERNE(event.current_df_i,
                                                event.current_energies['channels_dict_df_p'],
                                                channels,
                                                species='p',
                                                sensor='HED')

            if event.sensor == 'ephin':
                # convert single-element "channels" list to integer
                if type(channels) == list:
                    if len(channels) == 1:
                        channels = channels[0]
                    else:
                        print("No multi-channel support for SOHO/EPHIN included yet! Select only one single channel.")
                if event.species == 'e':
                    df_flux = event.current_df_e[f'E{channels}']
                    en_channel_string = event.current_energies[f'E{channels}']

        except KeyError:
            raise Exception(f"{channels} is an invalid channel or a combination of channels!")

    if event.spacecraft == 'wind':
        if event.sensor == '3dp':
            # convert single-element "channels" list to integer
            if type(channels) == list:
                if len(channels) == 1:
                    channels = channels[0]
                else:
                    print("No multi-channel support for Wind/3DP included yet! Select only one single channel.")
            if event.species in ['p', 'i']:
                if viewing != "omnidirectional":
                    df_flux = event.current_df_i.filter(like=f"FLUX_E{channels}_P{event.viewing[-1]}")
                else:
                    df_flux = event.current_df_i.filter(like=f'FLUX_{channels}')
                # extract pd.Series for further use:
                df_flux = df_flux[df_flux.columns[0]]
                # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                df_flux = df_flux*1e6
                en_channel_string = event.current_i_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']
            elif event.species == 'e':
                if viewing != "omnidirectional":
                    df_flux = event.current_df_e[f"FLUX_E{channels}_P{event.viewing[-1]}"]
                    #df_flux = event.current_df_e.filter(like=f'FLUX_E{channels}')
                else:
                    df_flux = event.current_df_e[f"FLUX_{channels}"]
                    #df_flux = event.current_df_e.filter(like=f'FLUX_{channels}')
                # extract pd.Series for further use:
                #df_flux = df_flux[df_flux.columns[0]]
                # change flux units from '#/cm2-ster-eV-sec' to '#/cm2-ster-MeV-sec'
                df_flux = df_flux*1e6
                en_channel_string = event.current_e_energies['channels_dict_df']['Bins_Text'][f'ENERGY_{channels}']

    if event.spacecraft.lower() == 'bepi':
        if type(channels) == list:
            if len(channels) == 1:
                # convert single-element "channels" list to integer
                channels = channels[0]
                if event.species == 'e':
                    df_flux = event.current_df_e[f'E{channels}']
                    en_channel_string = event.current_energies['Energy_Bin_str'][f'E{channels}']
                if event.species in ['p', 'i']:
                    df_flux = event.current_df_i[f'P{channels}']
                    en_channel_string = event.current_energies['Energy_Bin_str'][f'P{channels}']
            else:
                if event.species == 'e':
                    df_flux, en_channel_string = calc_av_en_flux_sixs(event.current_df_e, channels, event.species)
                if event.species in ['p', 'i']:
                    df_flux, en_channel_string = calc_av_en_flux_sixs(event.current_df_i, channels, event.species)

    if event.spacecraft.lower() == 'psp':
        if event.sensor.lower() == 'isois-epihi':
            if event.species in ['p', 'i']:
                # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                df_flux, en_channel_string =\
                    calc_av_en_flux_PSP_EPIHI(df=event.current_df_i,
                                                energies=event.current_i_energies,
                                                en_channel=channels,
                                                species='p',
                                                instrument='het',
                                                viewing=viewing.upper())
            if event.species == 'e':
                # We're using here only the HET instrument of EPIHI (and not LET1 or LET2)
                df_flux, en_channel_string =\
                    calc_av_en_flux_PSP_EPIHI(df=event.current_df_e,
                                                energies=event.current_e_energies,
                                                en_channel=channels,
                                                species='e',
                                                instrument='het',
                                                viewing=viewing.upper())
        if event.sensor.lower() == 'isois-epilo':
            if event.species == 'e':
                # We're using here only the F channel of EPILO (and not E or G)
                df_flux, en_channel_string =\
                    calc_av_en_flux_PSP_EPILO(df=event.current_df_e,
                                                en_dict=event.current_e_energies,
                                                en_channel=channels,
                                                species='e',
                                                mode='pe',
                                                chan='F',
                                                viewing=viewing)

    if event.spacecraft == 'solo':
        flux_series = df_flux #[channels]
    if event.spacecraft[:2].lower() == 'st':
        flux_series = df_flux  # [channel]'
    if event.spacecraft.lower() == 'soho':
        flux_series = df_flux  # [channel]
    if event.spacecraft.lower() == 'wind':
        flux_series = df_flux  # [channel]
    if event.spacecraft.lower() == 'psp':
        flux_series = df_flux #[channels]
    if event.spacecraft.lower() == 'bepi':
        flux_series = df_flux  # [channel]


    # Before returning, make sure that the type is pandas series, and not a 1-dimensional dataframe
    if not isinstance(flux_series, pd.core.series.Series):
        flux_series = flux_series.squeeze()

    return flux_series, en_channel_string

def print_energies(event, return_df=False):
    """
    Prints out the channel name / energy range pairs.

    THIS IS A COPYPASTE OF A METHOD WITH IDENTICAL NAME IN SEPPY.
    ACTS AS A PLACEHOLDER FUNCTION UNTIL SEPPY HAS BEEN PATCHED.

    Parameter:
    ---------
    return_df : {bool} default False. If True, returns the df instead of displaying it.
    """

    import numpy as np
    from IPython.display import display

    # This has to be run first, otherwise event.current_df does not exist
    # Note that PSP will by default have its viewing=='all', which does not yield proper dataframes
    if event.viewing != 'all':
        if event.spacecraft == 'solo' and not event.viewing:
            raise Warning("For this operation the instrument's 'viewing' direction must be defined in the call of 'Event'! Please define and re-run.")
            return
        else:
            event.choose_data(event.viewing)
    else:
        if event.sensor == "isois-epihi":
            # Just choose data with either ´A´ or ´B´. I'm not sure if there's a difference
            event.choose_data('A')
        if event.sensor == "isois-epilo":
            # ...And here with ´3´ or ´7´. Again I don't know if it'll make a difference
            event.choose_data('3')

    if event.species in ['e', "electron"]:
        channel_names = event.current_df_e.columns
        SOLO_EPT_CHANNELS_AMOUNT = 34
        SOLO_HET_CHANNELS_AMOUNT = 4
    if event.species in ['p', 'i', 'H', "proton", "ion"]:
        channel_names = event.current_df_i.columns
        SOLO_EPT_CHANNELS_AMOUNT = 64
        SOLO_HET_CHANNELS_AMOUNT = 36

    # Extract only the numbers from channel names
    if event.spacecraft == "solo":

        if event.sensor == "step":
            channel_names = list(channel_names)
            channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

        if event.sensor == "ept":
            channel_names = [name[1] for name in channel_names[:SOLO_EPT_CHANNELS_AMOUNT]]
            channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

        if event.sensor == "het":
            channel_names = [name[1] for name in channel_names[:SOLO_HET_CHANNELS_AMOUNT]]
            channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

    if event.spacecraft in ["sta", "stb"] or event.sensor == "erne":
        channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

    if event.sensor == "ephin":
        channel_numbers = np.array([int(name.split('E')[-1]) for name in channel_names])

    if event.sensor in ["ephin-5", "ephin-15"]:
        channel_numbers = [5, 15]

    if event.sensor == "isois-epihi":
        channel_numbers = np.array([int(name.split('_')[-1]) for name in channel_names])

    if event.sensor == "isois-epilo":
        channel_numbers = [int(name.split('_E')[-1].split('_')[0]) for name in channel_names]

    if event.sensor == "3dp":
        channel_numbers = [int(name.split('_')[1][-1]) for name in channel_names]

    # Remove any duplicates from the numbers array, since some dataframes come with, e.g., 'ch_2' and 'err_ch_2'
    channel_numbers = np.unique(channel_numbers)
    energy_strs = event.get_channel_energy_values("str")

    # Assemble a pandas dataframe here for nicer presentation
    column_names = ("Channel", "Energy range")
    column_data = {
        column_names[0]: channel_numbers,
        column_names[1]: energy_strs}

    df = pd.DataFrame(data=column_data)

    # Set the channel number as the index of the dataframe
    df = df.set_index(column_names[0])

    # Finally display the dataframe such that ALL rows are shown
    if not return_df:
        with pd.option_context('display.max_rows', None,
                                'display.max_columns', None,
                                ):
            display(df)
    else:
        return df

def _proper_sc_name(name) -> str:
    """
    Returns the proper name instead of the seppy inside name.
    """

    if name=="psp":
        return "Parker Solar Probe"
    if name=="soho":
        return "SOHO"
    if name=="solo":
        return "Solar Orbiter"
    if name=="sta":
        return "STEREO-A"
    if name=="stb":
        return "STEREO-B"
    if name=="wind":
        return "Wind"

    # If none of the identified names:
    return "Unidentified Spacecraft"

def _proper_species_name(name) -> str:
    """
    Returns the proper particle species name instead of 'e', 'p', etc like inside seppy.
    """

    if name=='p':
        return "Protons"
    if name=='e':
        return "Electrons"
    if name in ('i', 'H'):
        return "Ions"

    # If none of the identified particle species
    return "Unidentified particle"