
"""
A file that contains functions for the 'external' usage, e.g., functionality ran in a notebook.

"""

__author__ = "Christian Palmroos"


import pandas as pd

def export_seppy_data(event, viewing=None, species=None) -> pd.DataFrame:
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
    if event.spacecraft=="psp":
        if species=='e':
            return event.df_e
        else:
            return event.df_i

    # Solar Orbiter data files have a MultiIndex-structure:
    if event.spacecraft=="solo":
        if species=='e':
            return event.current_df_e.copy(deep=True)["Electron_Flux"]
        else:
            if event.sensor=="ept":
                return event.current_df_i.copy(deep=True)["Ion_Flux"]
            if event.sensor=="het":
                return event.current_df_i.copy(deep=True)["H_Flux"]

    # Now there are a maximum of two dataframes, one for positive and one 
    # for negative particle charges
    if species=='e':
        return event.current_df_e.copy(deep=True)
    else:
        return event.current_df_i.copy(deep=True)


def save_figure(results:dict, name:str, facecolor="white", transparent=False) -> None:
    """
    Saves a figure to local directory with name.
    """

    FIGURE_KEY = "fig"

    try:
        _ = results[FIGURE_KEY]
    except KeyError:
        print("There is no figure in results to save!")
        return None

    results[FIGURE_KEY].savefig(name, facecolor=facecolor, transparent=transparent, bbox_inches="tight")
