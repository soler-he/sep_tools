
"""
Contains plotting utility functions and constants.

"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Constants:
STANDARD_QUICKLOOK_FIGSIZE = (10,6)
STANDARD_FIGSIZE = (26,11)

STANDARD_TITLE_FONTSIZE = 32
STANDARD_FONTSIZE = 26
STANDARD_AXIS_LABELSIZE = 20

STANDARD_TICK_LABELSIZE = 25

STANDARD_MAJOR_TICKLEN = 11
STANDARD_MINOR_TICKLEN = 8

STANDARD_MAJOR_TICKWIDTH = 2.8
STANDARD_MINOR_TICKWIDTH = 2.1

DEFAULT_SELECTION_ALPHA = 0.12
BREAKPOINT_SHADING_ALPHA = 0.18

LATEX_PM = r"$\pm$"

def set_standard_ticks(ax:plt.Axes, labelsize:int=None) -> None:
    """
    Handles tickmarks, their sizes etc...

    Default labelsize = 25
    """

    if labelsize is None:
        labelsize = STANDARD_TICK_LABELSIZE

    ax.tick_params(which="major", length=STANDARD_MAJOR_TICKLEN, width=STANDARD_MAJOR_TICKWIDTH, labelsize=labelsize)
    ax.tick_params(which="minor", length=STANDARD_MINOR_TICKLEN, width=STANDARD_MINOR_TICKWIDTH, labelsize=labelsize-5)


def set_ylims(ax:plt.Axes, series:pd.Series=None, ylim:list=None):
    """
    Sets the vertical axis limits of the figure, given an Axes and a tuple or list of y-values.

    Parameters:
    -----------
    ax : {plt.Axes} Axes of the figure.
    series : {pd.Series} The 10-based logarithm of intensity time series.
    ylim : {list | tuple} The limits of the y-axis.
    """

    # If there are less than 2 entries in the series, no point in adjusting ylimits.
    if len(series) < 2:
        return None

    FIG_LOW_LIM_COEFF = 0.1
    FIG_HIGH_LIM_COEFF = 0.2
    MAX_ORDERS_OF_MAGNITUDE = 10
    MIN_FIG_INTENSITY = 1e-4

    # In case not otherwise specified, set the lower limit to 0.1 orders of magnitude less than the smallest value,
    # and upper limit as 0.2 orders of magnitude higher than the largest value
    if ylim is None:
        # ylim = [np.nanmin(series) - abs(np.nanmin(series) * FIG_LOW_LIM_COEFF),
        #         np.nanmax(series) + abs(np.nanmax(series) * FIG_HIGH_LIM_COEFF)]
        ylim = [np.nanmin(series) - FIG_LOW_LIM_COEFF,
                np.nanmax(series) + FIG_HIGH_LIM_COEFF]

        # Check if there is more than 10 orders of magnitude difference between the max and min values.
        # If the lower y-boundary is some ridiculously small number, adjust the y-axis a little
        if ylim[1]-ylim[0] > MAX_ORDERS_OF_MAGNITUDE:
            ylim[0] = MIN_FIG_INTENSITY

    ylim = ax.set_ylim(ylim)


def set_xlims(ax:plt.Axes, data:pd.DataFrame, xlim:list[str]) -> None:
    """
    Sets the x-axis boundaries for the plot

    Parameters:
    -----------
    ax : {plt.Axes} The axes of the figure.
    data : {pd.DataFrame} The data being plotted.
    xlim : {list[str]} A pair of datetime strings to set the plot boundaries.
    """

    if xlim is None:
        ax.set_xlim(data.index.values[0], data.index.values[-1])
    else:
        ax.set_xlim(pd.to_datetime(xlim[0]), pd.to_datetime(xlim[1]))


def fabricate_yticks(ax:plt.Axes, series:pd.Series) -> None:
    """
    Changes the y-axis labels to their 10-base exponential counterparts.

    ax : {plt.Axes} The axis object of the figure.
    """

    AXIS_STEPSIZE = 1
    BUFFER_REL_COEFF = 1.

    # This little helper function formats labels as "x * 10^y"
    def sci_notation(val):
        exponent = int(np.floor(np.log10(val)))
        coeff = val / (10**exponent)
        #return rf"${coeff:.1f} \times 10^{{{exponent}}}$"
        return rf"$10^{{{exponent}}}$"

    # Assert tick range:
    log_min = np.nanmin(series)
    log_max = np.nanmax(series)

    # Use a 100% buffer to push the boundaries (up and down)
    buffer = BUFFER_REL_COEFF * (log_max - log_min)
    log_min -= buffer
    log_max += buffer

    # Create the ticks
    first_tick = np.ceil(log_min/AXIS_STEPSIZE) * AXIS_STEPSIZE
    last_tick = np.floor(log_max/AXIS_STEPSIZE) * AXIS_STEPSIZE
    tick_values = np.arange(start=first_tick, stop=last_tick + AXIS_STEPSIZE, step=AXIS_STEPSIZE)

    tick_exponents = 10 ** tick_values

    tick_labels = [sci_notation(tickval) for tickval in tick_exponents]

    #integer_tick_values = [int(val) for val in ax.get_yticks() if val%1==0]
    ax.set_yticks(ticks=tick_values, labels=tick_labels)
