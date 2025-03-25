
"""
Contains the first development version for an SEP event onset finding tool that utilizes 
segmented linear regression.
piecewise-regression: Pilgrim, 2021.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import ipympl

from matplotlib.dates import DateFormatter

import piecewise_regression

# Relative imports cannot be used with "import .a" form; use "from . import a" instead. -Pylance
from . import calc_utilities as calc
from .plotting_utilities import set_standard_ticks, set_xlims, fabricate_yticks, STANDARD_QUICKLOOK_FIGSIZE, \
                                STANDARD_TITLE_FONTSIZE, STANDARD_FIGSIZE, STANDARD_LEGENDSIZE, DEFAULT_SELECTION_ALPHA, \
                                BREAKPOINT_SHADING_ALPHA, LATEX_PM

from .validate import _validate_index_choice, _validate_plot_style, _validate_fit_convergence, _validate_selection

from .externals import export_seppy_data

from .select_data import data_file

DEFAULT_NUM_OF_BREAKPOINTS = 1
SECONDS_PER_DAY = 86400


class Reg:

    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.selection_max_x = pd.NaT
        self.selection_max_y = np.nan

        self.selection_min_x = pd.NaT
        self.selection_min_y = np.nan

        # To keep track of how many times self._onclick() has been run
        self.times_clicked = 0

    def set_selection_max(self, x, y=None) -> None:
        """
        Sets the parameters by which data selection maximum will be applied when running
        regression analysis.

        Parameters:
        -----------
        x : {pd.Timestamp | datetime}
        y : {float}
        """
        if isinstance(x,str):
            x = pd.to_datetime(x)
        self.selection_max_x = x
        self.selection_max_y = y


    def _set_selection_min(self, x, y) -> None:
        """
        Sets the parameters by which data selection minimum will be applied when running
        regression analysis.

        Parameters:
        -----------
        x : {pd.Timestamp | datetime}
        y : {float}
        """
        if isinstance(x,str):
            x = pd.to_datetime(x)
        self.selection_min_x = x
        self.selection_min_y = y


    def _onclick(self, event) -> None:
        """
        Store coordinates to class attributes when clicking the interactive plot.
        Also draws a vertical line marking the end of the selection criterion.
        """
        # Update counter before doing anything
        self.times_clicked += 1
        if event.xdata is not None and event.ydata is not None:
            # First convert matplotlib's xdata (days after epoch) to seconds and then to datetime
            x = pd.to_datetime(event.xdata*SECONDS_PER_DAY, unit='s')
            self.set_selection_max(x=x, y=event.ydata)
            self._draw_selection_line_marker(x=x)
        else:
            raise TypeError("Event xdata or ydata was None")


    def _draw_selection_line_marker(self, x) -> None:
        self.quicklook_ax.axvline(x=x, color="green", zorder=10)


    def quicklook(self, channel:str=None, resample:str=None, xlim:list=None, selection:list[str]|str=None) -> None:
        """
        Makes a quicklook plot of one or more channels for a given dataframe.
        Meant to be used in interactive mode, so that the user can apply data selection
        by clicking.

        Comprehensive example of ipympl: https://matplotlib.org/ipympl/examples/full-example.html

        Parameters:
        --------------
        channel : str, list
        resample : str
        xlim : list
        selection : {list[str] or str} format: %Y-%m-%d %H:%M%S.
                    If given a pair of timestamps, apply selection between them. If one timestamp, start selection
                    from the beginning of the input data and select up to the given timestamp.
        """

        QUICKLOOK_TICK_LABELSIZE = 18
        QUICKLOOK_LEGENDSIZE = STANDARD_LEGENDSIZE-5

        # Apply resampling if asked to
        if isinstance(resample,str):
            data = calc.resample_df(df=self.data, avg=resample)
        else:
            data = self.data.copy(deep=True)

        # Exclude data outside figure boundaries:
        if isinstance(xlim, tuple|list):
            data = data.loc[(data.index >= xlim[0])&(data.index <= xlim[1])]

        # Make sure that channel is a list to iterate over
        if channel is None:
            channel = list(data.columns)
        if isinstance(channel,(str,int)):
            channel = [channel]

        # Attach the fig and axes to class attributes
        self.quicklook_fig, self.quicklook_ax = plt.subplots(figsize=STANDARD_QUICKLOOK_FIGSIZE)

        # Next block is about determining the selection for the data. It will be done either by
        # a click on the interactive plot or by giving "selection" parameter as an input.
        #
        # Attach the onclick() -method to a mouse button press event for the interactive plot if
        # the selection parameter was not provided
        if selection is None:
            cid = self.quicklook_fig.canvas.mpl_connect(s="button_press_event", func=self._onclick)
        else:
            # First make sure that selection is of correct type
            _validate_selection(selection=selection)

            # The numerical index of channel is needed to access the right selection y values
            idx_of_channel = data.columns.get_indexer(target=[channel[0]])[0]

            # Check for selection; is it a single str or a pair of strs?
            if isinstance(selection, str):
                selection = [selection]
            else:
                selection_min_dt = pd.to_datetime(selection[0])
                closest_min_dt_index = data.index.get_indexer(target=[selection_min_dt], method="nearest")[0]
                self._set_selection_min(x=selection_min_dt,
                                        y=data.iat[closest_min_dt_index, idx_of_channel])
                self._draw_selection_line_marker(x=selection_min_dt)

            # This is ran regardless of wether selection was str or list
            selection_max_dt = pd.to_datetime(selection[-1])
            closest_max_dt_index = data.index.get_indexer(target=[selection_max_dt], method="nearest")[0]
            self.set_selection_max(x=selection_max_dt,
                                    y=data.iat[closest_max_dt_index, idx_of_channel])
            self._draw_selection_line_marker(x=selection_max_dt)


        # Set the axis settings
        self.quicklook_ax.set_yscale("log")
        set_xlims(ax=self.quicklook_ax, data=data, xlim=xlim)
        set_standard_ticks(ax=self.quicklook_ax, labelsize=QUICKLOOK_TICK_LABELSIZE)

        # Plot the curves
        for ch in channel:
            self.quicklook_ax.step(data.index.values, data[ch].values, where="mid", label=ch)

        # Formatting the x-axis and setting the axis labels
        self.quicklook_ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d"))
        self.quicklook_ax.set_xlabel(f"Date of {data.index[len(data.index)//2].strftime('%b, %Y')}", fontsize=QUICKLOOK_LEGENDSIZE)
        self.quicklook_ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]", fontsize=QUICKLOOK_LEGENDSIZE)

        # Add the legend and show the figure
        self.quicklook_ax.legend(fontsize=QUICKLOOK_LEGENDSIZE)

        self.quicklook_fig.tight_layout()
        plt.show()


    def find_breakpoints(self, channel:str, resample:str=None, xlim:list=None, window:int=None, 
                        threshold:float=None, plot:bool=True, diagnostics=False, index_choice="time_s", 
                        plot_style="step", breaks=1, title:str=None, fill_zeroes=True):
        """
        If not using manual selection, then seeks for the first peak in the given data. Cuts the data there 
        and only considers that part which comes before the first peak. In this chosen part, seek (a) break/s 
        in the linear trend that is the background of the event. The break corresponds to the start of the 
        event, and the second linear fit corresponds to the slope of the rising phase of the event (the linear 
        slope of the 10-based logarithm).

        Parameters:
        -----------
        channel : {str} The ID of the channel.
        resample : {str}
        xlim : {list}
        window : {str}
        threshold : {float}
        plot : {bool} Draws the plot of breakpoints and intensity time series.
        diagnostics : {bool}
        index_choice : {str} Either 'counting_numbers' or 'time_s'
        plot_style : {str} Either 'step' or 'scatter'
        breaks : {int} Number of breaks to search for.
        title : {str} The title string.
        fill_zeroes : {bool} Fills zero intensity bins with a filler value, described in calc.fill_zeros().

        Returns:
        ----------
        results_dict : {dict} A dictionary of results that contains 'const', 'slope1', 'slope2', 
                            'break_point' and 'break_errors'.
        """

        # Clears the past (interactive) figure
        plt.close("all")

        # Run checks
        _validate_index_choice(index_choice=index_choice)
        _validate_plot_style(plot_style=plot_style)

        # Choose resampling:
        if isinstance(resample, str):
            data = calc.resample_df(df=self.data, avg=resample)
        # If no resampling, just take a copy of the original data to avert 
        # modifying it by accident
        else:
            data = self.data.copy(deep=True)
            resample = calc.infer_cadence(series=data[channel])

        # Exclude data outside figure boundaries:
        if isinstance(xlim, tuple|list):
            data = data.loc[(data.index >= xlim[0])&(data.index <= xlim[1])]

        # Select the channel and produce indices for them. The indices are stored in the 
        # column "time_s", for they read seconds since the Unix epoch (1970-01-01 00:00:00).
        # The index numbers can be used for the regression algorithm instead of datetime values. 
        data = calc.produce_index_numbers(df=data)

        # Choose how to treat 0 intensities before taking logarithms
        if fill_zeroes:
            data = calc.add_ordinal_numbers(df=data)
            series = calc.fill_zeros(series=data[channel])
        else:
            # This function handles also adding ordinal numbers
            data = calc.select_channel_nonzero_ints(df=data, channel=channel)
            series = data[channel]

        # Convert to log
        series = calc.ints2log10(intensity=series)
        # This is what's getting plotted
        plot_series = series.copy(deep=True)

        # selection_max was left undefined -> try to seek for the first peak 
        if pd.isnull(self.selection_max_x):
            # Get the numerical index of the first peak to choose the selection from 
            # background to first peak. Also generate numerical index to run from 0 to max_idx
            print("Notice that you're running regression analysis without a user-set selection of data.")
            print(f"Manually searching for the peak of the event with given parameters: window={window}, threshold={threshold}")
            max_val, max_idx = calc.search_first_peak(ints=series, window=window, threshold=threshold)
            print(f"Found peak with given parameters at: {data.index[max_idx]}")
            min_idx = 0
        else:
            max_idx = data.index.get_indexer(target=[self.selection_max_x], method="nearest")[0]
            max_val = self.selection_max_y
            if not pd.isnull(self.selection_min_x):
                min_idx = data.index.get_indexer(target=[self.selection_min_x], method="nearest")[0]
            else:
                min_idx = 0


        # Apply a slice/selection to the data series and the numerical indices (seconds since Epoch)
        # according to the first peak found
        series = series[min_idx:max_idx]
        numerical_indices = data[index_choice].values[min_idx:max_idx]

        # Get the fit results
        fit_results = break_regression(ints=series.values, indices=numerical_indices, num_of_breaks=breaks)

        # The results are a dictionary, extract values here. Also check that the result converged.
        estimates = fit_results["estimates"]
        regression_converged = fit_results["converged"]

        _validate_fit_convergence(regression_converged=regression_converged)

        const, list_of_alphas, list_of_breakpoints, list_of_breakpoint_errs = calc.unpack_fit_results(fit_results=estimates,
                                                                                                num_of_breaks=breaks)

        # Finds corresponding timestamps to the numerical indices
        list_of_dt_breakpoints, list_of_dt_breakpoint_errs = calc.breakpoints_to_datetime(series=series, numerical_indices=numerical_indices,
                                                                                    list_of_breakpoints=list_of_breakpoints,
                                                                                    list_of_breakpoint_errs=list_of_breakpoint_errs,
                                                                                    index_choice=index_choice)

        # Compile a results dictionary to eventually return
        results_dict = {"const": const}
        for i, alpha in enumerate(list_of_alphas):
            results_dict[f"alpha{i}"] = alpha
        for i, bp in enumerate(list_of_dt_breakpoints):
            results_dict[f"breakpoint{i}"] = bp
        for i, bp_errs in enumerate(list_of_dt_breakpoint_errs):
            results_dict[f"breakpoint{i}_errors"] = bp_errs

        if plot:

            # Init figure
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

            if diagnostics:
                print(f"Data selection: {series.index[0]}, {series.index[-1]}")
                print(f"Regression converged: {regression_converged}")
                # Generate the fit lines to display on the plot
                list_of_fit_series = calc.generate_fit_lines(data_df=data, indices=numerical_indices, const=const,
                                                            list_of_alphas=list_of_alphas, 
                                                            list_of_breakpoints=list_of_breakpoints, index_choice=index_choice)

                # Plot the fit results on the real data
                for line in list_of_fit_series:
                    ax.plot(line.index, line.values, lw=2.8, ls="--", c="maroon", zorder=3)

                # Apply a span over xmin=start and xmax=max_idx to display the are considered for the fit
                ax.axvspan(xmin=series.index[0], xmax=series.index[-1], facecolor="green", alpha=DEFAULT_SELECTION_ALPHA, label="selection area")

            # Plot the intensities
            if plot_style=="step":
                ax.step(plot_series.index, plot_series.values, label=channel, zorder=1, where="mid")
            if plot_style=="scatter":
                ax.scatter(plot_series.index, plot_series.values, label=channel, zorder=1)

            for i, breakpoint_dt in enumerate(list_of_dt_breakpoints):

                # One has to use the notoriously awkward triple curly parenthesis here to be able to
                # employ LateX formalism in an f-string:
                err_delta_plusminus = str(breakpoint_dt - list_of_dt_breakpoint_errs[i][0])[7:7+8]
                #err_delta_minus = str(list_of_dt_breakpoint_errs[i][1] - breakpoint_dt)[7:7+8]
                #bp_label = f"breakpoint$_{{{i}}}$: "+f"{breakpoint_dt.strftime('%H:%M:%S')}$_{{-{err_delta_minus}}}^{{+{err_delta_plus}}}$"
                bp_label = f"breakpoint$_{{{i}}}$: "+f"{breakpoint_dt.strftime('%H:%M:%S')}{LATEX_PM}{err_delta_plusminus}"
                ax.axvspan(xmin=list_of_dt_breakpoint_errs[i][0], xmax=list_of_dt_breakpoint_errs[i][1], alpha=BREAKPOINT_SHADING_ALPHA, color="red")
                ax.axvline(x=breakpoint_dt, c="red", lw=1.8, label=bp_label)

            # Sets the yticklabels to their exponential form (e.g., 10^5 instead of 5)
            fabricate_yticks(ax=ax)
            set_standard_ticks(ax=ax)

            # Format the x-axis, name the y-axis and set the x-axis span
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d"))
            ax.set_xlabel(f"Date of {breakpoint_dt.strftime('%b, %Y')}", fontsize=STANDARD_LEGENDSIZE)
            ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]", fontsize=STANDARD_LEGENDSIZE)
            set_xlims(ax=ax, data=data, xlim=xlim)

            ax.set_title(title, fontsize=STANDARD_TITLE_FONTSIZE)
            ax.legend(fontsize=STANDARD_LEGENDSIZE)
            plt.show()

            results_dict["fig"] = fig
            results_dict["ax"] = ax

        # When diagnostics is enabled, return additional info about the run
        if diagnostics:
            results_dict["series"] = series
            results_dict["indices"] = numerical_indices
            results_dict["data_df"] = data
            for i, line in enumerate(list_of_fit_series):
                results_dict[f"line{i}"] = line

        return results_dict


def break_regression(ints, indices, starting_values:list=None, num_of_breaks:int=None) -> dict:
    """
    Initializes the Fit of piecewise_regression package, effectively running the algorithm for
    given data.

    Parameters:
    -----------
    ints : {array-like} The intensity (logarithms)
    indices : {array-like} The x-axis values (ordinal numbers or such)
    starting_values : {list}
    num_of_breaks : {int} Number of expected breakpoints.

    Returns:
    --------
    fit_results : {dict} A dictionary that contains the results of analysis.
    """

    if num_of_breaks is None:
        num_of_breaks = DEFAULT_NUM_OF_BREAKPOINTS

    fit = piecewise_regression.Fit(xx=indices,
                                   yy=ints,
                                   start_values=starting_values,
                                   n_breakpoints=num_of_breaks)

    return fit.get_results()

