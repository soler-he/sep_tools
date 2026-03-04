
"""
Contains the first development version for an SEP event onset finding tool that utilizes 
segmented linear regression.
piecewise-regression: Pilgrim, 2021.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter

import piecewise_regression

# Relative imports cannot be used with "import .a" form; use "from . import a" instead. -Pylance
from . import calc_utilities as calc
from .plotting_utilities import set_standard_ticks, set_xlims, set_ylims, fabricate_yticks, STANDARD_QUICKLOOK_FIGSIZE, \
                                STANDARD_TITLE_FONTSIZE, STANDARD_FIGSIZE, STANDARD_FONTSIZE, \
                                BREAKPOINT_SHADING_ALPHA, SELECTION_SHADE_COLOR, SELECTION_SHADE_ALPHA, LATEX_PM

from .validate import _validate_index_choice, _validate_plot_style, _validate_fit_convergence, _validate_selection

from .externals import export_seppy_data, generate_column_indices, combine_energy_channels

from .select_data import data_file, SOURCE_OPTIONS

DEFAULT_NUM_OF_BREAKPOINTS = 1
DEFAULT_NUM_OF_TRIALS = 5
SECONDS_PER_DAY = 86400

QUICKLOOK_TICK_LABELSIZE = 18
QUICKLOOK_LEGENDSIZE = STANDARD_FONTSIZE-5

# SEPpy is the first of the source options; the other is 'User defined'
SEPPY = SOURCE_OPTIONS[0]

# Most channels have an "energy range", but some have an "effective energy" instead
SEPPY_ENERGY_COLUMN_NAMES = ("Energy range", "Effective energy")

class Reg:

    def __init__(self, data:pd.DataFrame, data_source:str, meta_df:pd.DataFrame=None, meta_dict:dict=None):
        """"
        
        data : {pd.Dataframe} The intensity data.
        data_source : {str} Either 'SEPpy' or 'User defined'
        meta_df : {pd.DataFrame} optional. Contains the channel energy strings.
        meta_dict : {dict} A dictionary containing the names 
        """

        self.data = data

        # Check for the df index timezone-awareness:
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)

        # Check that the data source is valid
        if data_source not in SOURCE_OPTIONS:
            raise ValueError(f"{data_source} is not a valid data source; choose either one of the following: {SOURCE_OPTIONS}.")

        self.data_source = data_source
        self.meta_df = meta_df
        self.meta_dict = meta_dict

        self.selection_max_x = pd.NaT
        self.selection_max_y = np.nan

        self.selection_min_x = pd.NaT
        self.selection_min_y = np.nan

        # To keep track of how many times self._onclick() has been run
        self.times_clicked = 0

        # For SEPpy data we change the column names
        if data_source==SEPPY:
            new_columns = generate_column_indices(columns=data.columns, meta_index=meta_df.index)
            self.data.rename(columns=new_columns, inplace=True)


    def _restart_clicks(self) -> None:
        self.times_clicked = 0


    def _title_str(self, channel_index) -> str:
        """
        Generates a title string for figures from SEPpy meta data.
        """
        energy_id = SEPPY_ENERGY_COLUMN_NAMES[0]
        # Unpack the metadata
        spacecraft = self.meta_dict["Spacecraft"]
        sensor = self.meta_dict["Sensor"]
        viewing = self.meta_dict["Viewing"]
        species = self.meta_dict["Species"]
        try:
            energy = self.meta_df[energy_id][channel_index]
        # This KeyError here is most probably caused by the energy column being
        # identified by "Effective energy" instead of "Energy range"
        except KeyError as e:
            energy_id = SEPPY_ENERGY_COLUMN_NAMES[1]
            energy = self.meta_df[energy_id][channel_index]
        # A check for the energy string; it may be an element of the dataframe or the string:
        if len(energy)==1:
            energy = energy.values[0]
        # A check for the viewing str; if the viewing is None, there should be no print output.
        viewing = viewing if viewing != "None" or viewing is None else ""
        # A further check for viewing; if sc==wind, then add "sector" to sectored viewing
        if spacecraft=="Wind":
            if len(viewing)==1:
                viewing = f"Sector {viewing}"
        # A similar check for BepiColombo
        if spacecraft=="BepiColombo":
            viewing = f"Side {viewing}"
        title_str = f"{spacecraft} / {sensor}$^{{\\mathrm{{{viewing}}}}}$\n{energy} {species}"
        return title_str


    def _set_selection_max(self, x, y=None) -> None:
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

        # This method only works for two clicks; the start and the end of the selection
        if self.times_clicked==0:

            if event.xdata is not None and event.ydata is not None:
                # First convert matplotlib's xdata (days after epoch) to seconds and then to datetime
                x = pd.to_datetime(event.xdata*SECONDS_PER_DAY, unit='s')
                self._set_selection_min(x=x, y=event.ydata)
                self._draw_selection_line_marker(x=x)
                print("First selection marked")
            else:
                raise TypeError("Event xdata or ydata was None")

        if self.times_clicked==1:

            if event.xdata is not None and event.ydata is not None:
                # Convert matplotlib's xdata (days after epoch) to seconds and then to datetime
                x = pd.to_datetime(event.xdata*SECONDS_PER_DAY, unit='s')
                self._set_selection_max(x=x, y=event.ydata)
                self._draw_selection_line_marker(x=x)
                print("Second selection marked")
                self._apply_selection_shading(ax=self.quicklook_ax)
            else:
                raise TypeError("Event xdata or ydata was None")

        # Finally update counter
        self.times_clicked += 1


    def _draw_selection_line_marker(self, x) -> None:
        self.quicklook_ax.axvline(x=x, color="green", zorder=10)


    def _apply_selection_shading(self, ax:plt.Axes) -> None:
        """
        Applies a matplotlib axhspan over xmin and xmax on the give Axes if they both exist.
        """
        if not isinstance(self.selection_min_x, pd._libs.tslibs.nattype.NaTType) and not isinstance(self.selection_max_x, pd._libs.tslibs.nattype.NaTType):
            ax.axvspan(xmin=self.selection_min_x, xmax=self.selection_max_x,
                       color=SELECTION_SHADE_COLOR, alpha=SELECTION_SHADE_ALPHA)


    def quicklook(self, channel:int|str=None, resample:str=None, xlim:list=None, selection:list[str]|str=None) -> None:
        """
        Makes a quicklook plot of one or more channels for a given dataframe.
        Meant to be used in interactive mode, so that the user can apply data selection
        by clicking.

        Comprehensive example of ipympl: https://matplotlib.org/ipympl/examples/full-example.html

        Parameters:
        --------------
        channel : int or str
        resample : str
        xlim : list
        selection : {list[str] or str} format: %Y-%m-%d %H:%M%S.
                    If given a pair of timestamps, apply selection between them. If one timestamp, start selection
                    from the beginning of the input data and select up to the given timestamp.
        """

        # Apply resampling if asked to
        if isinstance(resample,str):
            data = calc.resample_df(df=self.data, avg=resample)
        else:
            data = self.data.copy(deep=True)

        # Exclude data outside figure boundaries:
        if isinstance(xlim, tuple|list):
            data = data.loc[(data.index >= xlim[0])&(data.index <= xlim[1])]

        # A list for the channel means combining channels
        if isinstance(channel, (tuple,list)):
            raise NotImplementedError("Combining channels on the fly not yet implemented! Use the 'combine_channels()' -method to combine channels.")

        # Attach the fig and axes to class attributes
        self.quicklook_fig, self.quicklook_ax = plt.subplots(figsize=STANDARD_QUICKLOOK_FIGSIZE)

        # Next block is about determining the selection for the data. It will be done either by
        # a click on the interactive plot or by giving "selection" parameter as an input.
        #
        # Attach the onclick() -method to a mouse button press event for the interactive plot if
        # the selection parameter was not provided
        if selection is None:
            self._restart_clicks()
            cid = self.quicklook_fig.canvas.mpl_connect(s="button_press_event", func=self._onclick)
        else:
            # First make sure that selection is of correct type
            _validate_selection(selection=selection)

            # The numerical index of channel is needed to access the right selection y values
            idx_of_channel = data.columns.get_indexer(target=[channel])[0]

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
            self._set_selection_max(x=selection_max_dt,
                                    y=data.iat[closest_max_dt_index, idx_of_channel])
            self._draw_selection_line_marker(x=selection_max_dt)
            self._apply_selection_shading(ax=self.quicklook_ax)


        # Set the axis settings
        self.quicklook_ax.set_yscale("log")
        set_xlims(ax=self.quicklook_ax, data=data, xlim=xlim)
        set_standard_ticks(ax=self.quicklook_ax, labelsize=QUICKLOOK_TICK_LABELSIZE)

        # Plot the curve
        intensity_label = channel if not self.data_source==SEPPY else None
        self.quicklook_ax.step(data.index.values, data[channel].values, where="mid", label=intensity_label)

        # Formatting the x-axis and setting the axis labels
        self.quicklook_ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d"))
        self.quicklook_ax.set_xlabel(f"Date of {data.index[len(data.index)//2].strftime('%b, %Y')}", fontsize=QUICKLOOK_LEGENDSIZE)
        self.quicklook_ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]", fontsize=QUICKLOOK_LEGENDSIZE)

        # Add the legend and show the figure
        # self.quicklook_ax.legend(fontsize=QUICKLOOK_LEGENDSIZE)

        if self.data_source==SEPPY:
            title = self._title_str(channel_index=channel)
            self.quicklook_ax.set_title(title, fontsize=QUICKLOOK_LEGENDSIZE)

        # self.quicklook_fig.tight_layout()
        plt.show()


    def find_breakpoints(self, channel:str, resample:str=None, xlim:list=None, ylim:list=None, window:int=None, 
                        threshold:float=None, plot:bool=True, diagnostics=False, index_choice="time_s", 
                        plot_style="step", breaks=DEFAULT_NUM_OF_BREAKPOINTS, title:str=None, fill_zeroes=True,
                        convergence_trials=DEFAULT_NUM_OF_TRIALS):
        """
        If not using manual selection, then seeks for the first peak in the given data. Cuts the data there 
        and only considers that part which comes before the first peak. In this chosen part, seek (a) break/s 
        in the linear trend that is the background of the event. The break corresponds to the start of the 
        event, and the second linear fit corresponds to the slope of the rising phase of the event (the linear 
        slope of the 10-based logarithm).

        Parameters:
        -----------
        channel : {str} The ID of the channel.
        resample : {str} Time-averaging str to apply, e.g., '5 min' for 5 -minute time-averaging.
        xlim : {list | tuple} List of two timestamps. Sets the boundaries for the x-axis.
        ylim : {list | tuple} List of two floats/integers. Sets the boundaries for the y-axis.
        window : {str} For peak finder: The amount of data points to look forward from last found peak.
        threshold : {float} For peak finder: The minimum value to consider the peak.
        plot : {bool} Draws the plot of breakpoints and intensity time series.
        diagnostics : {bool} Enables diagnostic mode.
        index_choice : {str} Either 'counting_numbers' or 'time_s'
        plot_style : {str} Either 'step' or 'scatter'
        breaks : {int} Number of breaks to search for.
        title : {str} The title string.
        fill_zeroes : {bool} Fills zero intensity bins with a filler value, described in calc.fill_zeros().
        convergence_trials : {int} The number of trials to find a convergent solution before giving up. Default = 5

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

        # Get the fit results (try a number of times if converged == False)
        while convergence_trials > 0:

            # Runs regression model with given parameters
            fit_results, fit_curve = break_regression(ints=series.values, indices=numerical_indices, num_of_breaks=breaks)

            # The results are a dictionary, extract values here. Also check that the result converged.
            estimates = fit_results["estimates"]
            regression_converged:bool = fit_results["converged"]

            # Let's not validate convergence anymore
            # _validate_fit_convergence(regression_converged=regression_converged)
            # If regression converged to a solution, simply exit the loop and continue as usual
            if regression_converged:
                convergence_trials = 0

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
                results_dict["fit_curve"] = fit_curve

            # If not, notify the user and deduct 1 from the amount of trials -> try again
            else:
                convergence_trials -= 1
                print(f"Regression converged: {regression_converged}. Retries left: {convergence_trials}")

        # Check here if regression did not converge.
        # Results_dict still needs to be initialized even for empty result, so that figure and axes can be loaded into it
        if not regression_converged:
            results_dict = {}
            print("It could be a good idea to try a different amount of breakpoints and/or adjust the time-averaging")


        if plot:

            # Init figure
            fig, ax = plt.subplots(figsize=STANDARD_FIGSIZE)

            # Plot the intensities
            intensity_label = channel if not self.data_source==SEPPY else None
            if plot_style=="step":
                ax.step(plot_series.index, plot_series.values, label=intensity_label, zorder=2, where="mid")
            if plot_style=="scatter":
                ax.scatter(plot_series.index, plot_series.values, label=intensity_label, zorder=2)

            # The fits and breakpoints only exists if regression converged
            if regression_converged:

                # Plot the fit curve on top of the data:
                ax.plot(fit_curve, lw=2.8, ls="--", color="maroon", zorder=3)

                # Loop through the breakpoints
                for i, breakpoint_dt in enumerate(list_of_dt_breakpoints):

                    # One has to use the notoriously awkward triple curly parenthesis here to be able to
                    # employ LateX formalism in an f-string:
                    err_delta_plusminus = str(breakpoint_dt - list_of_dt_breakpoint_errs[i][0])[7:7+8]
                    #err_delta_minus = str(list_of_dt_breakpoint_errs[i][1] - breakpoint_dt)[7:7+8]
                    #bp_label = f"breakpoint$_{{{i}}}$: "+f"{breakpoint_dt.strftime('%H:%M:%S')}$_{{-{err_delta_minus}}}^{{+{err_delta_plus}}}$"
                    bp_label = f"breakpoint$_{{{i}}}$: "+f"{breakpoint_dt.strftime('%H:%M:%S')}{LATEX_PM}{err_delta_plusminus}"
                    ax.axvspan(xmin=list_of_dt_breakpoint_errs[i][0], xmax=list_of_dt_breakpoint_errs[i][1], 
                               alpha=BREAKPOINT_SHADING_ALPHA, color="red", zorder=3)
                    ax.axvline(x=breakpoint_dt, c="red", lw=1.8, label=bp_label, zorder=4)

            # Sets the yticklabels to their exponential form (e.g., 10^5 instead of 5). This has to be done
            # IMMEDIATELY after plotting the intensity, fits and breakpoints, not later, because otherwise it will mess up the 
            # spacing of the yticks for an unknown reason.
            fabricate_yticks(ax=ax, series=plot_series)
            set_ylims(ax=ax, series=plot_series, ylim=ylim)
            set_standard_ticks(ax=ax)

            # Some extra if diagnostics are enabled:
            if diagnostics:

                # Print some useful values that describe the data
                print(f"Data min: {np.min(series.values)}, max: {np.max(series.values)}")
                print(f"Data selection: {series.index[0]}, {series.index[-1]}")
                print(f"Regression converged: {regression_converged}")

                # Apply a span over xmin=start and xmax=max_idx to display the are considered for the fit
                #ax.axvspan(xmin=series.index[0], xmax=series.index[-1], facecolor="green",
                 #          alpha=DEFAULT_SELECTION_ALPHA, label="selection area", zorder=1)
                self._apply_selection_shading(ax=ax)

                # Initialize a parallel y-axis to plot the original values to (invisible). This is to compare
                # that the original values align with 
                ax1 = ax.twinx()
                ax1.set_ylabel("log(Intensity)", fontsize=STANDARD_FONTSIZE)
                ax1.step(plot_series.index, plot_series.values, zorder=1, where="mid", alpha=0.4)
                set_ylims(ax=ax1, series=plot_series, ylim=ylim)
                set_standard_ticks(ax=ax1)

                # Enable grid for easier comparison of axes
                ax.grid(visible=True, which="both")

            # Format the x-axis, name the y-axis and set the x-axis span
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d"))
            ax.set_xlabel(f"Date of {plot_series.index[0].strftime('%b, %Y')}", fontsize=STANDARD_FONTSIZE)
            ax.set_ylabel(r"Intensity [1/(cm$^{2}$ sr s MeV)]", fontsize=STANDARD_FONTSIZE)
            set_xlims(ax=ax, data=data, xlim=xlim)

            ax.legend(fontsize=STANDARD_FONTSIZE)

            if self.data_source==SEPPY:
                seppy_title = self._title_str(channel_index=channel)
                ax.set_title(seppy_title, fontsize=STANDARD_TITLE_FONTSIZE)
            if title is not None:
                ax.set_title(title, fontsize=STANDARD_TITLE_FONTSIZE)

            plt.show()

            results_dict["fig"] = fig
            results_dict["ax"] = ax

        # When diagnostics is enabled, return additional info about the run
        if diagnostics:
            results_dict["series"] = series
            results_dict["indices"] = numerical_indices
            results_dict["data_df"] = data

        return results_dict


    def combine_channels(self, seppy_data, channels:list) -> None:
        """
        Combines a range of energy channels into one channel.

        seppy_data : {seppy.Event}
        channels: {list|tuple} A pair of integer numbers corresponding to the energy range one 
                                wants to combine. Accepts None, having no effect.
        """

        if channels is None:
            return None

        if self.data_source != SEPPY:
            raise NotImplementedError("Combining channels so far only implemented for SEPpy missions!")

        f, d = combine_energy_channels(event=seppy_data, channels=channels)

        self.data[self.meta_df.index+1] = f
        self.meta_df.loc[self.meta_df.iloc[-1].name+1] = d


def break_regression(ints, indices, starting_values:list=None, num_of_breaks:int=None) -> tuple[dict, pd.Series]:
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
    fit_curve : {pd.Series} A series of the broken linear curve that approximates the data. Is None if the algorithm
                            did not converge to a solution.
    """

    if num_of_breaks is None:
        num_of_breaks = DEFAULT_NUM_OF_BREAKPOINTS

    fit = piecewise_regression.Fit(xx=indices,
                                   yy=ints,
                                   start_values=starting_values,
                                   n_breakpoints=num_of_breaks)

    results = fit.get_results()

    if results["converged"]:
        fit_curve = get_fit_curve(fit=fit)
    else:
        fit_curve = None

    return results, fit_curve

def get_fit_curve(fit:piecewise_regression.Fit) -> pd.Series:
    """
    Returns the broken linear curve that approximates the original data.

    Parameters:
    -----------
    fit : {piecewise_regression.Fit}

    Returns:
    -----------
    series : {pd.Series} Broken fit curve indexed by linear space from selection(min) to selection(max)
    """

    # Generates a linear spacing of index numbers
    linx = np.linspace(np.nanmin(fit.xx), np.nanmax(fit.xx), 1000)

    # Generates the y-values for the fit
    y_vals = fit.predict(xx_predict=linx)

    # Convert index number (UNIX) to datetime
    return pd.Series(data=y_vals, index=pd.to_datetime(linx, unit='s'))
