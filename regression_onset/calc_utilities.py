
"""
Contains calculation utility functions and constants for linear regression model -based SEP event onset analysis
python package.
"""

__author__ = "Christian Palmroos"

import numpy as np
import pandas as pd


# Global constants
INDEX_NUMBER_COL_NAME = "time_s"
ORDINAL_NUMBERS_COL_NAME = "counting_numbers"


def select_channel_nonzero_ints(df:pd.DataFrame, channel:str, dropnan:bool=True):
    """
    Selects the intensities (values) from the dataframe[channel] selection such that
    no zeroes are left in the dataframe. Also drops nans if dropnan (default True) is 
    enabled.
    """

    # Work on a copy to not alter the original one
    df = add_ordinal_numbers(df=df)

    selection = df[[channel, INDEX_NUMBER_COL_NAME, ORDINAL_NUMBERS_COL_NAME]]
    selection = selection.loc[selection[channel]!=0]

    if dropnan:
        # Selects the entries for which "channel" column has no nans
        selection = selection[~selection[channel].isnull()]

    return selection


def add_ordinal_numbers(df:pd.DataFrame):
    """
    Adds ordinal numbers (0,1,2,3...,N) column to the dataframe.
    """
    df = df.copy(deep=True)

    counting_numbers =  np.linspace(start=0, stop=len(df)-1, num=len(df))
    df[ORDINAL_NUMBERS_COL_NAME] = counting_numbers.astype(int)

    return df

def produce_index_numbers(df:pd.DataFrame):
    # Work on a copy to not alter the original one
    df = df.copy(deep=True)
    #index_numbers = df.index.strftime("%s")
    # pd.Timedelta(seconds=(timestamp_dt.timestamp() - float(timestamp_dt.strftime("%s")))) == Timedelta(0 days, 02:00:00)
    # It seems strftime("%s") shows time 2 hours behind real POSIX time, for reason related to timezone differences between
    # my local timezone (UTC+2 at the moment) and that of UTC time.
    # Instead, utilize DatetimeIndex.astype(int) to get total nanoseconds after the EPOCH in UTC.
    index_numbers = df.index.astype(np.int64) // 1e9
    df[INDEX_NUMBER_COL_NAME] = index_numbers.astype(int)
    return df


def resample_df(df:pd.DataFrame, avg:str) -> pd.DataFrame:
    """
    Resamples a dataframe such that care is taken on the offset and origin of the data index.

    Parameters:
    ----------
    df : {pd.DataFrame}
    avg : {str} Resampling string.
    """

    if isinstance(avg,str):
        avg = pd.Timedelta(avg)

    copy_df = df.resample(rule=avg, origin="start", label="left").mean()

    # After resampling, a time offset must be added to preserve the correct alignment of the time bins
    copy_df.index = copy_df.index + pd.tseries.frequencies.to_offset(avg/2)

    return copy_df


def ints2log10(intensity) -> pd.Series:
    """
    Converts intensities to log(intensity).

    Parameters:
    -----------
    intensity : {pd.Series}

    Returns:
    ----------
    logints : {pd.Series}
    """

    # Takes the logarithm of the ints
    logints = np.log10(intensity)

    # There may be zeroes in the intensities, which get converted to -inf
    # Convert -infs to nan
    logints.replace([np.inf, -np.inf], np.nan, inplace=True)

    return logints


def generate_fit_lines(data_df:pd.DataFrame, indices:np.ndarray, const:float, list_of_alphas:list[float], 
                       list_of_breakpoints:list[float], index_choice:str) -> list[pd.Series]:
    """
    Generates a list of first order polynomials as pandas Series from given fit parameters.

    Parameters:
    ----------
    data_df : {pd.DataFrame} The intensity dataframe, indexed by time.
    indices : {array-like} The numerical indices of the data, the x-axis. They are either ordinal numbers or seconds.
    const : {float} The constant of the first linear fit.
    list_of_alphas : {list[float]} The slopes of the fits. Is always one longer than list_of_breakpoints.
    list_of_breakpoints : {list[float]} The breakpoints of the fit lines. Always one shorter than list_of_alphas.

    Returns:
    --------
    list_of_lines : {list[pd.Series]} The lines.
    """

    # Gather the index selections to this list. Each fit has its own selection of the total indices.
    # The selections are separated by breakpoints in the fits. Also collect the list of line values
    # to its own list.
    list_of_index_selections = []
    list_of_lines = []
    for i, alpha in enumerate(list_of_alphas):

        # Define the selection (start&end) and apply it to all indices. Save the selected slice to a list.
        # For the start of the selection, first take 0, and then always index i-1 from breakpoints.
        # For the end of the selection, always take ith breakpoint, except for the final take len(indices)==final index
        selection_start = list_of_breakpoints[i-1] if i > 0 else 0
        selection_end = list_of_breakpoints[i] if i < len(list_of_breakpoints) else len(indices) if index_choice==ORDINAL_NUMBERS_COL_NAME else indices[-1]

        index_selection = indices[(indices>=selection_start)&(indices<=selection_end)]

        # Add the first element of the part that was left OUTSIDE (right) the previous selection; this is
        # to make sure that all consecutive fits have one common data point.
        # The elements returned by np.setdiff1d() are only found in ar1, not in ar2.
        try:
            # Only add one more element on iterations other than the final iteration
            if i < len(list_of_alphas)-1:
                index_diff = np.setdiff1d(ar1=indices, ar2=index_selection)
                # Apply further selection from selection_start onward to discard every element BEFORE the current 
                # fit. (fit1_indices still exist in the diff(all_indices,fit2_indices))
                index_diff = index_diff[index_diff>=selection_start]
                # And here append the first element of the set of indices RIGHT to the current selection of indices.
                index_selection = np.append(index_selection, index_diff[0])
        except IndexError as ie:
            # Here IndexError would be caused by trying to access the 0th element of index_diff. If said
            # array is empty, then that element does not exist -> No need to do anything
            print(f"{ie} on iteration {i}.")
            pass

        # Add the currently selected segment of indices to the list of index selections.
        list_of_index_selections.append(index_selection)

        # Generate the line y = alpha * x, where alpha = const
        line = list_of_index_selections[i] * alpha

        # Subtraction term is the first value of the line to bring the start of the line to the y = 0 level.
        if i > 0:
            line = line - line[0]

        # Choose the constant term for the 1st order polynomial:
        if i == 0:
            line_const = const
        else:
            # Depending on the orientation of the previous line, the next line starts from the max or the 
            # min of the previous line.
            line_const = np.nanmax(list_of_lines[i-1]) if list_of_alphas[i-1] > 0 else np.nanmin(list_of_lines[i-1])
        
        # At this point the line is a simple y = alpha * x, where y pierces the x-axis at the first value of
        # the respective index selection. Now lift the line with line_const, and append to the list of lines
        line = line + line_const

        list_of_lines.append(line)

    # Generate a list of datetime selection to index the lines
    list_of_datetimes = _generate_fits_datetimes(list_of_indices=list_of_index_selections, data_df=data_df, index_choice=index_choice)

    # Generate the list of series from list of lines (list of numpy arrays that contain the values of the lines)
    # and from the list of indices (which contain the corresponding x-values to the lines)
    list_of_series = [pd.Series(list_of_lines[i], index=list_of_datetimes[i]) for i in range(len(list_of_alphas))]

    return list_of_series


def _generate_fits_datetimes(list_of_indices:list, data_df:pd.DataFrame, index_choice:str):
    """

    Parameters:
    -----------
    list_of_indices : {list[pd.Series]} Fits generated from fit parameters.
    data_df : {pd.DataFrame} The dataframe that contains the selected data, indexed by time.
    index_choice : {str} Either 'counting_numbers' or 'time_s'

    Returns:
    -----------
    list_of_datetimes : {list[datetime]}
    """

    list_of_datetimes = []
    if index_choice=="counting_numbers":
        for indices in list_of_indices:
            datetimes_selection = data_df.loc[data_df[ORDINAL_NUMBERS_COL_NAME].isin(indices)].index
            list_of_datetimes.append(datetimes_selection)
    else:
        for indices in list_of_indices:
            datetimes_selection = pd.to_datetime(indices, unit='s')
            list_of_datetimes.append(datetimes_selection)

    return list_of_datetimes


def get_interpolated_timestamp(datetimes, break_point) -> pd.Timestamp:
    """
    Finds a timestamp from a series that relates to a floating-point index rather than integer.

    Parameters:
    -----------
    datetimes : {DatetimeIndex or similar}
    break_point : {float}

    Returns:
    ----------
    interpolated_timestamp : {pd.Timestamp}
    """

    # The "floor" of the index and the fractional part separately
    lower_index = int(break_point)
    fractional_part = break_point - lower_index

    # State the two timestamps to interpolate between
    lower_timestamp = datetimes[lower_index]
    try:
        upper_timestamp = datetimes[lower_index+1]
    except IndexError as ie:
        print(ie, fractional_part)
        upper_timestamp = lower_timestamp

    # Calculate the interpolated timestamp
    interpolated_timestamp = lower_timestamp + fractional_part * (upper_timestamp - lower_timestamp)

    return interpolated_timestamp


def search_first_peak(ints, window=None, threshold=None) -> tuple[float, int]:
    """
    Searches for a local maximum for a given window.

    Parameters:
    -----------
    ints : {array-like}
    window : {int}
    threshold : {float}

    Returns:
    ---------
    max_val : {float}
    max_idx : {int}
    """

    # Check that there are no nans
    if np.isnan(ints).any():
        raise ValueError("NaN values are not permitted!")

    # Default window length is 30 data points
    if window is None:
        window = 30

    # Default threshold value is very small
    if threshold is None:
        max_val = -1e5
    else:
        max_val = threshold

    warnings = 0
    threshold_hit = False
    for idx, val in enumerate(ints):

        # Just do nothing until we hit threshold
        if val < max_val and not threshold_hit:
            warnings = 0
            continue

        if val >= max_val:
            threshold_hit = True
            max_val = val
            max_idx = idx
            warnings = 0
        else:
            warnings += 1

        if warnings == window:
            return max_val, max_idx

    # If the peak was not found within the given window, return the 
    # values that were found. Unless the threshold was set too high, in which case
    # raise an exception.
    try:
        _ = max_idx
    except UnboundLocalError as ule:
        print(ule)
        raise ValueError("The parameter 'threshold' was set higher than any value in the intensity time series. Either set the threshold lower, or don't give it as an input.")
    return max_val, max_idx


def infer_cadence(series:pd.Series) -> str:
    """
    Returns the time resolution of the input series.

    Parameters:
    -----------
    series: {pd.Series}

    Returns:
    ----------
    resolution: {str}
            Pandas-compatible freqstr
    """

    # If series has no set frequency, infer it from the 
    # differential between consecutive timestamps:
    if series.index.freq is None:
        
        # The time differentials
        index_diffs = series.index.diff()
        
        # There might be unregular time differences, pick the most
        # appearing one (mode).
        diffs, counts = np.unique(index_diffs, return_counts=True)
        mode_dt = pd.Timedelta(diffs[np.argmax(counts)])

        # Round up to the nearest second, because otherwise e.g., STEREO / SEPT data
        # that may have cadence of '59.961614005' seconds is interpreted to have nanosecond
        # precision.
        mode_dt = mode_dt.round(freq='s')

        # If less than 1 minute, express in seconds
        divisor = 60 if mode_dt.resolution_string == "min" else 1
        
        return f"{mode_dt.seconds//divisor} {mode_dt.resolution_string}"

    
    else:
        freq_str = series.index.freq.freqstr
        return freq_str if freq_str!="min" else f"1 {freq_str}"


def breakpoints_to_datetime(series:pd.Series, numerical_indices:np.ndarray, list_of_breakpoints:list, 
                                list_of_breakpoint_errs:list, index_choice:str):
    """
    Converts breakpoints (along with their errors) that are floats to datetimes.

    Parameters:
    -----------
    series : {pd.Series} The data series indexed by time.
    numerical_indices : {np.ndarray} The numerical indices of data, either ordinal numbers or seconds.

    list_of_breakpoints : {list[float]}
    list_of_breakpoint_errs : {list[tuple]}
    index_choice : {str} Either 'counting_numbers' or 'time_s'

    Returns:
    -----------
    list_of_dt_breakpoints : {list[datetime]}
    list_of_dt_breakpoint_errs : {list[tuple]}
    """

    list_of_dt_breakpoints = []
    list_of_dt_breakpoint_errs = []

    if index_choice == "counting_numbers":
        # Choose the LAST entry of a linear space of integers that map to numerical_indices smaller than
        # the break_point. This is "how manieth" data point break_point is in series.
        lin_idx = np.linspace(start=0, stop=len(series)-1, num=len(series))
        for i, break_point in enumerate(list_of_breakpoints):
            break_point_idx = lin_idx[numerical_indices<break_point][-1]
            break_point_err_minus_idx = lin_idx[numerical_indices<list_of_breakpoint_errs[i][0]][-1]
            break_point_err_plus_idx = lin_idx[numerical_indices<list_of_breakpoint_errs[i][1]][-1]
            breakpoint_dt = get_interpolated_timestamp(datetimes=series.index, break_point=break_point_idx)
            breakpoint_dt_minus_err = get_interpolated_timestamp(datetimes=series.index, break_point=break_point_err_minus_idx)
            breakpoint_dt_plus_err = get_interpolated_timestamp(datetimes=series.index, break_point=break_point_err_plus_idx)
            list_of_dt_breakpoints.append(breakpoint_dt)
            list_of_dt_breakpoint_errs.append((breakpoint_dt_minus_err, breakpoint_dt_plus_err))
    else:
        for i, break_point in enumerate(list_of_breakpoints):
            breakpoint_dt = pd.to_datetime(break_point, unit='s')
            breakpoint_dt_minus_err = pd.to_datetime(list_of_breakpoint_errs[i][0], unit='s')
            breakpoint_dt_plus_err = pd.to_datetime(list_of_breakpoint_errs[i][1], unit='s')
            list_of_dt_breakpoints.append(breakpoint_dt)
            list_of_dt_breakpoint_errs.append((breakpoint_dt_minus_err, breakpoint_dt_plus_err))

    return list_of_dt_breakpoints, list_of_dt_breakpoint_errs


def unpack_fit_results(fit_results:dict, num_of_breaks:int) -> tuple:
    """

    Parameters:
    -----------
    fit_results : {dict}

    Returns:
    -----------
    const : {float} The constant of the first fit
    list_of_alphas : {list[float]} A list of slopes for the polynomial fits.
    list_of_breakpoints : {list[float]} A list of breakpoints for the fits.
    """

    # The constant and slope of the first fit are always there.
    const = fit_results["const"]["estimate"]
    alpha = fit_results["alpha1"]["estimate"]

    # Initialize lists to collect values into
    list_of_alphas = [alpha]
    list_of_breakpoints = []
    list_of_breakpoint_errs = []

    # For a single break, there will be one iteration in the loop -> one additional slope. 
    for i in range(num_of_breaks):

        # Access the i+2th index in fit_results, because that package's indexing somehow starts from 1, not 0.
        alpha = fit_results[f"alpha{i+2}"]["estimate"]
        break_point = fit_results[f"breakpoint{i+1}"]["estimate"]
        break_point_errs = fit_results[f"breakpoint{i+1}"]["confidence_interval"]

        list_of_alphas.append(alpha)
        list_of_breakpoints.append(break_point)
        list_of_breakpoint_errs.append(break_point_errs)

    return const, list_of_alphas, list_of_breakpoints, list_of_breakpoint_errs


def fill_zeros(series:pd.Series) -> pd.Series:
    """
    Replaces the zeros of a series with a filler value f, which obeys the equation:
    log_true_mean = 1/(num_all_values) *  ( sum_log_nonzeros + (num_all_values - num_nonzeros) * log(f) )
    """

    # Work on a copy of the series to not alter the original one by accident
    series = series.copy(deep=True)

    ordinal_idx = _find_last_0_index(series=series)

    # -1 is an invalid index and therefore we exit here
    if ordinal_idx == -1:
        return series

    # Apply a selection to the series, up to the final 0.
    series_sel = series.iloc[:ordinal_idx]

    filler = _calculate_filler(series=series_sel)
    
    # Replace zeroes with the filler value and return
    new_values = np.where(series.values > 0, series.values, filler)

    return pd.Series(new_values, index=series.index)


def _find_last_0_index(series:pd.Series) -> int:
    """
    Finds the ordinal index of the final 0 in a series.
    Example: series = [1,2,0,3,4,0,5] -> returns 5.
    """
    MIN_SERIES_LEN = 1000
    TARGET_VALUE = 0.

    # Search for the last 0:
    # Start by cutting half the series if iẗ́'s short enough; the tail may sometimes have zeroes
    series_half = series.iloc[:len(series)//2] if len(series) > MIN_SERIES_LEN else series

    try:
        final_0_index = series_half[series_half == TARGET_VALUE].index[-1]

    # IndexError is cause by the value being serched for not existing in the series.
    except IndexError:
        # Nothing to do, so just return
        return -1

    return series.index.get_indexer(target=[final_0_index])[0]


def _calculate_filler(series:pd.Series) -> float:
    """
    Calculates the filler value for fill_zeros() -function.
    Returns: filler {float}
    """

    # Sets to nan all zeroes
    series_nonzeros = np.where(series.values>0, series.values, np.nan)

    # Assert the number of nonzero values and all values
    num_nonzeros = np.count_nonzero(series)
    num_all = len(series)

    # This mean (including zeroes) is the true mean of log(ints) we are aiming for when filling the zeroes
    log_true_mean = np.log10(series.mean())

    #mean_log_nonzeros = np.nanmean(np.log10(series_nonzeros))

    sum_log_nonzeros = np.nansum(np.log10(series_nonzeros))

    nominator = num_all * log_true_mean - sum_log_nonzeros
    denominator = num_all - num_nonzeros

    log_filler = nominator / denominator

    filler = np.power(10, log_filler)
    
    return filler