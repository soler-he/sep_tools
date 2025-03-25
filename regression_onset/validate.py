
"""
Contains validations for function/method parameters.

"""

__author__ = "Christian Palmroos"

VALID_INDEX_CHOICES = ("counting_numbers", "time_s")
VALID_PLOT_STYLES = ("step", "scatter")

def _validate_index_choice(index_choice:str) -> None:
    if index_choice not in VALID_INDEX_CHOICES:
        raise ValueError(f"{index_choice} is not a valid index_choice!\nValid index_choice options are {VALID_INDEX_CHOICES}")

def _validate_plot_style(plot_style:str) -> None:
    if plot_style not in VALID_PLOT_STYLES:
        raise ValueError(f"{plot_style} is not a valid plot_style!\nValid plot_style options are {VALID_PLOT_STYLES}")

def _validate_fit_convergence(regression_converged:bool) -> None:
    if not regression_converged:
        raise ValueError(f"Regression converged: {regression_converged}. Try other settings.")

def _validate_selection(selection:list[str]|str) -> None:
    if not isinstance(selection, (list,str)):
        raise TypeError(f"The selection parameter must be a pandas-compatible datetime string or a pair of string (list), but {type(selection)} was provided!")
