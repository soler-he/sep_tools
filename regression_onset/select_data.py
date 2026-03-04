
"""
Interactive user-interface module for selecting between SEPpy-based data loading and user-defined local data in the 
Regression-Onset-Tool notebook.
"""

__author__ = "Christian Palmroos"


import ipywidgets as widgets

HANDLE_NAME = "Data source: "
SOURCE_OPTIONS = ("SEPpy (online)", "User defined")
BUTTON_STYLES = ("success", "info", "warning", "")

TOGGLEBUTTON_TOOLTIPS = ("Select SEPpy for data loading",
                         "Load data from your own file")

data_file0 = widgets.Select(
                options=SOURCE_OPTIONS,
                value=SOURCE_OPTIONS[0],
                description=HANDLE_NAME,
                disabled=False
                )

data_file = widgets.ToggleButtons(
                options=SOURCE_OPTIONS,
                value=SOURCE_OPTIONS[1],
                description=HANDLE_NAME,
                button_style=BUTTON_STYLES[3],
                tooltips=TOGGLEBUTTON_TOOLTIPS,
                disabled=False
                )

#
def _seppy_selected(data_file:widgets.ToggleButtons) -> bool:
    """
    Returns True if SEPpy is chosen. Otherwise returns False.
    """
    if data_file.value == SOURCE_OPTIONS[0]:
        return True
    return False
