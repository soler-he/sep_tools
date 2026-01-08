import os
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from fit_spectra.run_spectra_fit import run_the_fit
from fit_spectra.functions_for_spectra_fit import plot_spectrum


"""
Install dependencies for tests:
pip install flake8 pytest pytest-doctestplus pytest-cov pytest-mpl

To create/update the baseline images, run the following command from the base package dir:
pytest --mpl-generate-path=fit_spectra/tests/baseline fit_spectra/tests/test_.py

To run the tests locally, go to the base directory of the repository and run:
pytest -ra --mpl --mpl-baseline-path=baseline --mpl-baseline-relative --mpl-generate-summary=html --durations=0 fit_spectra/tests/test_.py
"""


@pytest.mark.parametrize("which_fit", [('single'), ('double'), ('best_sb'), ('cut'), ('double_cut'), ('best_cb'), ('triple'), ('best')])
@pytest.mark.mpl_image_compare(remove_text=False, deterministic=True)
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning")
# @pytest.mark.filterwarnings("ignore::UserWarning:seppy")
# @pytest.mark.filterwarnings("ignore::UserWarning:solo_epd_loader")
# @pytest.mark.filterwarnings("ignore::UserWarning:sunpy")
def test_SEP_Fit_Spectra(which_fit):
    path = f"{os.getcwd()}{os.sep}output_spectra{os.sep}spectrum_integral_SOLO_EPT_sun_electrons.csv"
    data = pd.read_csv(path)  # or pd.read_excel() for xlsx data

    plot_spectrum(data)

    # set by function call:
    # which_fit = 'single'  # single`, `double`, `best_sb`, `cut`, `double_cut`, `best_cb`, `triple`, `best`

    # initial guesses:
    intensity_zero_guess = 1e14  # peak flux
    gamma_1_guess = -1.7  # gamma 1 -> spectral index before the break (or single pl)
    gamma_2_guess = -2.  # gamma 2 -> spectral index after the first break (for broken pls)
    gamma_3_guess = -4.5  # gamma 3 -> spectral index after the second break (for triple pl)
    alpha_guess = 7.16  # sharpness of the first break
    beta_guess = 10  # sharpness of the second break
    break_energy_low_guess = 0.1  # in MeV
    break_energy_high_guess = 0.12  # in MeV
    cutoff_energy_guess = 0.12  # in MeV
    exponent_guess = 2
    e_min = None  # in MeV
    e_max = None  # in MeV
    exclude_channels = [31, 32, 33]  # None or list of indices correspong to the channels e.g. [1,3, 24]
    use_random = True
    iterations = 100
    legend_details = False
    plot_title = ''
    x_label = 'Energy (MeV)'
    y_label = 'Intensity\n/(s cm² sr MeV)'  # use for peak spectrum
    # y_label = 'Intensity\n/(cm² sr MeV)'  # use for integrated spectrum
    legend_title = ''
    use_filename_as_title = True
    save_plot = True

    run_the_fit(path, data, save_plot, use_filename_as_title, channels_to_exclude=exclude_channels, plot_title=plot_title, x_label=x_label, y_label=y_label, legend_title=legend_title, which_fit=which_fit,  e_min=e_min, e_max=e_max, g1_guess=gamma_1_guess, g2_guess=gamma_2_guess, g3_guess=gamma_3_guess, I0_guess=intensity_zero_guess, alpha_guess=alpha_guess, beta_guess=beta_guess, break_guess_low=break_energy_low_guess, break_guess_high=break_energy_high_guess, cut_guess=cutoff_energy_guess, exponent_guess=exponent_guess, use_random=use_random, iterations=iterations, legend_details=legend_details)
    fig = plt.gcf()

    return fig
