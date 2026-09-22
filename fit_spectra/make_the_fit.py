# solo_functions.py
import numpy as np
import pandas as pd
import fit_spectra.odrpack_functions as pl_fit
from scipy.stats import t as studentt
import pickle


def closest_values(array, value):
    """Find the values closest to a given value.

    The number of returned values depends on the length of the input
    array:

    - Up to 10 values: return all available values.
    - 11 to 20 values: return half of the input values, rounded.
    - More than 20 values: return 10 values.

    The value equal to ``value`` is excluded from the search.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing the values to search.
    value : float
        Value for which the closest values are found.

    Returns
    -------
    list
        Sorted values closest to ``value``.
    """
    array_length = len(array)

    if array_length <= 10:
        n_closest = array_length
    elif array_length <= 20:
        n_closest = round(array_length / 2)
    else:
        n_closest = 10

    array = np.delete(array, np.where(array == value))

    closest_values_array = np.array(())

    for _ in range(n_closest):
        absolute_val_array = np.abs(array - value)
        smallest_difference_index = absolute_val_array.argmin()
        closest_element = array[smallest_difference_index]

        closest_values_array = np.append(closest_values_array, closest_element,)

        array = np.delete(array, np.where(array == closest_element))

    return sorted(closest_values_array)



# ============================================================================
# Helpers
# ============================================================================

def _break_energy_interval(spec_e, e_err, break_energy):
    """Return the energy interval surrounding the data point nearest a break."""

    absolute_val_array = np.abs(spec_e - break_energy)
    smallest_difference_index = absolute_val_array.argmin()

    if smallest_difference_index == len(spec_e) - 1:
        low = (spec_e[smallest_difference_index - 1]
            - e_err[smallest_difference_index - 1])
        high = (spec_e[smallest_difference_index]
            + e_err[smallest_difference_index])
    else:
        low = (spec_e[smallest_difference_index]
            - e_err[smallest_difference_index])
        high = (spec_e[smallest_difference_index + 1]
            + e_err[smallest_difference_index + 1])

    return low, high
           

def check_redchi(
    spec_e,
    spec_flux,
    e_err,
    flux_err,
    gamma1=-1,
    gamma2=-2,
    gamma3=-4,
    c1=1000,
    alpha=10,
    beta=10,
    E_break_low=0.06,
    E_break_high=0.1,
    E_cut=None,
    exponent=2,
    fit="best",
    maxit=10000,
    e_min=None,
    e_max=None,
    ):
    """Check reduced chi-squared values for different spectral models.

    Parameters
    ----------
    spec_e, spec_flux : array-like
        Energy and flux data to be fitted.
    e_err, flux_err : array-like
        Uncertainties in energy and flux, respectively.
    gamma1, gamma2, gamma3 : float, optional
        Initial guesses for the power-law indices.
    c1 : float, optional
        Initial guess for the normalization.
    alpha, beta : float, optional
        Initial guesses for the smoothness parameters.
    E_break_low, E_break_high : float, optional
        Initial guesses for the lower and upper break energies.
    E_cut : float or None, optional
        Initial guess for the cut-off energy.
    exponent : float, optional
        Initial guess for the cut-off exponent.
    fit : str, optional
        Determines which fitting procedure is used.
    maxit : int, optional
        Maximum number of ODR iterations.
    e_min, e_max : float or None, optional
        Minimum and maximum energy limits used when checking fitted
        break and cut-off positions.

    Returns
    -------
    list
        A list containing the selected fit name, reduced chi-squared
        value, and ODR result.
    """
    # The function also checks if the break point is outside of the
    # energy array (also the cutoff point).
    # The min and max energies cannot be the last and/or first points
    # because it wouldn't be a physical result.

    emin = spec_e[2]
    emax = spec_e[len(spec_e) - 3]

    if e_min is None or e_min == spec_e[0]:
        emin = spec_e[2]
    else:
        emin = e_min

    if e_max is None or e_max == spec_e[len(spec_e) - 1]:
        emax = spec_e[len(spec_e) - 3]
    else:
        emax = e_max

    # ------------------------------------------------------------------
    # Select the best model based on reduced chi-squared.
    # ------------------------------------------------------------------
    if fit == "best":
        result_triple = pl_fit.triple_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            c1=c1,
            alpha=alpha,
            beta=beta,
            E_break_low=E_break_low,
            E_break_high=E_break_high,
            maxit=maxit,
            )

        if _is_valid_fit(result_triple):
            redchi_triple = result_triple.res_var
            breakp_low = result_triple.beta[6]
            breakp_high = result_triple.beta[7]
            difference_triple = np.abs(breakp_high - breakp_low)

            alpha = result_triple.beta[4]
            beta = result_triple.beta[5]

            if alpha > 0:
                gamma1 = result_triple.beta[1]
            elif alpha <= 0:
                gamma1 = result_triple.beta[2]
        else:
            redchi_triple = np.inf

        result_cut_break = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            E_cut=E_cut,
            exponent=exponent,
            print_report=False,
            maxit=maxit,
            )

        if _is_valid_fit(result_cut_break):
            redchi_cut_break = result_cut_break.res_var
            breakp_cut = result_cut_break.beta[4]
            cut_b = result_cut_break.beta[5]
            difference_cut = np.abs(breakp_cut - cut_b)
            exponent_cut_break = result_cut_break.beta[6]
        else:
            redchi_cut_break = np.inf
            breakp_cut = None
            cut_b = None
            difference_cut = None
            exponent_cut_break = None

        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
            )

        if _is_valid_fit(result_cut):
            redchi_cut = result_cut.res_var
            cut = result_cut.beta[2]
            exponent_cut = result_cut.beta[3]
        else:
            redchi_cut = np.inf
            cut = None
            exponent_cut = None

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
            )

        if _is_valid_fit(result_double):
            redchi_double = result_double.res_var
            breakp = result_double.beta[4]
        else:
            redchi_double = np.inf
            breakp = None

        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            )

        if _is_valid_fit(result_single_pl):
            redchi_single = result_single_pl.res_var
        else:
            redchi_single = np.inf

        chis = {
            "triple": redchi_triple,
            "double_cut": redchi_cut_break,
            "cut": redchi_cut,
            "double": redchi_double,
            "single": redchi_single,
            }

        sorted_chis = dict(sorted(chis.items(), key=lambda item: item[1]))

        if exponent_cut == 0:
            sorted_chis.pop("cut")

        if exponent_cut_break == 0:
            sorted_chis.pop("double_cut")

        # Check whether the ODR fits converged successfully.
        if not _is_valid_fit(result_triple):
            sorted_chis.pop("triple")

        if not _is_valid_fit(result_cut_break):
            sorted_chis.pop("double_cut")

        if not _is_valid_fit(result_cut):
            sorted_chis.pop("cut")

        if not _is_valid_fit(result_double):
            sorted_chis.pop("double")

        if not _is_valid_fit(result_single_pl):
            sorted_chis.pop("single")

        # Check if there are values with zero chi-squared.
        # If so, remove them from the dictionary.
        list_zero_chi = []

        for model_name in sorted_chis:
            if sorted_chis[model_name] == 0.0:
                list_zero_chi.append(model_name)

        for model_name in list_zero_chi:
            sorted_chis.pop(model_name)

        if not sorted_chis:
            return None

        smallest_value = list(sorted_chis.keys())[0]

        for i in range(len(sorted_chis)):
            # Check the candidate models in order of increasing
            # reduced chi-squared. The physical validity of each
            # fitted break/cutoff is checked before accepting it.

            if smallest_value == "triple":
                if (breakp_low < emax
                    and breakp_low > emin
                    and breakp_high < emax
                    and breakp_high > emin
                    and breakp_low < breakp_high):

                    low, high = _break_energy_interval(spec_e, e_err, breakp_low)
                    difference_triple_energy = high - low

                    # The triple PL is defined so that the two breaks are
                    # actually interchangeable. This means that it can
                    # happen that the 'high' break becomes the low break.
                    # These cases have to be deleted because it messes with
                    # the meaning of the parameters of the fit.

                    if (difference_triple > difference_triple_energy
                        and gamma1 < 0):

                        if alpha > 0 or beta > 0:
                            return [ "triple", redchi_triple, result_triple]
                        else:
                            smallest_value = list(sorted_chis.keys())[i]

                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double_cut":
                if (cut_b > emin
                    and cut_b < emax
                    and breakp_cut > emin
                    and breakp_cut < emax
                    and cut_b > breakp_cut):

                    low, high = _break_energy_interval(spec_e, e_err, breakp_cut,)
                    difference_cut_energy = high - low

                    if (gamma1 < 0
                        and difference_cut > difference_cut_energy):

                        return ["double_cut", redchi_cut_break, result_cut_break,]
                    else:
                        smallest_value = list(sorted_chis.keys())[i]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "cut":
                if cut >= emin and cut <= emax:
                    return ["cut", redchi_cut, result_cut]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "double":
                if breakp >= emin and breakp <= emax:
                    return ["double", redchi_double, result_double]
                else:
                    smallest_value = list(sorted_chis.keys())[i]

            if smallest_value == "single":
                return ["single", redchi_single, result_single_pl]


    # ------------------------------------------------------------------
    # Explicit triple power law fit
    # ------------------------------------------------------------------
    if fit == "triple":
        result_triple = pl_fit.triple_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            c1=c1,
            alpha=alpha,
            beta=beta,
            E_break_low=E_break_low,
            E_break_high=E_break_high,
            maxit=maxit,
            )

        # Check ODR convergence BEFORE accessing result parameters.
        if not _is_valid_fit(result_triple):
            fit = "double_cut"
        else:
            redchi_triple = result_triple.res_var
            breakp_low = result_triple.beta[6]
            breakp_high = result_triple.beta[7]
            difference_triple = breakp_high - breakp_low

            if (breakp_low < emax
                and breakp_low > emin
                and breakp_high < emax
                and breakp_high > emin):

                low, high = _break_energy_interval(spec_e, e_err, breakp_low)
                difference_triple_energy = high - low

                if (breakp_high > breakp_low
                    and difference_triple > difference_triple_energy):

                    return [ "triple", redchi_triple, result_triple]

                else:
                    fit = "double_cut"

            else:
                fit = "double_cut"


    # ------------------------------------------------------------------
    # Explicit double power law with exponential cutoff
    # ------------------------------------------------------------------
    if fit == "double_cut":
        result_cut_break = pl_fit.cut_break_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            E_cut=E_cut,
            exponent=exponent,
            print_report=False,
            maxit=maxit,
            )

        # Check ODR convergence BEFORE accessing result parameters.
        if not _is_valid_fit(result_cut_break):
            fit = "best_cb"
        else:
            redchi_cut_break = result_cut_break.res_var
            breakp_cut = result_cut_break.beta[4]

            # The cut of the break + cutoff.
            cut_b = result_cut_break.beta[5]
            difference_cut = breakp_cut - cut_b

            if (breakp_cut <= emax
                and breakp_cut > emin
                and cut_b <= emax
                and cut_b > emin):

                low, high = _break_energy_interval(spec_e, e_err, breakp_cut)
                difference_cut_energy = high - low

                if (cut_b > breakp_cut
                    and difference_cut > difference_cut_energy):

                    return [ "double_cut", redchi_cut_break, result_cut_break]

                else:
                    fit = "best_cb"

            else:
                fit = "best_cb"


    # ------------------------------------------------------------------
    # Select between a single and double power law
    # ------------------------------------------------------------------
    if fit == "best_sd":
        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            )

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
            )

        # Check convergence of BOTH fits.
        single_valid = _is_valid_fit(result_single_pl)
        double_valid = _is_valid_fit(result_double)

        if single_valid:
            redchi_single = result_single_pl.res_var
        else:
            redchi_single = np.inf

        if double_valid:
            redchi_double = result_double.res_var
            breakp = result_double.beta[4]
        else:
            redchi_double = np.inf
            breakp = None

        # If neither fit converged, there is no valid result.
        if not single_valid and not double_valid:
            return None

        if redchi_double <= redchi_single:
            if not double_valid:
                return ["single", redchi_single, result_single_pl]

            if breakp < emin or breakp > emax:
                return ["single", redchi_single, result_single_pl]

            if breakp >= emin and breakp <= emax:
                return ["double", redchi_double, result_double]

        if redchi_double > redchi_single:
            if single_valid:
                return ["single", redchi_single, result_single_pl]

            return ["double", redchi_double, result_double]


    # ------------------------------------------------------------------
    # Select between a single PL + exp cutoff and double power law
    # ------------------------------------------------------------------
    if fit == "best_cb":
        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
            )

        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
            )

        # Check convergence of BOTH fits.
        cut_valid = _is_valid_fit(result_cut)
        double_valid = _is_valid_fit(result_double)

        if cut_valid:
            redchi_cut = result_cut.res_var
            cut = result_cut.beta[2]
        else:
            redchi_cut = np.inf
            cut = None

        if double_valid:
            redchi_double = result_double.res_var
            breakp = result_double.beta[4]
        else:
            redchi_double = np.inf
            breakp = None

        # If neither fit converged, there is no valid result.
        if not cut_valid and not double_valid:
            return None

        if redchi_double <= redchi_cut:
            if not double_valid:
                if cut_valid and cut >= emin and cut <= emax:
                    return ["cut", redchi_cut, result_cut]
                return None

            if breakp < emin or breakp > emax:
                if cut_valid and cut >= emin and cut <= emax:
                    return ["cut", redchi_cut, result_cut]
                return None

            if breakp >= emin and breakp <= emax:
                return ["double", redchi_double, result_double]

        if redchi_double > redchi_cut:
            if not cut_valid:
                if double_valid and breakp >= emin and breakp <= emax:
                    return ["double", redchi_double, result_double]
                return None

            if cut < emin or cut > emax:
                if double_valid and breakp >= emin and breakp <= emax:
                    return ["double", redchi_double, result_double]
                return None

            if cut >= emin and cut <= emax:
                return ["cut", redchi_cut, result_cut]


    # ------------------------------------------------------------------
    # Explicit single PL + exponential cutoff fit
    # ------------------------------------------------------------------
    if fit == "cut":
        result_cut = pl_fit.cut_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
            E_cut=E_cut,
            exponent=exponent,
            maxit=maxit,
            )

        # Check ODR convergence BEFORE accessing result parameters.
        if not _is_valid_fit(result_cut):
            fit = "single"
        else:
            redchi_cut = result_cut.res_var

            # Should maybe make distinction between cut from cut PL
            # and cut from cut double PL.
            cut = result_cut.beta[2]

            if cut < emin or cut > emax:
                fit = "single"

            if cut >= emin and cut <= emax:
                return [
                    "cut",
                    redchi_cut,
                    result_cut
                ]


    # ------------------------------------------------------------------
    # Explicit double power-law fit
    # ------------------------------------------------------------------
    if fit == "double":
        result_double = pl_fit.double_pl_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            gamma2=gamma2,
            c1=c1,
            alpha=alpha,
            E_break=E_break_low,
            maxit=maxit,
            )

        # Check ODR convergence BEFORE accessing result parameters.
        if not _is_valid_fit(result_double):
            fit = "single"
        else:
            redchi_double = result_double.res_var
            breakp = result_double.beta[4]

            if breakp < emin or breakp > emax:
                fit = "single"

            if breakp >= emin and breakp <= emax:
                return ["double", redchi_double, result_double]


    # ------------------------------------------------------------------
    # Single power-law
    # ------------------------------------------------------------------
    if fit == "single":
        result_single_pl = pl_fit.power_law_fit(
            x=spec_e,
            y=spec_flux,
            xerr=e_err,
            yerr=flux_err,
            gamma1=gamma1,
            c1=c1,
        )

        # Check ODR convergence.
        if not _is_valid_fit(result_single_pl):
            return None

        redchi_single = result_single_pl.res_var

        return ["single", redchi_single, result_single_pl]



def _get_fit_errors(result):
    """
    Return the 95% confidence-interval-scaled ODR parameter errors.

    This preserves the original behavior:
        errors = student_t * result.sd_beta
    """
    dof = result.yest.size - len(result.beta)

    t_val = studentt.interval(0.95, dof)[1]
    return t_val * result.sd_beta


def _update_result_dataframe(
    result_dataframe,
    *,
    redchi,
    c1,
    c1_err,
    gamma1,
    gamma1_err,
    gamma2=None,
    gamma2_err=None,
    gamma3=None,
    gamma3_err=None,
    breakp_1=None,
    breakp_1_err=None,
    breakp_2=None,
    breakp_2_err=None,
    cutoff=None,
    cutoff_err=None,
    alpha=None,
    beta=None,
    exponent=None,
    e_min=None,
    e_max=None,
    ):
    """
    Store the final reported/physical fit parameters in result_dataframe.

    IMPORTANT:
    The gamma values passed here are the reordered/reporting convention,
    not necessarily result.beta[1:4].
    """

    result_dataframe["Reduced chi sq"] = redchi

    result_dataframe["c1"] = c1
    result_dataframe["c1 err"] = c1_err

    result_dataframe["Gamma1"] = gamma1
    result_dataframe["Gamma1 err"] = gamma1_err

    result_dataframe["Gamma2"] = gamma2
    result_dataframe["Gamma2 err"] = gamma2_err

    result_dataframe["Gamma3"] = gamma3
    result_dataframe["Gamma3 err"] = gamma3_err

    result_dataframe["Break point 1 [MeV]"] = breakp_1
    result_dataframe["Break point 1 err [MeV]"] = breakp_1_err

    result_dataframe["Break point 2 [MeV]"] = breakp_2
    result_dataframe["Break point 2 err [MeV]"] = breakp_2_err

    result_dataframe["Exponential cutoff point [MeV]"] = cutoff
    result_dataframe["Cutoff err [MeV]"] = cutoff_err

    result_dataframe["Alpha"] = alpha
    result_dataframe["Beta"] = beta
    result_dataframe["Exponent"] = exponent

    result_dataframe["E min [MeV]"] = e_min
    result_dataframe["E max [MeV]"] = e_max


def _plot_energy_marker(
    ax,
    energy,
    error,
    *,
    color,
    linestyle="--",
    label_prefix="E_b=",
    ):
    """
    Plot an energy marker in MeV or keV.

    The original code chose the unit according to whether the energy
    was below or above 1 MeV. This helper preserves that behavior.
    """

    if energy >= 1:
        value = energy
        value_err = error
        unit = "MeV"
    else:
        value = energy * 1e3
        value_err = error * 1e3
        unit = "keV"

    ax.axvline(x=energy,
                color=color,
                linestyle=linestyle,
                label=(
                    rf"$\mathregular{{{label_prefix}}}$ "
                    + f"{value:.1f}"
                    + "\n"
                    + r"$\pm$"
                    + f"{value_err:.1f}"
                    + f" {unit}"
                ),
                )


def _reorder_double_gammas(
    raw_gamma1,
    raw_gamma2,
    raw_gamma1_err,
    raw_gamma2_err,
    alpha,
    ):
    """
    Convert the raw mathematical-function gamma parameters into the
    physical/plot-order Gamma1 and Gamma2 convention.

    IMPORTANT:
    raw_gamma1/raw_gamma2 refer to the parameters in result.beta.
    They do NOT necessarily correspond to the first/second spectral
    power law in the plotted spectrum.

    For a double power law, the correspondence depends on alpha.
    """

    if alpha > 0:
        gamma1 = raw_gamma1
        gamma1_err = raw_gamma1_err

        gamma2 = raw_gamma2
        gamma2_err = raw_gamma2_err

    else:
        gamma1 = raw_gamma2
        gamma1_err = raw_gamma2_err

        gamma2 = raw_gamma1
        gamma2_err = raw_gamma1_err

    return (gamma1, gamma1_err, gamma2, gamma2_err)


def _reorder_triple_gammas(
    raw_gamma1,
    raw_gamma2,
    raw_gamma3,
    raw_gamma1_err,
    raw_gamma2_err,
    raw_gamma3_err,
    alpha,
    beta,
    ):
    """
    Convert the raw ODR gamma parameters into the physical/plot-order
    Gamma1, Gamma2 and Gamma3 convention.

    IMPORTANT
    ---------
    The triple convolution can exchange the spectral components depending
    on the signs of alpha and beta.

    Therefore:

        result.beta[1] -> raw_gamma1
        result.beta[2] -> raw_gamma2
        result.beta[3] -> raw_gamma3

    are the gamma parameters of the MATHEMATICAL FUNCTION, whereas
    Gamma1/Gamma2/Gamma3 returned here are the physical/plot-order
    spectral indices.

    The plotted function must therefore continue to use:

        pl_fit.triple_pl_func(result.beta, xplot)

    and NOT the reordered Gamma1/Gamma2/Gamma3.
    """

    if alpha > 0 and beta > 0:

        gamma1 = raw_gamma1
        gamma1_err = raw_gamma1_err

        gamma2 = raw_gamma2
        gamma2_err = raw_gamma2_err

        gamma3 = raw_gamma3
        gamma3_err = raw_gamma3_err

    elif alpha < 0 and beta > 0:

        gamma1 = raw_gamma2
        gamma1_err = raw_gamma2_err

        gamma2 = raw_gamma1
        gamma2_err = raw_gamma1_err

        gamma3 = raw_gamma3
        gamma3_err = raw_gamma3_err

    elif beta < 0 and alpha > 0:

        gamma1 = raw_gamma1
        gamma1_err = raw_gamma1_err

        gamma2 = raw_gamma3
        gamma2_err = raw_gamma3_err

        gamma3 = raw_gamma2
        gamma3_err = raw_gamma2_err

    elif alpha < 0 and beta < 0:

        gamma1 = raw_gamma3
        gamma1_err = raw_gamma3_err

        gamma2 = raw_gamma2
        gamma2_err = raw_gamma2_err

        gamma3 = raw_gamma1
        gamma3_err = raw_gamma1_err

    else:
        raise ValueError(
            "alpha and beta cannot be zero when determining "
            "the triple-power-law gamma ordering.")

    return (gamma1, gamma1_err, gamma2, 
            gamma2_err, gamma3, gamma3_err)


def _apply_triple_sign_convention(gamma1, gamma2, gamma3, alpha, beta, ):
    """
    Preserve the existing alpha/beta sign convention used by the code.

    This function ONLY changes the reported alpha/beta convention.

    It does NOT modify the raw ODR parameter vector and therefore does
    not modify the plotted mathematical function.
    """

    if gamma1 > gamma2 and gamma2 > gamma3:

        if alpha < 0 and beta > 0:
            alpha = abs(alpha)

        if alpha > 0 and beta < 0:
            beta = abs(beta)

        if alpha < 0 and beta < 0:
            alpha = abs(alpha)
            beta = abs(beta)

    elif gamma1 > gamma2 and gamma2 < gamma3:

        if alpha > 0 and beta > 0:
            beta = -abs(beta)

        if alpha < 0 and beta > 0:
            a = alpha
            b = beta
            alpha = b
            beta = a

        if alpha < 0 and beta < 0:
            alpha = abs(alpha)

    elif gamma1 < gamma2 and gamma2 > gamma3:

        if alpha > 0 and beta > 0:
            alpha = -abs(alpha)

        if alpha > 0 and beta < 0:
            a = alpha
            b = beta
            alpha = b
            beta = a

        if alpha < 0 and beta < 0:
            beta = -abs(beta)

    return alpha, beta


def _is_valid_fit(result):
    """Check whether an ODR result exists and reports successful convergence."""
    if result is None:
        return False

    return pl_fit.check_odr_output(result)


def _make_result_dataframe(which_fit_final):
    """Create the one-row result DataFrame."""
    return pd.DataFrame({"Final fit type": which_fit_final},
        index=[0],
        )

def MAKE_THE_FIT(
    spec_e,
    spec_flux,
    e_err,
    flux_err,
    ax,
    direction="sun",
    which_fit="best",
    e_min=None,
    e_max=None,
    g1_guess=-2.0,
    g2_guess=None,
    g3_guess=None,
    alpha_guess=5.0,
    beta_guess=5,
    break_low_guess=0.065,
    break_high_guess=0.12,
    cut_guess=0.12,
    c1_guess=None,
    exponent_guess=2,
    use_random=False,
    iterations=10,
    maxit=10000,
    path=None,
    path2=None,
    detailed_legend=False,
    ):
    """
    Fit the data to a single, double, triple, cutoff, or
    double-power-law + cutoff model.

    The mathematical ODR parameters and the reported physical parameters
    are deliberately kept separate.

    In particular, for the double and triple models:

        result_final.beta[1:4]

    are the raw gamma parameters of the mathematical convolution.

    The reported Gamma1/Gamma2/Gamma3 values are reordered according
    to alpha and beta so that they correspond to the physical/plot-order
    spectral components.

    The plotted curve ALWAYS uses the raw ODR parameter vector.
    """

    # ========================================================================
    # 1. Prepare guesses
    # ========================================================================

    if g2_guess is None:
        g2_guess = g1_guess - 0.1

    if g3_guess is None:
        g3_guess = g2_guess - 0.1

    spec_e = np.asarray(spec_e)
    spec_flux = np.asarray(spec_flux)
    e_err = np.asarray(e_err)
    flux_err = np.asarray(flux_err)

    if e_min is None:
        e_min = spec_e[0]

    if e_max is None:
        e_max = spec_e[-1]

    if c1_guess is None:
        absolute_val_array = np.abs(spec_e - 1)
        smallest_difference_index = absolute_val_array.argmin()
        c1_guess = spec_flux[smallest_difference_index]

    # ========================================================================
    # 2. Generate random-search arrays
    # ========================================================================

    g1_start_value = -np.abs(g1_guess) * 10.0
    g2_start_value = -np.abs(g2_guess) * 10.0
    g3_start_value = -np.abs(g3_guess) * 10.0

    g1_end_value = np.abs(g1_guess) * 10.0
    g2_end_value = np.abs(g2_guess) * 10.0
    g3_end_value = np.abs(g3_guess) * 10.0

    g1_step = np.abs(g1_guess / 4.0)
    g2_step = np.abs(g2_guess / 4.0)
    g3_step = np.abs(g3_guess / 4.0)

    if use_random:

        gamma1_array = closest_values(
            np.arange(g1_start_value, g1_end_value, g1_step),
            g1_guess)

        gamma2_array = closest_values(
            np.arange(g2_start_value, g2_end_value, g2_step),
            g2_guess)

        gamma3_array = closest_values(
            np.arange(g3_start_value, g3_end_value, g3_step),
            g3_guess)

        c1_array = np.arange(
            c1_guess / 100.0,
            c1_guess * 100.0,
            c1_guess / 500.0)

        a1_array = np.arange(0.01, 0.1, 0.01)
        a2_array = np.arange(0.1, 1.0, 0.05)
        a3_array = np.arange(1, 10, 0.5)
        a4_array = np.arange(10, 100, 10)
        a5_array = np.arange(100, 220, 20)

        alpha_array = np.hstack((a1_array, a2_array, a3_array, a4_array, a5_array))
        alpha_array = closest_values(alpha_array, alpha_guess)

        beta_array = np.hstack((a1_array, a2_array, a3_array, a4_array, a5_array))
        beta_array = closest_values(beta_array, beta_guess)

        if e_max < 0.1:

            break_array_low = np.arange( e_min, e_max, 0.001)

        elif e_max < 1.0:

            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, e_max, 0.005)

            break_array_low = np.hstack((b1_array, b2_array))

        elif e_max < 10:

            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, 1, 0.005)
            b3_array = np.arange(1, e_max, 0.01)

            break_array_low = np.hstack((b1_array, b2_array, b3_array))

        else:

            b1_array = np.arange(e_min, 0.1, 0.001)
            b2_array = np.arange(0.1, 1, 0.005)
            b3_array = np.arange(1, 10, 0.01)
            b4_array = np.arange(10, e_max, 1)

            break_array_low = np.hstack((b1_array, b2_array, b3_array, b4_array))

        break_array_high = break_array_low[1:]
        cut_array = break_array_low[1:]

        break_array_low = closest_values(break_array_low, break_low_guess)

        break_array_high = closest_values(break_array_high, break_high_guess)

        cut_array = closest_values(cut_array, cut_guess)

    # ========================================================================
    # 3. Plot/data preparation
    # ========================================================================

    color = {"sun": "crimson", "asun": "orange", "north": "darkslateblue", "south": "c"}

    xplot = np.logspace(
        np.log10(np.nanmin(spec_e)),
        np.log10(np.nanmax(spec_e)),
        num=500,
        )

    xplot = xplot[
        np.where(
            (xplot >= e_min)
            & (xplot <= e_max)
        )[0]]

    fit_ind = np.where(
        (spec_e >= e_min)
        & (spec_e <= e_max)
        & np.isfinite(spec_flux)
        & np.isfinite(flux_err)
        )[0]

    spec_e = spec_e[fit_ind]
    spec_flux = spec_flux[fit_ind]
    e_err = e_err[fit_ind]
    flux_err = flux_err[fit_ind]

    # ========================================================================
    # 4. Variables used to select the final fit
    # ========================================================================

    which_fit_final = ""
    redchi_final = np.inf
    result_final = None

    # ========================================================================
    # 5. BEST
    # ========================================================================

    if which_fit == "best":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            beta=beta_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            gamma3=g3_guess,
            E_break_low=break_low_guess,
            E_break_high=break_high_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="best",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                # Keep [0]
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                g3_random = np.random.choice(gamma3_array, 1)[0]
                alpha_random = np.random.choice(alpha_array, 1)[0]
                beta_random = np.random.choice(beta_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                break_high_random = np.random.choice(break_array_high, 1)[0]

                if break_high_random < break_low_random:

                    b = break_low_random
                    break_low_random = break_high_random
                    break_high_random = b

                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    beta=beta_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    gamma3=g3_random,
                    E_break_low=break_low_random,
                    E_break_high=break_high_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]
                #convergence is checked by check_redchi
                #convergence = _is_valid_fit(result_random)

                if (redchi_random < redchi_final):

                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 6. TRIPLE
    # ========================================================================

    if which_fit == "triple":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            beta=beta_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            gamma3=g3_guess,
            E_break_low=break_low_guess,
            E_break_high=break_high_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="triple",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):
                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                g3_random = np.random.choice(gamma3_array, 1)[0]
                alpha_random = np.random.choice(alpha_array, 1)[0]
                beta_random = np.random.choice(beta_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                break_high_random = np.random.choice(break_array_high, 1)[0]

                if break_high_random < break_low_random:
                    b = break_low_random
                    break_low_random = break_high_random
                    break_high_random = b

                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    beta=beta_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    gamma3=g3_random,
                    E_break_low=break_low_random,
                    E_break_high=break_high_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="triple",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                
                if (redchi_random < redchi_final):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 7. BEST_CB
    # ========================================================================

    if which_fit == "best_cb":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="best_cb",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="best_cb",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                
                if (redchi_random < redchi_final):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 8. BEST_SD
    # ========================================================================

    if which_fit == "best_sd":

        which_fit_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            fit="best_sd",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )

        if which_fit_guess is not None:

            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    fit="best_sd",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                if (redchi_random < redchi_final):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 9. DOUBLE_CUT
    # ========================================================================

    if which_fit == "double_cut":

        result_cut_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="double_cut",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )
        
        if result_cut_guess is not None:
            redchi_final = which_fit_guess[1]
            which_fit_final = which_fit_guess[0]
            result_final = which_fit_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="double_cut",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                
                if (redchi_random < redchi_final):
                    result_final = result_random
                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]

    # ========================================================================
    # 10. DOUBLE
    # ========================================================================

    if which_fit == "double":

        result_double_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            alpha=alpha_guess,
            gamma1=g1_guess,
            gamma2=g2_guess,
            E_break_low=break_low_guess,
            exponent=exponent_guess,
            fit="double",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
        )

        if result_double_guess is not None:
            redchi_final = result_double_guess[1]
            which_fit_final = result_double_guess[0]
            result_final = result_double_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(gamma1_array, 1)[0]
                g2_random = np.random.choice(gamma2_array, 1)[0]
                alpha_random = np.random.choice(alpha_array, 1)[0]
                break_low_random = np.random.choice(break_array_low, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    alpha=alpha_random,
                    gamma1=g1_random,
                    gamma2=g2_random,
                    E_break_low=break_low_random,
                    exponent=exponent_guess,
                    fit="double",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                if (redchi_random < redchi_final):
                    which_fit_final = which_fit_random[0]
                    redchi_final = redchi_random
                    result_final = result_random

    # ========================================================================
    # 11. CUT
    # ========================================================================

    if which_fit == "cut":

        result_cut_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            gamma1=g1_guess,
            E_cut=cut_guess,
            exponent=exponent_guess,
            fit="cut",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )

        if result_cut_guess is not None:
            redchi_final = result_cut_guess[1]
            which_fit_final = result_cut_guess[0]
            result_final = result_cut_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(gamma1_array, 1)[0]
                cut_random = np.random.choice(cut_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    gamma1=g1_random,
                    E_cut=cut_random,
                    exponent=exponent_guess,
                    fit="cut",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                    )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                if (redchi_random < redchi_final):
                    which_fit_final = which_fit_random[0]
                    redchi_final = redchi_random
                    result_final = result_random

                    
    # ========================================================================
    # 12. SINGLE
    # ========================================================================

    if which_fit == "single":

        result_single_guess = check_redchi(
            spec_e,
            spec_flux,
            e_err,
            flux_err,
            c1=c1_guess,
            gamma1=g1_guess,
            fit="single",
            maxit=maxit,
            e_min=e_min,
            e_max=e_max,
            )

        if result_single_guess is not None:
            redchi_final = result_single_guess[1]
            which_fit_final = result_single_guess[0]
            result_final = result_single_guess[2]

        if use_random:

            for i in range(iterations):

                g1_random = np.random.choice(gamma1_array, 1)[0]
                c1_random = np.random.choice(c1_array, 1)[0]

                which_fit_random = check_redchi(
                    spec_e,
                    spec_flux,
                    e_err,
                    flux_err,
                    c1=c1_random,
                    gamma1=g1_random,
                    fit="single",
                    maxit=maxit,
                    e_min=e_min,
                    e_max=e_max,
                )

                if which_fit_random is None:
                    continue

                redchi_random = which_fit_random[1]
                result_random = which_fit_random[2]

                if redchi_random < redchi_final:

                    redchi_final = redchi_random
                    which_fit_final = which_fit_random[0]
                    result_final = result_random
                    
    # ========================================================================
    # 13. Check that something was actually fitted
    # ========================================================================

    if result_final is None:

        raise RuntimeError("No valid fit was obtained for "
            f"which_fit='{which_fit}'.")

    result = result_final

    result_dataframe = _make_result_dataframe(which_fit_final)

    # ========================================================================
    # 14. SINGLE RESULT
    # ========================================================================

    if which_fit_final == "single":

        result_single_pl = result_final
        redchi_single = result_single_pl.res_var

        c1 = result_single_pl.beta[0]
        gamma1 = result_single_pl.beta[1]

        errors = _get_fit_errors(result_single_pl)

        gamma1_err = errors[1]

        if detailed_legend:

            ax.plot([], [], " ", 
                    label="Single pl")

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_single:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
                )

        fit_plot = pl_fit.simple_pl([c1, gamma1], xplot)

        ax.plot(
            xplot,
            fit_plot,
            "-",
            color=color[direction],
            label=(
                r"$\mathregular{\gamma=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
            ),
            )

        ax.plot(
            xplot,
            fit_plot,
            "--k",
            zorder=10,
            )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_single,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            e_min=e_min,
            e_max=e_max,
            )

    # ========================================================================
    # 15. DOUBLE RESULT
    # ========================================================================

    elif which_fit_final == "double":

        result_double = result_final

        breakp_1 = result_double.beta[4]
        alpha = result_double.beta[3]
        c1 = result_double.beta[0]

        redchi_double = result_double.res_var

        errors = _get_fit_errors(result_double)

        breakp_1_err = errors[4]

        # ------------------------------------------------------------
        # Raw mathematical gamma parameters.
        #
        # These are NOT necessarily the first and second physical
        # power-law segments in the plotted spectrum.
        # ------------------------------------------------------------

        raw_gamma1 = result_double.beta[1]
        raw_gamma2 = result_double.beta[2]

        raw_gamma1_err = errors[1]
        raw_gamma2_err = errors[2]

        
        (   gamma1,
            gamma1_err,
            gamma2,
            gamma2_err,
        ) = _reorder_double_gammas(
            raw_gamma1,
            raw_gamma2,
            raw_gamma1_err,
            raw_gamma2_err,
            alpha,
        )

        # Preserve the existing alpha convention.
        if gamma1 < gamma2 and alpha > 0:
            alpha = -abs(alpha)

        elif gamma2 < gamma1 and alpha < 0:
            alpha = abs(alpha)

        # IMPORTANT:
        # The plot uses the ORIGINAL ODR parameter vector.
        fit_plot = pl_fit.double_pl_func(
            result_double.beta,
            xplot,
            )

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot([], [], " ",
                    label="Broken pl")

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_double:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
                )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_2=}$"
                f"{gamma2:.2f}"
                r"$\pm$"
                f"{gamma2_err:.2f}"
                "\n"
                r"$\mathregular{\alpha=}$"
                f"{alpha:.2f}"
            ),
            )   

        _plot_energy_marker(
            ax,
            breakp_1,
            breakp_1_err,
            color="blue",
            label_prefix="E_b=",
            )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_double,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            gamma2=gamma2,
            gamma2_err=gamma2_err,
            breakp_1=breakp_1,
            breakp_1_err=breakp_1_err,
            alpha=alpha,
            e_min=e_min,
            e_max=e_max,
            )

    # ========================================================================
    # 16. CUT RESULT
    # ========================================================================

    elif which_fit_final == "cut":

        result_cut = result_final
        cut = result_cut.beta[2]

        redchi_cut = result_cut.res_var

        errors = _get_fit_errors(result_cut)

        c1 = result_cut.beta[0]
        gamma1 = result_cut.beta[1]

        gamma1_err = errors[1]
        cut_err = errors[2]
        exponent = result_cut.beta[3]

        fit_plot = pl_fit.cut_pl_func(result_cut.beta, xplot)

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot([], [], " ",
                label="Single pl + exp cutoff",
                )

            ax.plot([], [], " ",
                label=(
                    "exponent: "
                    f"{exponent:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_cut:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
                )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
            ),
            )

        _plot_energy_marker(
            ax,
            cut,
            cut_err,
            color="purple",
            label_prefix="E_c=",
            )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_cut,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            cutoff=cut,
            cutoff_err=cut_err,
            exponent=exponent,
            e_min=e_min,
            e_max=e_max,
            )

    # ========================================================================
    # 17. DOUBLE + CUTOFF RESULT
    # ========================================================================

    elif which_fit_final == "double_cut":

        result_cut = result_final

        breakp_1 = result_cut.beta[4]
        cut = result_cut.beta[5]
        alpha = result_cut.beta[3]

        redchi_cut = result_cut.res_var

        errors = _get_fit_errors(result_cut)

        breakp_1_err = errors[4]
        cut_err = errors[5]

        c1 = result_cut.beta[0]
        exponent = result_cut.beta[6]

        # ------------------------------------------------------------
        # Raw mathematical gamma parameters.
        # ------------------------------------------------------------

        raw_gamma1 = result_cut.beta[1]
        raw_gamma2 = result_cut.beta[2]

        raw_gamma1_err = errors[1]
        raw_gamma2_err = errors[2]

        (   gamma1,
            gamma1_err,
            gamma2,
            gamma2_err,
        ) = _reorder_double_gammas(
            raw_gamma1,
            raw_gamma2,
            raw_gamma1_err,
            raw_gamma2_err,
            alpha,
        )

        # Preserve existing alpha sign convention.
        if gamma1 < gamma2 and alpha > 0:
            alpha = -abs(alpha)

        elif gamma2 < gamma1 and alpha < 0:
            alpha = abs(alpha)

        # IMPORTANT:
        # Use raw ODR parameters for the mathematical function.
        fit_plot = pl_fit.cut_break_pl_func(
            result_cut.beta,
            xplot,
            )

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot([], [], " ",
                label="Broken pl + exp cutoff",
                )

            ax.plot([], [], " ",
                label=(
                    "exponent: "
                    f"{exponent:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_cut:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
                )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_2=}$"
                f"{gamma2:.2f}"
                r"$\pm$"
                f"{gamma2_err:.2f}"
                "\n"
                r"$\mathregular{\alpha=}$"
                f"{alpha:.2f}"
            ),
            )

        _plot_energy_marker(
            ax,
            breakp_1,
            breakp_1_err,
            color="blue",
            label_prefix="E_b=",
            )

        _plot_energy_marker(
            ax,
            cut,
            cut_err,
            color="purple",
            label_prefix="E_c=",
            )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_cut,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            gamma2=gamma2,
            gamma2_err=gamma2_err,
            breakp_1=breakp_1,
            breakp_1_err=breakp_1_err,
            cutoff=cut,
            cutoff_err=cut_err,
            alpha=alpha,
            exponent=exponent,
            e_min=e_min,
            e_max=e_max,
            )

    # ========================================================================
    # 18. TRIPLE RESULT
    # ========================================================================

    elif which_fit_final == "triple":

        result_triple = result_final

        breakp_1 = result_triple.beta[6]
        breakp_2 = result_triple.beta[7]

        alpha = result_triple.beta[4]
        beta = result_triple.beta[5]

        redchi_triple = result_triple.res_var

        errors = _get_fit_errors(result_triple)

        breakp_1_err = errors[6]
        breakp_2_err = errors[7]

        c1 = result_triple.beta[0]

        # ================================================================
        # RAW MATHEMATICAL GAMMAS
        # ================================================================
        #
        # These are the parameters that belong to the mathematical
        # convolution function.
        #
        # They are deliberately NOT assumed to be the physical spectral
        # Gamma1/Gamma2/Gamma3 ordering.
        #
        # The plotted curve below uses these parameters directly through
        #
        #     pl_fit.triple_pl_func(result_triple.beta, xplot)
        #
        # ================================================================

        raw_gamma1 = result_triple.beta[1]
        raw_gamma2 = result_triple.beta[2]
        raw_gamma3 = result_triple.beta[3]

        raw_gamma1_err = errors[1]
        raw_gamma2_err = errors[2]
        raw_gamma3_err = errors[3]

        # ================================================================
        # PHYSICAL / PLOT-ORDER GAMMAS
        # ================================================================
        #
        # Because alpha and beta can be positive or negative, the
        # convolution can exchange which raw mathematical gamma appears
        # as the first, second, or third physical power-law component.
        #
        # Therefore the reported Gamma1/Gamma2/Gamma3 are reordered here.
        #
        # THIS DOES NOT CHANGE THE FUNCTION BEING PLOTTED.
        # ================================================================

        (   gamma1,
            gamma1_err,
            gamma2,
            gamma2_err,
            gamma3,
            gamma3_err,
        ) = _reorder_triple_gammas(
            raw_gamma1,
            raw_gamma2,
            raw_gamma3,
            raw_gamma1_err,
            raw_gamma2_err,
            raw_gamma3_err,
            alpha,
            beta,
        )

        # ================================================================
        # Reported alpha/beta convention
        # ================================================================

        alpha, beta = _apply_triple_sign_convention(
            gamma1,
            gamma2,
            gamma3,
            alpha,
            beta,
            )

        # ================================================================
        # PLOT THE ACTUAL MATHEMATICAL FUNCTION
        # ================================================================
        #
        # DO NOT replace result_triple.beta by gamma1/gamma2/gamma3.
        #
        # The plotted function is the actual ODR mathematical model.
        # ================================================================

        fit_plot = pl_fit.triple_pl_func(result_triple.beta, xplot,)

        fit_plot[fit_plot == 0] = np.nan

        if detailed_legend:

            ax.plot([], [], " ", label="Triple pl",
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{\chi²=}$"
                    f"{redchi_triple:.2f}"
                ),
                )

            ax.plot([], [], " ",
                label=(
                    r"$\mathregular{I_0=}$"
                    + f"{c1:.2e}"
                    + "/(s cm² sr MeV)"
                ),
                )

        ax.plot(
            xplot,
            fit_plot,
            "-b",
            label=(
                r"$\mathregular{\gamma_1=}$"
                f"{gamma1:.2f}"
                r"$\pm$"
                f"{gamma1_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_2=}$"
                f"{gamma2:.2f}"
                r"$\pm$"
                f"{gamma2_err:.2f}"
                "\n"
                r"$\mathregular{\gamma_3=}$"
                f"{gamma3:.2f}"
                r"$\pm$"
                f"{gamma3_err:.2f}"
                "\n"
                r"$\mathregular{\alpha=}$"
                f"{alpha:.2f}"
                "\n"
                r"$\mathregular{\beta=}$"
                f"{beta:.2f}"
            ),
            )

        _plot_energy_marker(
            ax,
            breakp_1,
            breakp_1_err,
            color="blue",
            label_prefix="E_b1=",
            )

        _plot_energy_marker(
            ax,
            breakp_2,
            breakp_2_err,
            color="purple",
            label_prefix="E_b2=",
            )

        _update_result_dataframe(
            result_dataframe,
            redchi=redchi_triple,
            c1=c1,
            c1_err=errors[0],
            gamma1=gamma1,
            gamma1_err=gamma1_err,
            gamma2=gamma2,
            gamma2_err=gamma2_err,
            gamma3=gamma3,
            gamma3_err=gamma3_err,
            breakp_1=breakp_1,
            breakp_1_err=breakp_1_err,
            breakp_2=breakp_2,
            breakp_2_err=breakp_2_err,
            alpha=alpha,
            beta=beta,
            e_min=e_min,
            e_max=e_max,
            )

    # ========================================================================
    # 19. Save results
    # ========================================================================

    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(result, f)

    if path2 is not None:
        result_dataframe.to_csv(path2, sep=";")

    return result





