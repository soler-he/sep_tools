import numpy as np
from odrpack import odr_fit

def check_odr_output(result, printing=False):
    """Return True only for clean ODR convergence."""

    # These are the ONLY acceptable ODRPACK return codes.
    converged = result.info in (1, 2, 3)

    if printing:
        if converged:
            print("Fit converged successfully.")
        else:
            print(f"Fit rejected: {result.stopreason}")
            print(f"ODR info code: {result.info}")

    return converged


def simple_pl(x, p, E_0):
    """Evaluate a simple power-law model.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(I_0, gamma1)``, where ``I_0`` is the
        normalization at ``x = E_0`` and ``gamma1`` is the power-law
        index.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value calculated as ``I_0 * (x / E_0) ** gamma1``.
    """
    I_0, gamma1 = p
    return I_0 * (x / E_0) ** gamma1


def power_law_fit(x, y, xerr, yerr, gamma1=-1.8, I_0=None, E_0 = 0.1, print_report=False):
    """Fit a power-law model to data using odrpack.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the power-law index.
    I_0 : float or None, default=None
        Initial guess for the normalization. If ``None``, the last
        value of ``y`` is used.
    print_report : bool, default=False
        If ``True``, print the ODR fit report.

    Returns
    -------
    OdrResult
        The result returned by the ODR fit.
    """
    I_0 = y[-1] if I_0 is None else I_0

    # odrpack expects the model signature f(x, beta), whereas
    # simple_pl uses the original f(beta, x) signature.
    #def plmodel(x, beta):
    #    return simple_pl(beta, x)

    
    # Convert standard deviations to ODR weights.
    weight_x = 1 / np.asarray(xerr) ** 2
    weight_y = 1 / np.asarray(yerr) ** 2

    beta0=[I_0, gamma1]
    plmodel = lambda x, p: simple_pl(x, p, E_0)

    result = odr_fit(
        plmodel,
        x,
        y,
        beta0=beta0,
        weight_x=weight_x,
        weight_y=weight_y,
        report="short" if print_report else "none",
    )

    convergence = check_odr_output(result)
    if not convergence:
        result = odr_fit(
            plmodel,
            x,
            y,
            beta0=beta0,
            weight_x=weight_x,
            weight_y=weight_y,
            report="short" if print_report else "none",
        )
        convergence = check_odr_output(result)

    return result

def double_pl_func(x, p, E_0):
    """Evaluate a smoothly broken double power-law model.

    This is based on function 25 from Prinsloo (2019), without the
    exponential roll-over.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(I_0, gamma1, gamma2, alpha, E_break)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    I_0, gamma1, gamma2, alpha, E_break = p

    y = (I_0 * (x / E_0) ** gamma1 
         * ((x**alpha + E_break**alpha)
         / (E_0**alpha + E_break**alpha)) ** ((gamma2 - gamma1) / alpha))

    return y

def double_pl_fit(
    x,
    y,
    xerr,
    yerr,
    gamma1=-1.8,
    gamma2=-2,
    I_0=None,
    E_0 = 0.1,
    alpha=None,
    E_break=0.1,
    print_report=False,
    maxit=200,
):
    """Fit a smoothly broken double power-law model using odrpack.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the first power-law index.
    gamma2 : float, default=-2
        Initial guess for the second power-law index.
    I_0 : float or None, default=None
        Initial guess for the normalization. If ``None``, the fourth
        value of ``y`` is used.
    alpha : float or None, default=None
        Initial guess for the smoothness parameter. If ``None``,
        ``0.1`` is used.
    E_break : float, default=0.1
        Initial guess for the break energy.
    print_report : bool, default=False
        If ``True``, print the ODR fit report.
    maxit : int, default=200
        Maximum number of ODR iterations.

    Returns
    -------
    OdrResult
        The result returned by the ODR fit.
    """
    I_0 = y[3] if I_0 is None else I_0
    alpha = 0.1 if alpha is None else alpha

    # odrpack expects the model signature f(x, beta), whereas
    # double_pl_func uses the original f(beta, x) signature.
    #def plmodel(x, beta):
    #    return double_pl_func(beta, x)

    # Convert standard deviations to ODR weights.
    weight_x = 1 / np.asarray(xerr) ** 2
    weight_y = 1 / np.asarray(yerr) ** 2
    

    beta0 = [I_0, gamma1, gamma2, alpha, E_break]
    plmodel = lambda x, p: double_pl_func(x, p, E_0)

    result = odr_fit(
        plmodel,
        x,
        y,
        beta0=beta0,
        weight_x=weight_x,
        weight_y=weight_y,
        maxit=maxit,
        report="short" if print_report else "none",
    )

    convergence = check_odr_output(result)
    if not convergence:
        result = odr_fit(
            plmodel,
            x,
            y,
            beta0=beta0,
            weight_x=weight_x,
            weight_y=weight_y,
            maxit=maxit,
            report="short" if print_report else "none",
        )
        convergence = check_odr_output(result)

    return result

def triple_pl_func(x, p, E_0):
    """Evaluate a triple power-law model.

    Parameters
    ----------
    p : sequence of float
        Model parameters
        ``(I_0, gamma1, gamma2, gamma3, alpha, beta,
        E_break_low, E_break_high)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    I_0, gamma1, gamma2, gamma3, alpha, beta, E_break_low, E_break_high = p

    y = (I_0
        * (x / E_0) ** gamma1
        * ((x**alpha + E_break_low**alpha)
            / (E_0**alpha + E_break_low**alpha)) ** ((gamma2 - gamma1) / alpha)
        * ((x**beta + E_break_high**beta)
            / (E_0**beta + E_break_high**beta)) ** ((gamma3 - gamma2) / beta))
    
    return y

def triple_pl_fit(
    x,
    y,
    xerr,
    yerr,
    gamma1=-1.8,
    gamma2=-2,
    gamma3=-3,
    I_0=None,
    E_0 = 0.1,
    alpha=None,
    beta=None,
    E_break_low=0.06,
    E_break_high=0.12,
    print_report=False,
    maxit=200,
):
    """Fit a smoothly broken triple power-law model using odrpack.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the first power-law index.
    gamma2 : float, default=-2
        Initial guess for the second power-law index.
    gamma3 : float, default=-3
        Initial guess for the third power-law index.
    I_0 : float or None, default=None
        Initial guess for the normalization. If ``None``, the fourth
        value of ``y`` is used.
    alpha : float or None, default=None
        Initial guess for the first smoothness parameter. If ``None``,
        ``0.1`` is used.
    beta : float or None, default=None
        Initial guess for the second smoothness parameter. If ``None``,
        ``0.1`` is used.
    E_break_low : float, default=0.06
        Initial guess for the lower break energy.
    E_break_high : float, default=0.12
        Initial guess for the upper break energy.
    print_report : bool, default=False
        If ``True``, print the ODR fit report.
    maxit : int, default=200
        Maximum number of ODR iterations.

    Returns
    -------
    OdrResult
        The result returned by the ODR fit.
    """
    I_0 = y[3] if I_0 is None else I_0
    alpha = 0.1 if alpha is None else alpha
    beta = 0.1 if beta is None else beta

    # odrpack expects the model signature f(x, beta), whereas
    # triple_pl_func uses the original f(beta, x) signature.
    #def plmodel(x, beta):
    #    return triple_pl_func(beta, x)

    # Convert standard deviations to ODR weights.
    weight_x = 1 / np.asarray(xerr) ** 2
    weight_y = 1 / np.asarray(yerr) ** 2

    beta0 = [I_0, gamma1, gamma2, gamma3, alpha, beta, E_break_low, E_break_high]
    plmodel = lambda x, p: triple_pl_func(x, p, E_0)

    result = odr_fit(
        plmodel,
        x,
        y,
        beta0=beta0,
        weight_x=weight_x,
        weight_y=weight_y,
        maxit=maxit,
        report="short" if print_report else "none",
    )

    convergence = check_odr_output(result)
    if not convergence:
        result = odr_fit(
            plmodel,
            x,
            y,
            beta0=beta0,
            weight_x=weight_x,
            weight_y=weight_y,
            maxit=maxit,
            report="short" if print_report else "none",
        )
        convergence = check_odr_output(result)

    return result


def cut_pl_func(x, p, E_0):
    """Evaluate a power law with an exponential cut-off.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(I_0, gamma1, E_cut, exponent)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    I_0, gamma1, E_cut, exponent = p

    y = I_0 * (x / E_0) ** gamma1 * np.exp(-(x / E_cut) ** exponent)

    return y


def cut_pl_fit(
    x,
    y,
    xerr,
    yerr,
    gamma1=-1.8,
    I_0=None,
    E_0 = 0.1,
    E_cut=0.35,
    exponent=2,
    print_report=False,
    maxit=200,
):
    """Fit a power law with an exponential cut-off using odrpack.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the power-law index.
    I_0 : float or None, default=None
        Initial guess for the normalization. If ``None``, the fifth
        value of ``y`` is used.
    E_cut : float, default=0.35
        Initial guess for the cut-off energy.
    exponent : float, default=2
        Initial guess for the exponent of the exponential cut-off.
    print_report : bool, default=False
        If ``True``, print the ODR fit report.
    maxit : int, default=200
        Maximum number of ODR iterations.

    Returns
    -------
    OdrResult
        The result returned by the ODR fit.
    """
    I_0 = y[4] if I_0 is None else I_0

    # odrpack expects the model signature f(x, beta), whereas
    # cut_pl_func uses the original f(beta, x) signature.
    #def plmodel(x, beta):
    #    return cut_pl_func(beta, x)

    # Convert standard deviations to ODR weights.
    weight_x = 1 / np.asarray(xerr) ** 2
    weight_y = 1 / np.asarray(yerr) ** 2

    beta0 = [I_0, gamma1, E_cut, exponent]
    plmodel = lambda x, p: cut_pl_func(x, p, E_0)

    result = odr_fit(
        plmodel,
        x,
        y,
        beta0=beta0,
        weight_x=weight_x,
        weight_y=weight_y,
        maxit=maxit,
        report="short" if print_report else "none",
    )

    convergence = check_odr_output(result)
    if not convergence:
        result = odr_fit(
            plmodel,
            x,
            y,
            beta0=beta0,
            weight_x=weight_x,
            weight_y=weight_y,
            maxit=maxit,
            report="short" if print_report else "none",
        )
        convergence = check_odr_output(result)

    return result

def cut_break_pl_func(x, p, E_0):
    """Evaluate a smoothly broken power law with an exponential cut-off.

    Parameters
    ----------
    p : sequence of float
        Model parameters
        ``(I_0, gamma1, gamma2, alpha, E_break, E_cut, exponent)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    I_0, gamma1, gamma2, alpha, E_break, E_cut, exponent = p

    y = (I_0
        * (x / E_0) ** gamma1
        * ((x**alpha + E_break**alpha)
            / (E_0**alpha + E_break**alpha)) ** ((gamma2 - gamma1) / alpha)
        * np.exp(-(x / E_cut) ** exponent))

    return y


def cut_break_pl_fit(
    x,
    y,
    xerr,
    yerr,
    gamma1=-1.8,
    gamma2=-2,
    I_0=None,
    E_0 = 0.1,
    alpha=None,
    E_break=0.1,
    E_cut=0.35,
    exponent=2,
    print_report=False,
    maxit=200,
):
    """Fit a smoothly broken power law with an exponential cut-off using odrpack.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the first power-law index.
    gamma2 : float, default=-2
        Initial guess for the second power-law index.
    I_0 : float or None, default=None
        Initial guess for the normalization. If ``None``, the fifth
        value of ``y`` is used.
    alpha : float or None, default=None
        Initial guess for the smoothness parameter. If ``None``,
        ``0.1`` is used.
    E_break : float, default=0.1
        Initial guess for the break energy.
    E_cut : float, default=0.35
        Initial guess for the cut-off energy.
    exponent : float, default=2
        Initial guess for the exponent of the exponential cut-off.
    print_report : bool, default=False
        If ``True``, print the ODR fit report.
    maxit : int, default=200
        Maximum number of ODR iterations.

    Returns
    -------
    OdrResult
        The result returned by the ODR fit.
    """
    I_0 = y[4] if I_0 is None else I_0
    alpha = 0.1 if alpha is None else alpha

    # odrpack expects the model signature f(x, beta), whereas
    # cut_break_pl_func uses the original f(beta, x) signature.
    #def plmodel(x, beta):
    #    return cut_break_pl_func(beta, x)

    # Convert standard deviations to ODR weights.
    weight_x = 1 / np.asarray(xerr) ** 2
    weight_y = 1 / np.asarray(yerr) ** 2

    beta0 = [I_0, gamma1, gamma2, alpha, E_break, E_cut, exponent]
    plmodel = lambda x, p: cut_break_pl_func(x, p, E_0)

    result = odr_fit(
        plmodel,
        x,
        y,
        beta0=beta0,
        weight_x=weight_x,
        weight_y=weight_y,
        maxit=maxit,
        report="short" if print_report else "none",
    )

    convergence = check_odr_output(result)
    if not convergence:
        result = odr_fit(
            plmodel,
            x,
            y,
            beta0=beta0,
            weight_x=weight_x,
            weight_y=weight_y,
            maxit=maxit,
            report="short" if print_report else "none",
        )
        convergence = check_odr_output(result)

    return result



