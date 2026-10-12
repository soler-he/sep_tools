import numpy as np
from scipy.odr import *



def check_odr_output(result, printing = False):
    """Check whether an ODR fit converged successfully.

    Parameters
    ----------
    result : scipy.odr.Output
        ODR fit result containing the ``stopreason`` attribute.

    Returns
    -------
    bool
        ``True`` if the fit converged according to one of the recognised
        ODR convergence messages, otherwise ``False``.
    """

    # Check for convergence.
    success_messages = ["Sum of squares convergence", "Parameter convergence", "Sum of squares and parameter convergence",]

    converged = any(message in result.stopreason[0] for message in success_messages)

    if not converged and printing:
        print(f"Fit failed or did not converge: {result.stopreason}")
        print("Re-running the fit...")
    elif converged and printing:
        print("Fit converged successfully.")

    return converged

def simple_pl(p, x):
    """Evaluate a simple power-law model.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(c1, gamma1)``, where ``c1`` is the
        normalization at ``x = 0.1`` and ``gamma1`` is the power-law
        index.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value calculated as ``c1 * (x / 0.1) ** gamma1``.
    """
    c1, gamma1 = p
    return c1 * (x / 0.1) ** gamma1


def power_law_fit(x, y, xerr, yerr, gamma1=-1.8, c1=None, print_report=False):
    """Fit a power-law model to data using scipy.odr.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the power-law index.
    c1 : float or None, default=None
        Initial guess for the normalization. If ``None``, the last
        value of ``y`` is used.
    print_report : bool, default=False
        If ``True``, print the ODR fit report.

    Returns
    -------
    scipy.odr.Output
        The result returned by the ODR fit.
    """
    c1 = y[-1] if c1 is None else c1

    plmodel = Model(simple_pl)
    data = RealData(x, y, sx=xerr, sy=yerr)

    # Set up ODR with the initial parameter guesses.
    odr = ODR(data, plmodel, beta0=[c1, gamma1])

    # Run the regression.
    result = odr.run()

    convergence = check_odr_output(result)
    if not convergence:
        result = odr.run()
        convergence = check_odr_output(result)

    if print_report:
        result.pprint()

    return result


def double_pl_func(p, x):
    """Evaluate a smoothly broken double power-law model.

    This is based on function 25 from Prinsloo (2019), without the
    exponential roll-over.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(c1, gamma1, gamma2, alpha, E_break)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    c1, gamma1, gamma2, alpha, E_break = p

    y = (c1 * (x / 0.1) ** gamma1 
         * ((x**alpha + E_break**alpha)
         / (0.1**alpha + E_break**alpha)) ** ((gamma2 - gamma1) / alpha))

    return y

def double_pl_fit(x, y, xerr, yerr, gamma1=-1.8, gamma2=-2, c1=None, alpha=None, E_break=0.1,  print_report=False, maxit=200,):
    """Fit a smoothly broken double power-law model using scipy.odr.

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
    c1 : float or None, default=None
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
    scipy.odr.Output
        The result returned by the ODR fit.
    """
    c1 = y[3] if c1 is None else c1
    alpha = 0.1 if alpha is None else alpha

    plmodel = Model(double_pl_func)

    # Create the data object for ODR
    data = RealData(x, y, sx=xerr, sy=yerr)

    # Set up ODR with the initial parameter guesses.
    odr = ODR( data, plmodel, 
              beta0=[c1, gamma1, gamma2, alpha, E_break], 
              ifixb=[1, 1, 1, 1, 1], maxit=maxit,)

    # Run the regression
    result = odr.run()

    convergence = check_odr_output(result)
    if not convergence:
        result = odr.run()
        convergence = check_odr_output(result)

    if print_report:
        result.pprint()

    return result

def triple_pl_func(p, x):
    """Evaluate a triple power-law model.

    Parameters
    ----------
    p : sequence of float
        Model parameters
        ``(c1, gamma1, gamma2, gamma3, alpha, beta,
        E_break_low, E_break_high)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    c1, gamma1, gamma2, gamma3, alpha, beta, E_break_low, E_break_high = p

    y = (c1
        * (x / 0.1) ** gamma1
        * ((x**alpha + E_break_low**alpha)
            / (0.1**alpha + E_break_low**alpha)) ** ((gamma2 - gamma1) / alpha)
        * ((x**beta + E_break_high**beta)
            / (0.1**beta + E_break_high**beta)) ** ((gamma3 - gamma2) / beta))
    
    return y

def triple_pl_fit(x, y, xerr, yerr, gamma1=-1.8, gamma2=-2, gamma3=-3, c1=None,
 alpha=None, beta=None, E_break_low=0.06, E_break_high=0.12, print_report=False, maxit=200,):
    """Fit a smoothly broken triple power-law model using scipy.odr.

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
    c1 : float or None, default=None
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
    scipy.odr.Output
        The result returned by the ODR fit.
    """
    c1 = y[3] if c1 is None else c1
    alpha = 0.1 if alpha is None else alpha
    beta = 0.1 if beta is None else beta

    plmodel = Model(triple_pl_func)

    # Create the data object for ODR.
    data = RealData(x, y, sx=xerr, sy=yerr)

    # Set up ODR with the initial parameter guesses.
    odr = ODR(data, plmodel, 
              beta0=[c1, gamma1, gamma2, gamma3, alpha, beta, 
                     E_break_low, E_break_high,], 
                     ifixb=[1, 1, 1, 1, 1, 1, 1, 1], 
                     maxit=maxit,)

    # Run the regression.
    result = odr.run()

    convergence = check_odr_output(result)
    if not convergence:
        result = odr.run()
        convergence = check_odr_output(result)

    if print_report:
        result.pprint()

    return result

def cut_pl_func(p, x):
    """Evaluate a power law with an exponential cut-off.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(c1, gamma1, E_cut, exponent)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    c1, gamma1, E_cut, exponent = p

    y = c1 * (x / 0.1) ** gamma1 * np.exp(-(x / E_cut) ** exponent)

    return y
	
def cut_pl_fit(x, y, xerr, yerr, gamma1=-1.8, c1=None,
E_cut=0.35, exponent=2, print_report=False, maxit=200,):
    """Fit a power law with an exponential cut-off using scipy.odr.

    Parameters
    ----------
    x, y : array-like
        Data to fit.
    xerr, yerr : array-like
        Uncertainties in ``x`` and ``y``, respectively.
    gamma1 : float, default=-1.8
        Initial guess for the power-law index.
    c1 : float or None, default=None
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
    scipy.odr.Output
        The result returned by the ODR fit.
    """
    c1 = y[4] if c1 is None else c1

    plmodel = Model(cut_pl_func)

    # Create the data object for ODR.
    data = RealData(x, y, sx=xerr, sy=yerr)

    # Set up ODR with the initial parameter guesses.
    odr = ODR( data, plmodel, 
              beta0=[c1, gamma1, E_cut, exponent], 
              ifixb=[1, 1, 1, 1], 
              maxit=maxit,)

    # Run the regression.
    result = odr.run()

    convergence = check_odr_output(result)
    if not convergence:
        result = odr.run()
        convergence = check_odr_output(result)

    if print_report:
        result.pprint()

    return result


def cut_break_pl_func(p, x):
    """Evaluate a smoothly broken power law with an exponential cut-off.

    Parameters
    ----------
    p : sequence of float
        Model parameters
        ``(c1, gamma1, gamma2, alpha, E_break, E_cut, exponent)``.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    c1, gamma1, gamma2, alpha, E_break, E_cut, exponent = p

    y = (c1
        * (x / 0.1) ** gamma1
        * ((x**alpha + E_break**alpha)
            / (0.1**alpha + E_break**alpha)) ** ((gamma2 - gamma1) / alpha)
        * np.exp(-(x / E_cut) ** exponent))

    return y

def cut_break_pl_fit(x, y, xerr, yerr, gamma1=-1.8, gamma2=-2,
c1=None, alpha=None, E_break=0.1, E_cut=0.35, exponent=2,
print_report=False, maxit=200,):
    """Fit a smoothly broken power law with an exponential cut-off using scipy.odr.

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
    c1 : float or None, default=None
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
    scipy.odr.Output
        The result returned by the ODR fit.
    """
    c1 = y[4] if c1 is None else c1
    alpha = 0.1 if alpha is None else alpha

    plmodel = Model(cut_break_pl_func)

    # Create the data object for ODR.
    data = RealData(x, y, sx=xerr, sy=yerr)

    # Set up ODR with the initial parameter guesses.
    odr = ODR(data, plmodel,
        beta0=[c1, gamma1, gamma2, alpha,
            E_break, E_cut, exponent,],
        ifixb=[1, 1, 1, 1, 1, 1, 1],
        maxit=maxit,)

    # Run the regression.
    result = odr.run()

    convergence = check_odr_output(result)
    if not convergence:
        result = odr.run()
        convergence = check_odr_output(result)

    if print_report:
        result.pprint()

    return result	

def line(p, x):
    """Evaluate a linear function.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(c1, gamma1)``, where ``c1`` is the
        intercept and ``gamma1`` is the slope.
    x : float or array-like
        Independent variable.

    Returns
    -------
    float or array-like
        Model value.
    """
    c1, gamma1 = p
    return c1 + x * gamma1

def line_intersect(g1, c1, g2, c2):
    """Find the intersection point of two lines.

    Parameters
    ----------
    g1, c1 : float
        Slope and intercept of the first line.
    g2, c2 : float
        Slope and intercept of the second line.

    Returns
    -------
    tuple of float or None
        ``(x, y)`` coordinates of the intersection point. Returns
        ``None`` if the two lines are parallel.
    """
    if g1 == g2:
        print("These lines are parallel!!!")
        return None

    x = (c2 - c1) / (g1 - g2)
    y = g1 * x + c1

    return x, y


def double_line(p, x):
    """Evaluate a piecewise linear model with a predefined break point.

    Parameters
    ----------
    p : sequence of float
        Model parameters ``(c1, c2, gamma1, gamma2, E_break)``.
    x : array-like
        Independent variable.

    Returns
    -------
    numpy.ndarray
        Model values. For ``x < E_break``, the first linear model is
        used; for ``x >= E_break``, the second linear model is used.
    """
    c1, c2, gamma1, gamma2, E_break = p

    Xmaskd = x < E_break
    Xmasku = x >= E_break

    y = np.zeros(x.shape)

    y[Xmaskd] = c1 + gamma1 * x[Xmaskd]
    y[Xmasku] = c2 + gamma2 * x[Xmasku]

    return y




