import numpy as np
from scipy.odr import *



def power_law_fit(x,y,xerr = None,yerr = None, gamma1=-1.8, I0=None, print_report=False):
	'''
	fits a power law to the data using scipy.odr

    x,y: data to fit, should be np.log() of Energy and Intensity
    unc: y-error
    gamma1, I0: guess-values for the fit
    '''
	
	#covMatrix = np.cov(xerr,bias=False)

	

	I0 = y[-1] if I0==None else I0
	plmodel = Model(simple_pl)
	#data = RealData(x, y, covx=covMatrix, sy=yerr)
	data = RealData(x, y, sx=xerr, sy=yerr)
    # Set up ODR with the model and data.
	odr = ODR(data, plmodel, beta0=[I0, gamma1])
    # Run the regression.
	result = odr.run()
	
	if print_report:
		
		result.pprint()
		result.stopreason
		#print(type(result))
		#print(result.keys)
		
	return result

def double_pl_func(p, x):#, I0, gamma1, gamma2, alpha, E_break):
    '''
    Mar 2020: functin 25 of prinsloo 2019 paper but withoug exponential roll-over
    '''

    I0, gamma1, gamma2, alpha, E_break = p

    y = I0 * (x/0.1)**gamma1  * ((x**alpha + E_break**alpha)/(0.1**alpha+E_break**alpha))**((gamma2-gamma1)/alpha)

    return y


def double_pl_fit(x,y, xerr = None, yerr = None, gamma1=-1.8, gamma2=-2, I0=None, alpha=None, E_break=0.1, print_report=False, maxit=20):
	#covMatrix = np.cov(xerr,bias=False)

	I0 = y[3] if I0==None else I0
	alpha = 0.1 if alpha==None else alpha

	plmodel = Model(double_pl_func)
	#print(broken_pl_func)
	
	# Create a RealData object using our initiated data from above.
	data = RealData(x, y, sx=xerr, sy=yerr)
	# Set up ODR with the model and data.
	odr = ODR(data, plmodel, beta0=[I0, gamma1, gamma2, alpha, E_break], ifixb=[1,1,1,1,1], maxit=maxit)

	# Run the regression.
	result = odr.run()
	#iprint = odr.set_iprint(init=2,  iter=2, iter_step = 1, final=2)
	if print_report:
		result.pprint()
		result.stopreason
		#print(result.keys)

	return result

def triple_pl_func(p, x):#, I0, gamma1, gamma2, alpha, E_break):
    '''
    Mar 2020: functin 25 of prinsloo 2019 paper but withoug exponential roll-over
    '''

    I0, gamma1, gamma2, gamma3, alpha, beta, E_break_low, E_break_high = p

    y = I0 * (x/0.1)**gamma1  * ((x**alpha + E_break_low**alpha)/(0.1**alpha+E_break_low**alpha))**((gamma2-gamma1)/alpha)* ((x**beta + E_break_high**beta)/(0.1**beta+E_break_high**beta))**((gamma3-gamma2)/beta)

    return y


def triple_pl_fit(x,y, xerr = None, yerr = None, gamma1=-1.8, gamma2=-2, gamma3 = -3, I0=None, alpha=None, beta = None, E_break_low=0.06, E_break_high = 0.12, print_report=False, maxit=20):
	#covMatrix = np.cov(xerr,bias=False)

	I0 = y[3] if I0==None else I0
	alpha = 0.1 if alpha==None else alpha
	beta = 0.1 if beta==None else beta

	plmodel = Model(triple_pl_func)
	#print(broken_pl_func)
	
	# Create a RealData object using our initiated data from above.
	data = RealData(x, y, sx=xerr, sy=yerr)
	# Set up ODR with the model and data.
	odr = ODR(data, plmodel, beta0=[I0, gamma1, gamma2, gamma3, alpha, beta, E_break_low, E_break_high], ifixb=[1,1,1,1,1,1,1,1], maxit=maxit)

	# Run the regression.
	result = odr.run()
	#iprint = odr.set_iprint(init=2,  iter=2, iter_step = 1, final=2)
	if print_report:
		result.pprint()
		result.stopreason
		#print(result.keys)

	return result

def cut_break_pl_func(p, x): #I0, gamma1, gamma2, alpha, E_break, E_cut

	I0, gamma1, gamma2, alpha, E_break, E_cut, exponent = p
	
	y = I0*(x/0.1)**gamma1 * ((x**alpha + E_break**alpha)/(0.1**alpha+E_break**alpha))**((gamma2-gamma1)/alpha)*np.exp(-(x/E_cut)**exponent)
	
	return y

	
def cut_break_pl_fit(x,y, xerr = None, yerr = None, gamma1=-1.8, gamma2=-2, I0=None, alpha=None, E_break=0.1, E_cut = 0.35, exponent = 2, print_report=False, maxit=20):
	I0 = y[4] if I0==None else I0
	#c2 = y[-1]*1e-2 if c2==None else c2
	alpha = 0.1 if alpha==None else alpha
	#covMatrix = np.cov(xerr,bias=False)

	plmodel = Model(cut_break_pl_func)
	# Create a RealData object using our initiated data from above.
	data = RealData(x, y, sx=xerr, sy=yerr)
	# Set up ODR with the model and data.
	odr = ODR(data, plmodel, beta0=[I0, gamma1, gamma2, alpha, E_break, E_cut, exponent], ifixb=[1,1,1,1,1,1,1], maxit = maxit)

	# Run the regression.
	result = odr.run()

	if print_report:
		result.pprint()
		result.stopreason

	return result
	
def cut_pl_func(p, x): #I0, gamma1, gamma2, alpha, E_break, E_cut

	I0, gamma1, E_cut, exponent = p
	
	y = I0*(x/0.1)**gamma1 *np.exp(-(x/E_cut)**exponent)
	
	return y

	
def cut_pl_fit(x,y, xerr = None, yerr = None, gamma1=-1.8, I0=None, E_cut = 0.35, exponent = 2, print_report=False, maxit=20):
	I0 = y[4] if I0==None else I0
	#c2 = y[-1]*1e-2 if c2==None else c2
	#alpha = 0.1 if alpha==None else alpha
	#covMatrix = np.cov(xerr,bias=False)

	plmodel = Model(cut_pl_func)
	# Create a RealData object using our initiated data from above.
	data = RealData(x, y, sx=xerr, sy=yerr)
	# Set up ODR with the model and data.
	
	odr = ODR(data, plmodel, beta0=[I0, gamma1, E_cut, exponent], ifixb=[1,1,1,1], maxit = maxit)

	# Run the regression.
	result = odr.run()

	if print_report:
		result.pprint()
		result.stopreason

	return result




def line_intersect(g1, I0, g2, c2):
    if g1 == g2:
        print ("These lines are parallel!!!")
        return None
    x = (c2 - I0) / (g1 - g2)
    y = g1 * x + I0
    return x,y


def line(p,x):#, I0, gamma1):
    I0, gamma1 = p
    return I0 + x * gamma1


def double_line(p, x):#, I0, c2, gamma1, gamma2, E_break):
    '''
    a double line model with predefined break-point
    '''

    I0,c2, gamma1, gamma2, E_break = p
    Xmaskd = x < E_break
    Xmasku = x >= E_break

    y = np.zeros(x.shape)

    y[Xmaskd] = I0 + gamma1 * x[Xmaskd]
    y[Xmasku] = c2 + gamma2 * x[Xmasku]

    return y


def simple_pl(p,x):#, I0, gamma1):
	I0, gamma1 = p
	y = I0*(x/0.1)**gamma1
	return y#I0*x**gamma1



