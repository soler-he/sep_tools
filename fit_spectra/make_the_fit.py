# solo_functions.py
import numpy as np
import pandas as pd
import fit_spectra.my_power_law_fits_odr as pl_fit
from scipy.stats import t as studentt
# from lmfit.models import GaussianModel
import pickle
# from scipy.odr import *


def closest_values(array, value):
	"""This function finds n closest values to the guess value and returns an array with the closest values.
	the number of values in the closest_values_array depends on the length of the initial array.
	When used in MAKE_THE_FIT the value is going to be g1_guess, g2_guess, alpha_guess, break_guess, c1_guess, cut_guess.

	Args:
		array (np.array): array of values from which to choose
		value (float): value close to which the values need to be
	"""

		
	
	if len(array)<=10:
		array_size = len(array)
		
	if len(array)>10 and len(array)<= 20:
		array_size = round(len(array)/2)
	
	if len(array) >20:
		array_size = 10
		
	
	array = np.delete(array, np.where(array ==value))
	#if len(new_array)!=0:
	#	array = new_array
		
	closest_values_array = np.array(())
	
	for i in range(array_size):
		absolute_val_array = np.abs(array - value)
		smallest_difference_index = absolute_val_array.argmin()
		closest_element = array[smallest_difference_index]
		closest_values_array = np.append(closest_values_array, closest_element)
		array = np.delete(array, np.where(array ==closest_element))
	
	closest_values_array = sorted(closest_values_array)
	
	return(closest_values_array)

	
def check_redchi(spec_e, spec_flux, e_err, flux_err, gamma1 = -1, gamma2 = -2, gamma3 = -4, c1 = 1000, alpha = 10, beta = 10, E_break_low = 0.06, E_break_high = 0.1,  E_cut= None, exponent = 2, fit = 'best',  maxit=10000, e_min=None, e_max=None):
	"""This function compares the reduced chi sq from the different fits. The fit iteration results come from this function. 
	the function also checks if the break point is outside of the energy array (also the cutoff point)
	the min and max energies cannot be last and/or first points because it wouldn't be a physical result
	

	Args:
		spec_e (list): list of values corresponding to the energy
		spec_flux (list): list of values corresponding to the intensity/flux
		e_err (list): uncertainty values for the energy
		flux_err (list): uncertainty values for the intensity/flux
		gamma1 (int, optional): value for spectral index 1. Defaults to -1.
		gamma2 (int, optional): value for spetcral index 2. Defaults to -2.
		gamma3 (int, optional): value for spectral index 3. Defaults to -4.
		c1 (int, optional): Intensity at 100 keV. Defaults to 1000.
		alpha (int, optional): sharpness of the transition from spectral index 1 to 2. Defaults to 10.
		beta (int, optional): sharpness of the transition from spectral index 2 to 3. Defaults to 10.
		E_break_low (float, optional): first spectral break in MeV. Defaults to 0.06.
		E_break_high (float, optional): second spectral break in MeV. Defaults to 0.1.
		E_cut (float, optional): exponential cutoff in MeV. Defaults to None.
		fit (str, optional): thw type of fit preferred. The options are 'single', 'double', 'triple', 'cut', . Defaults to 'best'.
		maxit (int, optional): number of maximum iteration for ODR. Defaults to 10000.
		e_min (float, optional): minimum energy for the fit. Defaults to None.
		e_max (float, optional): maximum energy for the fit. Defaults to None.
	"""
	

	emin = spec_e[2]
	emax = spec_e[len(spec_e)-3]
	
	if e_min is None or e_min == spec_e[0]:
		emin = spec_e[2]
	else:
		emin = e_min

	if e_max is None or e_max == spec_e[len(spec_e)-1]:
		emax = spec_e[len(spec_e)-3]

	else:
		emax = e_max

	#print(emin, emax)

	if fit == 'best':
		#print('143 best')
		result_triple = pl_fit.triple_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, gamma3 = gamma3, c1 = c1, alpha = alpha, beta = beta, E_break_low = E_break_low, E_break_high = E_break_high, maxit=10000)
		redchi_triple = result_triple.res_var
		breakp_low    = result_triple.beta[6]	
		breakp_high   = result_triple.beta[7]
		difference_triple = np.abs(breakp_high-breakp_low)
		alpha         = result_triple.beta[4]
		beta          = result_triple.beta[5]
		if alpha>0:
			gamma1     = result_triple.beta[1]
		elif alpha <= 0:
			gamma1     = result_triple.beta[2]

	
		result_cut_break = pl_fit.cut_break_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=gamma1, gamma2=gamma2, c1=c1, alpha=alpha, E_break=E_break_low, E_cut = E_cut, exponent = exponent, print_report=False, maxit=10000)
		redchi_cut_break = result_cut_break.res_var
		breakp_cut = result_cut_break.beta[4]
		cut_b = result_cut_break.beta[5]
		difference_cut = np.abs(breakp_cut-cut_b)
		exponent_cut_break = result_cut_break.beta[6]


		result_cut = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1, E_cut = E_cut, exponent = exponent, maxit=10000)
		redchi_cut= result_cut.res_var
		cut        = result_cut.beta[2]	#shoud maybe make distinction between cut from cut pl and cut from cut double pl
		exponent_cut = result_cut.beta[3]

		result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, c1 = c1, alpha = alpha, E_break = E_break_low, maxit=10000)
		redchi_double = result_double.res_var
		breakp        = result_double.beta[4]	
		

		result_single_pl = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1)
		redchi_single  = result_single_pl.res_var  

		chis = {"triple":redchi_triple, "double_cut":redchi_cut_break, "cut":redchi_cut, "double":redchi_double, "single":redchi_single}
		sorted_chis = dict(sorted(chis.items(), key=lambda x: x[1], reverse=False))
		if exponent_cut == 0:
			sorted_chis.pop("cut")
		if exponent_cut_break == 0:
			sorted_chis.pop("double_cut")
		#smallest_value = list(sorted_chis.keys())[0]
		#print(sorted_chis)
		#print('Smallest chis value ' + smallest_value)

		# check if there are values with zero chi sq. If so, delete from dict. Then check if dict is empty. If yes: loop again through the results
		# aka redo them until not empty. Use that as dict. 
		
		list_zero_chi = []
		for i in sorted_chis:
			if sorted_chis[i] == 0.:
				list_zero_chi.append(i)
		
		for i in list_zero_chi:
			sorted_chis.pop(i)

		smallest_value = list(sorted_chis.keys())[0]

		#print(sorted_chis)
		#print(smallest_value)



		for i in range(len(sorted_chis)):
			# make if statemenys to check values etc of breaks snd so on change smallest value ofter checking everything
			if smallest_value == 'triple':
				if breakp_low < emax and breakp_low > emin and breakp_high < emax and breakp_high > emin and breakp_low<breakp_high:
					absolute_val_array = np.abs(spec_e - breakp_low)
					smallest_difference_index = absolute_val_array.argmin()
					low = ''
					high = ''
					if smallest_difference_index == len(spec_e)-1:
						low = spec_e[smallest_difference_index-1]-e_err[smallest_difference_index-1]
						high = spec_e[smallest_difference_index]+e_err[smallest_difference_index]
					else:
						low = spec_e[smallest_difference_index]-e_err[smallest_difference_index]
						high = spec_e[smallest_difference_index+1]+e_err[smallest_difference_index+1]
					difference_triple_energy = high-low

					#print('TRIPLE')
					#print(absolute_val_array)
					#print(smallest_difference_index)
					#print(low)
					#print(high)
					#print(difference_triple_energy)
					#print(difference_triple)

					#if breakp_high > breakp_low and difference_triple>difference_triple_energy:
					#The triple pl is defined so that the two breaks are actually interchangable 
					# this means that it can happen that the 'high' break becomes the low break
					# these cases have to be deleted because it messes with the meaning of the parameters of the fit
						
					if difference_triple>difference_triple_energy and gamma1 <0:
						if alpha >0 or beta >0:
							which_fit = 'triple'
							redchi = redchi_triple
							result = result_triple
							return([which_fit, redchi, result])
						else:
							smallest_value = list(sorted_chis.keys())[i] 
							# There are cases in which these statements lead to no return because all options are bad.
							# This needs to be fixed by somehow adding options to the list 1.11.23
					else:
						smallest_value = list(sorted_chis.keys())[i]
				else:
					smallest_value = list(sorted_chis.keys())[i]

			if smallest_value == 'double_cut':
				if cut_b > emin and cut_b< emax and breakp_cut> emin and breakp_cut<emax and cut_b>breakp_cut:
					absolute_val_array = np.abs(spec_e - breakp_cut)
					smallest_difference_index = absolute_val_array.argmin()
					low = ''
					high = ''
					if smallest_difference_index == len(spec_e)-1:
						low = spec_e[smallest_difference_index-1]-e_err[smallest_difference_index-1]
						high = spec_e[smallest_difference_index]+e_err[smallest_difference_index]
					else:
						low = spec_e[smallest_difference_index]-e_err[smallest_difference_index]
						high = spec_e[smallest_difference_index+1]+e_err[smallest_difference_index+1]
					difference_cut_energy = high-low

					#print('BC')
					#print(absolute_val_array)
					#print(smallest_difference_index)
					#print(low)
					#print(high)
					#print(difference_cut_energy)
					#print(difference_cut)

					if gamma1 <0 and difference_cut>difference_cut_energy:
						which_fit = 'double_cut'
						redchi = redchi_cut_break
						result = result_cut_break
						return([which_fit, redchi, result])
						
					else:
						smallest_value = list(sorted_chis.keys())[i]
				else:
					smallest_value = list(sorted_chis.keys())[i]
			
			if smallest_value == 'cut':
				if cut >= emin and cut <=emax:	
					which_fit = 'cut'
					redchi = redchi_cut
					result = result_cut
					return([which_fit, redchi, result])
				else:
					smallest_value = list(sorted_chis.keys())[i]
			
			if smallest_value == 'double':
				if breakp >= emin and breakp <=emax:	
					which_fit = 'double'
					redchi = redchi_double
					result = result_double
					return([which_fit, redchi, result])
				else:
					smallest_value = list(sorted_chis.keys())[i]
			if smallest_value == 'single':
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			
		# redo loop either because list is already empty or because none of the previous options worked
		
					
		
	if fit == 'triple':
		result_triple = pl_fit.triple_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, gamma3 = gamma3, c1 = c1, alpha = alpha, beta = beta, E_break_low = E_break_low, E_break_high = E_break_high, maxit=10000)
		redchi_triple = result_triple.res_var
		breakp_low    = result_triple.beta[6]	
		breakp_high   = result_triple.beta[7]
		difference_triple = breakp_high-breakp_low

		if breakp_low < emax and breakp_low > emin and breakp_high < emax and breakp_high > emin:
			absolute_val_array = np.abs(spec_e - breakp_low)
			smallest_difference_index = absolute_val_array.argmin()
			low = ''
			high = ''
			if smallest_difference_index == len(spec_e)-1:
				low = spec_e[smallest_difference_index-1]-e_err[smallest_difference_index-1]
				high = spec_e[smallest_difference_index]+e_err[smallest_difference_index]
			else:
				low = spec_e[smallest_difference_index]-e_err[smallest_difference_index]
				high = spec_e[smallest_difference_index+1]+e_err[smallest_difference_index+1]
			
			difference_triple_energy = high-low
			
			if breakp_high > breakp_low and difference_triple>difference_triple_energy:
				which_fit = 'triple'
				redchi = redchi_triple
				result = result_triple
				return([which_fit, redchi, result])
					
			else:
				fit = 'double_cut'
				
		else:
			fit = 'double_cut'



			
	if fit == 'double_cut':
		result_cut_break = pl_fit.cut_break_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=gamma1, gamma2=gamma2, c1=c1, alpha=alpha, E_break=E_break_low, E_cut = E_cut, exponent = exponent, print_report=False, maxit=10000)
		redchi_cut_break = result_cut_break.res_var
		breakp_cut = result_cut_break.beta[4]
		#The cut of the break + cutoff
		cut_b = result_cut_break.beta[5]
		difference_cut = breakp_cut-cut_b

		if breakp_cut <= emax and breakp_cut > emin and cut_b <= emax and cut_b > emin:	
			absolute_val_array = np.abs(spec_e - breakp_cut)
			smallest_difference_index = absolute_val_array.argmin()
			low = ''
			high = ''
			if smallest_difference_index == len(spec_e)-1:
				low = spec_e[smallest_difference_index-1]-e_err[smallest_difference_index-1]
				high = spec_e[smallest_difference_index]+e_err[smallest_difference_index]
			else:
				low = spec_e[smallest_difference_index]-e_err[smallest_difference_index]
				high = spec_e[smallest_difference_index+1]+e_err[smallest_difference_index+1]
			
			difference_cut_energy = high-low

			if cut_b > breakp_cut and difference_cut > difference_cut_energy:
				which_fit = 'double_cut'
				redchi = redchi_cut_break
				result = result_cut_break
				return([which_fit, redchi, result])
			
			else:
				fit = 'best_cb'
		
		else:
			fit = 'best_cb'



	
	if fit == 'best_sb':
		result_single_pl = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1)
		redchi_single  = result_single_pl.res_var  

		result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, c1 = c1, alpha = alpha, E_break = E_break_low, maxit=10000)
		redchi_double = result_double.res_var
		breakp        = result_double.beta[4]	

		if redchi_double<=redchi_single:
			if breakp < emin or breakp > emax:
				which_fit = 'single'
				redchi = redchi_single
				result = result_single_pl
				return([which_fit, redchi, result])
			if breakp >= emin and breakp <=emax:	
				which_fit = 'double'
				redchi = redchi_double
				result = result_double
				return([which_fit, redchi, result])
		if redchi_double>redchi_single:
			which_fit = 'single'
			redchi = redchi_single
			result = result_single_pl
			return([which_fit, redchi, result])

	if fit == 'best_cb':
		result_cut = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1, E_cut = E_cut, exponent = exponent, maxit=10000)
		redchi_cut= result_cut.res_var
		cut        = result_cut.beta[2]	#shoud maybe make distinction between cut from cut pl and cut from cut double pl

		result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, c1 = c1, alpha = alpha, E_break = E_break_low, maxit=10000)
		redchi_double = result_double.res_var
		breakp        = result_double.beta[4]	

		if redchi_double<=redchi_cut:
			if breakp < emin or breakp > emax:
				fit = 'single'
			if breakp >= emin and breakp <=emax:	
				which_fit = 'double'
				redchi = redchi_double
				result = result_double
				return([which_fit, redchi, result])
		if redchi_double>redchi_cut:
			if cut < emin or cut > emax:
				fit = 'single'
			if cut >= emin and cut <=emax:	
				which_fit = 'cut'
				redchi = redchi_cut
				result = result_cut
				return([which_fit, redchi, result])
			
	
	
	if fit == 'cut':
		result_cut = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1, E_cut = E_cut,exponent = exponent, maxit=10000)
		redchi_cut= result_double.res_var
		cut        = result_cut.beta[2]	#shoud maybe make distinction between cut from cut pl and cut from cut double pl
		if cut < emin or cut > emax:
			fit = 'single'

		if cut >= emin and cut <=emax:	
			which_fit = 'cut'
			redchi = redchi_cut
			result = result_cut
			return([which_fit, redchi, result])
		

	if fit == 'double':
		result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, gamma2 = gamma2, c1 = c1, alpha = alpha, E_break = E_break_low, maxit=10000)
		redchi_double = result_double.res_var
		breakp        = result_double.beta[4]	

		if breakp < emin or breakp > emax:
			fit = 'single'
		if breakp >= emin and breakp <=emax:	
			which_fit = 'double'
			redchi = redchi_double
			result = result_double
			return([which_fit, redchi, result])
		
	if fit == 'single':
		result_single_pl = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = gamma1, c1 = c1)
		redchi_single  = result_single_pl.res_var  
	
		which_fit = 'single'
		redchi = redchi_single
		result = result_single_pl
		return([which_fit, redchi, result])

	


	
	
def find_c1(spec_e, spec_flux, e_min, e_max):
	"""_summary_

	Args:
		spec_e (_type_): _description_
		spec_flux (_type_): _description_
		e_min (_type_): _description_
		e_max (_type_): _description_
	"""
	absolute_val_array = np.abs(spec_e - e_max)
	smallest_difference_index = absolute_val_array.argmin()
	closest_element = spec_e[smallest_difference_index]
	
	x1 = np.log10(spec_e[smallest_difference_index-5])
	y1 = np.log10(spec_flux[smallest_difference_index-5])
	
	x2 = np.log(spec_e[smallest_difference_index])
	y2 = np.log10(spec_flux[smallest_difference_index])
	
	m = (y1-y2)/(x1-x2)
	q = (x1*y2-x2*y1)/(x1-x2)
	
	c1 = m*1.0+q
	
	print('x1', x1, 'y1', y1, 'x2', x2, 'y2', y2)
	return(c1)
	
		
	
	

def MAKE_THE_FIT(spec_e, spec_flux, e_err, flux_err, ax, direction='sun', which_fit='best', e_min=None, e_max=None, g1_guess=-2., g2_guess=None, g3_guess=None, alpha_guess=5., beta_guess = 5,  break_low_guess=0.065, break_high_guess=0.12, cut_guess = 0.12, c1_guess=None, exponent_guess = 2, use_random = False, iterations = 10, path = None, path2 = None, detailed_legend = False):
	"""This function fit the data to a single, double or break+cut power law. 
	The fit type can be chosen between: single,double, cut or best. 
	The best option checks between all the options and chooses between the three by checking the reduced chisqr.
	Also when the double or cut options are chosen, the function checks if the break or cutoff points are outside of the energy range.
	In such case, a sigle pl will be fit to the data and the function will output that the breakpoint is outside of the energy range.

	Args:
		spec_e (list): list of values corresponding to the energy
		spec_flux (list): list of values corresponding to the intensity/flux
		e_err (list): uncertainty values for the energy
		flux_err (list): uncertainty values for the intensity/flux
		ax (axis): _description_
		direction (str, optional): _description_. Defaults to 'sun'.
		which_fit (str, optional): which_fit options: 'single' will force a single pl fit to the data
		  			'double' will force a double pl fit to the data but ONLY if the break point is within the energy range otherwise a sigle pl fit will be produced instead
		  			'best_sb' will choose automatically the best fit type between single and double by comparing the redchis of the fits
		    		'cut' will produce a single pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a double or single pl will be fit instead
			  		'double_cut' will produce a double pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a double or single pl will be fit instead
				  	'best_cb'. Defaults to 'best'.
					'triple' will force a triple pl fit. If this is not possible, the function will check which is the next best option.
					'best' will choose automatically the best fit type by comparing the redchis of the fits.
					Defaults to 'best'.
		e_min (float, optional): The lower energy limit for the fit. Defaults to None.
		e_max (float, optional): The upper energy limit for the fit. Defaults to None.
		g1_guess (float, optional): The slope of the single pl fit or the first part of a double/triple pl fit. Defaults to -1.9.
		g2_guess (float, optional): The slope of the second part of a double/triple pl fit. gamma2 < gamma1. Defaults to -2.5. 
		g3_guess (int, optional): The slope of the third part of a double/triple pl fit. gamma3 < gamma2 < gamma1. Defaults to -4.
		c1_guess (int, optional): The intensity/flux value at 0.1 MeV. Defaults to 1000.
		alpha_guess (int, optional): The smoothness of the transition between gamma1 and gamma2. Defaults to 10.
		beta_guess (int, optional): The smoothness of the transition between gamma3 and gamma2. Defaults to 10.
		break_guess_low (float, optional): Guess value for the energy correponding to the break in the double pl and first break for the triple pl. Input in MeV.  Defaults to 0.6.
		break_guess_high (float, optional): Guess value for the energy correponding to the second break for the triple pl.Input in MeV. Defaults to 1.2.
		cut_guess (float, optional): Guess value for the energy corresponding to the exponential cutoff.Input in MeV.  Defaults to 1.2.
		use_random (bool, optional): If True the fitting function will, in addition to the guess values, choose random values from a predifined list of values for each variable. 
					These values are chosen close to the guess values. Defaults to False.
		iterations (int, optional): The number of times the function will choose random values to use in the fit to the data. Defaults to 10.
		path (_type_, optional): _description_. Defaults to None.
		path2 (_type_, optional): _description_. Defaults to None.
		detailed_legend (bool, optional): _description_. Defaults to False.

			"""

	#print(spec_e)
	#print(spec_flux)
	# CHANGE GUESS VALUES OF GAMMA1
	if g2_guess is None:
		g2_guess = g1_guess - 0.1

	if g3_guess is None:
		g3_guess = g2_guess - 0.1
		
	if e_min is None:
		#e_min = min(spec_e)
		e_min = spec_e[0]
	if e_max is None:
		#e_max = max(spec_e)
		e_max = spec_e[len(spec_e)-1]
	
	if c1_guess is None:
		absolute_val_array = np.abs(spec_e - 1)
		smallest_difference_index = absolute_val_array.argmin()
		c1_guess = spec_flux[smallest_difference_index]
		
		
	
	# the break guess should be between min and max energy
	
	# have to construct the guesses logarithmically
	# have to construct the guesses logarithmically
	g1_start_value = -np.abs(g1_guess)*10.
	g2_start_value = -np.abs(g2_guess)*10.
	g3_start_value = -np.abs(g3_guess)*10.

	g1_end_value = np.abs(g1_guess)*10.
	g2_end_value = np.abs(g2_guess)*10.
	g3_end_value = np.abs(g3_guess)*10.


	g1_step = np.abs(g1_guess/4.)
	g2_step = np.abs(g2_guess/4.)
	g3_step = np.abs(g3_guess/4.)
	

	if use_random :	
		gamma1_array = closest_values(np.arange(g1_start_value,g1_end_value,g1_step), g1_guess)
		gamma2_array = closest_values(np.arange(g2_start_value,g2_end_value,g2_step), g2_guess)
		gamma3_array = closest_values(np.arange(g3_start_value,g3_end_value,g3_step), g3_guess)
		
	# c1_array...  we want to get a good approximation of the flux at 1, whatever 1 is in your plot. 
		c1_array = np.arange(c1_guess/100.,c1_guess*100., c1_guess/500.)
		
	# alpha array
		a1_array = np.arange(0.01,0.1,0.01)
		a2_array = np.arange(0.1,1.0,0.05)
		a3_array = np.arange(1,10,0.5)
		a4_array = np.arange(10,100,10)
		a5_array = np.arange(100,220,20)
		alpha_array = np.hstack((a1_array,a2_array,a3_array,a4_array,a5_array))
		alpha_array = closest_values(alpha_array, alpha_guess)
		beta_array = np.hstack((a1_array,a2_array,a3_array,a4_array,a5_array))
		beta_array = closest_values(beta_array, beta_guess)
	# break array
	# cut array = break_array *1.8
		
		if e_max<0.1:
			break_array_low = np.arange(e_min, e_max, 0.001)
		if e_max>=0.1 and e_max<1.0:
			b1_array = np.arange(e_min, 0.1, 0.001)
			b2_array = np.arange(0.1, e_max, 0.005)
			break_array_low = np.hstack((b1_array, b2_array))
		if e_max >=1 and e_max < 10:
			b1_array = np.arange(e_min, 0.1, 0.001)
			b2_array = np.arange(0.1, 1, 0.005)
			b3_array = np.arange(1, e_max, 0.01)
			break_array_low = np.hstack((b1_array, b2_array, b3_array))
		if e_max>=10:
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

	#print(c1_guess)
	
	color = {'sun':'crimson', 'asun':'orange','north':'darkslateblue','south':'c'}
	spec_e = np.array(spec_e)
	spec_flux = np.array(spec_flux)

		
	xplot = np.logspace(np.log10(np.nanmin(spec_e)), np.log10(np.nanmax(spec_e)), num=500)
	xplot = xplot[np.where((xplot >= e_min) & (xplot <= e_max))[0]]
	
	# 05.12.2025 made changes to block below. If issues check.
	fit_ind = []

	if flux_err is None:
		fit_ind   = np.where((spec_e >= e_min) & (spec_e <= e_max) & (np.isfinite(spec_flux) == True))[0]
		spec_e    = spec_e[fit_ind]
		spec_flux = spec_flux[fit_ind]
		if e_err is not None:
			e_err = np.array(e_err)
			e_err = e_err[fit_ind]
		#flux_err  = flux_err[fit_ind]
	
	elif flux_err is not None:
		print('HERE')
		fit_ind   = np.where((spec_e >= e_min) & (spec_e <= e_max) & (np.isfinite(spec_flux) == True) & (np.isfinite(flux_err) == True))[0]
		spec_e    = spec_e[fit_ind]
		spec_flux = spec_flux[fit_ind]
		flux_err = np.array(flux_err)
		flux_err  = flux_err[fit_ind]
		if e_err is not None:
			e_err = np.array(e_err)
			e_err = e_err[fit_ind]


	# everything is in a for loop that chooses random values between the 
	# closest_values (n times) and checks the redchis and chooses the best one
	# try separately the input guesses and then the random ones
	
	# everything is done first with input guess values and then with randoms
	
	# parameters used as final inputs !!!(not the fit result but input)!!!
	
	which_fit_final = ''
	
	redchi_final = 0
	
	result_final = None
	
# spec_e, spec_flux, e_err, flux_err, gamma1, gamma2, gamma3, c1, alpha, beta, E_break_low, E_break_high,  E_cut= None, fit = 'best',  maxit=10000, e_min=None, e_max=None):
	if which_fit == 'best':
		#print('572 best')
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, beta = beta_guess, gamma1=g1_guess, gamma2=g2_guess, gamma3 = g3_guess, E_break_low=break_low_guess, E_break_high = break_high_guess, E_cut = cut_guess, exponent = exponent_guess, fit = 'best', maxit=10000, e_min = e_min, e_max = e_max)
		redchi_guess = which_fit_guess[1]
		
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]

		if use_random :
			#print('USING RANDOM BEST')
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0] 
				g3_random = np.random.choice(gamma3_array, 1)[0]
# POSSIBLE ISSUES
				gammas = [g1_random, g2_random, g3_random]
				gammas.sort()
				g1_random = gammas[0]
				g2_random = gammas[1]
				g3_random = gammas[2]

				alpha_random = np.random.choice(alpha_array, 1)[0]
				beta_random = np.random.choice(beta_array, 1)[0]
				break_low_random = np.random.choice(break_array_low, 1)[0]
				break_high_random = np.random.choice(break_array_high, 1)[0]
				if break_high_random < break_low_random:
					b = break_low_random
					break_low_random = break_high_random
					break_high_random = b
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, beta = beta_random, gamma1=g1_random, gamma2=g2_random, gamma3 = g3_random, E_break_low=break_low_random, E_break_high = break_high_random, E_cut = cut_random, exponent = exponent_guess, maxit=10000, e_min = e_min, e_max = e_max)
				#print(which_fit_random)
				#while which_fit_random is None:
				#	which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, beta = beta_random, gamma1=g1_random, gamma2=g2_random, gamma3 = g3_random, E_break_low=break_low_random, E_break_high = break_high_random, E_cut = cut_random, maxit=10000, e_min = e_min, e_max = e_max)
				#print(which_fit_random is None)
				if which_fit_random is None:
					break
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					redchi_final = redchi_random
					which_fit_final = which_fit_random[0]

	if which_fit == 'triple':
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, beta = beta_guess, gamma1=g1_guess, gamma2=g2_guess, gamma3 = g3_guess, E_break_low=break_low_guess, E_break_high = break_high_guess, E_cut = cut_guess, exponent = exponent_guess, fit = 'triple', maxit=10000, e_min = e_min, e_max = e_max)
		# if for some reason the fit is not doable, the result will be None. In that case you cannot use redchi_guess = which_fit_guess[1] because you cannot call a None value. 
		# so you need to repeat the fit. Maybe this could be checked already in the function file.
		#while which_fit_guess is None:
		#	which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, beta = beta_guess, gamma1=g1_guess, gamma2=g2_guess, gamma3 = g3_guess, E_break_low=break_low_guess, E_break_high = break_high_guess, E_cut = cut_guess, fit = 'best', maxit=10000, e_min = e_min, e_max = e_max)
		#print(which_fit_guess)
		redchi_guess = which_fit_guess[1]
		#print(redchi_guess)
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]

		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0] 
				g3_random = np.random.choice(gamma3_array, 1)[0]

######## SHOULD MAKE SIMILAR STATEMENT FOR GAMMA 3 TO CHECK GAMMAS
######## OR FIRST RANDOMIZE GAMMA 1 THEN CUT THE ARRAYS OF GAMMA2 AND GAMMA3 SO THAT THE CHOSEN VALUES ARE NEVER SMALLER THAN THE PREVIOUS GAMMA 
######## OR AS AN ALTERNATIVE IF THE RANDOM VALUE IS BIGGER FOR GAMMA2 THAB GAMMA1 ETC JUST REDO THE RANDOM CHOICE OR EXCHANGE GAMMA VALUES! THIS IS JUST RANDOM VALUES NOT THE 
######## FINAL VALUES SO IT'S OK TO EXCHANGE THE RANDOM VALUES.

######## BELOW : WHEN CHECKING REDCHIS REMEMBER TO ADD ALL NEW VARIABLES INTO THE FUNCTION AND ALSO MAKE NEW VARIABLES FOR THE RANDOM VALUES OF THE NEW TRIPLE PL FUNCTION
######## NEW VARIABLES: BETA, GAMMA 3, BREAK LOW AND HIGH ! REMEMBER TO CHANGE THE NAMES OF THE OLD BREAK TO THE NEW ONE

				#gamma2 should always be more negative (smaller) than gamma1	
				#if g1_random < g2_random and g1_random < g3_random:
				#	gamma = g1_random
				#	g1_random = g2_random
				#	g2_random = gamma
				gammas = [g1_random, g2_random, g3_random]
				gammas.sort()
				g1_random = gammas[0]
				g2_random = gammas[1]
				g3_random = gammas[2]

				alpha_random = np.random.choice(alpha_array, 1)[0]
				beta_random = np.random.choice(beta_array, 1)[0]
				break_low_random = np.random.choice(break_array_low, 1)[0]
				break_high_random = np.random.choice(break_array_high, 1)[0]
				if break_high_random < break_low_random:
					b = break_low_random
					break_low_random = break_high_random
					break_high_random = b
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, beta = beta_random, gamma1=g1_random, gamma2=g2_random, gamma3 = g3_random, E_break_low=break_low_random, E_break_high = break_high_random, E_cut = cut_random, exponent = exponent_guess, fit = 'triple', maxit=10000, e_min = e_min, e_max = e_max)
				#print(which_fit_random)
				#while which_fit_random is None:
				#	which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, beta = beta_random, gamma1=g1_random, gamma2=g2_random, gamma3 = g3_random, E_break_low=break_low_random, E_break_high = break_high_random, E_cut = cut_random, maxit=10000, e_min = e_min, e_max = e_max)
				if which_fit_random is None:
					break
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					redchi_final = redchi_random
					which_fit_final = which_fit_random[0]




	if which_fit == 'best_cb':
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		#which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_low_guess, E_cut = cut_guess, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess,  gamma1=g1_guess, gamma2=g2_guess, E_break_low=break_low_guess, E_cut = cut_guess, exponent = exponent_guess, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
		
		redchi_guess = which_fit_guess[1]
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]
		
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0] 
				#gamma2 should always be more negative (smaller) than gamma1
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_low_random = np.random.choice(break_array_low,1)[0]
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				#which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_low_random, E_cut = cut_random, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random,  gamma1=g1_random, gamma2=g2_random, E_break_low=break_low_random, E_cut = cut_random, exponent = exponent_guess, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
				if which_fit_random is None:
					break
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					if which_fit_random[0] == 'double':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					if which_fit_random[0] == 'cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					

	if which_fit == 'best_sb':
	#first check the redchi and if the break is outside of the energy range using the guess values then compare the random values to these 
	#if redchi is better, substitute values
		#which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_low_guess, E_cut = None, fit = 'best_sb', maxit=10000, e_min = e_min, e_max = e_max)
		which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break_low=break_low_guess, fit = 'best_sb', maxit=10000, e_min = e_min, e_max = e_max)
		
		redchi_guess = which_fit_guess[1]
		redchi_final = redchi_guess
		which_fit_final = which_fit_guess[0]
		result_final = which_fit_guess[2]
		
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0]
				#gamma2 should always be more negative (smaller) than gamma1
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_low_random = np.random.choice(break_array_low,1)[0]
				c1_random = np.random.choice(c1_array,1)[0]
				#which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_low_random, E_cut = None, fit = 'best_sb', maxit=10000, e_min = e_min, e_max = e_max)
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break_low=break_low_random, fit = 'best_sb', maxit=10000, e_min = e_min, e_max = e_max)
				if which_fit_random is None:
					break
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					if which_fit_random[0] == 'double':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						

	
	if which_fit == 'double_cut':
		result_cut_guess = pl_fit.cut_break_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, gamma2=g2_guess, c1=c1_guess, alpha=alpha_guess, E_break=break_low_guess, E_cut = cut_guess, exponent = exponent_guess, print_report=False, maxit=10000)
		breakp_cut = result_cut_guess.beta[4]
		cut_b = result_cut_guess.beta[5]
	
		if breakp_cut < e_min or breakp_cut > e_max:
			print('The break point is outside of the energy range')
			which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break_low=break_low_guess, E_cut = cut_guess, exponent = exponent_guess, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
			redchi_guess = which_fit_guess[1]
			redchi_final = redchi_guess
			which_fit_final = which_fit_guess[0]
			result_final = which_fit_guess[2]
			
		if breakp_cut >= e_min and breakp_cut <=e_max:
			if cut_b <=e_min or cut_b >=e_max:
				# The breaks are checked by redchi
				#which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, gamma1=g1_guess, gamma2=g2_guess, E_break=break_low_guess, E_cut = None, fit = 'best_cb', maxit=10000, e_min = e_min, e_max = e_max)
				which_fit_guess = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_guess, alpha=alpha_guess, beta = beta_guess, gamma1=g1_guess, gamma2=g2_guess, E_break_low=break_low_guess, E_cut = cut_b, exponent = exponent_guess, fit = 'double_cut', maxit=10000, e_min = e_min, e_max = e_max)
		
				redchi_guess = which_fit_guess[1]
				redchi_final = redchi_guess
				which_fit_final = which_fit_guess[0]
				result_final = which_fit_guess[2]
				
			if cut_b>e_min and cut_b< e_max:
				which_fit_final = 'double_cut'
				result_final = result_cut_guess
				redchi_guess  = result_cut_guess.res_var
				redchi_final = redchi_guess
			
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0]
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_low_random = np.random.choice(break_array_low,1)[0]
				cut_random = np.random.choice(cut_array, 1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]

				#which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break=break_low_random, E_cut = cut_random, fit = 'double_cut', maxit=10000, e_min = e_min, e_max = e_max)
				which_fit_random = check_redchi(spec_e, spec_flux, e_err, flux_err, c1=c1_random, alpha=alpha_random, gamma1=g1_random, gamma2=g2_random, E_break_low=break_low_random, E_cut = cut_random, exponent = exponent_guess, fit = 'double_cut', maxit=10000, e_min = e_min, e_max = e_max)
				if which_fit_random is None:
					break
				redchi_random = which_fit_random[1]
				if redchi_random < redchi_final:
					result_final = which_fit_random[2]
					if which_fit_random[0] == 'single':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					if which_fit_random[0] == 'double':		
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					if which_fit_random[0] == 'cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
						
					if which_fit_random[0] == 'double_cut':
						redchi_final = redchi_random
						which_fit_final = which_fit_random[0]
					
	
	if which_fit == 'double':
		# even if the which_fit is double we need to check first if the break point is outside of the energy range. In that case we have to change it to single.
		result_double_guess = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, gamma2=g2_guess, c1 = c1_guess, alpha = alpha_guess, E_break = break_low_guess, maxit=10000)
		breakp_1 = result_double_guess.beta[4]
		
		if breakp_1 < e_min or breakp_1 > e_max:
			print('The break point is outside of the energy range')
			which_fit_final = 'single'
			result_single_pl_guess = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, c1=c1_guess)
			result_final = result_single_pl_guess
			redchi_guess  = result_single_pl_guess.res_var  
			redchi_final = redchi_guess
			
		if breakp_1 >= e_min and breakp_1 <=e_max:
			which_fit_final = 'double'
			result_final = result_double_guess
			redchi_guess  = result_double_guess.res_var
			redchi_final = redchi_guess
		
		if use_random :
			#print('USING RANDOM double')
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				g2_random = np.random.choice(gamma2_array, 1)[0]
				if g1_random<g2_random:
					gamma = g1_random
					g1_random = g2_random
					g2_random = gamma
				alpha_random = np.random.choice(alpha_array, 1)[0]
				break_low_random = np.random.choice(break_array_low,1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]
				result_double_random = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = g1_random, gamma2 = g2_random, c1 = c1_random, alpha = alpha_random, E_break = break_low_random, maxit=10000)
				breakp_1 = result_double_random.beta[4]
				if breakp_1 < e_min or breakp_1 > e_max:
					result_single_pl_random = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_random, c1=c1_random)
					redchi_random  = result_single_pl_random.res_var  
					if redchi_random < redchi_final:
						which_fit_final = 'single'
						redchi_final = redchi_random
						result_final = result_single_pl_random
						
				if breakp_1 >= e_min and breakp_1 <=e_max:
					redchi_random = result_double_random.res_var
					if redchi_random < redchi_final:
						which_fit_final = 'double'
						redchi_final = redchi_random
						result_final =result_double_random
						

	if which_fit == 'cut':
		# even if the which_fit is double we need to check first if the break point is outside of the energy range. In that case we have to change it to single.
		result_cut_guess = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess,  c1 = c1_guess,  E_cut = cut_guess, exponent = exponent_guess, maxit=10000)
		cut = result_cut_guess.beta[2]
		
		if cut < e_min or cut > e_max:
			print('The cutoff point is outside of the energy range')
			which_fit_final = 'single'
			result_single_pl_guess = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, c1=c1_guess)
			result_final = result_single_pl_guess
			redchi_guess  = result_single_pl_guess.res_var  
			redchi_final = redchi_guess
			
		if cut >= e_min and cut <=e_max:
			which_fit_final = 'cut'
			result_final = result_cut_guess
			redchi_guess  = result_cut_guess.res_var
			redchi_final = redchi_guess
		
		if use_random :
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				cut_random = np.random.choice(cut_array,1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]
				result_cut_random = pl_fit.cut_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1 = g1_random,  c1 = c1_random,  E_cut = cut_random, exponent = exponent_guess, maxit=10000)
				cut = result_cut_random.beta[2]
				if cut < e_min or cut > e_max:
					result_single_pl_random = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_random, c1=c1_random)
					redchi_random  = result_single_pl_random.res_var  
					if redchi_random < redchi_final:
						which_fit_final = 'single'
						redchi_final = redchi_random
						result_final = result_single_pl_random
						
				if cut >= e_min and cut <=e_max:
					redchi_random = result_cut_random.res_var
					if redchi_random < redchi_final:
						which_fit_final = 'cut'
						redchi_final = redchi_random
						result_final =result_cut_random

	
	if which_fit == 'single':
		which_fit_final = 'single'
		result_single_pl_guess = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_guess, c1=c1_guess)
		result_final = result_single_pl_guess
		redchi_guess  = result_single_pl_guess.res_var  
		redchi_final = redchi_guess
		
		if use_random:
			for i in range(iterations):
				#need [0] because it's an array
				g1_random = np.random.choice(gamma1_array, 1)[0]
				c1_random = np.random.choice(c1_array, 1)[0]
				result_single_pl_random = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=g1_random, c1=c1_random)
				redchi_random  = result_single_pl_random.res_var  
				if redchi_random < redchi_final:
					redchi_final = redchi_random
					result_final = result_single_pl_random
					
	
	result_dataframe = pd.DataFrame({"Final fit type":which_fit_final}, index = [0])
	result = result_final
	#result.pprint()
	#print(which_fit_final)
	if which_fit_final == 'single':
		#result_single_pl = pl_fit.power_law_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err, gamma1=gamma1_final, c1=c1_final)
		result_single_pl = result_final
		result        = result_final
		# dof_single                   = len(x) - len(result_single_pl.beta)
		redchi_single  = result_single_pl.res_var  #result_single_pl.sum_square / dof_single
		c1          = result_single_pl.beta[0]
		gamma1      = result_single_pl.beta[1]
		dof         = len(spec_e) - len(result_single_pl.beta)
		t_val       = studentt.interval(0.95, dof)[1]
		errors      = t_val * result_single_pl.sd_beta  #np.sqrt(np.diag(final_fit.cov_beta))
		#errors      =  result_single_pl.sd_beta
		gamma1_err  = errors[1]

		

		if detailed_legend:
			ax.plot([], [], ' ', label="Single pl")
			ax.plot([], [], ' ', label=r'$\mathregular{\chi²=}$%5.2f' %round(redchi_single, ndigits=2))
			ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$' +"{:.2e}".format(c1)+"/(s cm² sr MeV)")
			#ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$%5.2f' %round(c1, ndigits=2)+"/(s cm² sr MeV)")
			


		ax.plot(xplot, pl_fit.simple_pl([c1, gamma1], xplot), '-', color=color[direction], label=r'$\mathregular{\gamma=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err))
		ax.plot(xplot, pl_fit.simple_pl([c1, gamma1], xplot), '--k', zorder=10)




		result_dataframe["Reduced chi sq"] = redchi_single
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = None
		result_dataframe["Gamma2 err"] = None
		result_dataframe["Gamma3"] = None
		result_dataframe["Gamma3 err"] = None
		result_dataframe["Break point 1 [MeV]"] = None
		result_dataframe["Break point 1 err [MeV]"] = None 
		result_dataframe["Break point 2 [MeV]"] = None
		result_dataframe["Break point 2 err [MeV]"] = None 
		result_dataframe["Exponential cutoff point [MeV]"] = None
		result_dataframe["Cutoff err [MeV]"] = None
		result_dataframe["Alpha"] = None
		result_dataframe["Beta"] = None
		result_dataframe["Exponent"] = None

	
	if which_fit_final == 'double':
		#result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_double = result_final
		result        = result_final
		breakp_1        = result_double.beta[4]
		alpha         = result_double.beta[3]
		dof           = len(spec_e) - len(result_double.beta)
		redchi_double = result_double.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_double.sd_beta  #np.sqrt(np.diag(result_double.cov_beta))
		#errors     = result_double.sd_beta
		breakp_1_err = errors[4]
		c1         = result_double.beta[0]
# NINA SAID THERE IS AN ISSUE HERE. THE CODE BREAKS SOMEHOW.gamma1 is not found. 02.11.23
		if alpha > 0 :
			gamma1     = result_double.beta[1]
			gamma1_err = errors[1]
			gamma2     = result_double.beta[2]
			gamma2_err = errors[2]
			

		if alpha < 0 :
			gamma1     = result_double.beta[2]
			gamma1_err = errors[2]
			gamma2     = result_double.beta[1]
			gamma2_err = errors[1]


		
#Added this on Nov 18 2024 change if causes problems !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			
		if gamma1<gamma2 and alpha>0:
			alpha = -abs(alpha)

		elif gamma2<gamma1 and alpha<0:
			alpha = abs(alpha)

			
		##if gamma1<gamma2:
		#	gamma_temp = gamma1
		#	gamma_temp_err = gamma1_err 
		#	gamma1 = gamma2
		#	gamma1_err = gamma2_err
		#	gamma2 = gamma_temp
		#	gamma2_err = gamma_temp_err
			
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! result double seems to be empty sometimes and it causes the fit to crash
		fit_plot = pl_fit.double_pl_func(result_double.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan

		if detailed_legend:
			ax.plot([], [], ' ', label="double pl")
			ax.plot([], [], ' ', label=r'$\mathregular{\chi²=}$%5.2f' %round(redchi_double, ndigits=2))
			ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$' +"{:.2e}".format(c1)+"/(s cm² sr MeV)")
			#ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$%5.2f' %round(c1, ndigits=2)+"/(s cm² sr MeV)")
			


		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\gamma_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{\gamma_2=}$%5.2f' %round(gamma2, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma2_err)+'\n'+r'$\mathregular{\alpha=}$%5.2f' %round(alpha, ndigits=2))#, lw=lwd)
		if len(str(breakp_1*1e3).split('.')[0])>3:
			ax.axvline(x=breakp_1, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp_1, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_1_err, ndigits=1))+' MeV')
		elif len(str(breakp_1*1e3).split('.')[0])<=3:
			ax.axvline(x=breakp_1, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp_1*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_1_err*1e3, ndigits=1))+' keV')
	

		result_dataframe["Reduced chi sq"] = redchi_double
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = gamma2
		result_dataframe["Gamma2 err"] = gamma2_err
		result_dataframe["Gamma3"] = None
		result_dataframe["Gamma3 err"] = None
		result_dataframe["Break point 1 [MeV]"] = breakp_1
		result_dataframe["Break point 1 err [MeV]"] = breakp_1_err
		result_dataframe["Break point 2 [MeV]"] = None
		result_dataframe["Break point 2 err [MeV]"] = None
		result_dataframe["Exponential cutoff point [MeV]"] = None
		result_dataframe["Cutoff err [MeV]"] = None
		result_dataframe["Alpha"] = alpha
		result_dataframe["Beta"] = None
		result_dataframe["Exponent"] = None



	if which_fit_final == 'cut':
		#result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_cut = result_final
		result        = result_final
		cut        = result_cut.beta[2]
			#shoud maybe make distinction between cut from cut pl and cut from cut double pl
		dof           = len(spec_e) - len(result_cut.beta)
		redchi_cut = result_cut.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_cut.sd_beta  #np.sqrt(np.diag(result_double.cov_beta))
		#errors     = result_cut.sd_beta
		c1         = result_cut.beta[0]
		gamma1     = result_cut.beta[1]
		gamma1_err = errors[1]
		cut_err = errors[2]
		exponent = result_cut.beta[3]
		
			
		fit_plot = pl_fit.cut_pl_func(result_cut.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan

		if detailed_legend:
			ax.plot([], [], ' ', label="Single pl + exp cutoff")
			ax.plot([], [], ' ', label="exponent: "+str(round(exponent, ndigits=2)))
			ax.plot([], [], ' ', label=r'$\mathregular{\chi²=}$%5.2f' %round(redchi_cut, ndigits=2))
			#ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$%5.2f' %round(c1, ndigits=2)+"/(s cm² sr MeV)")
			ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$' +"{:.2e}".format(c1)+"/(s cm² sr MeV)")
			
			


		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\gamma_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err))#, lw=lwd)
		if len(str(cut*1e3).split('.')[0])>3:
			ax.axvline(x=cut, color='purple', linestyle='--', label=r'$\mathregular{E_c=}$ '+str(round(cut, ndigits=1))+'\n'+r"$\pm$"+str(round(cut_err, ndigits=1))+' MeV')
		elif len(str(cut*1e3).split('.')[0])<=3:
			ax.axvline(x=cut, color='purple', linestyle='--', label=r'$\mathregular{E_c=}$ '+str(round(cut*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(cut_err*1e3, ndigits=1))+' keV')
		

		result_dataframe["Reduced chi sq"] = redchi_cut
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = None
		result_dataframe["Gamma2 err"] = None
		result_dataframe["Gamma3"] = None
		result_dataframe["Gamma3 err"] = None
		result_dataframe["Break point 1 [MeV]"] = None
		result_dataframe["Break point 1 err [MeV]"] = None
		result_dataframe["Break point 2 [MeV]"] = None
		result_dataframe["Break point 2 err [MeV]"] = None
		result_dataframe["Exponential cutoff point [MeV]"] = cut
		result_dataframe["Cutoff err [MeV]"] = cut_err
		result_dataframe["Alpha"] = None
		result_dataframe["Beta"] = None
		result_dataframe["Exponent"] = exponent

		
		
	if which_fit_final == 'double_cut':
		#result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_cut = result_final
		result        = result_final
		cut			  = result_cut.beta[5]
		breakp_1        = result_cut.beta[4]
		alpha         = result_cut.beta[3]
		dof           = len(spec_e) - len(result_cut.beta)
		redchi_cut = result_cut.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_cut.sd_beta  #np.sqrt(np.diag(result_cut.cov_beta))
		#errors     = result_cut.sd_beta
		breakp_1_err = errors[4]
		cut_err = errors[5]
		c1         = result_cut.beta[0]
		exponent = result_cut.beta[6]
	
		if alpha > 0 :
			gamma1     = result_cut.beta[1]
			gamma1_err = errors[1]
			gamma2     = result_cut.beta[2]
			gamma2_err = errors[2]
			
			
		if alpha < 0 :
			gamma1     = result_cut.beta[2]
			gamma1_err = errors[2]
			gamma2     = result_cut.beta[1]
			gamma2_err = errors[1]
			

		if gamma1<gamma2 and alpha>0:
			alpha = -abs(alpha)

		elif gamma2<gamma1 and alpha<0:
			alpha = abs(alpha)

	
			
		fit_plot = pl_fit.cut_break_pl_func(result_cut.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan

		if detailed_legend:
			ax.plot([], [], ' ', label="double pl + exp cutoff")
			ax.plot([], [], ' ', label="exponent: "+str(round(exponent, ndigits=2)))
			ax.plot([], [], ' ', label=r'$\mathregular{\chi²=}$%5.2f' %round(redchi_cut, ndigits=2))
			#ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$%5.2f' %round(c1, ndigits=2)+"/(s cm² sr MeV)")
			ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$' +"{:.2e}".format(c1)+"/(s cm² sr MeV)")
			
	

		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\gamma_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{\gamma_2=}$%5.2f' %round(gamma2, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma2_err)+'\n'+r'$\mathregular{\alpha=}$%5.2f' %round(alpha, ndigits=2))#, lw=lwd)
		if len(str(breakp_1*1e3).split('.')[0])>3:
			ax.axvline(x=breakp_1, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp_1, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_1_err, ndigits=1))+' MeV')
		if len(str(cut*1e3).split('.')[0])>3:
			ax.axvline(x=cut, color='purple', linestyle='--', label=r'$\mathregular{E_c=}$ '+str(round(cut, ndigits=1))+'\n'+r"$\pm$"+str(round(cut_err, ndigits=1))+' MeV')
		if len(str(breakp_1*1e3).split('.')[0])<=3:
			ax.axvline(x=breakp_1, color='blue', linestyle='--', label=r'$\mathregular{E_b=}$ '+str(round(breakp_1*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_1_err*1e3, ndigits=1))+' keV')
		if len(str(cut*1e3).split('.')[0])<=3:
			ax.axvline(x=cut, color='purple', linestyle='--', label=r'$\mathregular{E_c=}$ '+str(round(cut*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(cut_err*1e3, ndigits=1))+' keV')



		result_dataframe["Reduced chi sq"] = redchi_cut
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
		result_dataframe["Gamma1"] = gamma1
		result_dataframe["Gamma1 err"] = gamma1_err
		result_dataframe["Gamma2"] = gamma2
		result_dataframe["Gamma2 err"] = gamma2_err
		result_dataframe["Gamma3"] = None
		result_dataframe["Gamma3 err"] = None
		result_dataframe["Break point 1 [MeV]"] = breakp_1
		result_dataframe["Break point 1 err [MeV]"] = breakp_1_err
		result_dataframe["Break point 2 [MeV]"] = None
		result_dataframe["Break point 2 err [MeV]"] = None
		result_dataframe["Exponential cutoff point [MeV]"] = cut
		result_dataframe["Cutoff err [MeV]"] = cut_err
		result_dataframe["Alpha"] = alpha
		result_dataframe["Beta"] = None
		result_dataframe["Exponent"] = exponent

		
	


	if which_fit_final == 'triple':
		#result_double = pl_fit.double_pl_fit(x = spec_e, y = spec_flux, xerr = e_err, yerr = flux_err,  gamma1=gamma1_final, gamma2=gamma2_final, c1 = c1_final, alpha=alpha_final,E_break=break_final, maxit=10000)
		result_triple = result_final
		result        = result_final
		breakp_2		  = result_triple.beta[7]
		breakp_1        = result_triple.beta[6]
		alpha         = result_triple.beta[4]
		beta          = result_triple.beta[5]
		dof           = len(spec_e) - len(result_triple.beta)
		redchi_triple = result_triple.res_var
		t_val      = studentt.interval(0.95, dof)[1]
		errors     = t_val * result_triple.sd_beta  #np.sqrt(np.diag(result_cut.cov_beta))
		#errors     =  result_triple.sd_beta
		breakp_1_err = errors[6]
		breakp_2_err = errors[7]
		c1         = result_triple.beta[0]
		gamma1     = result_triple.beta[1]
		gamma1_err = errors[1]
		gamma2     = result_triple.beta[2]
		gamma2_err = errors[2]
		gamma3 = result_triple.beta[3]
		gamma3_err = errors[3]


	
		if alpha > 0 and beta > 0 :
			gamma1     = result_triple.beta[1]
			gamma1_err = errors[1]
			gamma2     = result_triple.beta[2]
			gamma2_err = errors[2]
			gamma3 = result_triple.beta[3]
			gamma3_err = errors[3]
			
		if alpha < 0 and beta> 0:
			gamma1     = result_triple.beta[2]
			gamma1_err = errors[2]
			gamma2     = result_triple.beta[1]
			gamma2_err = errors[1]
			gamma3 = result_triple.beta[3]
			gamma3_err = errors[3]
			

		if beta < 0 and alpha >0:
			gamma1     = result_triple.beta[1]
			gamma1_err = errors[1]
			gamma2     = result_triple.beta[3]
			gamma2_err = errors[3]
			gamma3 = result_triple.beta[2]
			gamma3_err = errors[2]

		if alpha < 0 and beta < 0 :
			gamma1     = result_triple.beta[3]
			gamma1_err = errors[3]
			gamma2     = result_triple.beta[2]
			gamma2_err = errors[2]
			gamma3 = result_triple.beta[1]
			gamma3_err = errors[1]
			
	
		
		if gamma1>gamma2 and gamma2>gamma3:
			#if alpha>0 and beta>0:
				#all ok
			if alpha<0 and beta>0:
				alpha = abs(alpha)
			if alpha>0 and beta<0:
				beta = abs(beta)
			if alpha<0 and beta<0:
				#something went wrong
				alpha = abs(alpha)
				beta = abs(beta)

		if gamma1>gamma2 and gamma2<gamma3:
			if alpha>0 and beta>0:
				beta = -abs(beta)
			#if alpha >0 and beta<0:
				#all ok
			if alpha<0 and beta>0:
				a = alpha
				b = beta
				alpha = b
				beta = a
			if alpha <0 and beta<0:
				alpha = abs(alpha)

		if gamma1<gamma2 and gamma2>gamma3:
			if alpha >0 and beta >0 :
				alpha = -abs(alpha)
			if alpha >0 and beta <0:
				a = alpha
				b = beta
				alpha = b
				beta = a
			#if alpha<0 and beta>0:
				#all ok
			if alpha<0 and beta<0:
				beta = -abs(beta)


			

			
		fit_plot = pl_fit.triple_pl_func(result_triple.beta, xplot)
		fit_plot[fit_plot == 0] = np.nan

		if detailed_legend:
			ax.plot([], [], ' ', label="Triple pl")
			ax.plot([], [], ' ', label=r'$\mathregular{\chi²=}$%5.2f' %round(redchi_triple, ndigits=2))
			#ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$%5.2f' %round(c1, ndigits=2)+"/(s cm² sr MeV)")
			ax.plot([], [], ' ', label=r'$\mathregular{I_0=}$' +"{:.2e}".format(c1)+"/(s cm² sr MeV)")
			
			

		ax.plot(xplot, fit_plot, '-b', label=r'$\mathregular{\gamma_1=}$%5.2f' %round(gamma1, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma1_err)+'\n'+r'$\mathregular{\gamma_2=}$%5.2f' %round(gamma2, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma2_err)+'\n'+r'$\mathregular{\gamma_3=}$%5.2f' %round(gamma3, ndigits=2)+r"$\pm$"+'{0:.2f}'.format(gamma3_err)+'\n'+r'$\mathregular{\alpha=}$%5.2f' %round(alpha, ndigits=2)+'\n'+r'$\mathregular{\beta=}$%5.2f' %round(beta, ndigits=2))#, lw=lwd)
		if len(str(breakp_1*1e3).split('.')[0])>3:
			ax.axvline(x=breakp_1, color='blue', linestyle='--', label=r'$\mathregular{E_b1=}$ '+str(round(breakp_1, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_1_err, ndigits=1))+' MeV')
		
		if len(str(breakp_1*1e3).split('.')[0])<=3:
			ax.axvline(x=breakp_1, color='blue', linestyle='--', label=r'$\mathregular{E_b1=}$ '+str(round(breakp_1*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_1_err*1e3, ndigits=1))+' keV')
		
		
		if len(str(breakp_2*1e3).split('.')[0])>3:
			ax.axvline(x=breakp_2, color='purple', linestyle='--', label=r'$\mathregular{E_b2=}$ '+str(round(breakp_2, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_2_err, ndigits=1))+' MeV')

		if len(str(breakp_2*1e3).split('.')[0])<=3:
			ax.axvline(x=breakp_2, color='purple', linestyle='--', label=r'$\mathregular{E_b2=}$ '+str(round(breakp_2*1e3, ndigits=1))+'\n'+r"$\pm$"+str(round(breakp_2_err*1e3, ndigits=1))+' keV')

		result_dataframe["Reduced chi sq"] = redchi_triple
		result_dataframe["c1"] = c1
		result_dataframe["c1 err"] = errors[0]
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
		result_dataframe["Exponential cutoff point [MeV]"] = None
		result_dataframe["Cutoff err [MeV]"] = None
		result_dataframe["Alpha"] = alpha
		result_dataframe["Beta"] = beta
		result_dataframe["Exponent"] = None


		
		#print('WHICH FIT   '+which_fit)

	


	result_dataframe['E min [MeV]'] = e_min
	result_dataframe['E max [MeV]'] = e_max
	#Debugging in case of errors
	#print(result.beta, 'beta')
	#print(result.sd_beta, 'sd_beta')
	#print(result.cov_beta, 'cov_beta')
	#print(result.delta, 'delta')
	#print(result.eps, 'eps')
	#print(result.xplus, 'xplus')
	#print(result.y, 'y')
	#print(result.res_var, 'res_var')
	#print(result.sum_square, 'sum_square')
	#print(result.sum_square_delta, 'sum_square_delta')
	#print(result.sum_square_eps, 'sum_square_eps')
	#print(result.inv_condnum, 'inv_condnum')
	#print(result.rel_error, 'rel_error')
	#print(result.work, 'work')
	#print(result.work_ind, 'work_ind')
	#print(result.info, 'info')
	#print(result.stopreason, 'stopreason')
	
	# save result to pickle file 
	if path != None:
		#pfname =  '.p'
		with open(path, 'wb') as f:
			pickle.dump(result, f)

	# save the fitting variables
	if path2 != None:
		result_dataframe.to_csv(path2, sep = ";")

	#c1 = result.beta[0]
	#print('The fitting variable c1 is ' ,c1)
	return result
	
	
def DETERMINE_PEAK_SPEC(flux, flux_err):
    '''
    returns the peak flux of each energy channel in flux
    returns the assocated peak times
    flux: flux array, flux time series of all energy channels
    flux covers only the time interval of interest and is already time averaged
    '''
    # spec_energy = flux.columns
    peak_times = []
    peak_flux = []
    peak_flux_err = []
    for ch in range(flux.values.shape[1]):
        if any(np.isfinite(flux.values[:,ch])):
            max_ind = np.where(np.nanmax(flux.values[:,ch]) == flux.values[:,ch])[0][0]
            peak_times.append(flux.index[max_ind])
            peak_flux.append(flux.values[max_ind,ch])
            peak_flux_err.append(flux_err.values[max_ind,ch])
        else:
            peak_flux.append(np.nan)
            peak_times.append(pd.NaT)
            peak_flux_err.append(np.nan)
            # TO DO: ADD ALSO METADATA TO spec_data, multiply by 1e2

    spec_data = pd.DataFrame({'peak_time':peak_times, 'flux':peak_flux, 'flux_err':peak_flux_err})#, 'spec_energy':spec_energy})
    return spec_data
