# NEW VERSION OF THE GENERAL VERSION. iN THE MAKING.

# from turtle import title
# import numpy as np
import pandas as pd
# import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as pltt
# from sunpy.coordinates import get_horizons_coord
import fit_spectra.make_the_fit as fitting
# from make_the_fit_tripl import  MAKE_THE_FIT
# from make_the_fit import closest_values
# from make_the_fit import find_c1
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import fit_spectra.functions_for_spectra_fit as sf
import os
from pathlib import Path


def run_the_fit(path, data, save, use_filename_as_title = False, channels_to_exclude = None, plot_title = '', x_label = 'Intensity [/]', y_label = 'Energy [MeV]', legend_title = '', data_label_for_legend = 'data', which_fit = 'best', e_min = None, e_max = None, g1_guess = -1.9, g2_guess = -2.5, g3_guess = -4., c1_guess = 1000, alpha_guess = 10, beta_guess = 10, break_guess_low = 0.6, break_guess_high = 1.2, cut_guess = 1.2, exponent_guess = 2, use_random = True, iterations = 20 , legend_details = False):
    """This function calls the make_the_fit functoin that creates the fit. It plots and saves the results of the fit.

    Args:
        path (string): The full path to the file that will be used for the fit inclufing the file name. e.g. '/home/admin/folder1/folder2/file.csv' 
        data (dataframe): The data that will be fit (energy, energy uncertainty, intensity and intensity uncertainty). The columns should be named 'Energy', 'Intensity', 'E_err', 'I_err'
                          the order does not matter.
        save (bool): if True the plots and fit results will be saved. Note: the original filename and possible new title of the plot will be used as the file name. Possible spaces will be replaced by '_'
        channels_to_exclude (list): list of channels to be excluded. List of indices corresponding to the cannels. Defaults to 'None'
        plot_title (str, optional): The title of the plot, will also be used when saving the results. Defaults to ''.
        x_label (str, optional): label for the x axis. Defaults to 'Intensity [/]'.
        y_label (str, optional): label for the y axis. Defaults to 'Energy [MeV]'.
        legend_title (str, optional): title for the legend. Defaults to ''.
        which_fit (str, optional): which_fit options: 'single' will force a single pl fit to the data
		  			'double' will force a double pl fit to the data but ONLY if the break point is within the energy range otherwise a sigle pl fit will be produced instead
		  			'best_sb' will choose automatically the best fit type between single and double by comparing the redchis of the fits
		    		'cut' will produce a single pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a double or single pl will be fit instead
			  		'double_cut' will produce a double pl fit with an exponential cutoff point. If the cutoff point is outside of the energy range a double or single pl will be fit instead
				  	'best_cb'. Defaults to 'best'.
					'triple' will force a triple pl fit. If this is not possible, the function will check which is the next best option.
					'best' will choose automatically the best fit type by comparing the redchis of the fits
                    Defaults to 'best'
        e_min (float, optional): The lower energy limit for the fit. Defaults to None.
		e_max (float, optional): The upper energy limit for the fit. Defaults to None.
        g1_guess (float, optional): The slope of the single pl fit or the first part of a double/triple pl fit. Defaults to -1.9.
		g2_guess (float, optional): The slope of the second part of a double/triple pl fit. gamma2 < gamma1. Defaults to -2.5. 
		g3_guess (int, optional): The slope of the third part of a double/triple pl fit. gamma3 < gamma2 < gamma1. Defaults to -4.
		c1_guess (int, optional): The intensity/flux value at 0.1 MeV. Defaults to 1000.
		alpha_guess (int, optional): The smoothness of the transition between gamma1 and gamma2. Defaults to 10.
		beta_guess (int, optional): The smoothness of the transition between gamma3 and gamma2. Defaults to 10.
		break_guess_low (float, optional): Guess value for the energy correponding to the break in the double pl and first break for the triple pl. Input in MeV. Defaults to 0.6.
		break_guess_high (float, optional): Guess value for the energy correponding to the second break for the triple pl.Input in MeV.  Defaults to 1.2.
		cut_guess (float, optional): Guess value for the energy corresponding to the exponential cutoff. Input in MeV.  Defaults to 1.2.
		use_random (bool, optional): If True the fitting function will, in addition to the guess values, choose random values from a predifined list of values for each variable. 
					These values are chosen close to the guess values. Defaults to True.
		iterations (int, optional): The number of times the function will choose random values to use in the fit to the data. Defaults to 20.

    """
    
    #file_name = os.path.basename(path) this gives you the filename with the extention e.g. .csv
    file_name = Path(path).stem # this gives you the filename without extension

    folder_path = os.path.dirname(path) # path to the folder without the last /
    if use_filename_as_title:
        plot_title = file_name.replace("_", " ")

    # in make the fit we have two paths. one for pickle files (deleted from here) and path2 to save the fit variables. Needs to be updated in the future
    
    name_string = ''
    if use_filename_as_title == False:
        name_string = plot_title.replace(" ", "_")

    fit_var_path = folder_path+'/'+file_name+'_'+name_string+'_fit-result-variables_'+which_fit+'.csv'

    all_data = data

    dataframe_to_fit = data
    dataframe_to_exclude = pd.DataFrame()

    if channels_to_exclude != None:
        args = sf.exclude_channels(data, channels_to_exclude) #returns two dataframes #1 has the good data (to fit) #2 has the excluded channels
        dataframe_to_fit = args[0]
        dataframe_to_exclude = args[1]
       

    x_data = dataframe_to_fit['Energy'] # energy for spectra
    y_data   = dataframe_to_fit['Intensity']
      
    x_err = None
    y_err = None

    if 'E_err' in dataframe_to_fit:
        x_err  = dataframe_to_fit['E_err']
    
    if 'I_err' in dataframe_to_fit:
        y_err    = dataframe_to_fit['I_err'] 

    #checking if uncertainties for energy and intensity are NaNs
    if x_err.isnull().all():
        x_err = None
        
    if y_err.isnull().all():
        y_err = None
       
    

    f, ax = plt.subplots(1, figsize=(6, 5), dpi = 300)
    
    fit = fitting.MAKE_THE_FIT(x_data, y_data, x_err, y_err, ax, direction='sun', e_min = e_min, e_max = e_max, which_fit=which_fit, g1_guess=g1_guess, g2_guess=g2_guess, g3_guess = g3_guess, alpha_guess=alpha_guess, beta_guess = beta_guess, break_low_guess=break_guess_low, break_high_guess = break_guess_high, cut_guess = cut_guess, c1_guess = c1_guess, exponent_guess = exponent_guess, use_random = use_random, iterations = iterations, path = None, path2 = fit_var_path, detailed_legend = legend_details)
    
    
    ax.errorbar(x_data, y_data, xerr = x_err, yerr=y_err, marker='o', markersize= 3 , linestyle='', color='red', alpha = 0.5, label=data_label_for_legend, zorder = -1)
    if channels_to_exclude != None:
        ax.errorbar(dataframe_to_exclude['Energy'], dataframe_to_exclude['Intensity'], xerr = dataframe_to_exclude['E_err'], yerr=dataframe_to_exclude['I_err'], marker='o', markersize= 3 , linestyle='', color='gray', alpha = 0.5, label='excluded channels', zorder = -1)
   
 # REMEMBER: when choosing the ranges don't use uncertainties because they can be None
    x_range_min = min(all_data['Energy'])
    x_range_max = max(all_data['Energy'])
    

    ax.set_xscale('log')
    ax.set_yscale('log')
        
    ax.set_xlim(x_range_min-(x_range_min/2), x_range_max+(x_range_max/2))
        #ax.set_ylim(y_range_min-(y_range_min/2), y_range_max+(y_range_max/2))
    
    locmin = pltt.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(pltt.NullFormatter())

    plt.legend(title=''+legend_title+'',  prop={'size': 7})
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(plot_title)
    
    if save:
        plt.savefig(folder_path+'/'+file_name+'_'+name_string+'_fit-plot_'+which_fit+'.png', dpi=300)
        

    plt.show()

    results = pd.read_csv(fit_var_path, sep = ';')

    #print(results.columns)
    print('STOP REASON : ', fit.stopreason)
    sf.print_results(results)
    
