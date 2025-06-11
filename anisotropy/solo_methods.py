import numpy as np
import pandas as pd
from numba import njit, prange
from solo_epd_loader import epd_load
import datetime as dt
from anisotropy.polarity_plotting import polarity_rtn
from pandas.tseries.frequencies import to_offset
from sunpy.coordinates import get_horizons_coord, HeliographicStonyhurst
from tqdm.auto import tqdm
from seppy.loader.solo import mag_load


def solo_download_and_prepare(instrument, startdate, enddate, path, averaging, species, en_channel, bg_start, bg_end, solo_ept_ion_contamination_correction=False):
    data_product = "l2"
    df_sun_p, df_asun_p, df_north_p, df_south_p, df_sun_e, df_asun_e, df_north_e, df_south_e, energies = epd_load_data(instrument, data_product, startdate, enddate, path)
    
    if solo_ept_ion_contamination_correction:
        if (instrument == "EPT"):
            print("Correcting for ion contamination.")
            df_sun_e_orig = df_sun_e.copy()
            df_asun_e_orig = df_asun_e.copy()
            df_south_e_orig = df_south_e.copy()
            df_north_e_orig = df_north_e.copy()
            df_sun_e, df_asun_e, df_north_e, df_south_e = calc_ept_ion_contamination_correction(df_sun_e, df_asun_e, df_north_e, df_south_e, df_sun_p, df_asun_p, df_north_p, df_south_p)
            
    if species == "p":
        df_sun = df_sun_p
        df_asun = df_asun_p
        df_north = df_north_p
        df_south = df_south_p
        t_str = "DELTA_EPOCH"
        if instrument == "EPT":
            ch_string = 'Ions'
            sp_str = "Ion"
        elif instrument == "HET":
            ch_string = "Protons"
            sp_str = "H"
    elif species == "e":
        df_sun = df_sun_e
        df_asun = df_asun_e
        df_north = df_north_e
        df_south = df_south_e
        ch_string = 'Electrons'
        sp_str = "Electron"
        if instrument == "EPT":
            t_str = "DELTA_EPOCH_1"
        elif instrument == "HET":
            t_str = "DELTA_EPOCH_4"
        
    en_ch_df = pd.DataFrame(energies[f'{sp_str}_Bins_Text'], columns=['energy'])
    en_ch_df.index.names = ['channel']
    en_ch_df.to_csv(f'anisotropy/channels_Solar_Orbiter_{instrument}_{species}.csv')

    df_sun = df_sun.loc[(df_sun.index >= startdate) & (df_sun.index <= enddate)]
    df_asun = df_asun.loc[(df_asun.index >= startdate) & (df_asun.index <= enddate)]
    df_north = df_north.loc[(df_north.index >= startdate) & (df_north.index <= enddate)]
    df_south = df_south.loc[(df_south.index >= startdate) & (df_south.index <= enddate)]

    if averaging is None:
        av_min = None
    else:
        split_strings = averaging.split("min")
        if len(split_strings) == 1:
            split_strings = averaging.split("s")
            if len(split_strings) == 2:
                av_min = float(split_strings[0])/60
            else:
                raise ValueError("Invalid averaging.")
        elif len(split_strings) == 2:
            av_min = float(split_strings[0])
        else:
            raise ValueError("Invalid averaging.")
            
    delta_t_mean = np.mean(np.diff(df_sun.index)/np.timedelta64(1,"s"))
    print("Cadence is {:.1f} s.".format(delta_t_mean))
    if av_min is not None:
        if av_min*60 <= 1.9*delta_t_mean:
            print("No need for averaging (cadence is {:.1f} s).".format(delta_t_mean))
            averaging = None
            av_min = None

    df_dict = epd_prepare(instrument,sp_str,df_sun.copy(),df_asun.copy(),df_north.copy(),df_south.copy(),averaging=averaging)
    df_dict_counts = epd_prepare_counts(instrument,sp_str,t_str,df_sun.copy(),df_asun.copy(),df_north.copy(),df_south.copy(),averaging=averaging)
    if solo_ept_ion_contamination_correction:
        df_sun_e_orig = df_sun_e_orig.loc[(df_sun_e_orig.index>=startdate) & (df_sun_e_orig.index<=enddate)]
        df_asun_e_orig = df_asun_e_orig.loc[(df_asun_e_orig.index>=startdate) & (df_asun_e_orig.index<=enddate)]
        df_north_e_orig = df_north_e_orig.loc[(df_north_e_orig.index>=startdate) & (df_north_e_orig.index<=enddate)]
        df_south_e_orig = df_south_e_orig.loc[(df_south_e_orig.index>=startdate) & (df_south_e_orig.index<=enddate)]
        if (instrument == "EPT" and species == "e"):
            df_dict_orig = epd_prepare(instrument,sp_str,df_sun_e_orig.copy(),df_asun_e_orig.copy(),df_north_e_orig.copy(),df_south_e_orig.copy(),averaging=averaging)
            df_dict_counts_orig = epd_prepare_counts(instrument,sp_str,t_str,df_sun_e_orig.copy(),df_asun_e_orig.copy(),df_north_e_orig.copy(),df_south_e_orig.copy(),averaging=averaging)

    # Download magnetic field data, calculate polarities, and mu_coverage
    mag_srf = solo_mag_srf(startdate,enddate,path,av_min=1/60)
    # mag_srf = solo_mag_srf(startdate,enddate,path,av_min=av_min)
    longest = "sun"
    length = len(df_dict["sun"])
    for view in ["asun","north","south"]:
        if len(df_dict[view])>length:
            longest = view
        
    # mag_srf = resample_mag_to_fluxes(df_dict[longest].index,mag_srf,averaging=averaging)
    delta_t_arr = np.diff(df_dict[longest].index)/np.timedelta64(1,"s")
    delta_t_arr = np.insert(delta_t_arr,0,delta_t_arr[0])
    mag_srf = resample_mag_to_fluxes_delta_t(df_dict[longest].index,mag_srf,delta_t_arr)

    if av_min is not None:
        pol, phi_relative, pol_times, mag_rtn = solo_polarity_preparation(startdate,enddate,av_min=av_min,path=path,mag_data_product="l2",V=400)
    else:
        pol, phi_relative, pol_times, mag_rtn = solo_polarity_preparation(startdate,enddate,av_min=1,path=path,mag_data_product="l2",V=400)
    mag_data = mag_rtn.rename(columns={'B_r':'b_rtn_0', 'B_t':'b_rtn_1','B_n':'b_rtn_2'})
    mag_data["b"] = np.linalg.norm(mag_data[["b_rtn_0","b_rtn_1","b_rtn_2"]].to_numpy(),axis=1)
    mag_data_coord = "RTN"
    coverage = calc_pa_coverage(instrument, mag_srf)
    mu_times, mu_data = mu_from_coverage(coverage)

    I_times, I_data, I_unc, sectors, en_channel_string = epd_combine_intensities(instrument,df_dict,species,en_channel,energies)
    
    I_data[I_data<0] = 0 #prevent negative values after ion contamination correction
    
    I_unc[I_unc==0] = np.nanmin(I_unc[I_unc>0]) #prevent zero uncertainties as weights
    delta_E = delta_E_array(instrument,species,energies,en_channel)
    count_str = f"{sp_str}_Rate"; flux_str=f"{sp_str}_Flux"

    bg_times, bg_I_data, bg_I_unc, bg_mu_data = get_background_data(I_times,I_data,I_unc,mu_times,mu_data,bg_start,bg_end)

    count_arr = np.nan*np.zeros((np.shape(I_data)[0],np.shape(I_data)[1],delta_E.size)); flux_arr = count_arr.copy(); t_arr = count_arr.copy()
    if isinstance(en_channel,int):
        bins = en_channel
        for j,view in enumerate(["sun","asun","north","south"]):
            if len(df_dict_counts[view])==len(I_times):
                #summed counts
                count_arr[:,j,0] = df_dict_counts[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                #summed fluxes, used to infer the effective geometric factors
                flux_arr[:,j,0] = df_dict[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                #summed accumulation times (s)
                t_arr[:,j,0] = df_dict_counts[view][t_str][t_str].to_numpy()
            elif df_dict_counts[view].index[0] == I_times[0]:
                count_arr[:-1,j,0] = df_dict_counts[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                flux_arr[:-1,j,0] = df_dict[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                t_arr[:-1,j,0] = df_dict_counts[view][t_str][t_str].to_numpy()
            elif df_dict_counts[view].index[1] == I_times[0]:
                count_arr[1:,j,0] = df_dict_counts[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                flux_arr[1:,j,0] = df_dict[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                t_arr[1:,j,0] = df_dict_counts[view][t_str][t_str].to_numpy()
            else:
                raise Exception("Timestamps not matching.")
    else:
        for i,bins in enumerate(np.arange(en_channel[0], en_channel[-1]+1)):
            for j,view in enumerate(["sun","asun","north","south"]):
                if len(df_dict_counts[view])==len(I_times):
                    count_arr[:,j,i] = df_dict_counts[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                    flux_arr[:,j,i] = df_dict[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                    t_arr[:,j,i] = df_dict_counts[view][t_str][t_str].to_numpy()
                elif df_dict_counts[view].index[0] == I_times[0]:
                    count_arr[:-1,j,i] = df_dict_counts[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                    flux_arr[:-1,j,i] = df_dict[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                    t_arr[:-1,j,i] = df_dict_counts[view][t_str][t_str].to_numpy()
                elif df_dict_counts[view].index[1] == I_times[0]:
                    count_arr[1:,j,i] = df_dict_counts[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                    flux_arr[1:,j,i] = df_dict[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                    t_arr[1:,j,i] = df_dict_counts[view][t_str][t_str].to_numpy()
                else:
                    raise Exception("Timestamps not matching.")

    #Calculate the geometric factors of each telescope+each energy channel directly from the intensity and count data.
    gf_arr = np.nanmedian(count_arr/flux_arr/delta_E/t_arr,axis=0)
    gf_arr_max = np.nanmax(count_arr/flux_arr/delta_E/t_arr,axis=0)

    #prevent negative values after ion contamination correction
    count_arr[count_arr<0] = 0
    flux_arr[flux_arr<0] = 0

    if solo_ept_ion_contamination_correction:
        if (instrument == "EPT" and species == "e"):
            count_arr_orig = np.nan*np.zeros((np.shape(I_data)[0],np.shape(I_data)[1],delta_E.size)); flux_arr_orig = count_arr_orig.copy(); t_arr_orig = count_arr_orig.copy()
            if isinstance(en_channel,int):
                bins = en_channel
                for j,view in enumerate(["sun","asun","north","south"]):
                    if len(df_dict_counts_orig[view])==len(I_times):
                        #summed counts
                        count_arr_orig[:,j,0] = df_dict_counts_orig[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                        #summed fluxes, used to infer the effective geometric factors
                        flux_arr_orig[:,j,0] = df_dict_orig[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                        #summed accumulation times (s)
                        t_arr_orig[:,j,0] = df_dict_counts_orig[view][t_str][t_str].to_numpy()
                    elif df_dict_counts_orig[view].index[0] == I_times[0]:
                        count_arr_orig[:-1,j,0] = df_dict_counts_orig[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                        flux_arr_orig[:-1,j,0] = df_dict_orig[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                        t_arr_orig[:-1,j,0] = df_dict_counts_orig[view][t_str][t_str].to_numpy()
                    elif df_dict_counts_orig[view].index[1] == I_times[0]:
                        count_arr_orig[1:,j,0] = df_dict_counts_orig[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                        flux_arr_orig[1:,j,0] = df_dict_orig[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                        t_arr_orig[1:,j,0] = df_dict_counts_orig[view][t_str][t_str].to_numpy()
                    else:
                        raise Exception("Timestamps not matching.")
            else:
                for i,bins in enumerate(np.arange(en_channel[0], en_channel[-1]+1)):
                    for j,view in enumerate(["sun","asun","north","south"]):
                        if len(df_dict_counts_orig[view])==len(I_times):
                            count_arr_orig[:,j,i] = df_dict_counts_orig[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                            flux_arr_orig[:,j,i] = df_dict_orig[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                            t_arr_orig[:,j,i] = df_dict_counts_orig[view][t_str][t_str].to_numpy()
                        elif df_dict_counts_orig[view].index[0] == I_times[0]:
                            count_arr_orig[:-1,j,i] = df_dict_counts_orig[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                            flux_arr_orig[:-1,j,i] = df_dict_orig[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                            t_arr_orig[:-1,j,i] = df_dict_counts_orig[view][t_str][t_str].to_numpy()
                        elif df_dict_counts_orig[view].index[1] == I_times[0]:
                            count_arr_orig[1:,j,i] = df_dict_counts_orig[view][count_str][f"%s_%i"%(count_str,bins)].to_numpy()
                            flux_arr_orig[1:,j,i] = df_dict_orig[view][flux_str][f"%s_%i"%(flux_str,bins)].to_numpy()
                            t_arr_orig[1:,j,i] = df_dict_counts_orig[view][t_str][t_str].to_numpy()
                        else:
                            raise Exception("Timestamps not matching.")
            # Replace gf_arr with the one calculated for original values before ion contamination correction
            gf_arr = np.nanmedian(count_arr_orig/flux_arr_orig/delta_E/t_arr_orig,axis=0)
            gf_arr_max = np.nanmax(count_arr_orig/flux_arr_orig/delta_E/t_arr_orig,axis=0)
            
    if np.any(gf_arr==0):
        print("WARNING: Zeroes in the gf_arr:")
        print(gf_arr)
        print("Using max instead:")
        print(gf_arr_max)
        gf_arr = gf_arr_max

    pointing_arr = telescope_pointing(instrument.lower())

    mag_sc_arr = mag_srf[["Bx","By","Bz"]].to_numpy()

    return I_times, I_data, I_unc, sectors, en_channel_string, delta_E, count_str, mu_times, mu_data, mag_data, pol, phi_relative, pol_times, bg_times, bg_I_data, bg_I_unc, bg_mu_data, sp_str, ch_string, mag_data_coord, coverage, flux_arr, count_arr, t_arr, gf_arr, mag_sc_arr, averaging, av_min


def get_background_data(I_times,I_data,I_unc,mu_times,mu_data,backsub_start,backsub_end):
    if len(mu_times) == len(I_times) + 1:
        mu_times = mu_times[1:]
        mu_data = mu_data[1:,:]
    bg_sun = I_data[(I_times>=backsub_start) & (I_times<=backsub_end),0]
    bg_asun = I_data[(I_times>=backsub_start) & (I_times<=backsub_end),1]
    bg_north = I_data[(I_times>=backsub_start) & (I_times<=backsub_end),2]
    bg_south = I_data[(I_times>=backsub_start) & (I_times<=backsub_end),3]
    bg_unc_sun = I_unc[(I_times>=backsub_start) & (I_times<=backsub_end),0]
    bg_unc_asun = I_unc[(I_times>=backsub_start) & (I_times<=backsub_end),1]
    bg_unc_north = I_unc[(I_times>=backsub_start) & (I_times<=backsub_end),2]
    bg_unc_south = I_unc[(I_times>=backsub_start) & (I_times<=backsub_end),3]
    bg_mu_sun = mu_data[(mu_times>=backsub_start) & (mu_times<=backsub_end),0]
    bg_mu_asun = mu_data[(mu_times>=backsub_start) & (mu_times<=backsub_end),1]
    bg_mu_north = mu_data[(mu_times>=backsub_start) & (mu_times<=backsub_end),2]
    bg_mu_south = mu_data[(mu_times>=backsub_start) & (mu_times<=backsub_end),3]
    bg_I_data = np.column_stack((bg_sun, bg_asun, bg_north, bg_south))
    bg_I_unc = np.column_stack((bg_unc_sun, bg_unc_asun, bg_unc_north, bg_unc_south))
    bg_mu_data = np.column_stack((bg_mu_sun, bg_mu_asun, bg_mu_north, bg_mu_south))
    bg_times = I_times[(I_times>=backsub_start) & (I_times<=backsub_end)]
    return bg_times, bg_I_data, bg_I_unc, bg_mu_data


def epd_combine_intensities(instrument,df_dict,species,en_channel,energies):
    df_sun = df_dict["sun"]
    df_asun = df_dict["asun"]
    df_north = df_dict["north"]
    df_south = df_dict["south"]
    
    if species in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        ch_string = 'Electrons'
        
    if species in ['p', 'i', 'protons', 'ions']:
        if instrument in ['STEP', 'EPT']: 
            en_str = energies['Ion_Bins_Text']
            ch_string = 'Ions'
        if instrument == 'HET': 
            en_str = energies['H_Bins_Text']
            ch_string = 'Protons'

    if instrument == 'STEP':
        if species in ['e', 'electrons']:
            en_channel_string = energies['Electron_Bins_Text'][en_channel][0]
        if species in ['p', 'i', 'protons', 'ions']:
            en_channel_string = energies['Bins_Text'][en_channel]

    if instrument in ['EPT', 'HET']:  # average several energy channels; needs to be implemented for STEP 
        I_sun, en_channel_string = calc_av_en_flux_EPD(df_sun, energies, en_channel, species, instrument)
        I_asun, en_channel_string = calc_av_en_flux_EPD(df_asun, energies, en_channel, species, instrument)
        I_north, en_channel_string = calc_av_en_flux_EPD(df_north, energies, en_channel, species, instrument)
        I_south, en_channel_string = calc_av_en_flux_EPD(df_south, energies, en_channel, species, instrument)
        I_sun_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_EPD(df_sun, energies, en_channel, species, instrument)
        I_asun_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_EPD(df_asun, energies, en_channel, species, instrument)
        I_north_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_EPD(df_north, energies, en_channel, species, instrument)
        I_south_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_EPD(df_south, energies, en_channel, species, instrument)

        try:
            I_data = np.column_stack((I_sun, I_asun, I_north, I_south))
            I_unc = np.column_stack((I_sun_uncertainty, I_asun_uncertainty, I_north_uncertainty, I_south_uncertainty))
            I_times = I_sun.index
        except:
            freq = "{:d}ms".format(int(10))
            I_sun.index = I_sun.index.round(freq=freq)
            I_asun.index = I_asun.index.round(freq=freq)
            I_south.index = I_south.index.round(freq=freq)
            I_north.index = I_north.index.round(freq=freq)
            N_arr = np.array([len(I_sun),len(I_asun),len(I_north),len(I_south)])
            df_arr = [I_sun,I_asun,I_north,I_south]
            idx = np.argsort(N_arr)
            I_df = pd.merge(df_arr[0], df_arr[1], how='outer', left_index=True, right_index=True, suffixes = ('_a', '_b'))
            I_df = pd.merge(I_df, df_arr[2], how='outer', left_index=True, right_index=True, suffixes = ('_c', '_d'))
            I_df = pd.merge(I_df, df_arr[3], how='outer', left_index=True, right_index=True, suffixes = ('_e', '_f'))
            I_data = I_df.to_numpy()
            I_times = I_df.index
            df_arr = [I_sun_uncertainty,I_asun_uncertainty,I_north_uncertainty,I_south_uncertainty]
            I_df = pd.merge(df_arr[0], df_arr[1], how='outer', left_index=True, right_index=True, suffixes = ('_a', '_b'))
            I_df = pd.merge(I_df, df_arr[2], how='outer', left_index=True, right_index=True, suffixes = ('_c', '_d'))
            I_df = pd.merge(I_df, df_arr[3], how='outer', left_index=True, right_index=True, suffixes = ('_e', '_f'))
            I_unc = I_df.to_numpy()
        sectors = ["sun","asun","north","south"]
    return I_times, I_data, I_unc, sectors, en_channel_string


def calc_av_en_flux_EPD(df, energies, en_channel, species, instrument):  # original from Nina Slack Feb 9, 2022, rewritten Jan Apr 8, 2022
    """This function averages the flux of several energy channels of HET into a combined energy channel
    channel numbers counted from 0

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """
    if species.lower() in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Flux'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        if instrument.lower() == 'het':
            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Flux'
        if instrument.lower() == 'ept':
            en_str = energies['Ion_Bins_Text']
            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Flux'
    if type(en_channel) is list:
        energy_low = en_str[en_channel[0]].flat[0].split('-')[0]

        energy_up = en_str[en_channel[-1]].flat[0].split('-')[-1]

        en_channel_string = energy_low + '-' + energy_up

        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            try:
                df = df[flux_key]
            except (AttributeError, KeyError):
                None
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[f'{flux_key}_{bins}'] * DE[bins]
                else:
                    I_all = I_all + df[f'{flux_key}_{bins}'] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
            en_channel_string = en_str[en_channel].flat[0]
    else:
        en_channel_string = en_str[en_channel].flat[0]
        flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
    return flux_out, en_channel_string


def calc_av_en_flux_uncertainty_EPD(df, energies, en_channel, species, instrument):  # Laura edited from the function above
    """This function averages the uncertainty of several energy channels of HET into a combined energy channel
    channel numbers counted from 0

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """
    if species.lower() in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Uncertainty'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        if instrument.lower() == 'het':
            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Uncertainty'
        if instrument.lower() == 'ept':
            en_str = energies['Ion_Bins_Text']
            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Uncertainty'
    if type(en_channel) == list:
        energy_low = en_str[en_channel[0]].flat[0].split('-')[0]

        energy_up = en_str[en_channel[-1]].flat[0].split('-')[-1]

        en_channel_string = energy_low + '-' + energy_up

        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            try:
                df = df[flux_key]
            except (AttributeError, KeyError):
                None
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = (df[f'{flux_key}_{bins}'] * DE[bins])**2
                else:
                    I_all = I_all + (df[f'{flux_key}_{bins}'] * DE[bins])**2
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': np.sqrt(I_all)/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
            en_channel_string = en_str[en_channel].flat[0]
    else:
        en_channel_string = en_str[en_channel].flat[0]
        flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
    return flux_out, en_channel_string


def delta_E_array(instrument,species,energies,en_channel):
    if species.lower() in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        if instrument.lower() == 'het':
            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
        if instrument.lower() == 'ept':
            en_str = energies['Ion_Bins_Text']
            bins_width = 'Ion_Bins_Width'
    if type(en_channel) == list:
        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            delta_E = []
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                delta_E.append(DE[bins])
            delta_E = np.array(delta_E)
        if len(en_channel) == 1:
            delta_E = energies[bins_width][en_channel[0]]
    else:
        delta_E = energies[bins_width][en_channel]
    return np.array(delta_E)


@njit
def calc_resample_mag_to_fluxes(timestamps,mag_times,mag_data_arr,tlimits):
    arr = np.nan*np.zeros((len(timestamps),np.shape(mag_data_arr)[1]))
    for i in range(len(timestamps)):
        for j in prange(np.shape(mag_data_arr)[1]):
            arr[i,j] = np.nanmean(mag_data_arr[(mag_times >= tlimits[i,0]) & (mag_times < tlimits[i,1]),j])
    return arr


def resample_mag_to_fluxes_delta_t(datetimes,df_mag,delta_t_arr,pos_timestamp="center"):
    #delta_t_arr in seconds
    timestamps = np.array([t.timestamp() for t in datetimes])
    if pos_timestamp == "center":
        tlimits = np.column_stack([[(datetimes[i]-pd.Timedelta(seconds=delta_t_arr[i])/2).timestamp() for i in range(len(datetimes))],[(datetimes[i]+pd.Timedelta(seconds=delta_t_arr[i])/2).timestamp() for i in range(len(datetimes))]])
    elif pos_timestamp == "left":
        tlimits = np.column_stack([[datetimes[i].timestamp() for i in range(len(datetimes))],[(datetimes[i]+pd.Timedelta(seconds=delta_t_arr[i])).timestamp() for i in range(len(datetimes))]])
    elif pos_timestamp == "right":
        tlimits = np.column_stack([[(datetimes[i]-pd.Timedelta(seconds=delta_t_arr[i])).timestamp() for i in range(len(datetimes))],[datetimes[i].timestamp() for i in range(len(datetimes))]])
    mag_times = np.array([t.timestamp() for t in df_mag.index])
    arr = calc_resample_mag_to_fluxes(timestamps,mag_times,df_mag.to_numpy(),tlimits)
    df_mag_resampled = pd.DataFrame(index=datetimes,data=arr,columns=df_mag.columns)
    return df_mag_resampled


def resample_mag_to_fluxes(datetimes,df_mag,averaging,pos_timestamp="center"):
    if averaging is None:
        #For SolO particle fluxes cadence is 1 s
        averaging = "1s"
    timestamps = np.array([t.timestamp() for t in datetimes])
    if pos_timestamp == "center":
        tlimits = np.column_stack([[t.timestamp() for t in datetimes-pd.Timedelta(averaging)/2],[t.timestamp() for t in datetimes+pd.Timedelta(averaging)/2]])
    elif pos_timestamp == "left":
        tlimits = np.column_stack([[t.timestamp() for t in datetimes],[t.timestamp() for t in datetimes+pd.Timedelta(averaging)]])
    elif pos_timestamp == "right":
        tlimits = np.column_stack([[t.timestamp() for t in datetimes-pd.Timedelta(averaging)],[t.timestamp() for t in datetimes]])
    mag_times = np.array([t.timestamp() for t in df_mag.index])
    arr = calc_resample_mag_to_fluxes(timestamps,mag_times,df_mag.to_numpy(),tlimits)
    df_mag_resampled = pd.DataFrame(index=datetimes,data=arr,columns=df_mag.columns)
    return df_mag_resampled


def solo_mag_srf(startdate,enddate,path,av_min=None,mag_data_product="l2"):
    # MAG and PA coverage
    msdate = dt.datetime.combine(startdate.date(), dt.time.min)
    medate = dt.datetime.combine(enddate.date()+ dt.timedelta(days=1), dt.time.min)
    try:
        mag_srf = solo_mag_loader(msdate, medate, level=mag_data_product, frame='srf', av=av_min, path=path)
    except:
        raise
        print('changing mag data product to LL')
        mag_type = 'LL'
        mag_srf = solo_mag_loader(msdate, medate, level=mag_data_product,frame='srf', av=av_min, path=path)
    
    mag_srf = mag_srf[(mag_srf.index <= enddate) & (mag_srf.index >= startdate)]
    return mag_srf


def epd_load_data(instrument,data_product,startdate,enddate,path):
    df_sun_p, df_sun_e, energies = epd_load(sensor=instrument, level=data_product, startdate=startdate, enddate=enddate, viewing='sun', path=path, autodownload=True)
    df_asun_p, df_asun_e, energies = epd_load(sensor=instrument, level=data_product, startdate=startdate, enddate=enddate, viewing='asun', path=path, autodownload=True)
    df_north_p, df_north_e, energies = epd_load(sensor=instrument, level=data_product, startdate=startdate, enddate=enddate, viewing='north', path=path, autodownload=True)
    df_south_p, df_south_e, energies = epd_load(sensor=instrument, level=data_product, startdate=startdate, enddate=enddate, viewing='south', path=path, autodownload=True)
    return df_sun_p,df_asun_p,df_north_p,df_south_p,df_sun_e,df_asun_e,df_north_e,df_south_e, energies


def epd_prepare_counts(instrument,sp_str,t_str,df_sun,df_asun,df_north,df_south,averaging=None):
    if averaging is not None:
        split_strings = averaging.split("min")
        if len(split_strings) == 1:
            split_strings = averaging.split("s")
            if len(split_strings) == 2:
                av_min = float(split_strings[0])/60
            else:
                print("Invalid averaging.")
                exit()
        elif len(split_strings) == 2:
            av_min = float(split_strings[0])
        else:
            print("Invalid averaging.")
            exit()
        if instrument in ['EPT', 'HET']:
            #count rates (counts/second) to counts
            df_sun[f"{sp_str}_Rate"] = df_sun[f"{sp_str}_Rate"].copy().to_numpy()*df_sun[f"{t_str}"].to_numpy()
            df_asun[f"{sp_str}_Rate"] = df_asun[f"{sp_str}_Rate"].copy().to_numpy()*df_asun[f"{t_str}"].to_numpy()
            df_north[f"{sp_str}_Rate"] = df_north[f"{sp_str}_Rate"].copy().to_numpy()*df_north[f"{t_str}"].to_numpy()
            df_south[f"{sp_str}_Rate"] = df_south[f"{sp_str}_Rate"].copy().to_numpy()*df_south[f"{t_str}"].to_numpy()
            df_sun   = df_sun.resample(averaging,label='left').sum()
            df_sun.index = df_sun.index + to_offset(pd.Timedelta(averaging)/2)
            df_asun  = df_asun.resample(averaging,label='left').sum()
            df_north = df_north.resample(averaging,label='left').sum()
            df_south = df_south.resample(averaging,label='left').sum()
            df_asun.index = df_asun.index + to_offset(pd.Timedelta(averaging)/2)
            df_north.index = df_north.index + to_offset(pd.Timedelta(averaging)/2)
            df_south.index = df_south.index + to_offset(pd.Timedelta(averaging)/2)
        else:
            print("WARNING: Only EPT and HET implemented for averaging!")
    else:
        if instrument in ['EPT', 'HET']:
            #count rates (counts/second) to counts
            df_sun[f"{sp_str}_Rate"] = df_sun[f"{sp_str}_Rate"].copy().to_numpy()*df_sun[f"{t_str}"].to_numpy()
            df_asun[f"{sp_str}_Rate"] = df_asun[f"{sp_str}_Rate"].copy().to_numpy()*df_asun[f"{t_str}"].to_numpy()
            df_north[f"{sp_str}_Rate"] = df_north[f"{sp_str}_Rate"].copy().to_numpy()*df_north[f"{t_str}"].to_numpy()
            df_south[f"{sp_str}_Rate"] = df_south[f"{sp_str}_Rate"].copy().to_numpy()*df_south[f"{t_str}"].to_numpy()
        else:
            print("WARNING: Only EPT and HET implemented for averaging!")

    return {"sun": df_sun, "asun": df_asun, "north": df_north, "south": df_south}


def epd_prepare(instrument,sp_str,df_sun,df_asun,df_north,df_south,averaging=None):
    if averaging is not None:
        split_strings = averaging.split("min")
        if len(split_strings) == 1:
            split_strings = averaging.split("s")
            if len(split_strings) == 2:
                av_min = float(split_strings[0])/60
            else:
                print("Invalid averaging.")
                exit()
        elif len(split_strings) == 2:
            av_min = float(split_strings[0])
        else:
            print("Invalid averaging.")
            exit()
        if instrument in ['EPT', 'HET']:
            #Prepare uncertainties for correct resampling
            delta_t = np.mean(np.diff(df_sun.index))/np.timedelta64(1, 's')
            N = np.round(av_min*60/delta_t)
            df_sun[f"{sp_str}_Uncertainty"] = df_sun[f"{sp_str}_Uncertainty"].copy().to_numpy()**2
            df_asun[f"{sp_str}_Uncertainty"] = df_asun[f"{sp_str}_Uncertainty"].copy().to_numpy()**2 
            df_north[f"{sp_str}_Uncertainty"]= df_north[f"{sp_str}_Uncertainty"].copy().to_numpy()**2 
            df_south[f"{sp_str}_Uncertainty"] = df_south[f"{sp_str}_Uncertainty"].copy().to_numpy()**2

            df_sun   = df_sun.resample(averaging,label='left').mean()
            df_sun.index = df_sun.index + to_offset(pd.Timedelta(averaging)/2)
            df_asun  = df_asun.resample(averaging,label='left').mean()
            df_north = df_north.resample(averaging,label='left').mean()
            df_south = df_south.resample(averaging,label='left').mean()
            df_asun.index = df_asun.index + to_offset(pd.Timedelta(averaging)/2)
            df_north.index = df_north.index + to_offset(pd.Timedelta(averaging)/2)
            df_south.index = df_south.index + to_offset(pd.Timedelta(averaging)/2)

            df_sun[f"{sp_str}_Uncertainty"] = np.sqrt(df_sun[f"{sp_str}_Uncertainty"].copy().to_numpy()/N)
            df_asun[f"{sp_str}_Uncertainty"] = np.sqrt(df_asun[f"{sp_str}_Uncertainty"].copy().to_numpy()/N)
            df_north[f"{sp_str}_Uncertainty"]= np.sqrt(df_north[f"{sp_str}_Uncertainty"].copy().to_numpy()/N)
            df_south[f"{sp_str}_Uncertainty"] = np.sqrt(df_south[f"{sp_str}_Uncertainty"].copy().to_numpy()/N) 
        else:
            print("WARNING: Only EPT and HET implemented for averaging!")

    return {"sun": df_sun, "asun": df_asun, "north": df_north, "south": df_south}


def telescope_pointing(instrument):
    if instrument.lower() in ['ept', 'het']:
        # pointing directions of EPT in XYZ/SRF coordinates (!) (arrows point into the sensor)
        pointing_sun = np.array([-0.81915206, 0.57357645, 0.])
        pointing_asun = np.array([0.81915206, -0.57357645, 0.])
        pointing_north = np.array([0.30301532, 0.47649285, -0.8253098])
        pointing_south = np.array([-0.30301532, -0.47649285, 0.8253098])
        pointing_arr = np.vstack((pointing_sun,pointing_asun,pointing_north,pointing_south))
    elif instrument.lower() == 'step':
        # Particle flow direction (unit vector) in spacecraft XYZ coordinates for each STEP pixel ('XYZ_Pixels')
        pointing_arr = np.array([[-0.8412, 0.4396,  0.3149],
                                  [-0.8743, 0.457 ,  0.1635],
                                  [-0.8862, 0.4632, -0.    ],
                                  [-0.8743, 0.457 , -0.1635],
                                  [-0.8412, 0.4396, -0.315 ],
                                  [-0.7775, 0.5444,  0.3149],
                                  [-0.8082, 0.5658,  0.1635],
                                  [-0.8191, 0.5736,  0.    ],
                                  [-0.8082, 0.5659, -0.1634],
                                  [-0.7775, 0.5444, -0.3149],
                                  [-0.7008, 0.6401,  0.3149],
                                  [-0.7284, 0.6653,  0.1634],
                                  [-0.7384, 0.6744, -0.    ],
                                  [-0.7285, 0.6653, -0.1635],
                                  [-0.7008, 0.6401, -0.315 ]])
    else:
        raise Exception("Only EPT/HET and STEP implemented.")
    return pointing_arr


def calc_pa_coverage(instrument, mag_data):
    print(f'Calculating PA coverage for {instrument}...')
    if instrument not in ['ept', 'EPT', 'het', 'HET', 'step', 'STEP']:
        print("instrument not known, select 'EPT', 'HET', or 'STEP' ")
        coverage = pd.DataFrame(mag_data.index)
    else:
        if instrument.lower() == 'ept':
            opening = 30
        if instrument.lower() == 'het':
            opening = 43
        if instrument.lower() == 'step':
            print("Opening of STEP just a placeholder! Replace with real value! This affects the 'min' and 'max' values of the pitch-angle, not the 'center' ones.")
            opening = 10
        mag_vec = np.array([mag_data.Bx.to_numpy(), mag_data.By.to_numpy(), mag_data.Bz.to_numpy()])

        if instrument.lower() in ['ept', 'het']:
            # pointing directions of EPT in XYZ/SRF coordinates (!) (arrows point into the sensor)
            pointing_arr = telescope_pointing(instrument.lower())
            pointing_sun = pointing_arr[0,:]
            pointing_asun = pointing_arr[1,:]
            pointing_north = pointing_arr[2,:]
            pointing_south = pointing_arr[3,:]
            pa_sun = np.ones(len(mag_data.Bx.to_numpy())) * np.nan
            pa_asun = np.ones(len(mag_data.Bx.to_numpy())) * np.nan
            pa_north = np.ones(len(mag_data.Bx.to_numpy())) * np.nan
            pa_south = np.ones(len(mag_data.Bx.to_numpy())) * np.nan

            for i in tqdm(range(len(mag_data.Bx.to_numpy()))):
                pa_sun[i] = np.rad2deg(angle_between(pointing_sun, mag_vec[:, i]))
                pa_asun[i] = np.rad2deg(angle_between(pointing_asun, mag_vec[:, i]))
                pa_north[i] = np.rad2deg(angle_between(pointing_north, mag_vec[:, i]))
                pa_south[i] = np.rad2deg(angle_between(pointing_south, mag_vec[:, i]))

        if instrument.lower() == 'step':
            # Particle flow direction (unit vector) in spacecraft XYZ coordinates for each STEP pixel ('XYZ_Pixels')
            pointing_step = telescope_pointing(instrument.lower())
            pa_step = np.ones((len(mag_data.Bx.to_numpy()), pointing_step.shape[0])) * np.nan

            for i in tqdm(range(len(mag_data.Bx.to_numpy()))):
                for j in range(pointing_step.shape[0]):
                    pa_step[i, j] = np.rad2deg(angle_between(pointing_step[j], mag_vec[:, i]))

    if instrument.lower() in ['ept', 'het']:
        sun_min = pa_sun - opening/2
        sun_max = pa_sun + opening/2
        asun_min = pa_asun - opening/2
        asun_max = pa_asun + opening/2
        north_min = pa_north - opening/2
        north_max = pa_north + opening/2
        south_min = pa_south - opening/2
        south_max = pa_south + opening/2
        cov_sun = pd.DataFrame({'min': sun_min, 'center': pa_sun, 'max': sun_max}, index=mag_data.index)
        cov_asun = pd.DataFrame({'min': asun_min, 'center': pa_asun, 'max': asun_max}, index=mag_data.index)
        cov_north = pd.DataFrame({'min': north_min, 'center': pa_north, 'max': north_max}, index=mag_data.index)
        cov_south = pd.DataFrame({'min': south_min, 'center': pa_south, 'max': south_max}, index=mag_data.index)
        keys = [('sun'), ('asun'), ('north'), ('south')]
        coverage = pd.concat([cov_sun, cov_asun, cov_north, cov_south], keys=keys, axis=1)

    if instrument.lower() == 'step':
        pa_step_min = pa_step - opening/2
        pa_step_max = pa_step + opening/2

        cov = {}
        for i in range(pa_step.shape[1]):
            cov[f'Pixel_{i+1}'] = pd.DataFrame({'min': pa_step_min[:, i], 'center': pa_step[:, i], 'max': pa_step_max[:, i]}, index=mag_data.index)
        coverage = pd.concat(cov, keys=cov.keys(), axis=1)

    coverage[coverage > 180] = 180
    coverage[coverage < 0] = 0
    return coverage


def mu_from_coverage(coverage):
    mu_times = coverage.index
    mu_sun = np.cos(np.deg2rad(coverage['sun'].center.to_numpy()))
    mu_asun = np.cos(np.deg2rad(coverage['asun'].center.to_numpy()))
    mu_north = np.cos(np.deg2rad(coverage['north'].center.to_numpy()))
    mu_south = np.cos(np.deg2rad(coverage['south'].center.to_numpy()))
    mu_data = np.column_stack((mu_sun, mu_asun, mu_north, mu_south))
    return mu_times, mu_data


def solo_specieschannels(specieschannel):
    if specieschannel == "p 25 MeV":
        species = "p"
        instrument = "HET"
        channels = [19,24]
    elif specieschannel == "e- 100 keV":
        species = "e"
        instrument = "EPT"
        channels = [14,18]
    elif specieschannel == "e- 1 MeV":
        species = "e"
        instrument = "HET"
        channels = [0,1]
    else:
        raise ValueError("Invalid species and energy string: {}.".format(specieschannel))
    return instrument, species, channels


def calc_ept_ion_contamination_correction(df_sun_e,df_asun_e,df_north_e,df_south_e,df_sun_p,df_asun_p,df_north_p,df_south_p):
    corr_flux_e = calc_EPT_corrected_e(df_sun_e['Electron_Flux'].copy(), df_sun_p['Ion_Flux'].copy())
    corr_rate_e = df_sun_e['Electron_Rate']/df_sun_e['Electron_Flux'].values*corr_flux_e.values
    corr_rate_e[df_sun_e["Electron_Rate"]==0] = 0
    corr_rate_e = corr_rate_e + 0*corr_flux_e.values
    df_sun_e["Electron_Flux"] = corr_flux_e
    df_sun_e["Electron_Rate"] = corr_rate_e

    corr_flux_e = calc_EPT_corrected_e(df_asun_e['Electron_Flux'].copy(), df_asun_p['Ion_Flux'].copy())
    corr_rate_e = df_asun_e['Electron_Rate']/df_asun_e['Electron_Flux'].values*corr_flux_e.values
    corr_rate_e[df_asun_e["Electron_Rate"]==0] = 0
    corr_rate_e = corr_rate_e + 0*corr_flux_e.values
    df_asun_e["Electron_Flux"] = corr_flux_e
    df_asun_e["Electron_Rate"] = corr_rate_e
    
    corr_flux_e = calc_EPT_corrected_e(df_north_e['Electron_Flux'].copy(), df_north_p['Ion_Flux'].copy())
    corr_rate_e = df_north_e['Electron_Rate']/df_north_e['Electron_Flux'].values*corr_flux_e.values
    corr_rate_e[df_north_e["Electron_Rate"]==0] = 0
    corr_rate_e = corr_rate_e + 0*corr_flux_e.values
    df_north_e["Electron_Flux"] = corr_flux_e
    df_north_e["Electron_Rate"] = corr_rate_e

    corr_flux_e = calc_EPT_corrected_e(df_south_e['Electron_Flux'].copy(), df_south_p['Ion_Flux'].copy())
    corr_rate_e = df_south_e['Electron_Rate']/df_south_e['Electron_Flux'].values*corr_flux_e.values
    corr_rate_e[df_south_e["Electron_Rate"]==0] = 0
    corr_rate_e = corr_rate_e + 0*corr_flux_e.values
    df_south_e["Electron_Flux"] = corr_flux_e
    df_south_e["Electron_Rate"] = corr_rate_e
    return df_sun_e,df_asun_e,df_north_e,df_south_e


def solo_download_intensities(instrument,startdate,enddate,path,averaging,species,en_channel,solo_ept_ion_contamination_correction=False):
    data_product = "l2"
    df_sun_p,df_asun_p,df_north_p,df_south_p,df_sun_e,df_asun_e,df_north_e,df_south_e, energies = epd_load_data(instrument,data_product,startdate,enddate,path)
    
    if solo_ept_ion_contamination_correction:
        if (instrument == "EPT"):
            print("Correcting for ion contamination.")
            df_sun_e,df_asun_e,df_north_e,df_south_e = calc_ept_ion_contamination_correction(df_sun_e,df_asun_e,df_north_e,df_south_e,df_sun_p,df_asun_p,df_north_p,df_south_p)
            
    if species == "p":
        df_sun = df_sun_p; df_asun = df_asun_p; df_north = df_north_p; df_south = df_south_p
        t_str="DELTA_EPOCH"
        if instrument == "EPT": ch_string = 'Ions'; sp_str = "Ion"
        elif instrument == "HET": ch_string = "Protons"; sp_str = "H"
    elif species == "e":
        df_sun = df_sun_e; df_asun = df_asun_e; df_north = df_north_e; df_south = df_south_e
        ch_string = 'Electrons'; sp_str = "Electron"
        if instrument == "EPT": t_str="DELTA_EPOCH_1"
        elif instrument == "HET": t_str="DELTA_EPOCH_4"

    df_sun = df_sun.loc[(df_sun.index>=startdate) & (df_sun.index<=enddate)]
    df_asun = df_asun.loc[(df_asun.index>=startdate) & (df_asun.index<=enddate)]
    df_north = df_north.loc[(df_north.index>=startdate) & (df_north.index<=enddate)]
    df_south = df_south.loc[(df_south.index>=startdate) & (df_south.index<=enddate)]
    
    df_dict = epd_prepare(instrument,sp_str,df_sun,df_asun,df_north,df_south,averaging=averaging)

    I_times, I_data, I_unc, sectors, en_channel_string = epd_combine_intensities(instrument,df_dict,species,en_channel,energies)

    return I_times, I_data, I_unc, sectors, en_channel_string


def solo_mag_loader(sdate, edate, level='l2', type='normal', frame='rtn', av=None, path=None):
    """
    to do: implement higher resultion averaging ('1S' (seconds)) for burst data
    loads SolO/MAG data from soar using function mag_load() from Jan: 
    autodownloads if files are not there

    Parameters
    ----------
    sdate : int
        20210417
    edate : int
        20210418
    level : str, optional
        by default 'l2'
    type : str, optional
        'burst', 'normal-1-minute', by default 'normal'
    frame : str, optional
        'srf', by default 'rtn'
    av : int or None, optional
        number of minutes to average, by default None

    Returns
    -------
    [type]
        [description]
    """    
    print('Loading MAG...')
    mag_data = mag_load(sdate, edate, level=level, data_type=type, frame=frame, path=path)
    #mag_data = mag_load(sdate, edate, level=level, frame=frame, path=path)
    if frame == 'rtn':
        mag_data.rename(columns={'B_RTN_0':'B_r', 'B_RTN_1':'B_t', 'B_RTN_2':'B_n'}, inplace=True)
    if frame == 'srf':
        mag_data.rename(columns={'B_SRF_0':'Bx', 'B_SRF_1':'By', 'B_SRF_2':'Bz'}, inplace=True)
    if av is not None:
        mav = f'{av}min' 
        mag_offset = f'{av/2}min' 
        mag_data = mag_data.resample(mav,label='left').mean()
        mag_data.index = mag_data.index + to_offset(mag_offset)

    return mag_data


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calc_EPT_corrected_e(df_ept_e, df_ept_p):
    ion_cont_corr_matrix = np.loadtxt(r'C:\Users\lakavu\Desktop\Työ\jupyterlab\SEP analysis\EPT_ion_contamination_matrix_sun.dat')  # using the new calibration files (using the sun_matrix because they don't differ much)
    Electron_Flux_cont = np.zeros(np.shape(df_ept_e))*np.nan
    for tt in range(len(df_ept_e)):
        Electron_Flux_cont[tt, :] = np.sum(ion_cont_corr_matrix * np.ma.masked_invalid(df_ept_p.values[tt, :]), axis=1)
    df_ept_e_corr = df_ept_e - Electron_Flux_cont
    return df_ept_e_corr


def solo_polarity_preparation(startdate, enddate, av_min=1, path=None, mag_data_product="l2", V=400):
    msdate = dt.datetime.combine(startdate.date(), dt.time.min)
    medate = dt.datetime.combine(enddate.date() + dt.timedelta(days=1), dt.time.min)
    try:
        mag_rtn = solo_mag_loader(msdate, medate, level=mag_data_product, frame='rtn', av=av_min, path=path)
    except:
        print('changing mag data product to LL')
        mag_type = 'LL'
        mag_rtn = solo_mag_loader(msdate, medate, level=mag_data_product,frame='rtn', av=av_min, path=path)
    pos = get_horizons_coord('Solar Orbiter', time={'start':mag_rtn.index[0]-pd.Timedelta(minutes=15),'stop':mag_rtn.index[-1]+pd.Timedelta(minutes=15),'step':"1min"})  # (lon, lat, radius) in (deg, deg, AU)
    pos = pos.transform_to(HeliographicStonyhurst())
    #Interpolate position data to magnetic field data cadence
    r = np.interp([t.timestamp() for t in mag_rtn.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.radius.value)
    lat = np.interp([t.timestamp() for t in mag_rtn.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.lat.value)
    pol, phi_relative = polarity_rtn(mag_rtn.B_r.values, mag_rtn.B_t.values, mag_rtn.B_n.values,r,lat,V=V)
    pol_times = mag_rtn.index.values
    return pol, phi_relative, pol_times, mag_rtn
