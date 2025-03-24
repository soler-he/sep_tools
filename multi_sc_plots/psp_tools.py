import astropy.units as u
import datetime as dt
# import math
import numpy as np
import os
import pandas as pd
import warnings
import sunpy

from astropy.constants import e, k_B, m_p
from astropy.table import QTable
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, psp_isois_load
from seppy.tools import resample_df
#from stixdcpy.quicklook import LightCurves # https://github.com/i4Ds/stixdcpy
from sunpy.coordinates import frames, get_horizons_coord

from multi_sc_plots.polarity_plotting import polarity_rtn

# disable unused speasy data provider before importing to speed it up
os.environ['SPEASY_CORE_DISABLED_PROVIDERS'] = "sscweb,archive,csa"
import speasy as spz

from IPython.core.display import display

# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', category=UserWarning, message='no explicit representation of timezones available', module='speasy.core.data_containers')

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000


def mag_angles(B,Br,Bt,Bn):
    theta = np.arccos(Bn/B)
    alpha = 90-(180/np.pi*theta)

    r = np.sqrt(Br**2 + Bt**2 + Bn**2)
    phi = np.arccos(Br/np.sqrt(Br**2 + Bt**2))*180/np.pi

    sel = np.where(Bt < 0)
    count = len(sel[0])
    if count > 0:
        phi[sel] = 2*np.pi - phi[sel]
    sel = np.where(r <= 0)
    count = len(sel[0])
    if count > 0:
        phi[sel] = 0

    return alpha, phi

def load_data(options):
    """
    Load data used in plotting.
    
    Parameters
    ----------

    options : Options object

    Returns
    -------

    list of dataframes

    """
    

    #####################################################################
    ######## Data loading ###############################################
    #####################################################################
    
    global enddate
    global startdate
    global file_path

    global plot_epilo_e
    global plot_epihi_e
    global plot_epilo_p
    global plot_epihi_p
    global plot_electrons
    global plot_protons
    global plot_stix
    global plot_radio
    global plot_mag
    global plot_mag_angles
    global plot_Vsw
    global plot_N
    global plot_T
    global plot_p_dyn
    
    global epilo_ic_channel
    global epilo_channel
    global stix_ltc
    
    global psp_het_energies
    global psp_rfs_lfr_psd
    global psp_rfs_hfr_psd
    global df_psp_spani
    global df_psp_spc
    
    global psp_mag
    global stix_orig
    global psp_epilo_energies_org
    global psp_epilo_ic_energies_org
    global psp_het_org
    global psp_epilo_ic_org
    global psp_epilo_org

    enddate = options.enddate.value
    startdate = options.startdate.value

    if not isinstance(startdate, dt.datetime) or not isinstance(enddate, dt.datetime):
        raise ValueError("Invalid start/end date")
    
    file_path = options.path

    plot_epilo_e = options.psp_epilo_e.value
    plot_epihi_e = options.psp_epihi_e.value
    plot_epilo_p = options.psp_epilo_p.value
    plot_epihi_p = options.psp_epihi_p.value
    plot_stix = options.stix.value
    plot_radio = options.radio.value
    plot_mag = options.mag.value
    plot_mag_angles = options.mag_angles.value
    plot_Vsw = options.Vsw.value
    plot_N = options.N.value
    plot_T = options.T.value
    plot_p_dyn = options.p_dyn.value
    
    
    epilo_ic_channel = options.psp_epilo_ic_channel.value
    epilo_channel = options.psp_epilo_channel.value
    #stix_ltc = options.stix_ltc.value


    plot_electrons = plot_epilo_e or plot_epihi_e
    plot_protons = plot_epilo_p or plot_epihi_p


    # if plot_stix:
        
    #     if enddate-startdate > dt.timedelta(days=7):
    #         print('WARNING: STIX loading for more than 7 days not supported at the moment!')
    #         print('')
    #     lc = LightCurves.from_sdc(start_utc=startdate, end_utc=enddate, ltc=stix_ltc)
    #     stix_orig = lc.to_pandas()
    
    try:    
        if plot_epihi_p or plot_epihi_e:
            
            psp_het_org, psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate, enddate, 
                                                                        path=file_path, resample=None)

        if plot_epilo_e:
            psp_epilo_org, psp_epilo_energies_org = psp_isois_load('PSP_ISOIS-EPILO_L2-PE', startdate, enddate, 
                                                                                path=file_path, resample=None, epilo_channel=epilo_channel, 
                                                                                epilo_threshold=None)
            electron_countrate_keys = psp_epilo_org.filter(like='Electron_CountRate_ChanF_E').keys()
            psp_epilo_org[electron_countrate_keys] = psp_epilo_org[electron_countrate_keys].mask(psp_epilo_org[electron_countrate_keys] < 0.0)
            

        if plot_epilo_p:
            psp_epilo_ic_org, psp_epilo_ic_energies_org = psp_isois_load('PSP_ISOIS-EPILO_L2-IC', startdate, enddate, 
                                                                                        path=file_path, resample=None, epilo_channel=epilo_ic_channel, 
                                                                                        epilo_threshold=None)
    except AttributeError:
        print("No PSP ISOIS data found for given interval!")
        plot_epihi_e = False
        plot_epilo_e = False
        plot_epihi_p = False
        plot_epilo_p = False

    if plot_radio:
        psp_rfs_lfr_psd = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSP_FLD.RFS_LFR.PSP_FLD_L3_RFS_LFR.psp_fld_l3_rfs_lfr_PSD_SFU,
                                        startdate, enddate).replace_fillval_by_nan()
        psp_rfs_hfr_psd = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSP_FLD.RFS_HFR.PSP_FLD_L3_RFS_HFR.psp_fld_l3_rfs_hfr_PSD_SFU, 
                                        startdate, enddate).replace_fillval_by_nan()

        # Get frequency (MHz) bins, since metadata is lost upon conversion to df
        psp_rfs_lfr_freq = psp_rfs_lfr_psd.axes[1].values[0] / 1e6     
        psp_rfs_hfr_freq = psp_rfs_hfr_psd.axes[1].values[0] / 1e6

        # frequencies overlap, so leave the last seven out
        psp_rfs_lfr_psd = psp_rfs_lfr_psd.to_dataframe().iloc[:,:-6]
        psp_rfs_hfr_psd = psp_rfs_hfr_psd.to_dataframe()
    
        # put frequencies into column names for easier access
        psp_rfs_lfr_psd.columns = psp_rfs_lfr_freq[:-6]
        psp_rfs_hfr_psd.columns = psp_rfs_hfr_freq

        # Remove bar artifacts caused by non-NaN values before time jumps
        for i in range(len(psp_rfs_lfr_psd.index) - 1):
            if (psp_rfs_lfr_psd.index[i+1] - psp_rfs_lfr_psd.index[i]) > np.timedelta64(5, "m"):   
                psp_rfs_lfr_psd.iloc[i,:] = np.nan
        for i in range(len(psp_rfs_hfr_psd.index) - 1):
            if (psp_rfs_hfr_psd.index[i+1] - psp_rfs_hfr_psd.index[i]) > np.timedelta64(5, "m"):
                psp_rfs_hfr_psd.iloc[i,:] = np.nan

    if plot_mag or plot_mag_angles:
        df_psp_mag_rtn = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_mag_phi = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min_phi, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_mag_theta = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min_theta, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_mag_tot = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.FIELDS_MAG.psp_mag_1min.psp_b_1min_tot, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        
        psp_mag = pd.concat([df_psp_mag_rtn, df_psp_mag_phi, df_psp_mag_theta, df_psp_mag_tot], axis=1)
        psp_mag['phi_mod'] = ((psp_mag['phi'].values - 180) % 360) - 180


    if plot_mag_angles:
        theta, phi = mag_angles(psp_mag['|b|'].values, psp_mag['br'].values, psp_mag['bt'].values, psp_mag['bn'].values)
        psp_mag['theta2'] = theta
        psp_mag['phi2'] = phi

    if plot_Vsw or plot_N or plot_T or plot_p_dyn:
        # SPC
        df_psp_spc_np_tot = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_np_tot, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_spc_vp_tot_nrm = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_vp_tot_nrm, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_spc_vp_tot_rtn = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_vp_tot, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_spc_wp_tot = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_wp_tot, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_spc_GF = spz.get_data(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_flag.psp_spc_gf, 
                                    startdate, enddate, output_format="CDF_ISTP").replace_fillval_by_nan().to_dataframe()
        df_psp_spc = pd.concat([df_psp_spc_np_tot, df_psp_spc_vp_tot_nrm, df_psp_spc_vp_tot_rtn, df_psp_spc_wp_tot, df_psp_spc_GF], axis=1)

        # SPAN-i
        df_psp_spani_np = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.DENS, 
                                    startdate, enddate).replace_fillval_by_nan().to_dataframe()
        df_psp_spani_vp_rtn_sun = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.VEL_RTN_SUN, 
                                    startdate, enddate).replace_fillval_by_nan().to_dataframe()
        df_psp_spani_T = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.TEMP, 
                                    startdate, enddate).replace_fillval_by_nan().to_dataframe()
        df_psp_spani_QF = spz.get_data(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.QUALITY_FLAG, 
                                    startdate, enddate).replace_fillval_by_nan().to_dataframe()
        df_psp_spani = pd.concat([df_psp_spani_np, df_psp_spani_vp_rtn_sun, df_psp_spani_T, df_psp_spani_QF], axis=1)

        # Read units into dictionary

        df_psp_spc_units = {}
        df_psp_spc_units['np_tot'] = u.Unit(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_np_tot.units)
        df_psp_spc_units['|vp_tot|'] = u.Unit(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_vp_tot_nrm.units)
        for k in ['vp_totr', 'vp_tott', 'vp_totn']:
            df_psp_spc_units[k] = u.Unit(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_vp_tot.units)
        df_psp_spc_units['wp_tot'] = u.Unit(spz.inventories.data_tree.amda.Parameters.PSP.SWEAP_SPC.psp_spc_fit.psp_spc_wp_tot.units)

        df_psp_spani_units = {}
        df_psp_spani_units['Density'] = u.Unit(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.DENS.UNITS)
        for k in ['Vx RTN', 'Vy RTN', 'Vz RTN']:
            df_psp_spani_units[k] = u.Unit(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.VEL_RTN_SUN.UNITS)
        df_psp_spani_units['Temperature'] = u.Unit(spz.inventories.data_tree.cda.ParkerSolarProbe.PSPSWEAPSPAN.PSP_SWP_SPI_SF00_L3_MOM.TEMP.UNITS)

        # Convert to AstroPy QTables

        qt_psp_spc = QTable.from_pandas(df_psp_spc, index=True, units=df_psp_spc_units)
        # qt_psp_spc = QTable(qt_psp_spc, masked=True)
        qt_psp_spani = QTable.from_pandas(df_psp_spani, index=True, units=df_psp_spani_units)
        qt_psp_spc['T'] = (1/2*m_p/k_B*(qt_psp_spc['wp_tot'])**2).si
        qt_psp_spc['p_dyn'] = (m_p*qt_psp_spc['np_tot']*(qt_psp_spc['|vp_tot|'])**2).to(u.nPa)
        qt_psp_spani['V_tot_rtn'] = np.sqrt(qt_psp_spani['Vx RTN']**2+qt_psp_spani['Vy RTN']**2+qt_psp_spani['Vz RTN']**2)
        qt_psp_spani['T_K'] = (qt_psp_spani['Temperature']/k_B).si
        qt_psp_spani['p_dyn'] = (m_p*qt_psp_spani['Density']*(qt_psp_spani['V_tot_rtn'])**2).to(u.nPa)

        # Back to Pandas

        df_psp_spc = qt_psp_spc.to_pandas(index='index')
        df_psp_spc.index.name = None

        df_psp_spani = qt_psp_spani.to_pandas(index='index')
        df_psp_spani.index.name = None

        # Data cleaning

        df_psp_spc = df_psp_spc.mask(df_psp_spc['general_flag']!=0.0)
        df_psp_spani['Temperature'] = df_psp_spani['Temperature'].mask(df_psp_spani['Temperature']<0.0)
        df_psp_spani['T_K'] = df_psp_spani['T_K'].mask(df_psp_spani['T_K']<0.0)


        #### Filter data based on Quality Flags
        # The Quality flags mostly contain a description of the instrument activities and operational status. 
        # For those, I would recommend avoiding anything with the following quality bits set to 1:

        # - bit0 - counter overflow
        # - bit3 - spoiler test
        # - bit10 - bad energy table
        # - bit11 - MCP test
        # - bit14 - threshold test
        # - bit15 - commanding

        # (R. Livi, priv. comm.)

        df_psp_spani['Quality Flag binary'] = df_psp_spani['Quality Flag'].astype(int).map('{:b}'.format).astype(str)
        df_psp_spani['Quality Flag binary'] = df_psp_spani['Quality Flag binary'].str.zfill(16)

        qf_bits_list = ['Counter Overflow', 'Survey Snapshot ON (not applicable to archive products)', 'Alternate Energy Table', 'Spoiler Test', 'Attenuator Engaged', 'Highest Archive Rate', 'No Targeted Sweep',
                        'SPAN-Ion New Mass Table (not applicable to electrons)', 'Over-deflection', 'Archive Snapshot ON', 'Bad Energy Table', 'MCP Test', 'Survey Available', 'Archive Available', 
                        'Threshold Test', 'Commanding']
        qf_bits_list.reverse()

        for i in range(len(qf_bits_list)):
            df_psp_spani[qf_bits_list[i]] = df_psp_spani['Quality Flag binary'].str[i]
            df_psp_spani[qf_bits_list[i]] = df_psp_spani[qf_bits_list[i]].astype(int)

        cond1 = df_psp_spani['Counter Overflow']==1
        cond2 = df_psp_spani['Spoiler Test']==1
        cond3 = df_psp_spani['Bad Energy Table']==1
        cond4 = df_psp_spani['MCP Test']==1
        cond5 = df_psp_spani['Threshold Test']==1
        cond6 = df_psp_spani['Commanding']==1

        df_psp_spani = df_psp_spani.mask(cond1 | cond2 | cond3 | cond4 | cond5 | cond6)

        # Drop binary version of Quality Flag because otherwise resampling will crash later
        df_psp_spani.drop(columns='Quality Flag binary', inplace=True)

    

    #################################################################
    ############## Resampling #######################################
    #################################################################

    


def make_plot(options):
    """
    Plot chosen data with user-specified parameters.
    """
   
    plot_polarity = options.polarity.value
    plot_epihi_p_combined_pixels = options.psp_epihi_p_combined_pixels.value
    
    psp_het_viewing = options.psp_het_viewing.value
    epilo_ic_viewing = options.psp_epilo_ic_viewing.value
    epilo_viewing = options.psp_epilo_viewing.value

    ch_het_p = options.psp_ch_het_p.value
    ch_epilo_ic = options.psp_ch_epilo_ic.value
    ch_het_e = options.psp_ch_het_e.value
    ch_epilo_e = options.psp_ch_epilo_e.value
    
    legends_inside = options.legends_inside.value
    cmap = options.radio_cmap.value

    if options.plot_range is None:
        t_start = startdate
        t_end = enddate
    else:
        t_start = options.plot_range.children[0].value[0]
        t_end = options.plot_range.children[0].value[1]

    ####  Resampling
    
    resample = str(options.resample.value) + "min"         # convert to form that Pandas accepts
    resample_mag = str(options.resample_mag.value) + "min"

    if resample != "0min":
        if plot_epihi_e or plot_epihi_p:
            psp_het = resample_df(psp_het_org, resample)   
        if plot_epilo_e:
            psp_epilo = resample_df(psp_epilo_org, resample) 
        if plot_epilo_p:
            psp_epilo_ic = resample_df(psp_epilo_ic_org, resample) 
        # if plot_psp_pixel:
        #     df_psp_pixel = resample_df(df_psp_pixel_org, resample) 
        if plot_stix:
            stix = resample_df(stix_orig, resample)
    
    else:
        if plot_epihi_e or plot_epihi_p:
            psp_het = psp_het_org 
        if plot_epilo_e:
            psp_epilo = psp_epilo_org
        if plot_epilo_p:
            psp_epilo_ic = psp_epilo_ic_org
        # if plot_psp_pixel:
        #     df_psp_pixel = df_psp_pixel_org
        if plot_stix:
            stix = stix_orig

    if resample_mag != "0min":
        if plot_Vsw or plot_N or plot_T or plot_p_dyn:
            df_magplas_spani = resample_df(df_psp_spani, resample_mag) 
            df_magplas_spc = resample_df(df_psp_spc, resample_mag)
        if plot_mag:
            mag = resample_df(psp_mag, resample_mag) 

    else:
        if plot_Vsw or plot_N or plot_T or plot_p_dyn:
            df_magplas_spani = df_psp_spani 
            df_magplas_spc = df_psp_spc
        if plot_mag:
            mag = psp_mag

    print(f"Plotting PSP data for timerange {t_start} - {t_end}")
    
    ############################################################################
    ############## Energy channel ranges #######################################
    ############################################################################
    print("Chosen channels:")
    if plot_protons:  
        #Chosen channels
        if plot_epihi_p:
            print('HET protons:', ch_het_p, ',', len(ch_het_p))
        if plot_epilo_p:
            print('EPI-Lo ic:', ch_epilo_ic, ',', len(ch_epilo_ic))

    if plot_electrons:
        if plot_epihi_e:
            print('HET electrons:', ch_het_e, ',', len(ch_het_e))
        if plot_epilo_e:
            print('EPI-Lo electrons:', ch_epilo_e, ',', len(ch_epilo_e))
        
    

    panels = 1*plot_radio + 1*plot_stix + 1*plot_electrons + 1*plot_protons + 2*plot_mag_angles + 1*plot_mag + 1*plot_Vsw + 1*plot_N + 1*plot_T + 1*plot_p_dyn 

    panel_ratios = list(np.zeros(panels)+1)

    if plot_radio:
        panel_ratios[0] = 2

    if plot_electrons and plot_protons:
        panel_ratios[0+1*plot_radio] = 2
        panel_ratios[1+1*plot_radio] = 2
    if plot_electrons or plot_protons:    
        panel_ratios[0+1*plot_radio] = 2

    font_ylabel = 20
    font_legend = 10
    
    if panels == 3:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 4*panels])#, gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
    else:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 3*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")

    fig.subplots_adjust(hspace=0.1)
    
    i = 0

    if plot_radio:
        vmin, vmax = 500, 1e7
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        TimeHFR2D, FreqHFR2D = np.meshgrid(psp_rfs_hfr_psd.index, psp_rfs_hfr_psd.columns, indexing='ij')
        TimeLFR2D, FreqLFR2D = np.meshgrid(psp_rfs_lfr_psd.index, psp_rfs_lfr_psd.columns, indexing='ij')

        # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
        # when resampling isn't used
        mesh = axs[i].pcolormesh(TimeLFR2D, FreqLFR2D, psp_rfs_lfr_psd.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)
        axs[i].pcolormesh(TimeHFR2D, FreqHFR2D, psp_rfs_hfr_psd.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=font_ylabel)
        
        # Add inset axes for colorbar
        axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
        cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
        cbar.set_label("Intensity (sfu)", rotation=90, labelpad=10, fontsize=font_ylabel)
        i += 1
        
    
    if plot_stix:
        for key in stix.keys():
            axs[i].plot(stix.index, stix[key], ds="steps-mid", label=key)
        if stix_ltc:
            title = 'SolO/STIX (light travel time corrected)'
        else:
            title = 'SolO/STIX'
        if legends_inside:
            axs[i].legend(loc='upper right', title=title)
        else:
            # axs[i].legend(loc='upper right', title=title, bbox_to_anchor=(1, 0.5))
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', title=title)
        axs[i].set_ylabel('Counts', fontsize=font_ylabel)
        axs[i].set_yscale('log')
        i +=1  
    
    
    color_offset = 4 
    
    if plot_electrons:
        if plot_epilo_e:
            axs[i].set_prop_cycle('color', plt.cm.viridis_r(np.linspace(0, 1, len(ch_epilo_e)+color_offset)))
            for channel in ch_epilo_e:
                psp_epilo_energy = np.round(psp_epilo_energies_org[f'Electron_Chan{epilo_channel}_Energy'][f'Electron_Chan{epilo_channel}_Energy_E{channel}_P{epilo_viewing}'], 2).astype(str)
                axs[i].plot(psp_epilo.index, psp_epilo[f'Electron_CountRate_Chan{epilo_channel}_E{channel}_P{epilo_viewing}'],
                            ds="steps-mid", label=f'EPI-lo PE {epilo_channel}{epilo_viewing} {psp_epilo_energy} keV')
    
        if plot_epihi_e:
            axs[i].set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0, 1, len(ch_het_e)+color_offset)))
            for channel in ch_het_e:
                axs[i].plot(psp_het.index, psp_het[f'{psp_het_viewing}_Electrons_Rate_{channel}'],
                            ds="steps-mid", label=f'HET {psp_het_viewing}'+psp_het_energies['Electrons_ENERGY_LABL'].flatten()[channel])
                
        # axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=FONT_YLABEL)
        axs[i].set_ylabel("Count rates", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                          title=f'Electrons',
                          fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
                          title=f'Electrons',
                          fontsize=font_legend)
       
        axs[i].set_yscale('log')
        i +=1    
    
        
    color_offset = 2    
    if plot_protons:
        if plot_epilo_p:
            axs[i].set_prop_cycle('color', plt.cm.viridis_r(np.linspace(0, 1, len(ch_epilo_ic)+color_offset)))
            # [::-1] to reverse list
            for channel in ch_epilo_ic[::-1]:
                # print(f'H_Flux_Chan{epilo_ic_channel}_E{channel}_P{epilo_ic_viewing}')
                psp_epilo_ic_energy = np.round(psp_epilo_ic_energies_org[f'H_Chan{epilo_ic_channel}_Energy'][f'H_Chan{epilo_ic_channel}_Energy_E{channel}_P{epilo_ic_viewing}'], 2).astype(str)
                axs[i].plot(psp_epilo_ic.index, psp_epilo_ic[f'H_Flux_Chan{epilo_ic_channel}_E{channel}_P{epilo_ic_viewing}'],
                            ds="steps-mid", label=f'EPI-lo IC {epilo_ic_channel}{epilo_ic_viewing} {psp_epilo_ic_energy} keV')
    
        # if plot_psp_pixel:
        #     axs[i].set_prop_cycle('color', plt.cm.tab10(range(6)))
        #     for key in ['L2Ap', 'L3Ap', 'L4Ap', 'H2Ap', 'H3Ap', 'H4Ap']:
        #     # for key in ['L2Ap', 'L4Ap', 'H2Ap', 'H3Ap', 'H4Ap']:
        #         axs[i].plot(df_psp_pixel.index, df_psp_pixel[f'{key}_Flux'], label=f'{key} {energies_psp_pixel[key]}', drawstyle='steps-mid')
        
        if plot_epihi_p:    
            if plot_epihi_p_combined_pixels:
                # comb_channels = [[1,2], [3,5], [5,7], [4,5], [7], [9]]
                comb_channels = [[3,5], [5,7], [4,5], [7], [9]]
                axs[i].set_prop_cycle('color', plt.cm.Greys_r(np.linspace(0, 1, len(comb_channels)+5)))
                for channel in comb_channels:
                    df_psp_epihi, df_psp_epihi_name = calc_av_en_flux_PSP_EPIHI(psp_het, psp_het_energies, channel, 'p', 'het', psp_het_viewing)
                    axs[i].plot(df_psp_epihi.index, df_psp_epihi.flux, label=f'HET {psp_het_viewing}{df_psp_epihi_name}', lw=1, ds="steps-mid")
            else:
                axs[i].set_prop_cycle('color', plt.cm.plasma(np.linspace(0, 1, len(ch_het_p)+color_offset)))
                for channel in ch_het_p:
                    axs[i].plot(psp_het.index, psp_het[f'{psp_het_viewing}_H_Flux_{channel}'], label=f'HET {psp_het_viewing}'+psp_het_energies['H_ENERGY_LABL'].flatten()[channel], ds="steps-mid")
        
        axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=font_ylabel)
        # title = f'Ions (HET {psp_het_viewing})'
        title = f'Ions (Pixel)'
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                          title=title,
                          fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., 
                          title=title,
                          fontsize=font_legend)
        axs[i].set_yscale('log')
    
        # axs[i].set_ylim([5e-2, None])
        
        i +=1    
        
        
    # plot magnetic field
    if plot_mag:
        ax = axs[i]
        ax.plot(mag.index, mag['|b|'], label='B', color='k', linewidth=1)
        ax.plot(mag.index.values, mag['br'].values, label='Br', color='dodgerblue')
        ax.plot(mag.index.values, mag['bt'].values, label='Bt', color='limegreen')
        ax.plot(mag.index.values, mag['bn'].values, label='Bn', color='deeppink')
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        if legends_inside:
            ax.legend(loc='upper right', fontsize=font_legend)
        else:
            # ax.legend(loc='upper right', bbox_to_anchor=(1.01, 0.5))
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=font_legend)
            
        ax.set_ylabel('B [nT]', fontsize=font_ylabel)
        ax.tick_params(axis="x", direction="in", which='both')#, pad=-15)
        i += 1
        
        if plot_polarity:
            pos = get_horizons_coord(f'PSP', time={'start':mag.index[0]-pd.Timedelta(minutes=15), 'stop':mag.index[-1]+pd.Timedelta(minutes=15), 'step':"1min"})  # (lon, lat, radius) in (deg, deg, AU)
            pos = pos.transform_to(frames.HeliographicStonyhurst())
            #Interpolate position data to magnetic field data cadence
            r = np.interp([t.timestamp() for t in mag.index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.radius.value)
            lat = np.interp([t.timestamp() for t in mag.index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.lat.value)
            pol, phi_relative = polarity_rtn(mag['br'].values, mag['bt'].values, mag['bn'].values, r, lat, V=400)
            # create an inset axe in the current axe:
            pol_ax = inset_axes(ax, height="5%", width="100%", loc='upper center', bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
            pol_ax.get_xaxis().set_visible(False)
            pol_ax.get_yaxis().set_visible(False)
            pol_ax.set_ylim(0,1)
            pol_ax.set_xlim([mag.index.values[0], mag.index.values[-1]])
            pol_arr = np.zeros(len(pol))+1
            timestamp = mag.index.values[2] - mag.index.values[1]
            norm = Normalize(vmin=0, vmax=180, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
            pol_ax.bar(mag.index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
            pol_ax.bar(mag.index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
            pol_ax.set_xlim(t_start, t_end)
        
    if plot_mag_angles:
        ax = axs[i]
        #Bmag = np.sqrt(np.nansum((mag_data.B_r.values**2,mag_data.B_t.values**2,mag_data.B_n.values**2), axis=0))    
        # alpha, phi = mag_angles(mag.BFIELD_3, mag.BFIELD_0.values, mag.BFIELD_1.values,
        #                         mag.BFIELD_2.values)
        ax.plot(mag.index, mag['theta'], '.k', label='theta', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=font_ylabel)
        ax.tick_params(axis="x",direction="in", pad=-15)
    
        i += 1
        ax = axs[i]
        # ax.plot(mag.index, mag['phi'], '.k', label='phi', ms=1)
        ax.plot(mag.index, mag['phi_mod'], '.k', label='phi', ms=1)
        # ax.plot(mag.index, mag['phi2'], '.r', label='phi', ms=1)    
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=font_ylabel)
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        i += 1
        
    ### Temperature
    if plot_T:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['T_K'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['T'], '-r', label="SPC")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=font_ylabel)
        axs[i].set_yscale('log')
    
        # TODO: manually set lower boundary, remove at some point
        axs[i].set_ylim(np.nanmin(df_magplas_spc['T'])-0.1*np.nanmin(df_magplas_spc['T']), None)
    
        if legends_inside:
            axs[i].legend(loc='upper right', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=font_legend)
        i += 1
    
    ### Dynamic pressure
    if plot_p_dyn:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['p_dyn'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['p_dyn'], '-r', label="SPC")
        axs[i].set_ylabel(r"P$_\mathrm{dyn}$ [nPa]", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=font_legend)
        axs[i].set_yscale('log')
        i += 1
    
    ### Density
    if plot_N:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['Density'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['np_tot'], '-r', label="SPC")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=font_legend)
        axs[i].set_yscale('log')
        i += 1
    
    ### Vsw
    if plot_Vsw:
        axs[i].plot(df_magplas_spani.index, df_magplas_spani['V_tot_rtn'], '-k', label="SPAN-i")
        axs[i].plot(df_magplas_spc.index, df_magplas_spc['|vp_tot|'], '-r', label="SPC")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [kms$^{-1}$]", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=font_legend)
        # i += 1     
            
    axs[0].set_title(f'Parker Solar Probe', ha='center')
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {t_start.year}")#, fontsize=15)
    axs[-1].set_xlim(t_start, t_end)
    
    #plt.tight_layout()
    fig.patch.set_facecolor('white')
    fig.set_dpi(200)
    plt.show()

    return fig, axs

    