

# from IPython.core.display import display, HTML
# display(HTML(data="""<style> div#notebook-container { width: 80%; } div#menubar-container { width: 85%; } div#maintoolbar-container { width: 90%; } </style>"""))
from matplotlib.ticker import AutoMinorLocator#, LogLocator, NullFormatter, LinearLocator, MultipleLocator, 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000
import numpy as np
import os
import pandas as pd
import datetime as dt
import sunpy
import cdflib

from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates import frames
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from sunpy.net import Scraper
from sunpy.time import TimeRange
from sunpy.data.data_manager.downloader import ParfiveDownloader

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm

from multi_inst_plots.other_tools import polarity_rtn, mag_angles # , polarity_panel, polarity_colorwheel

from seppy.loader.wind import wind3dp_load
from seppy.loader.soho import soho_load
from seppy.tools import resample_df
# from other_loaders_py3 import wind_3dp_av_en  #, wind_mfi_loader, ERNE_HED_loader


#intensity_label = 'Intensity\n/(s cm² sr MeV)'
intensity_label = 'Intensity\n'+r'[(s cm² sr MeV)$^{-1}$]'



def download_wind_waves_cdf(freq, startdate, enddate, path=None):
    """
    Download Wind WAVES L2 data files for given time range.

    Parameters
    ----------
    freq: str
        RAD1 or RAD2 (lower case works as well)
    startdate, enddate: str or dt
        start and end dates as parse_time compatible strings or datetimes (see TimeRange docs)
    path : str (optional)
        Local download directory, defaults to sunpy's data directory
    
    Returns
    -------
    List of downloaded files
    """
    dl = ParfiveDownloader()
    
    timerange = TimeRange(startdate, enddate)

    try:
        from packaging.version import Version
        if hasattr(sunpy, "__version__") and Version(sunpy.__version__) >= Version("6.1.0"):
            pattern = ("https://spdf.gsfc.nasa.gov/pub/data/wind/waves/{freq}_l2/{{year:4d}}/wi_l2_wav_{freq}_{{year:4d}}{{month:2d}}{{day:2d}}_v{version}.cdf")

            scrap = Scraper(format=pattern, freq=freq.lower(), version="{:2d}") 
        else:
            pattern = "https://spdf.gsfc.nasa.gov/pub/data/wind/waves/{freq}_l2/%Y/wi_l2_wav_{freq}_%Y%m%d_{version}.cdf"
 
            scrap = Scraper(pattern=pattern, freq=freq.lower(), version="v\\d{2}")  # regex matching "v{any digit}{any digit}""
        
        
        filelist_urls = scrap.filelist(timerange=timerange)

        filelist_urls.sort()

        # After sorting, any multiple versions are next to each other in ascending order.
        # If there are files with same dates, assume multiple versions -> pop the first one and repeat.
        # Should end up with a list with highest version numbers. Magic number -7 is the index where 
        # version number starts
        # As of 13.2.2025, no higher versions than v01 exist in either rad1_l2 or rad2_l2 directory

        i = 0
        while i < len(filelist_urls) - 1:
            if filelist_urls[i+1][:-7] == filelist_urls[i][:-7]:
                filelist_urls.pop(i)
            else:
                i += 1

        filelist = [url.split('/')[-1] for url in filelist_urls]

        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        # Check if file with same name already exists in path
        for url, f in zip(filelist_urls, filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                dl.download(url=url, path=f)

    except (RuntimeError, IndexError):
        print(f'Unable to obtain Wind WAVES {freq} data for {startdate}-{enddate}!')
        downloaded_files = []

    return downloaded_files


def load_waves_rad(dataset, startdate, enddate, file_path=None):
    """
    Read Wind/WAVES data (assuming freq is 1D or identical rows if 2D)

    Parameters
    ----------
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    file_path : {str}, optional
        File path as a string. Defaults to sunpy's default download directory

    """
    
    files = download_wind_waves_cdf(dataset, startdate, enddate, path=file_path)

    # Read the frequency binning (assumed constant across all data)
    freq_hz  = cdflib.CDF(files[0]).varget("FREQUENCY")

    # If freq is 2D but each row is identical, take freq_raw[0,:]
    if freq_hz.ndim == 2:
        freq_hz = freq_hz[0, :]
    
    psd_v2hz = np.empty(shape=(0,len(freq_hz))) 
    time_dt = np.array([], dtype="datetime64")

    # append data 
    for file in files:
        cdf = cdflib.CDF(file)

        # PSD shape (nTime, nFreq)
        psd_raw = cdf.varget("PSD_V2_SP")
        # Time
        time_ns = cdf.varget("Epoch")  # shape (nTime,)

        time_dt = np.append(time_dt, cdflib.epochs.CDFepoch.to_datetime(time_ns))

        psd_v2hz = np.append(psd_v2hz, psd_raw, axis=0)

    # remove bar artifacts caused by non-NaN values before time jumps
    # for each time step except the last one:
    for i in range(len(time_dt)-1):
        # check if time increases by more than 5 min:
        if time_dt[i+1] - time_dt[i] > np.timedelta64(5, "m"):
            psd_v2hz[i,:] = np.nan

    # Some files use a fill value ~ -9.9999998e+30
    fill_val = -9.999999848243207e+30
    valid_mask = (freq_hz > 0) & (freq_hz != fill_val) 
    freq_hz = freq_hz[valid_mask]
    psd_v2hz = psd_v2hz[:, valid_mask]

    # Convert frequency to MHz
    freq_mhz = freq_hz / 1e6

    # Sort time
    if not sorted(time_dt):
        idx_t = np.argsort(time_dt)
        time_dt = time_dt[idx_t]
        psd_v2hz  = psd_v2hz[idx_t, :]

    # Remove duplicate times
    t_unique, t_uidx = np.unique(time_dt, return_index=True)
    if len(t_unique) < len(time_dt):
        time_dt = t_unique
        psd_v2hz  = psd_v2hz[t_uidx, :]

    # Sort freq
    if not sorted(freq_mhz):
        idx_f = np.argsort(freq_mhz)
        freq_mhz = freq_mhz[idx_f]
        psd_v2hz  = psd_v2hz[:, idx_f]

    # Remove duplicate freqs
    f_unique, f_uidx = np.unique(freq_mhz, return_index=True)
    if len(f_unique) < len(freq_mhz):
        freq_mhz = f_unique
        psd_v2hz  = psd_v2hz[:, f_uidx]

    data = pd.DataFrame(psd_v2hz, index=time_dt, columns=freq_mhz)

    return data

def wind_mfi_loader(startdate, enddate):

    dataset = 'WI_H3-RTN_MFI'  # 'WI_H2_MFI'
    cda_dataset = a.cdaweb.Dataset(dataset)

    trange = a.Time(startdate, enddate)

    # path = path_loc+'wind/mfi/'  # you can define here where the original data files should be saved, see 2 lines below
    path = None
    result = Fido.search(trange, cda_dataset)
    downloaded_files = Fido.fetch(result, path=path)  # use Fido.fetch(result, path='/ThisIs/MyPath/to/Data/{file}') to use a specific local folder for saving data files
    downloaded_files.sort()

    # read in data files to Pandas Dataframe
    data = TimeSeries(downloaded_files, concatenate=True)
    df = data.to_dataframe()

    # wind_datetime = np.arange(concat_df.shape[0]) * datetime.timedelta(hours=1)
    # for i in range(concat_df.shape[0]):
    #     dt64=df.index[i]
    #     ts=(dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    #     wind_datetime[i]=datetime.datetime.utcfromtimestamp(ts)

    # df['BR'], df['BT'], df['BN'] = cs.cxform('GSE','RTN', wind_datetime, x=df['BGSE_0'], y=df['BGSE_1'], z=df['BGSE_2'])
    # df['B'] = np.sqrt(df['BGSE_0'].values**2+df['BGSE_1'].values**2 +df['BGSE_2'].values**2)
    df['B'] = np.sqrt(df['BRTN_0'].values**2+df['BRTN_1'].values**2 +df['BRTN_2'].values**2)
    # return concat_df
    return  df



def load_data(options):
    
    ######### This is just for easier copy pasting from other versions (no need to change variables into dictionary references) #########
    global df_wind_wav_rad2
    global df_wind_wav_rad1
    global df_solwind
    global mag_data
    global plot_wind
    global plot_wind_e
    global plot_wind_p
    global plot_ephin
    global plot_erne
    global ephin_
    global erne_p_
    global edic_
    global pdic_
    global meta_ephin
    global meta_erne
    global meta_e
    global meta_p
    global l1_ch_eph_e
    global intercal
    global l1_ch_eph_p

    global startdate
    global enddate
    global plot_radio
    global plot_electrons
    global plot_protons
    #global plot_pad
    global plot_mag_angles
    global plot_mag
    global plot_Vsw
    global plot_N
    global plot_T
    global plot_polarity

    global path

    global av_mag
    global av_sep
    global av_erne

    plot_wind_e = options.l1_wind_e.value
    plot_wind_p = options.l1_wind_p.value

    plot_wind = plot_wind_e or plot_wind_p

    plot_ephin = options.l1_ephin.value
    plot_erne = options.l1_erne.value

    plot_electrons = plot_wind_e or plot_ephin
    plot_protons = plot_wind_p or plot_erne

    startdate = options.startdate.value
    enddate = options.enddate.value

    if not isinstance(startdate, dt.datetime) or not isinstance(enddate, dt.datetime):
        raise ValueError("Invalid start/end date")
    
    if plot_ephin:
        l1_ch_eph_e = options.l1_ch_eph_e.value
        intercal = options.l1_intercal.value
        l1_ch_eph_p = options.l1_ch_eph_p.value
    
    wind_flux_thres = None

    plot_radio = options.radio.value
    #plot_pad = options.pad.value
    plot_mag = options.mag.value
    plot_mag_angles = options.mag_angles.value
    plot_Vsw = options.Vsw.value
    plot_N = options.N.value
    plot_T = options.T.value
    plot_polarity = options.polarity.value
    path = options.path

    av_sep = str(options.l1_av_sep.value) + "min"
    av_mag =  str(options.resample_mag.value) + "min"
    av_erne = str(options.l1_av_erne.value) + "min"



    # LOAD DATA
    ####################################################################
    if plot_wind:
        edic_, meta_e = wind3dp_load(dataset="WI_SFSP_3DP",
                            startdate=startdate,
                            enddate=enddate,
                            resample=0,
                            multi_index=True,
                            path=path,
                            threshold=wind_flux_thres)
        pdic_, meta_p = wind3dp_load(dataset="WI_SOSP_3DP",
                            startdate=startdate,
                            enddate=enddate,
                            resample=0,
                            multi_index=True,
                            path=path,
                            threshold=wind_flux_thres)
        
    if plot_radio:
        try:
            df_wind_wav_rad2 = load_waves_rad(dataset="RAD2", startdate=startdate, enddate=enddate, file_path=path)
            df_wind_wav_rad1 = load_waves_rad(dataset="RAD1", startdate=startdate, enddate=enddate, file_path=path)
        except IndexError:
            print(f'Unable to obtain Wind WAVES data for {startdate} - {enddate}!')
            df_wind_wav_rad1 = []
            df_wind_wav_rad2 = []


    if plot_ephin:
        try:
            ephin_, meta_ephin = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN", startdate=startdate, enddate=enddate,
                            path=path, resample=None)
        except UnboundLocalError:   # soho_ephin_loader throws this error since it tries to access a variable (cs_e300) that hasn't been set
            print(f"Unable to obtain SOHO/EPHIN data for {startdate} - {enddate}!")
            ephin_ = []
            meta_ephin = []

    if plot_erne:
        erne_p_, meta_erne = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=startdate, enddate=enddate,
                            path=path, resample=None)
        

    if plot_mag or plot_mag_angles:
        try:
            mag_data = wind_mfi_loader(startdate, enddate)
            
        except IndexError:  # TimeSeries() call throws IndexError when trying to pop from an empty list
            print(f"No MFI data found for {startdate} - {enddate}!")
            mag_data = []
    
        
    #product = a.cdaweb.Dataset('AC_K0_SWE')
    #product = a.cdaweb.Dataset('WI_PLSP_3DP')
    #product = a.cdaweb.Dataset('WI_PM_3DP')  
    if plot_Vsw or plot_N or plot_T:
        try:
            product = a.cdaweb.Dataset('WI_K0_3DP')

            time = a.Time(startdate, enddate)
            result = Fido.search(time & product)
            files = Fido.fetch(result, path=path)
            sw_data = TimeSeries(files, concatenate=True)
            df_solwind = sw_data.to_dataframe()
            df_solwind['vsw'] = np.sqrt(df_solwind['ion_vel_0']**2 + df_solwind['ion_vel_1']**2 + df_solwind['ion_vel_2']**2)
        except IndexError:
            print(f"Unable to obtain WI_K0_3DP data for {startdate} - {enddate}!")
            df_solwind = []
            
    
    
        
    # add particles, SWE

def make_plot(options):
    
    global df_mag
    global df_vsw
    global edic
    global pdic

    global ephin
    global erne_p

    legends_inside = options.legends_inside.value

    

    # AVERAGING
    if plot_mag or plot_mag_angles:
        # If no data, mag_data is an empty list and resample_df would crash (no resample method). Else if no averaging is done, rename to df_mag.
        if isinstance(mag_data, pd.DataFrame) and av_mag != "0min":
            df_mag = resample_df(mag_data, av_mag)
        else:
            df_mag = mag_data
            
    if plot_Vsw or plot_N or plot_T:
        if isinstance(df_solwind, pd.DataFrame) and av_mag != "0min":
            df_vsw = resample_df(df_solwind, av_mag)
        else:
            df_vsw = df_solwind

    # else:
    #     if plot_mag or plot_mag_angles:
    #         df_mag = mag_data
            
    #     if plot_Vsw or plot_T or plot_N:
    #         df_vsw = df_solwind

    if plot_polarity and isinstance(mag_data, pd.DataFrame):
        df_mag_pol = resample_df(mag_data, '1min')  # resampling to 1min for polarity plot
    else:
        df_mag_pol = []
        
    
    if plot_wind:
        if isinstance(edic_, pd.DataFrame) and av_sep != "0min":
            edic = resample_df(edic_, av_sep)
        else:
            edic = edic_

        if isinstance(pdic_, pd.DataFrame) and av_sep != "0min":
            pdic = resample_df(pdic_, av_sep)
        else:
            pdic = pdic_
    
    if plot_ephin:
        if isinstance(ephin_, pd.DataFrame) and av_sep != "0min":
            ephin = resample_df(ephin_, av_sep)
        else:
            ephin = ephin_

    if plot_erne:
        if isinstance(erne_p_, pd.DataFrame) and av_erne != "0min":
            erne_p = resample_df(erne_p_, av_erne)
        else:
            erne_p = erne_p_
    

    wind_ev2MeV_fac = 1e6
    cmap = options.radio_cmap.value

    if options.plot_range is None:
        t_start = startdate
        t_end = enddate
    else:
        t_start = options.plot_range.children[0].value[0]
        t_end = options.plot_range.children[0].value[1]


    font_ylabel = 20
    font_legend = 10

    panels = 1*plot_radio + 1*plot_electrons + 1*plot_protons + 2*plot_mag_angles + 1*plot_mag + 1* plot_Vsw + 1* plot_N + 1* plot_T # + 1*plot_pad 

    if panels == 0:
        print("No instruments chosen!")
        return (None, None)
    
    print(f"Plotting Wind/SOHO data for timerange {t_start} - {t_end}")
    
    panel_ratios = list(np.zeros(panels)+1)
    if plot_radio:
        panel_ratios[0] = 2
    if plot_electrons and plot_protons:
        panel_ratios[0+1*plot_radio] = 2
        panel_ratios[1+1*plot_radio] = 2
    if plot_electrons or plot_protons:    
        panel_ratios[0+1*plot_radio] = 2

    
    if panels == 3:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 4*panels])#, gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
    else:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 3*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")
        #fig, axs = plt.subplots(nrows=panels, sharex=True, dpi=100, figsize=[7, 1.5*panels], gridspec_kw={'height_ratios': panel_ratios})# layout="constrained")

        
    fig.subplots_adjust(hspace=0.1)

    if panels == 1:
        axs = [axs]

    
    color_offset = 3
    i = 0

    if plot_radio:
        vmin, vmax = 1e-15, 1e-10
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        if isinstance(df_wind_wav_rad1, pd.DataFrame) and isinstance(df_wind_wav_rad2, pd.DataFrame):
            time_rad2_2D, freq_rad2_2D = np.meshgrid(df_wind_wav_rad2.index, df_wind_wav_rad2.columns, indexing='ij')
            time_rad1_2D, freq_rad1_2D = np.meshgrid(df_wind_wav_rad1.index, df_wind_wav_rad1.columns, indexing='ij')

            # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
            # when resampling isn't used
            mesh = axs[i].pcolormesh(time_rad1_2D, freq_rad1_2D, df_wind_wav_rad1.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)
            axs[i].pcolormesh(time_rad2_2D, freq_rad2_2D, df_wind_wav_rad2.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)

            # Add inset axes for colorbar
            axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
            cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
            cbar.set_label(r"Intensity ($\mathrm{V^2/Hz}$)", rotation=90, labelpad=10, fontsize=font_ylabel)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=font_ylabel)
        
        
        i += 1

    if plot_electrons:
        # electrons
        ax = axs[i]
        if plot_wind and isinstance(edic, pd.DataFrame):
            axs[i].set_prop_cycle('color', plt.cm.Greens_r(np.linspace(0,1, len(meta_e['channels_dict_df'])+color_offset)))
            for ch in np.arange(1, len(meta_e['channels_dict_df'])):
                ax.plot(edic.index, edic[f'FLUX_{ch}'] * wind_ev2MeV_fac, label='Wind/3DP '+meta_e['channels_dict_df']['Bins_Text'].values[ch], drawstyle='steps-mid')
        
        
        if plot_ephin and isinstance(ephin, pd.DataFrame):
            ax.plot(ephin.index, ephin[l1_ch_eph_e]*intercal, '-k', label='SOHO/EPHIN '+meta_ephin[l1_ch_eph_e]+f' / {intercal}', drawstyle='steps-mid')
        # ax.set_ylim(1e0, 1e4)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Electrons', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
                    title=f'Electrons', fontsize=font_legend)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=font_ylabel)
        i += 1
        
    color_offset = 2    
    if plot_protons:    
        # protons low en:
        ax = axs[i]
        if plot_wind and isinstance(pdic, pd.DataFrame):
            ax.set_prop_cycle('color', plt.cm.plasma(np.linspace(0,1, len(meta_p['channels_dict_df'])+color_offset)))
            for ch in np.arange(2, len(meta_p['channels_dict_df'])):
                ax.plot(pdic.index, pdic[f'FLUX_{ch}'] * wind_ev2MeV_fac, label='Wind/3DP '+meta_p['channels_dict_df']['Bins_Text'].values[ch],
                        drawstyle='steps-mid')
        ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=font_ylabel)

        # protons high en:
        if plot_erne and isinstance(erne_p, pd.DataFrame):
            ax.set_prop_cycle('color', plt.cm.YlOrRd(np.linspace(0.2,1,10))) #cm.RdPu_r
            for ch in np.arange(0, 10):
                ax.plot(erne_p.index, erne_p[f'PH_{ch}'], label='SOHO/ERNE/HED '+meta_erne['channels_dict_df_p']['ch_strings'][ch], 
                            drawstyle='steps-mid')
        if legends_inside:
            ax.legend(title='Protons', loc="upper right", fontsize=font_legend)
        else:
            ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_legend)
        ax.set_yscale('log')
        i += 1

    
    if plot_mag:    
        ax = axs[i]
        if isinstance(mag_data, pd.DataFrame):
            ax.plot(df_mag.index, df_mag.B.values, label='B', color='k', linewidth=1)
            ax.plot(df_mag.index, df_mag.BRTN_0.values, label='Br', color='dodgerblue', linewidth=1)
            ax.plot(df_mag.index, df_mag.BRTN_1.values, label='Bt', color='limegreen', linewidth=1)
            ax.plot(df_mag.index, df_mag.BRTN_2.values, label='Bn', color='deeppink', linewidth=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        if legends_inside:
            ax.legend(title='Protons', loc="upper right", fontsize=font_legend)
        else:
            ax.legend(title='Protons', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_legend)#, title='RTN')#, bbox_to_anchor=(1, 0.5))
        ax.set_ylabel('B [nT]', fontsize=font_ylabel)
        ax.tick_params(axis="x",direction="in", which='both') #, pad=-15
        
        i += 1
        
        if plot_polarity and isinstance(df_mag_pol, pd.DataFrame):
            pos = get_horizons_coord('Wind', time={'start':df_mag_pol.index[0]-pd.Timedelta(minutes=15),
                                                'stop':df_mag_pol.index[-1]+pd.Timedelta(minutes=15),'step':"1min"}) 
                                                    # (lon, lat, radius) in (deg, deg, AU)
            pos = pos.transform_to(frames.HeliographicStonyhurst())
            #Interpolate position data to magnetic field data cadence
            r = np.interp([t.timestamp() for t in df_mag_pol.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.radius.value)
            lat = np.interp([t.timestamp() for t in df_mag_pol.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.lat.value)
            pol, phi_relative = polarity_rtn(df_mag_pol.BRTN_0.values, df_mag_pol.BRTN_1.values, df_mag_pol.BRTN_2.values,r,lat,V=400)
            # create an inset axe in the current axe:
            pol_ax = inset_axes(ax, height="5%", width="100%", loc=9, bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
            pol_ax.get_xaxis().set_visible(False)
            pol_ax.get_yaxis().set_visible(False)
            pol_ax.set_ylim(0,1)
            pol_ax.set_xlim([df_mag_pol.index.values[0], df_mag_pol.index.values[-1]])
            pol_arr = np.zeros(len(pol))+1
            timestamp = df_mag_pol.index.values[2] - df_mag_pol.index.values[1]
            norm = Normalize(vmin=0, vmax=180, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
            pol_ax.bar(df_mag_pol.index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
            pol_ax.bar(df_mag_pol.index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
            pol_ax.set_xlim(t_start, t_end)
        
        
    if plot_mag_angles:
        ax = axs[i]
        if isinstance(df_mag, pd.DataFrame):
            alpha, phi = mag_angles(df_mag.B.values, df_mag.BRTN_0.values, df_mag.BRTN_1.values, df_mag.BRTN_2.values)
            ax.plot(df_mag.index, alpha, '.k', label='alpha', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=font_ylabel)
        ax.tick_params(axis="x",direction="in")
        i += 1
        
        ax = axs[i]
        if isinstance(df_mag, pd.DataFrame):
            ax.plot(df_mag.index, phi, '.k', label='phi', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=font_ylabel)
        ax.tick_params(axis="x",direction="in", which='both')
        i += 1
        
    ### Temperature
    if plot_T:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw['ion_temp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=font_ylabel)
        i += 1

    ### Density
    if plot_N:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw.ion_density,
                        '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=font_ylabel)
        i += 1

    ### Sws
    if plot_Vsw:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw.vsw,
                        '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [km/s]", fontsize=font_ylabel)
        i += 1
            

    axs[0].set_title('Near-Earth spacecraft (Wind, SOHO)', fontsize=font_ylabel)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {t_start.year}", fontsize=15)
    axs[-1].set_xlim(t_start, t_end)
    fig.patch.set_facecolor('white')
    fig.set_dpi(200)
    plt.show()
    # if save_fig:
    #     plt.savefig(f'{outpath}L1_multiplot_{str(startdate.date())}--{str(enddate.date())}_{av_sep}.png')

    return fig, axs