import os
import sunpy
import cdflib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.constants import e, k_B, m_p
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
from seppy.loader.wind import wind3dp_load
from seppy.loader.soho import soho_load
from seppy.tools import resample_df
# from other_loaders_py3 import wind_3dp_av_en  #, wind_mfi_loader, ERNE_HED_loader

from multi_inst_plots.other_tools import polarity_rtn, mag_angles, load_goes_xrs, \
    load_solo_stix, plot_goes_xrs, plot_solo_stix, make_fig_axs


plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000

#intensity_label = 'Intensity\n/(s cm² sr MeV)'
intensity_label = 'Intensity\n'+r'[(cm² sr s MeV)$^{-1}$]'



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
        try:
            cdf = cdflib.CDF(file)

            # PSD shape (nTime, nFreq)
            psd_raw = cdf.varget("PSD_V2_SP")  # TODO: use "PSD_V2_Z" instead?
            # Time
            time_ns = cdf.varget("Epoch")  # shape (nTime,)

            time_dt = np.append(time_dt, cdflib.epochs.CDFepoch.to_datetime(time_ns))

            psd_v2hz = np.append(psd_v2hz, psd_raw, axis=0)
        except ValueError:
            pass

    # Some files use a fill value ~ -9.9999998e+30
    fill_val = cdf.varattsget("FREQUENCY")['FILLVAL']
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

    # remove bar artifacts caused by non-NaN values before time jumps
    # for each time step except the last one:
    for i in range(len(time_dt)-1):
        # check if time increases by more than 5 min:
        if time_dt[i+1] - time_dt[i] > np.timedelta64(5, "m"):
            psd_v2hz[i,:] = np.nan


    data = pd.DataFrame(psd_v2hz, index=time_dt, columns=freq_mhz)

    return data

def wind_mfi_loader(startdate, enddate, path=None):

    dataset = 'WI_H3-RTN_MFI'  # 'WI_H2_MFI'
    cda_dataset = a.cdaweb.Dataset(dataset)

    trange = a.Time(startdate, enddate)

    result = Fido.search(trange, cda_dataset)
    downloaded_files = Fido.fetch(result, path=path)  
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
    data = {}
    metadata = {}

    global df_wind_wav_rad2
    global df_wind_wav_rad1
    global df_solwind
    global mag_data
    global ephin_
    global erne_p_
    global edic_
    global pdic_
    global meta_ephin
    global meta_erne
    global meta_e
    global meta_p
    global df_stix_
    global df_goes_
    global goes_sat

    startdate = options.startdt
    enddate = options.enddt
    path = options.path
    stix_ltc = options.stix_ltc.value

    options.plot_start = None
    options.plot_end = None

    if options.mag.value == False:
        options.polarity.value = False

    wind_flux_thres = None

    dataset_num = (options.l1_wind_e.value or options.l1_wind_p.value) + options.radio.value + options.l1_ephin.value \
                    + options.l1_erne.value + (options.mag.value or options.mag_angles.value) \
                    + (options.Vsw.value or options.N.value or options.T.value or options.p_dyn.value) \
                    + options.stix.value + options.goes.value

    dataset_index = 1
    # LOAD DATA
    ####################################################################
    if options.l1_wind_e.value or options.l1_wind_p.value:
        print(f"Loading Wind/3DP data... (dataset {dataset_index}/{dataset_num})")
        edic_, meta_e = wind3dp_load(dataset="WI_SFSP_3DP",
                            startdate=startdate,
                            enddate=enddate,
                            resample=0,
                            multi_index=False,
                            path=path,
                            threshold=wind_flux_thres)
        pdic_, meta_p = wind3dp_load(dataset="WI_SOSP_3DP",
                            startdate=startdate,
                            enddate=enddate,
                            resample=0,
                            multi_index=False,
                            path=path,
                            threshold=wind_flux_thres)

        data["3dp_e"] = edic_
        data["3dp_p"] = pdic_
        metadata["3dp_e"] = meta_e
        metadata["3dp_p"] = meta_p
        dataset_index += 1

    if options.radio.value == True:
        print(f"Loading Wind/WAVES data... (dataset {dataset_index}/{dataset_num})")
        try:
            df_wind_wav_rad1 = load_waves_rad(dataset="RAD1", startdate=startdate, enddate=enddate, file_path=path)
        except IndexError:
            print(f'Unable to obtain Wind RAD1 data for {startdate} - {enddate}!')
            df_wind_wav_rad1 = []

        try:
            df_wind_wav_rad2 = load_waves_rad(dataset="RAD2", startdate=startdate, enddate=enddate, file_path=path)
        except IndexError:
            print(f'Unable to obtain Wind RAD2 data for {startdate} - {enddate}!')
            df_wind_wav_rad2 = []

        data["wav_rad1"] = df_wind_wav_rad1
        data["wav_rad2"] = df_wind_wav_rad2
        dataset_index += 1

    if options.l1_ephin.value == True:
        print(f"Loading SOHO/EPHIN data... (dataset {dataset_index}/{dataset_num})")
        try: 
            ephin_, meta_ephin = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN", startdate=startdate, enddate=enddate,
                            path=path, resample=None)
        except UnboundLocalError:   
            # soho_ephin_loader throws this error since it 
            # tries to access a variable (cs_e300) that hasn't been set
            print(f"Unable to obtain SOHO/EPHIN data for {startdate} - {enddate}!")
            ephin_ = []
            meta_ephin = []

        data["ephin"] = ephin_
        metadata["ephin"] = meta_ephin
        dataset_index += 1

    if options.l1_erne.value == True:
        print(f"Loading SOHO/ERNE data... (dataset {dataset_index}/{dataset_num})")
        erne_p_, meta_erne = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=startdate, enddate=enddate,
                            path=path, resample=None)
        
        data["erne"] = erne_p_
        metadata["erne"] = meta_erne
        dataset_index += 1
        

    if options.mag.value or options.mag_angles.value:
        print(f"Loading Wind/MFI data... (dataset {dataset_index}/{dataset_num})")
        try:
            mag_data = wind_mfi_loader(startdate, enddate, path=path)
            # L1 mag has values surrounded by NaN's, meaning line plot's won't show
            mag_data = mag_data[~np.isnan(mag_data["B"]) & ~np.isnan(mag_data["BRTN_0"]) 
                              & ~np.isnan(mag_data["BRTN_1"]) & ~np.isnan(mag_data["BRTN_2"]) ]
            
        except IndexError:  # TimeSeries() call throws IndexError when trying to pop from an empty list
            print(f"No MFI data found for {startdate} - {enddate}!")
            mag_data = []

        data["mag"] = mag_data
        dataset_index += 1
    

    if options.Vsw.value or options.N.value or options.T.value or options.p_dyn.value:
        print(f"Loading Wind solar wind (WI_K0_3DP) data... (dataset {dataset_index}/{dataset_num})")
        try:
            product = a.cdaweb.Dataset('WI_K0_3DP')

            time = a.Time(startdate, enddate)
            result = Fido.search(time & product)
            files = Fido.fetch(result, path=path)
            sw_data = TimeSeries(files, concatenate=True)
            df_solwind = sw_data.to_dataframe()
            df_solwind['vsw'] = np.sqrt(df_solwind['ion_vel_0']**2 + df_solwind['ion_vel_1']**2 + df_solwind['ion_vel_2']**2)
            df_solwind['ion_temp'] = df_solwind['ion_temp'] * e / k_B   # Kelvin
            df_solwind['p_dyn'] = m_p * df_solwind['ion_density'] * 1e6 * (df_solwind['vsw'] * 1e3)**2 * 1e9    # nPa
        except IndexError:
            print(f"Unable to obtain WI_K0_3DP data for {startdate} - {enddate}!")
            df_solwind = []

        data["solwind"] = df_solwind
        dataset_index += 1

      
    if options.stix.value == True:
        print(f"Loading SolO/STIX data... (dataset {dataset_index}/{dataset_num})")
        df_stix_ = load_solo_stix(startdate, enddate, resample=None, ltc = stix_ltc)
        data["stix"] = df_stix_
        dataset_index += 1

    if options.goes.value == True:
        print(f"Loading GOES/XRS data... (dataset {dataset_index}/{dataset_num})")
        df_goes_, goes_sat = load_goes_xrs(startdate, enddate, man_select=options.goes_man_select.value, 
                                           resample=None, path=path)
        data["goes"] = df_goes_
        dataset_index += 1
    
    print("Data loaded!")
    
    return data, metadata
    

def energy_channel_selection(options):
    cols = []
    df = pd.DataFrame()

    try:
        if options.l1_wind_e.value == True:
            if isinstance(meta_e, dict):
                cols.append("3DP Electrons")
                series_e = meta_e['channels_dict_df']['Bins_Text'].iloc[1:].reset_index(drop=True)
                df = pd.concat([df, series_e], axis=1)
            
        if options.l1_wind_p.value == True:
            if isinstance(meta_p, dict):
                cols.append("3DP Protons")
                series_p = meta_p['channels_dict_df']['Bins_Text'].iloc[2:].reset_index(drop=True)
                df = pd.concat([df, series_p], axis=1)
        
        if options.l1_ephin.value == True:
            if isinstance(meta_ephin, dict):
                cols.append("EPHIN Electrons")
                energy_list = []
                for ch in ["E150", "E300", "E1300", "E3000"]:
                    energy_list.append(meta_ephin[ch])
                series_ephin = pd.Series(energy_list)
                df = pd.concat([df, series_ephin], axis=1)

        if options.l1_erne.value == True:
            if isinstance(meta_erne, dict):
                cols.append("ERNE Protons")
                series_erne = meta_erne['channels_dict_df_p']['ch_strings'].reset_index(drop=True)
                df = pd.concat([df, series_erne], axis=1)

    except NameError:
        print("Some particle data option was selected but not loaded. Run load_data() first!")
        
    df.columns = cols
    return df


def make_plot(options):

    wind_ev2MeV_fac = 1e6
    
    plot_electrons = options.l1_wind_e.value or options.l1_ephin.value
    plot_protons = options.l1_wind_p.value or options.l1_erne.value

    ### AVERAGING ###
    
    av_sep =  options.l1_av_sep.value
    av_mag =  options.resample_mag.value
    av_erne =  options.l1_av_erne.value
    av_stixgoes =  options.resample_stixgoes.value

    if options.mag.value or options.mag_angles.value:
        if av_mag > 0 and av_mag <= 1.5:
            print("Wind/MFI native cadence is 1.5 min, so no averaging was applied.")
        if isinstance(mag_data, pd.DataFrame) and av_mag > 1.5:
            df_mag = resample_df(mag_data, str(60 * av_mag) + "s")
        else:
            df_mag = mag_data

        if options.polarity.value:
            if isinstance(mag_data, pd.DataFrame):
                df_mag_pol = resample_df(mag_data, '1min')  # resampling to 1min for polarity plot
            else:
                df_mag_pol = []
            
    if options.Vsw.value or options.N.value or options.T.value or options.p_dyn.value:
        if av_mag > 0 and av_mag <= 1.5:
            print("WI_K0_3DP native cadence is 1.5 min, so no averaging was applied.")
        if isinstance(df_solwind, pd.DataFrame) and av_mag > 1.5:
            df_vsw = resample_df(df_solwind, str(60 * av_mag) + "s")
        else:
            df_vsw = df_solwind

    
    if options.l1_wind_e.value or options.l1_wind_p.value:
        if av_sep > 0 and av_sep <= 0.2:
            print("Wind/3DP native cadence is 12 s, so no averaging was applied.")
        if isinstance(edic_, pd.DataFrame) and av_sep > 0.2:
            edic = resample_df(edic_, str(60 * av_sep) + "s")
        else:
            edic = edic_

        if isinstance(pdic_, pd.DataFrame) and av_sep > 0.2:
            pdic = resample_df(pdic_, str(60 * av_sep) + "s")
        else:
            pdic = pdic_
    
    if options.l1_ephin.value:
        if av_sep > 0 and av_sep <= 1:
            print("EPHIN native cadence is 1 min, so no averaging was applied.")
        if isinstance(ephin_, pd.DataFrame) and av_sep > 1:
            ephin = resample_df(ephin_, str(60 * av_sep) + "s")
        else:
            ephin = ephin_

    if options.l1_erne.value:
        if av_erne > 0 and av_erne <= 1:
            print("ERNE native cadence is 1 min, so no averaging was applied.")
        if isinstance(erne_p_, pd.DataFrame) and av_erne > 1:
            erne_p = resample_df(erne_p_, str(60 * av_erne) + "s")
        else:
            erne_p = erne_p_

    if options.goes.value:
        if isinstance(df_goes_, pd.DataFrame) and av_stixgoes > 0:
            df_goes = resample_df(df_goes_, str(60 * av_stixgoes) + "s")
        else:
            df_goes = df_goes_
        
    if options.stix.value:
        if isinstance(df_stix_, pd.DataFrame) and av_stixgoes > 0:
            df_stix = resample_df(df_stix_, str(60 * av_stixgoes) + "s")
        else:
            df_stix = df_stix_

    wind_ch_e = options.l1_ch_wind_e.value
    wind_ch_p = options.l1_ch_wind_p.value
    erne_ch = options.l1_ch_erne_p.value
    ephin_ch_index = list(options.l1_ch_ephin_e.value)
    ephin_ch = np.array(["E150", "E300", "E1300", "E3000"])[ephin_ch_index]

    if plot_protons or plot_electrons:
        print("Chosen energy channels:")
        if options.l1_wind_e.value:
            print(f'Wind/3DP electrons: {wind_ch_e}, {len(wind_ch_e)}')
        if options.l1_ephin.value:
            print(f'EPHIN electrons: {tuple(ephin_ch_index)}, {len(ephin_ch_index)}')
        if options.l1_wind_p.value:
            print(f'Wind/3DP protons: {wind_ch_p}, {len(wind_ch_p)}')
        if options.l1_erne.value:
            print(f'ERNE-HED protons: {erne_ch}, {len(erne_ch)}')

    legends_inside = options.legends_inside.value
    cmap = options.radio_cmap.value

    intercal = 1

    font_ylabel = 20
    font_legend = 10

    fig, axs = make_fig_axs(options)

    color_offset = 3
    i = 0

    if options.radio.value == True:
        vmin, vmax = 1e-15, 1e-10
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        mesh = None
        
        if isinstance(df_wind_wav_rad1, pd.DataFrame):
            time_rad1_2D, freq_rad1_2D = np.meshgrid(df_wind_wav_rad1.index, df_wind_wav_rad1.columns, indexing='ij')
            mesh = axs[i].pcolormesh(time_rad1_2D, freq_rad1_2D, df_wind_wav_rad1.iloc[:-1,:-1], 
                                     shading='flat', cmap=cmap, norm=log_norm)

        if isinstance(df_wind_wav_rad2, pd.DataFrame):
            time_rad2_2D, freq_rad2_2D = np.meshgrid(df_wind_wav_rad2.index, df_wind_wav_rad2.columns, indexing='ij')
            mesh = axs[i].pcolormesh(time_rad2_2D, freq_rad2_2D, df_wind_wav_rad2.iloc[:-1,:-1], 
                                     shading='flat', cmap=cmap, norm=log_norm)

        if mesh is not None:
            # Add inset axes for colorbar
            axins = inset_axes(axs[i], width="100%", height="100%", loc="center", 
                               bbox_to_anchor=(1.01,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
            cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
            cbar.set_label(r"Intensity [$\mathrm{V^2/Hz}$]", rotation=90, labelpad=10, fontsize=font_ylabel)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency [MHz]", fontsize=font_ylabel)
        
        i += 1

    if options.stix.value == True:
        plot_solo_stix(df_stix, axs[i], options.stix_ltc.value, legends_inside, font_ylabel)
        i += 1 

    if options.goes.value == True:
        plot_goes_xrs(options=options, data=df_goes, sat=goes_sat, ax=axs[i], font_legend=font_legend)
        i += 1

    if plot_electrons:
        # electrons
        ax = axs[i]
        if options.l1_wind_e.value and isinstance(edic, pd.DataFrame):
            ax.set_prop_cycle('color', plt.cm.Greens_r(np.linspace(0,1, len(meta_e['channels_dict_df'])+color_offset)))
            for ch in wind_ch_e:
                ax.plot(edic.index, edic[f'FLUX_{ch}'] * wind_ev2MeV_fac, 
                        label='Wind/3DP '+meta_e['channels_dict_df']['Bins_Text'].iloc[1:].values[ch], drawstyle='steps-mid')
        
        color_offset = 2
        if options.l1_ephin.value and isinstance(ephin, pd.DataFrame):
                ax.set_prop_cycle('color', plt.cm.Blues_r(np.linspace(0, 1, 4+color_offset)))
                for ch in ephin_ch:
                    ax.plot(ephin.index, ephin[ch]*intercal, label='SOHO/EPHIN '+meta_ephin[ch], drawstyle='steps-mid')

        # ax.set_ylim(1e0, 1e4)
        if legends_inside:
            ax.legend(loc='upper right', borderaxespad=0., 
                    title=f'Electrons', fontsize=font_legend)
        else:
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0., 
                    title=f'Electrons', fontsize=font_legend)
            
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label, fontsize=font_ylabel)
        i += 1
        
    color_offset = 2    
    if plot_protons:    
        # protons low en:
        ax = axs[i]
        if options.l1_wind_p.value and isinstance(pdic, pd.DataFrame):
            ax.set_prop_cycle('color', plt.cm.Wistia_r(np.linspace(0,1, len(meta_p['channels_dict_df'])+color_offset)))
            for ch in wind_ch_p:
                ax.plot(pdic.index, pdic[f'FLUX_{ch}'] * wind_ev2MeV_fac, 
                        label='Wind/3DP '+meta_p['channels_dict_df']['Bins_Text'].iloc[2:].values[ch], drawstyle='steps-mid')

        # protons high en:
        if options.l1_erne.value and isinstance(erne_p, pd.DataFrame):
            ax.set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0.2,1,10))) #cm.RdPu_r
            for ch in erne_ch:
                ax.plot(erne_p.index, erne_p[f'PH_{ch}'], 
                        label='SOHO/ERNE/HED '+meta_erne['channels_dict_df_p']['ch_strings'][ch], drawstyle='steps-mid')
                
        if legends_inside:
            ax.legend(loc='upper right', borderaxespad=0., fontsize=font_legend, title="Protons/Ions")
        else:
            ax.legend(loc='upper left', borderaxespad=0., fontsize=font_legend, 
                      bbox_to_anchor=(1.01, 1), title="Protons/Ions")

        ax.set_ylabel(intensity_label, fontsize=font_ylabel)
        ax.set_yscale('log')
        i += 1

    
    if options.mag.value == True:    
        ax = axs[i]
        if isinstance(df_mag, pd.DataFrame):
            ax.plot(df_mag.index, df_mag.B.values, label='B', color='k', linewidth=1)
            ax.plot(df_mag.index, df_mag.BRTN_0.values, label='Br', color='dodgerblue', linewidth=1)
            ax.plot(df_mag.index, df_mag.BRTN_1.values, label='Bt', color='limegreen', linewidth=1)
            ax.plot(df_mag.index, df_mag.BRTN_2.values, label='Bn', color='deeppink', linewidth=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        if legends_inside:
            ax.legend(loc='upper right', borderaxespad=0., fontsize=font_legend)
        else:
            ax.legend(loc='upper left', borderaxespad=0., fontsize=font_legend, bbox_to_anchor=(1.01, 1))
        ax.set_ylabel('B [nT]', fontsize=font_ylabel)
        ax.tick_params(axis="x",direction="in", which='both') #, pad=-15
        
        i += 1
        
        if options.polarity.value and isinstance(df_mag_pol, pd.DataFrame):
            pos = get_horizons_coord('Wind', time={'start':df_mag_pol.index[0]-pd.Timedelta(minutes=15),
                                                'stop':df_mag_pol.index[-1]+pd.Timedelta(minutes=15),'step':"1min"}) 
                                                    # (lon, lat, radius) in (deg, deg, AU)
            pos = pos.transform_to(frames.HeliographicStonyhurst())

            #Interpolate position data to magnetic field data cadence
            r = np.interp([t.timestamp() for t in df_mag_pol.index],
                          [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.radius.value)
            lat = np.interp([t.timestamp() for t in df_mag_pol.index],
                            [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.lat.value)
            
            pol, phi_relative = polarity_rtn(df_mag_pol.BRTN_0.values, df_mag_pol.BRTN_1.values, 
                                             df_mag_pol.BRTN_2.values, r, lat, V=400)
            
            # create an inset axe in the current axe:
            pol_ax = inset_axes(ax, height="5%", width="100%", loc=9, 
                                bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
            
            pol_ax.get_xaxis().set_visible(False)
            pol_ax.get_yaxis().set_visible(False)
            pol_ax.set_ylim(0, 1)
            # pol_ax.set_xlim([df_mag_pol.index.values[0], df_mag_pol.index.values[-1]])
            pol_arr = np.zeros(len(pol)) + 1
            timestamp = df_mag_pol.index.values[2] - df_mag_pol.index.values[1]
            norm = Normalize(vmin=0, vmax=180, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
            
            pol_ax.bar(df_mag_pol.index.values[(phi_relative>=0) & (phi_relative<180)],
                       pol_arr[(phi_relative>=0) & (phi_relative<180)],
                       color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]), width=timestamp)
            pol_ax.bar(df_mag_pol.index.values[(phi_relative>=180) & (phi_relative<360)],
                       pol_arr[(phi_relative>=180) & (phi_relative<360)],
                       color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),
                       width=timestamp)
            
            pol_ax.set_xlim(options.plot_start, options.plot_end)
        
        
    if options.mag_angles.value == True:
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
    if options.T.value == True:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw['ion_temp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=font_ylabel)
        axs[i].set_yscale('log')
        i += 1

    ### Dynamic pressure
    if options.p_dyn.value == True:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw['p_dyn'], '-k', label="Dynamic pressure")
        axs[i].set_ylabel(r"P$_\mathrm{dyn}$ [nPa]", fontsize=font_ylabel)
        i += 1

    ### Density
    if options.N.value == True:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw.ion_density,
                        '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=font_ylabel)
        i += 1

    ### Vsw
    if options.Vsw.value == True:
        if isinstance(df_vsw, pd.DataFrame):
            axs[i].plot(df_vsw.index, df_vsw.vsw,
                        '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [km s$^{-1}$]", fontsize=font_ylabel)
        i += 1
            
    if options.showplot:
        plt.show()

    return fig, axs


