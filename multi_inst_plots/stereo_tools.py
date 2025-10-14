import numpy as np
import pandas as pd
import warnings
import cdflib
import sunpy

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize

from seppy.loader.stereo import stereo_load
from seppy.util import resample_df

from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates import frames


from multi_inst_plots.other_tools import polarity_rtn, mag_angles, load_goes_xrs, load_solo_stix, plot_goes_xrs, plot_solo_stix, make_fig_axs, cdaweb_download_fido


# define some plot settings
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000


# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='No units provided for variable', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='astropy did not recognize units of', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action="ignore", message="No artists with labels found to put in legend.")

# def load_stereo_mag(dataset, startdate, enddate, path=None, max_conn=5):
#     """
#     Workaround for loading weekly MAG data from monthly CDF files.
#     (doesnt work as of 2.5.2025, don't use this!)
#     """

#     trange = a.Time(startdate, enddate)
#     if trange.min==trange.max:
#         print(f'"startdate" and "enddate" might need to be different!')

#     # Catch old default value for pos_timestamp
#     if pos_timestamp is None:
#         pos_timestamp = 'center'

#     if not (pos_timestamp=='center' or pos_timestamp=='start' or pos_timestamp=='original'):
#         raise ValueError(f'"pos_timestamp" must be either "original", "center", or "start"!')
#     cda_dataset = a.cdaweb.Dataset(dataset)
#     try:
#         result = Fido.search(trange, cda_dataset)
#         filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
#         filelist.sort()
#         if path is None:
#             filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
#         elif type(path) is str:
#             filelist = [path + os.sep + f for f in filelist]
#         downloaded_files = filelist

#         for i, f in enumerate(filelist):
#             if os.path.exists(f) and os.path.getsize(f) == 0:
#                 os.remove(f)
#             if not os.path.exists(f):
#                 downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=max_conn)

#         # downloaded_files = Fido.fetch(result, path=path, max_conn=max_conn)
#         data = TimeSeries(downloaded_files, concatenate=True)
#         df = data.to_dataframe()

#         metadata = _get_metadata(dataset, downloaded_files[0])

#     except (RuntimeError, IndexError):
#             print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
#             downloaded_files = []
#             df = []
#             metadata = []

#     return df, metadata

def load_swaves(dataset, startdate, enddate, path=None):
    """
    Load STEREO/WAVES data from CDAWeb.

    Parameters
    ----------
    startdate, enddate : datetime or str
        start/end date in standard format (e.g. YYYY-mm-dd or YYYY/mm/dd, anything parseable by sunpy.time.parse_time)
    
    dataset : string
        dataset identifier
                        
    Returns
    -------
    psd_sfu : pd.DataFrame
        data
    """


    files = cdaweb_download_fido(dataset=dataset, startdate=startdate, enddate=enddate, path=path)

    if len(files) == 0:
        print(f"No {dataset} data found between {startdate} and {enddate}")
        return None
    else:
        freq_mhz = cdflib.CDF(files[0]).varget("FREQUENCY") / 1e6

        psd_sfu = np.empty(shape=(0,len(freq_mhz)))

        time = np.array([], dtype="datetime64")

        for file in files:
            try:
                cdf_file = cdflib.CDF(file)
                
                time_ns_1day = cdf_file.varget('Epoch')
                time_dt  = cdflib.epochs.CDFepoch.to_datetime(time_ns_1day)
                psd_sfu_1day  = cdf_file.varget('PSD_SFU')

                time = np.append(time, time_dt)
                psd_sfu = np.append(psd_sfu, psd_sfu_1day, axis=0)
            except ValueError:
                pass

        # remove bar artifacts caused by non-NaN values before time jumps
            # for each time step except the last one:
        for i in range(len(time)-1):
            # check if time increases by more than 5 min:
            if time[i+1] - time[i] > np.timedelta64(5, "m"):
                psd_sfu[i,:] = np.nan

        psd_sfu = pd.DataFrame(psd_sfu, index=time, columns=freq_mhz)

    return psd_sfu



def load_data(options):
    data = {}
    metadata = {}

    global df_sept_electrons_orig
    global df_sept_protons_orig
    global df_het_orig
    global df_waves_hfr
    global df_waves_lfr
    global df_stix_
    global df_goes_
    global goes_sat
    global df_mag_orig
    global df_magplasma
    global meta_magplas
    global meta_mag
    global meta_se
    global meta_sp
    global meta_het
    global sept_viewing

    sept_viewing = options.ster_sept_viewing.value
    sc = options.ster_sc.value
    stix_ltc = options.stix_ltc.value
    goes_man_select = options.goes_man_select.value
    startdate = options.startdt
    enddate = options.enddt
    path = options.path

    if options.mag.value == False:
        options.polarity.value = False

    options.plot_start = None
    options.plot_end = None

    dataset_num = (options.ster_het_e.value or options.ster_het_p.value) + (options.ster_sept_e.value or options.ster_sept_p.value) \
                     + options.radio.value + (options.mag.value or options.mag_angles.value) \
                    + (options.Vsw.value or options.N.value or options.T.value or options.p_dyn.value or options.polarity.value) \
                    + options.stix.value + options.goes.value
    
    dataset_index = 1

    if options.ster_sept_e.value == True or options.ster_sept_p.value == True:
        print(f"Loading SEPT... (dataset {dataset_index}/{dataset_num})")
        df_sept_electrons_orig, meta_se = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                                                      sept_species='e', sept_viewing=sept_viewing,
                                                      path=path, spacecraft=sc)
        data["sept_electrons"] = df_sept_electrons_orig
        metadata["sept_electrons"] = meta_se
        
        df_sept_protons_orig, meta_sp = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                                                    sept_species='p', sept_viewing=sept_viewing,
                                                    path=path, spacecraft=sc)
        
        data["sept_protons"] = df_sept_protons_orig
        metadata["sept_protons"] = meta_sp
        
        dataset_index += 1
    
    if options.ster_het_e.value == True or options.ster_het_p.value == True:
        print(f"Loading HET... (dataset {dataset_index}/{dataset_num})")
        df_het_orig, meta_het = stereo_load(instrument='HET', startdate=startdate, enddate=enddate, 
                                            path=path, spacecraft=sc)
        
        data["het"] = df_het_orig
        metadata["het"] = meta_het
        
        dataset_index += 1

    if options.mag.value == True or options.mag_angles.value == True:
        print(f"Loading MAG... (dataset {dataset_index}/{dataset_num})")
        df_mag_orig, meta_mag = stereo_load(spacecraft=sc, instrument='MAG', startdate=startdate, enddate=enddate, 
                                            mag_coord='RTN', path=path)
        
        data["mag"] = df_mag_orig
        metadata["mag"] = meta_mag

        dataset_index += 1
        
    if options.Vsw.value or options.N.value or options.T.value or options.p_dyn.value or options.polarity.value:
        print(f"Loading MAGPLASMA... (dataset {dataset_index}/{dataset_num})")
        df_magplasma, meta_magplas = stereo_load(instrument='MAGPLASMA', startdate=startdate, enddate=enddate, 
                            path=path, spacecraft=sc)
        
        data["magplasma"] = df_magplasma
        metadata["magplasma"] = meta_magplas
        
        dataset_index += 1

    if options.radio.value == True:
        print(f"Loading WAVES... (dataset {dataset_index}/{dataset_num})")
        df_waves_hfr = load_swaves(f"ST{sc}_L3_WAV_HFR", startdate=startdate, enddate=enddate, path=path)
        df_waves_lfr = load_swaves(f"ST{sc}_L3_WAV_LFR", startdate=startdate, enddate=enddate, path=path)

        data["waves_hfr"] = df_waves_hfr
        data["waves_lfr"] = df_waves_lfr
        
        dataset_index += 1

    if options.stix.value == True:
        print(f"Loading SolO/STIX... (dataset {dataset_index}/{dataset_num})")
        df_stix_ = load_solo_stix(start=startdate, end=enddate, ltc=stix_ltc, resample=None)
        data["stix"] = df_stix_

        dataset_index += 1

    if options.goes.value == True:
        print(f"Loading GOES/XRS... (dataset {dataset_index}/{dataset_num})")
        df_goes_, goes_sat = load_goes_xrs(startdate, enddate, man_select=goes_man_select, resample=None, path=path)

        data["goes"] = df_goes_
        metadata["goes_sat"] = goes_sat

    print("Data loaded!")

    return data, metadata
    

def energy_channel_selection(options):
    cols = []
    df = pd.DataFrame()
    try:
        if options.ster_sept_e.value == True:
            if isinstance(meta_se, pd.DataFrame):
                cols.append("SEPT Electrons")
                series_se = meta_se["ch_strings"].reset_index(drop=True)
                df = pd.concat([df, series_se], axis=1)
        
        if options.ster_sept_p.value == True:
            if isinstance(meta_sp, pd.DataFrame):
                cols.append("SEPT Protons")
                series_sp = meta_sp["ch_strings"].reset_index(drop=True)
                df = pd.concat([df, series_sp], axis=1)

        if options.ster_het_e.value == True:
            if isinstance(meta_het, dict):
                cols.append("HET Electrons")
                series_he = pd.Series(meta_het["Electron_Bins_Text"])
                df = pd.concat([df, series_he], axis=1)

        if options.ster_het_p.value == True:
            if isinstance(meta_het, dict):
                cols.append("HET Protons")
                series_hp = pd.Series(meta_het["Proton_Bins_Text"])
                df = pd.concat([df, series_hp], axis=1)
    except NameError:
        print("Some particle data option was selected but not loaded. Run load_data() first!")

    df.columns = cols
    return df



def make_plot(options):
    
    if options.ster_sept_viewing.value != sept_viewing:
        print(f"Data not loaded for chosen SEPT viewing direction ({options.ster_sept_viewing.value}).", 
              f"Replotting with previous selection ({sept_viewing}).")

    resample = options.resample.value
    resample_mag = options.resample_mag.value
    resample_stixgoes = options.resample_stixgoes.value

    if options.ster_sept_e.value == True:
        if isinstance(df_sept_electrons_orig, pd.DataFrame):
            if resample > 1:
                df_sept_electrons = resample_df(df_sept_electrons_orig, str(60 * resample) + "s")
            else:
                print("SEPT native cadence is 1 min, so no averaging was applied.")
                df_sept_electrons = df_sept_electrons_orig
        else:
            df_sept_electrons = df_sept_electrons_orig

    if options.ster_sept_p.value == True:
        if isinstance(df_sept_protons_orig, pd.DataFrame):
            if resample > 1:
                df_sept_protons = resample_df(df_sept_protons_orig, str(60 * resample) + "s")
            else:
                print("SEPT native cadence is 1 min, so no averaging was applied.")
                df_sept_protons = df_sept_protons_orig
        else:
            df_sept_protons = df_sept_protons_orig

    if options.ster_het_e.value or options.ster_het_p.value:
        if isinstance(df_het_orig, pd.DataFrame):
            if resample > 1:
                df_het = resample_df(df_het_orig, str(60 * resample) + "s") 
            else:
                print("HET native cadence is 1 min, so no averaging was applied.")
                df_het = df_het_orig
        else:
            df_het = df_het_orig
            
        
    if options.Vsw.value or options.N.value or options.T.value or options.polarity.value:
        if isinstance(df_magplasma, pd.DataFrame):
            if resample > 1:
                df_magplas = resample_df(df_magplasma, str(60 * resample_mag) + "s")
            else:
                print("MAGPLASMA native cadence is 1 min, so no averaging was applied.")
                df_magplas = df_magplasma
        else:
            df_magplas = df_magplasma

    if options.mag.value or options.mag_angles.value:
        if isinstance(df_mag_orig, pd.DataFrame):
            if resample == 0:
                df_mag = resample_df(df_mag_orig, "5s") # high cadence, resample to ease load
            else:
                df_mag = resample_df(df_mag_orig, str(60 * resample_mag) + "s")
        else:
            df_mag = df_mag_orig

    if options.goes.value == True:
        if isinstance(df_goes_, pd.DataFrame) and resample_stixgoes > 0:
            df_goes = resample_df(df_goes_, str(60 * resample_stixgoes) + "s")
        else:
            df_goes = df_goes_
        
    if options.stix.value == True:
        if isinstance(df_stix_, pd.DataFrame) and resample_stixgoes > 0:
            df_stix = resample_df(df_stix_, str(60 * resample_stixgoes) + "s")
        else:
            df_stix = df_stix_

    font_ylabel = 20
    font_legend = 10
    
    plot_electrons = options.ster_sept_e.value or options.ster_het_e.value
    plot_protons = options.ster_sept_p.value or options.ster_het_p.value
    
    ch_sept_e = options.ster_ch_sept_e.value
    ch_sept_p = options.ster_ch_sept_p.value
    ch_het_p = options.ster_ch_het_p.value
    ch_het_e = options.ster_ch_het_e.value

    cmap = options.radio_cmap.value
    legends_inside = options.legends_inside.value


    #Chosen channels
    if plot_protons or plot_electrons:
        print("Chosen energy channels:")
        if options.ster_sept_e.value:
            print(f'SEPT electrons: {ch_sept_e}, {len(ch_sept_e)}')
        if options.ster_het_e.value:
            print(f'HET electrons: {ch_het_e}, {len(ch_het_e)}')
        if options.ster_sept_p.value:
            print(f'SEPT protons: {ch_sept_p}, {len(ch_sept_p)}')
        if options.ster_het_p.value:
            print(f'HET protons: {ch_het_p}, {len(ch_het_p)}')

    fig, axs = make_fig_axs(options)

    i = 0


    color_offset = 4
    if options.radio.value == True:
        vmin, vmax = 500, 1e7
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        mesh = None

        if isinstance(df_waves_hfr, pd.DataFrame):
            TimeHFR2D, FreqHFR2D = np.meshgrid(df_waves_hfr.index, df_waves_hfr.columns, indexing='ij')
            mesh = axs[i].pcolormesh(TimeHFR2D, FreqHFR2D, df_waves_hfr.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)

        if isinstance(df_waves_lfr, pd.DataFrame):
            TimeLFR2D, FreqLFR2D = np.meshgrid(df_waves_lfr.index, df_waves_lfr.columns, indexing='ij')
            mesh = axs[i].pcolormesh(TimeLFR2D, FreqLFR2D, df_waves_lfr.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)

        if mesh is not None:    
            # Add inset axes for colorbar
            axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.01,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
            cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
            cbar.set_label("Intensity [sfu]", rotation=90, labelpad=10, fontsize=font_ylabel)

        axs[i].set_ylim((2.61e-3,1.60e1))
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
        axs[i].set_yscale('log')
        if options.ster_sept_e.value == True:
            # plot sept electron channels
            axs[i].set_prop_cycle('color', plt.cm.Greens_r(np.linspace(0,1,len(ch_sept_e)+color_offset)))
            if isinstance(df_sept_electrons, pd.DataFrame):
                for channel in ch_sept_e:
                    axs[i].plot(df_sept_electrons.index, df_sept_electrons[f'ch_{channel+2}'],
                                ds="steps-mid", label='SEPT '+meta_se.ch_strings[channel+2])
        if options.ster_het_e.value == True:
            # plot het electron channels
            axs[i].set_prop_cycle('color', plt.cm.Blues_r(np.linspace(0,1,4+color_offset)))
            if isinstance(df_het, pd.DataFrame):
                for channel in ch_het_e:
                    axs[i].plot(df_het[f'Electron_Flux_{channel}'], 
                                label='HET '+meta_het['channels_dict_df_e'].ch_strings[channel],
                            ds="steps-mid")
            axs[i].set_ylim(bottom=1e-3)
        
        axs[i].set_ylabel("Intensity\n"+r"[(cm$^2$ sr s MeV)$^{-1}$]", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Electrons (SEPT: {sept_viewing}', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0., 
                    title=f'Electrons (SEPT: {sept_viewing})', fontsize=font_legend)
        
        i +=1    

        
    color_offset = 2    
    if plot_protons:
        axs[i].set_yscale('log')
        if options.ster_sept_p.value == True:
            # plot sept proton channels
            num_channels = len(ch_sept_p)# + len(n_het_p)
            axs[i].set_prop_cycle('color', plt.cm.Wistia_r(np.linspace(0,1,num_channels+color_offset)))
            if isinstance(df_sept_protons, pd.DataFrame):
                for channel in ch_sept_p:
                    axs[i].plot(df_sept_protons.index, df_sept_protons[f'ch_{channel+2}'], 
                            label='SEPT '+meta_sp.ch_strings[channel+2], ds="steps-mid")
            
            
        color_offset = 3 
        if options.ster_het_p.value == True:
            # plot het proton channels
            axs[i].set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0.2,1,len(ch_het_p)+color_offset)))
            if isinstance(df_het, pd.DataFrame):
                for channel in ch_het_p:
                    axs[i].plot(df_het.index, df_het[f'Proton_Flux_{channel}'], 
                            label='HET '+meta_het['channels_dict_df_p'].ch_strings[channel], ds="steps-mid")
            axs[i].set_ylim(bottom=1e-4)
        
        axs[i].set_ylabel("Intensity\n"+r"[(cm$^2$ sr s MeV)$^{-1}$]", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Protons/Ions (SEPT: {sept_viewing})', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0., 
                    title=f'Protons/Ions (SEPT: {sept_viewing})', fontsize=font_legend)
        
        
        i +=1    
        
    # plot magnetic field
    if options.mag.value == True:
        ax = axs[i]
        if isinstance(df_mag, pd.DataFrame):
            ax.plot(df_mag.index, df_mag.BFIELD_3, label='B', color='k', linewidth=1)
            ax.plot(df_mag.index.values, df_mag.BFIELD_0.values, label='Br', color='dodgerblue')
            ax.plot(df_mag.index.values, df_mag.BFIELD_1.values, label='Bt', color='limegreen')
            ax.plot(df_mag.index.values, df_mag.BFIELD_2.values, label='Bn', color='deeppink')
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        if legends_inside:
            ax.legend(loc='upper right', borderaxespad=0., fontsize=font_legend)
        else:
            ax.legend(loc='upper left', borderaxespad=0., fontsize=font_legend, bbox_to_anchor=(1.01, 1))
            
        ax.set_ylabel('B [nT]', fontsize=font_ylabel)
        ax.tick_params(axis="x", direction="in", which='both')#, pad=-15)
        
        
        if options.polarity.value == True and isinstance(df_magplas, pd.DataFrame):
            pos = get_horizons_coord(f'STEREO-{options.ster_sc.value}', time={'start':df_magplas.index[0]-pd.Timedelta(minutes=15),'stop':df_magplas.index[-1]+pd.Timedelta(minutes=15),'step':"1min"})  # (lon, lat, radius) in (deg, deg, AU)
            pos = pos.transform_to(frames.HeliographicStonyhurst())
            #Interpolate position data to magnetic field data cadence
            r = np.interp([t.timestamp() for t in df_magplas.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.radius.value)
            lat = np.interp([t.timestamp() for t in df_magplas.index],[t.timestamp() for t in pd.to_datetime(pos.obstime.value)],pos.lat.value)
            pol, phi_relative = polarity_rtn(df_magplas.BFIELDRTN_0.values, df_magplas.BFIELDRTN_1.values, df_magplas.BFIELDRTN_2.values,r,lat,V=400)
            # create an inset axe in the current axe:
            pol_ax = inset_axes(ax, height="5%", width="100%", loc=9, bbox_to_anchor=(0.,0,1,1.1), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
            pol_ax.get_xaxis().set_visible(False)
            pol_ax.get_yaxis().set_visible(False)
            pol_ax.set_ylim(0,1)
            pol_ax.set_xlim([df_magplas.index.values[0], df_magplas.index.values[-1]])
            pol_arr = np.zeros(len(pol))+1
            timestamp = df_magplas.index.values[2] - df_magplas.index.values[1]
            norm = Normalize(vmin=0, vmax=180, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
            pol_ax.bar(df_magplas.index.values[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
            pol_ax.bar(df_magplas.index.values[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
            pol_ax.set_xlim(options.plot_start, options.plot_end)

        i += 1
        
    if options.mag_angles.value == True:
        ax = axs[i]
        #Bmag = np.sqrt(np.nansum((mag_data.B_r.values**2,mag_data.B_t.values**2,mag_data.B_n.values**2), axis=0)) 
        if isinstance(df_mag, pd.DataFrame):  
            alpha, phi = mag_angles(df_mag.BFIELD_3, df_mag.BFIELD_0.values, df_mag.BFIELD_1.values,
                                    df_mag.BFIELD_2.values)
            ax.plot(df_mag.index, alpha, '.k', label='alpha', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-90, 90)
        ax.set_ylabel(r"$\Theta_\mathrm{B}$ [°]", fontsize=font_ylabel)
        # ax.set_xlim(X1, X2)
        ax.tick_params(axis="x",direction="in", pad=-15)

        i += 1
        ax = axs[i]
        if isinstance(df_mag, pd.DataFrame):  
            ax.plot(df_mag.index, phi, '.k', label='phi', ms=1)
        ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim(-180, 180)
        ax.set_ylabel(r"$\Phi_\mathrm{B}$ [°]", fontsize=font_ylabel)
        # ax.set_xlim(X1, X2)
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        i += 1
        
    ### Temperature
    if options.T.value == True:
        if isinstance(df_magplas, pd.DataFrame):  
            axs[i].plot(df_magplas.index, df_magplas['Tp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=font_ylabel)
        axs[i].set_yscale('log')
        i += 1

    ### Dynamic pressure
    if options.p_dyn.value == True:
        if isinstance(df_magplas, pd.DataFrame):  
            axs[i].plot(df_magplas.index, df_magplas['Dynamic_Pressure'], '-k', label="Dynamic pressure")
        axs[i].set_ylabel(r"P$_\mathrm{dyn}$ [nPa]", fontsize=font_ylabel)
        i += 1

    ### Density
    if options.N.value == True:
        if isinstance(df_magplas, pd.DataFrame):
            axs[i].plot(df_magplas.index, df_magplas.Np,
                        '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=font_ylabel)
        
        i += 1

    ### Sws
    if options.Vsw.value == True:
        if isinstance(df_magplas, pd.DataFrame):
            axs[i].plot(df_magplas.index, df_magplas.Vp,
                        '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [km s$^{-1}$]", fontsize=font_ylabel)
        #i += 1
        
    if options.showplot:
        plt.show()
        
    return fig, axs

