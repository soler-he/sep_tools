# from IPython.core.display import display, HTML
# display(HTML(data="""<style> div#notebook-container { width: 80%; } div#menubar-container { width: 85%; } div#maintoolbar-container { width: 90%; } </style>"""))
import numpy as np
import os
import pandas as pd
import datetime as dt
import warnings
import math
import cdflib
import sys
import sunpy

from matplotlib import pyplot as plt
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rcParams['agg.path.chunksize'] = 20000

from matplotlib import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize

from seppy.loader.stereo import stereo_load
from seppy.util import resample_df

from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates import frames


from multi_inst_plots.other_tools import polarity_rtn, mag_angles, load_goes_xrs, load_solo_stix, plot_goes_xrs, plot_solo_stix, make_fig_axs
import multi_inst_plots.cdaweb as cdaweb


# omit Pandas' PerformanceWarning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='No units provided for variable', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='astropy did not recognize units of', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action="ignore", message="No artists with labels found to put in legend.")



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
    ndarray :
        1) timestamps in matplotlib format,
        2) frequencies in MHz,
        3) intensities in sfu for each (time, frequency) data point
    """


    files = cdaweb.cdaweb_download_fido(dataset=dataset, startdate=startdate, enddate=enddate, path=path)

    if len(files) == 0:
        print(f"No SWAVES radio data found between {startdate} and {enddate}")
        return None
    else:
        freq_mhz = cdflib.CDF(files[0]).varget("FREQUENCY") / 1e6

        psd_sfu = np.empty(shape=(0,len(freq_mhz)))

        time = np.array([], dtype="datetime64")

        for file in files:
            cdf_file = cdflib.CDF(file)
            
            time_ns_1day = cdf_file.varget('Epoch')
            time_dt  = cdflib.epochs.CDFepoch.to_datetime(time_ns_1day)
            psd_sfu_1day  = cdf_file.varget('PSD_SFU')

            time = np.append(time, time_dt)
            psd_sfu = np.append(psd_sfu, psd_sfu_1day, axis=0)

        # remove bar artifacts caused by non-NaN values before time jumps
            # for each time step except the last one:
        for i in range(len(time)-1):
            # check if time increases by more than 5 min:
            if time[i+1] - time[i] > np.timedelta64(5, "m"):
                psd_sfu[i,:] = np.nan

        psd_sfu = pd.DataFrame(psd_sfu, index=time, columns=freq_mhz)

    return psd_sfu


def load_data(options):
    global df_sept_electrons
    global df_sept_protons
    global df_het
    global df_waves_hfr
    global df_waves_lfr
    global df_stix
    global df_goes
    global goes_sat
    global df_mag
    global df_magplas

    global meta_magplas
    global meta_mag
    global meta_se
    global meta_sp
    global meta_het
    
    global startdate
    global enddate
    global sept_viewing
    global sc
    global plot_sept_p
    global plot_sept_e
    global plot_protons
    global plot_electrons
    global plot_radio
    global plot_stix
    global plot_goes
    global plot_mag
    global plot_mag_angles
    global plot_Vsw
    global plot_N
    global plot_T
    global plot_het
    global plot_het_e
    global plot_het_p
    global plot_polarity
    global stix_ltc

    startdate = options.startdt
    enddate = options.enddt
    
    sept_viewing = options.ster_sept_viewing.value
    sc = options.ster_sc.value
    plot_radio = options.radio.value
    plot_stix = options.stix.value
    stix_ltc = options.stix_ltc.value
    plot_goes = options.goes.value
    goes_pick_max = options.goes_pick_max.value
    plot_het_e = options.ster_het_e.value
    plot_het_p = options.ster_het_p.value
    plot_sept_e = options.ster_sept_e.value
    plot_sept_p = options.ster_sept_p.value
    plot_mag = options.mag.value
    plot_mag_angles = options.mag_angles.value
    plot_Vsw = options.Vsw.value
    plot_N = options.N.value
    plot_T = options.T.value
    plot_polarity = options.polarity.value
    
    path = options.path

    plot_electrons = plot_het_e or plot_sept_e

    plot_protons = plot_het_p or plot_sept_p

    plot_het = plot_het_p or plot_het_e

    resample = str(options.resample.value) + "min"
    resample_mag = str(options.resample_mag.value) + "min"
    resample_stixgoes = str(options.resample_stixgoes.value) + "min"

    if plot_sept_e:
        df_sept_electrons_orig, meta_se = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                            sept_species='e', sept_viewing=sept_viewing,
                            path=path, spacecraft=sc)
        
    if plot_sept_p:
        df_sept_protons_orig, meta_sp = stereo_load(instrument='SEPT', startdate=startdate, enddate=enddate, 
                                sept_species='p', sept_viewing=sept_viewing,
                                path=path, spacecraft=sc)
    
    if plot_het:
        df_het_orig, meta_het = stereo_load(instrument='HET', startdate=startdate, enddate=enddate,
                        path=path, spacecraft=sc)

    if plot_mag or plot_mag_angles:
        df_mag_orig, meta_mag = stereo_load(spacecraft=sc, instrument='MAG', startdate=startdate, enddate=enddate, mag_coord='RTN', 
                                        path=path)

    if plot_Vsw or plot_N or plot_T or plot_polarity:
        df_magplasma, meta_magplas = stereo_load(instrument='MAGPLASMA', startdate=startdate, enddate=enddate, 
                            path=path, spacecraft=sc)
        

    if plot_radio:
        df_waves_hfr = load_swaves(f"ST{sc}_L3_WAV_HFR", startdate=startdate, enddate=enddate, path=path)
        df_waves_lfr = load_swaves(f"ST{sc}_L3_WAV_LFR", startdate=startdate, enddate=enddate, path=path)

    if plot_stix:
        df_stix = load_solo_stix(start=startdate, end=enddate, ltc=stix_ltc, resample=resample_stixgoes)

    if plot_goes:
        df_goes, goes_sat = load_goes_xrs(start=startdate, end=enddate, pick_max=goes_pick_max, resample=resample_stixgoes)


    
    ### Resampling

    if plot_sept_e:
        if isinstance(df_sept_electrons_orig, pd.DataFrame) and resample != "0min":
            df_sept_electrons = resample_df(df_sept_electrons_orig, resample)
        else:
            df_sept_electrons = df_sept_electrons_orig

    if plot_sept_p:
        if isinstance(df_sept_protons_orig, pd.DataFrame) and resample != "0min":
            df_sept_protons = resample_df(df_sept_protons_orig, resample)
        else:
            df_sept_protons = df_sept_protons_orig

    if plot_het:
        if isinstance(df_het_orig, pd.DataFrame) and resample != "0min":
            df_het = resample_df(df_het_orig, resample)  
        else:
            df_het = df_het_orig
            
        
    if plot_Vsw or plot_N or plot_T or plot_polarity:
        if isinstance(df_magplasma, pd.DataFrame) and resample_mag != "0min":
            df_magplas = resample_df(df_magplasma, resample_mag)
             
        else:
            df_magplas = df_magplasma

    if plot_mag or plot_mag_angles:
        if isinstance(df_mag_orig, pd.DataFrame) and resample_mag != "0min":
            df_mag = resample_df(df_mag_orig, resample_mag)
            
        else:
            df_mag = df_mag_orig
            
    




def make_plot(options):
    
    font_ylabel = 20
    font_legend = 10
    
    
    ch_sept_e = options.ster_ch_sept_e.value
    ch_sept_p = options.ster_ch_sept_p.value
    ch_het_p = options.ster_ch_het_p.value
    ch_het_e = (0, 1, 2)

    cmap = options.radio_cmap.value
    legends_inside = options.legends_inside.value


    #Chosen channels
    if plot_protons or plot_electrons:
        print('Chosen energy channels:')
        if plot_electrons:
            if plot_sept_e:
                print(f'SEPT electrons: {ch_sept_e}, {len(ch_sept_e)}')
            if plot_het_e:
                print(f'HET electrons: {ch_het_e}, {len(ch_het_e)}')
        if plot_protons:
            if plot_sept_p:
                print(f'SEPT protons: {ch_sept_p}, {len(ch_sept_p)}')
            if plot_het_p:
                print(f'HET protons: {ch_het_p}, {len(ch_het_p)}')

    fig, axs = make_fig_axs(options)

    i = 0


    color_offset = 4
    if plot_radio:
        vmin, vmax = 500, 1e7
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        if isinstance(df_waves_hfr, pd.DataFrame) and isinstance(df_waves_lfr, pd.DataFrame):
            TimeHFR2D, FreqHFR2D = np.meshgrid(df_waves_hfr.index, df_waves_hfr.columns, indexing='ij')
            TimeLFR2D, FreqLFR2D = np.meshgrid(df_waves_lfr.index, df_waves_lfr.columns, indexing='ij')

            # Create colormeshes. Shading option flat and thus the removal of last row and column are there to solve the time jump bar problem, 
            # when resampling isn't used
            mesh = axs[i].pcolormesh(TimeLFR2D, FreqLFR2D, df_waves_lfr.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)
            axs[i].pcolormesh(TimeHFR2D, FreqHFR2D, df_waves_hfr.iloc[:-1,:-1], shading='flat', cmap=cmap, norm=log_norm)
            # Add inset axes for colorbar
            axins = inset_axes(axs[i], width="100%", height="100%", loc="center", bbox_to_anchor=(1.05,0,0.03,1), bbox_transform=axs[i].transAxes, borderpad=0.2)
            cbar = fig.colorbar(mesh, cax=axins, orientation="vertical")
            cbar.set_label("Intensity (sfu)", rotation=90, labelpad=10, fontsize=font_ylabel)

        axs[i].set_yscale('log')
        axs[i].set_ylabel("Frequency (MHz)", fontsize=font_ylabel)
        
        i += 1

    if plot_stix:
        plot_solo_stix(df_stix, axs[i], stix_ltc, legends_inside, font_ylabel)
        i += 1 

    if plot_goes:
        plot_goes_xrs(df_goes, goes_sat, axs[i], legends_inside, font_ylabel)
        i += 1

    if plot_electrons:
        if plot_sept_e:
            # plot sept electron channels
            axs[i].set_prop_cycle('color', plt.cm.Reds_r(np.linspace(0,1,len(ch_sept_e)+color_offset)))
            if isinstance(df_sept_electrons, pd.DataFrame):
                for channel in ch_sept_e:
                    axs[i].plot(df_sept_electrons.index, df_sept_electrons[f'ch_{channel+2}'],
                                ds="steps-mid", label='SEPT '+meta_se.ch_strings[channel+2])
        if plot_het_e:
            # plot het electron channels
            axs[i].set_prop_cycle('color', plt.cm.PuRd_r(np.linspace(0,1,4+color_offset)))
            if isinstance(df_het, pd.DataFrame):
                for channel in ch_het_e:
                    axs[i].plot(df_het[f'Electron_Flux_{channel}'], 
                                label='HET '+meta_het['channels_dict_df_e'].ch_strings[channel],
                            ds="steps-mid")
        
        axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Electrons (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., 
                    title=f'Electrons (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        axs[i].set_yscale('log')
        i +=1    

        
    color_offset = 2    
    if plot_protons:
        if plot_sept_p:
            # plot sept proton channels
            num_channels = len(ch_sept_p)# + len(n_het_p)
            axs[i].set_prop_cycle('color', plt.cm.plasma(np.linspace(0,1,num_channels+color_offset)))
            if isinstance(df_sept_protons, pd.DataFrame):
                for channel in ch_sept_p:
                    axs[i].plot(df_sept_protons.index, df_sept_protons[f'ch_{channel+2}'], 
                            label='SEPT '+meta_sp.ch_strings[channel+2], ds="steps-mid")
            
        color_offset = 0 
        if plot_het_p:
            # plot het proton channels
            axs[i].set_prop_cycle('color', plt.cm.YlOrRd(np.linspace(0.2,1,len(ch_het_p)+color_offset)))
            if isinstance(df_het, pd.DataFrame):
                for channel in ch_het_p:
                    axs[i].plot(df_het.index, df_het[f'Proton_Flux_{channel}'], 
                            label='HET '+meta_het['channels_dict_df_p'].ch_strings[channel], ds="steps-mid")
        
        axs[i].set_ylabel("Flux\n"+r"[(cm$^2$ sr s MeV)$^{-1}]$", fontsize=font_ylabel)
        if legends_inside:
            axs[i].legend(loc='upper right', borderaxespad=0., 
                    title=f'Ions (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        else:
            axs[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
                    title=f'Ions (SEPT: {sept_viewing}, HET: sun)', fontsize=font_legend)
        axs[i].set_yscale('log')
        i +=1    
        
    # plot magnetic field
    if plot_mag:
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
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=font_legend)
            
        ax.set_ylabel('B [nT]', fontsize=font_ylabel)
        ax.tick_params(axis="x", direction="in", which='both')#, pad=-15)
        
        
        if plot_polarity and isinstance(df_magplas, pd.DataFrame):
            pos = get_horizons_coord(f'STEREO-{sc}', time={'start':df_magplas.index[0]-pd.Timedelta(minutes=15),'stop':df_magplas.index[-1]+pd.Timedelta(minutes=15),'step':"1min"})  # (lon, lat, radius) in (deg, deg, AU)
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
        
    if plot_mag_angles:
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
    if plot_T:
        if isinstance(df_magplas, pd.DataFrame):  
            axs[i].plot(df_magplas.index, df_magplas['Tp'], '-k', label="Temperature")
        axs[i].set_ylabel(r"T$_\mathrm{p}$ [K]", fontsize=font_ylabel)
        axs[i].set_yscale('log')
        i += 1

    ### Density
    if plot_N:
        if isinstance(df_magplas, pd.DataFrame):
            axs[i].plot(df_magplas.index, df_magplas.Np,
                        '-k', label="Ion density")
        axs[i].set_ylabel(r"N$_\mathrm{p}$ [cm$^{-3}$]", fontsize=font_ylabel)
        i += 1

    ### Sws
    if plot_Vsw:
        if isinstance(df_magplas, pd.DataFrame):
            axs[i].plot(df_magplas.index, df_magplas.Vp,
                        '-k', label="Bulk speed")
        axs[i].set_ylabel(r"V$_\mathrm{sw}$ [kms$^{-1}$]", fontsize=font_ylabel)
        #i += 1
        
    plt.show()

    return fig, axs


