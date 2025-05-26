import numpy as np
import pandas as pd
import datetime as dt
import os
import sunpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from stixdcpy.quicklook import LightCurves
from seppy.tools import resample_df
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

from time import sleep


def polarity_rtn(Br,Bt,Bn,r,lat,V=400,delta_angle=10):
    """
    Calculates the magnetic field polarity sector for magnetic field data (Br, Bt, Bn) 
    from a spacecraft at a (spherical) distance r (AU) and heliographic latitude lat (deg). 
    Uses the nominal Parker spiral geometry for solar wind speed V (default 400 km/s) for reference.
    delta_angle determines the uncertain angles around 90([90-delta_angle,90+delta_angle]) and 270 
    ([270-delta_angle,270+delta_angle]).
    """
    au = 1.495978707e8 #astronomical units (km)
    omega = 2*np.pi/(25.38*24*60*60) #solar rotation rate (rad/s)
    #Nominal Parker spiral angle at distance r (AU)
    phi_nominal = np.rad2deg(np.arctan(omega*r*au/V))
    #Calculating By and Bx from the data (heliographical coordinates, where meridian centered at sc)
    Bx = Br*np.cos(np.deg2rad(lat)) - Bn*np.sin(np.deg2rad(lat))
    By = Bt
    phi_fix = np.zeros(len(Bx))
    phi_fix[(Bx>0) & (By>0)] = 360.0
    phi_fix[(Bx<0)] = 180.0
    phi = np.rad2deg(np.arctan(-By/Bx)) + phi_fix
    #Turn the origin to the nominal Parker spiral direction
    phi_relative = phi - phi_nominal
    phi_relative[phi_relative>360] -= 360
    phi_relative[phi_relative<0] += 360
    pol = np.nan*np.zeros(len(Br))
    
    pol[((phi_relative>=0) & (phi_relative<=90.-delta_angle)) | ((phi_relative>=270.+delta_angle) & (phi_relative<=360))] = 1
    pol[(phi_relative>=90.+delta_angle) & (phi_relative<=270.-delta_angle)] = -1
    pol[((phi_relative>=90.-delta_angle) & (phi_relative<=90.+delta_angle)) | ((phi_relative>=270.-delta_angle) & (phi_relative<=270.+delta_angle))] = 0
    return pol, phi_relative


def polarity_colorwheel():
    # Generate a figure with a polar projection
    fg = plt.figure(figsize=(1,1))
    ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

    n = 100  #the number of secants for the mesh
    norm = Normalize(0, np.pi) 
    t = np.linspace(0,np.pi,n)   #theta values
    r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = tg                         #define color values as theta value
    im = ax.pcolormesh(t, r, c.T,norm=norm,cmap="bwr")  #plot the colormesh on axis with colormap
    t = np.linspace(np.pi,2*np.pi,n)   #theta values
    r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = 2*np.pi-tg                         #define color values as theta value
    im = ax.pcolormesh(t, r, c.T,norm=norm,cmap="bwr")  #plot the colormesh on axis with colormap
    ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
    ax.tick_params(pad=0,labelsize=8)      #cosmetic changes to tick labels
    ax.spines['polar'].set_visible(False)    #turn off the axis spine.
    ax.grid(False)


def polarity_panel(ax,datetimes,phi_relative,bbox_to_anchor=(0.,0.22,1,1.1)):
    pol_ax = inset_axes(ax, height="8%", width="100%", loc=9,
                        bbox_to_anchor=bbox_to_anchor, 
                        bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
    pol_ax.get_xaxis().set_visible(False)
    pol_ax.get_yaxis().set_visible(False)
    pol_ax.set_ylim(0,1)
    pol_arr = np.zeros(len(phi_relative))+1
    timestamp = datetimes[2] - datetimes[1]
    norm = Normalize(vmin=0, vmax=180, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
    pol_ax.bar(datetimes[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],
               color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
    pol_ax.bar(datetimes[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],
               color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
    return pol_ax


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



def cdaweb_download_fido(dataset, startdate, enddate, path=None, max_conn=5):
    """
    Downloads dataset files via SunPy/Fido from CDAWeb

    Parameters
    ----------
    dataset : {str}
        Name of dataset:
        - 'PSP_FLD_L3_RFS_HFR'
        - 'PSP_FLD_L3_RFS_LFR'
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None
    max_conn : {int}, optional
        The number of parallel download slots used by Fido.fetch, by default 5

    Returns
    -------
    List of downloaded files
    """
    trange = a.Time(startdate, enddate)
    cda_dataset = a.cdaweb.Dataset(dataset)
    try:
        result = Fido.search(trange, cda_dataset)
        filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
        filelist.sort()
        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        # Check if file with same name already exists in path
        for i, f in enumerate(filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=max_conn)
    except (RuntimeError, IndexError):
        print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
        downloaded_files = []
    return downloaded_files



def load_solo_stix(start, end, ltc=True, resample=None):
    if end - start > dt.timedelta(7):
        print("STIX loading for more than 7 days not supported, no data was fetched")
        return []
    try:
        lc = LightCurves.from_sdc(start_utc=start, end_utc=end, ltc=ltc)
        df_stix = lc.to_pandas()

        if resample != "0min" and resample is not None:
            df_stix = resample_df(df_stix, resample=resample, pos_timestamp=None)

    except (TypeError, KeyError):
        print("Unable to load STIX data!")
        df_stix = []

    return df_stix


def plot_solo_stix(data, ax, ltc, legends_inside, font_ylabel):
    if isinstance(data, pd.DataFrame):
        for key in data.keys():
            ax.plot(data.index, data[key], ds="steps-mid", label=key)
    if ltc:
        title = 'SolO/STIX (light travel time corr.)'
    else:
        title = 'SolO/STIX'
    if legends_inside:
        ax.legend(loc='upper right', borderaxespad = 0., title=title, fontsize=10)
    else:
        # axs[i].legend(loc='upper right', title=title, bbox_to_anchor=(1, 0.5))
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad = 0., title=title, fontsize=10)
    ax.set_ylabel('Counts', fontsize=font_ylabel)
    ax.set_yscale('log')


def load_goes_xrs(start, end, man_select=False, resample=None, path=None):
    """
    Load GOES high-cadence XRS data with Fido. Picks largest satellite number available, if none specified.

    Parameters
    ----------

    start : str or dt.datetime
      start date in a parse_time-compatible format
    end : str or dt.datetime
      end date in a parse_time-compatible format
    man_select : bool (optional)
      allow manual selection of GOES satellite (print what's available)
    resample : in

    Returns
    -------

    df_goes : pd.DataFrame
        GOES data
    sat : int
        satellite number for which data was returned
    """
    
    result_goes = Fido.search(a.Time(start, end), a.Instrument("XRS"), a.Resolution("flx1s"))

    # No data found
    if len(result_goes["xrs"]) == 0:
        print(f"No GOES/XRS data found for {start} - {end}!")
        df_goes = []
        sat = ''
        return df_goes, sat

    if man_select:
        print(result_goes)
        sats = tuple(np.unique(result_goes["xrs"]["SatelliteNumber"]).tolist())
        sleep(1)

        while True:
            sat = input(f"Choose preferred GOES satellite number {sats}:")

            if sat == '':
                print("Aborting GOES satellite selection. No data will be plotted.")
                df_goes = []
                return df_goes, sat
            
            try:
                sat = int(sat)
                if sat not in result_goes["xrs"]["SatelliteNumber"]:
                    print("Not a valid option, try again.")
                    sleep(1)
                else:
                    break

            except ValueError:
                print("Not a valid option, try again.")
                sleep(1)

    else:
        sat = int(max(result_goes["xrs"]["SatelliteNumber"]))

    print(f"Fetching GOES-{sat} data for {start} - {end}")
    file_goes = Fido.fetch(result_goes["xrs"][result_goes["xrs", "SatelliteNumber"] == sat], path=path)    

    goes = ts.TimeSeries(file_goes, concatenate=True)
    df_goes = goes.to_dataframe()
    
    # Filter data
    df_goes['xrsa'] = df_goes['xrsa'].mask((df_goes['xrsa_quality'] != 0), other=np.nan)   # mask non-zero quality flagged entries as NaN
    df_goes['xrsb'] = df_goes['xrsb'].mask((df_goes['xrsb_quality'] != 0), other=np.nan)  
    df_goes = df_goes[(df_goes['xrsa_quality'] == 0) | (df_goes['xrsb_quality'] == 0)]     # keep entries that have at least one good quality flag

    # Resampling
    if resample != "0min" and resample is not None:
        df_goes = resample_df(df_goes, resample=resample)

    return df_goes, sat
    

def plot_goes_xrs(options, data, sat, ax, font_legend):
    ax.hlines([1e-7, 1e-6, 1e-5, 1e-4], color="#cccccc", xmin=options.plot_start, xmax=options.plot_end)
    peak = 0
    if isinstance(data, pd.DataFrame):
        peak = max(data["xrsb"])
        for channel, wavelength in zip(["xrsa", "xrsb"], ["0.5 - 4.0 Å", "1.0 - 8.0 Å"]):
            ax.plot(data.index, data[channel], ds="steps-mid", label=wavelength)
        title = f"GOES-{sat}/XRS"
        if options.legends_inside.value == True:
            ax.legend(loc="upper right", title=title, borderaxespad = 0., fontsize = 10)
        else:
            ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left', title=title, borderaxespad = 0., fontsize=10)

    ax.set_ylabel(r"[W m$^{-2}$]")
    ax.set_yscale('log')

    # flare class labels
    for i, cl in enumerate(["A", "B", "C", "M", "X"]):
        log_midpoint = 3.1e-8 * (10 ** i)
        ax.annotate(text=cl, xy=(options.plot_end, log_midpoint), xycoords="data", xytext=(5, 0), 
                    textcoords="offset points", fontsize=font_legend, va="center")
    
    # set minimum y-limits
    if peak > 1e-3:
        ax.set_ylim(bottom=1e-8)
    else:
        ax.set_ylim((1e-8, 1e-3))


def make_fig_axs(options):

    plot_radio = options.radio.value
    plot_mag = options.mag.value
    plot_mag_angles = options.mag_angles.value
    plot_Vsw = options.Vsw.value
    plot_N = options.N.value
    plot_T = options.T.value
    plot_Pdyn = options.p_dyn.value
    plot_stix = options.stix.value
    plot_goes = options.goes.value


    if options.plot_start is None:
        options.plot_start = options.startdt
    
    if options.plot_end is None:
        options.plot_end = options.enddt

    if options.plot_end < options.plot_start:
        print("Check plot ranges! End time cannot precede start time.")

    if ((options.plot_start > options.enddt and options.plot_end > options.enddt) \
        or (options.plot_start < options.startdt and options.plot_end < options.startdt)):
        print("Selected plot range is not between loaded range!")

    if options.plot_end > options.enddt or options.plot_start < options.startdt:
        print(f"Plot range exceeds loaded range {options.startdate.value} - {options.enddate.value}")
         

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        plot_wind_e = options.l1_wind_e.value
        plot_wind_p = options.l1_wind_p.value
        plot_ephin = options.l1_ephin.value
        plot_erne = options.l1_erne.value
        plot_electrons = plot_wind_e or plot_ephin
        plot_protons = plot_wind_p or plot_erne

    if options.spacecraft.value == "Parker Solar Probe":
        plot_epilo_e = options.psp_epilo_e.value
        plot_epihi_e = options.psp_epihi_e.value
        plot_epilo_p = options.psp_epilo_p.value
        plot_epihi_p = options.psp_epihi_p.value
        plot_electrons = plot_epilo_e or plot_epihi_e
        plot_protons = plot_epilo_p or plot_epihi_p

    if options.spacecraft.value == "Solar Orbiter":
        plot_het_e = options.solo_het_e.value
        plot_het_p = options.solo_het_p.value
        plot_ept_e = options.solo_ept_e.value
        plot_ept_p = options.solo_ept_p.value
        plot_electrons = plot_het_e or plot_ept_e
        plot_protons = plot_het_p or plot_ept_p

    if options.spacecraft.value == "STEREO":
        plot_het_e = options.ster_het_e.value
        plot_het_p = options.ster_het_p.value
        plot_sept_e = options.ster_sept_e.value
        plot_sept_p = options.ster_sept_p.value
        plot_electrons = plot_het_e or plot_sept_e
        plot_protons = plot_het_p or plot_sept_p


    font_ylabel = 20
    font_legend = 10

    if options.spacecraft.value == "Solar Orbiter":
        panels = 1*plot_stix + 1*plot_goes + 1*plot_electrons + 1*plot_protons + 2*plot_mag_angles + 1*plot_mag + 1* plot_Vsw + 1* plot_N + 1* plot_T + 1* plot_Pdyn
        
    else: 
        panels = 1*plot_radio + 1*plot_stix + 1*plot_goes + 1*plot_electrons + 1*plot_protons + 2*plot_mag_angles + 1*plot_mag + 1* plot_Vsw + 1* plot_N + 1* plot_T + 1*plot_Pdyn 

    panel_ratios = list(np.zeros(panels)+1)

    if options.spacecraft.value == "Solar Orbiter":      # TODO remove this once RPW is included
        # if plot_radio:
        #     panel_ratios[0] = 2
        if plot_electrons and plot_protons:
            panel_ratios[0 + 1*plot_stix + 1*plot_goes] = 2
            panel_ratios[1 + 1*plot_stix + 1*plot_goes] = 2
        if plot_electrons or plot_protons:    
            panel_ratios[0 + 1*plot_stix + 1*plot_goes] = 2

    else:
        if plot_radio:
            panel_ratios[0] = 2
        if plot_electrons and plot_protons:
            panel_ratios[0 + 1*plot_radio + 1*plot_stix + 1*plot_goes] = 2
            panel_ratios[1 + 1*plot_radio + 1*plot_stix + 1*plot_goes] = 2
        if plot_electrons or plot_protons:    
            panel_ratios[0 + 1*plot_radio + 1*plot_stix + 1*plot_goes] = 2
    
    if panels == 3:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 4*panels])
    else:
        fig, axs = plt.subplots(nrows=panels, sharex=True, figsize=[12, 3*panels], 
                                gridspec_kw={'height_ratios': panel_ratios})
        

    if panels == 1:
        axs = [axs]
    
    if panels == 0:
        print("No instruments chosen!")
        return (None, None)

    pad = 12
    if options.spacecraft.value == "L1 (Wind/SOHO)":
        axs[0].set_title('Near-Earth spacecraft (Wind, SOHO)', pad=pad, fontsize=font_ylabel)
    elif options.spacecraft.value == "Parker Solar Probe":
        axs[0].set_title('Parker Solar Probe', pad=pad, fontsize=font_ylabel)
    elif options.spacecraft.value == "STEREO":
        axs[0].set_title(f'STEREO {options.ster_sc.value}', pad=pad, fontsize=font_ylabel)
    else:
        axs[0].set_title(f'Solar Orbiter', pad=pad, fontsize=font_ylabel)

    
    axs[-1].xaxis.minorticks_on()
    axs[-1].xaxis.set(major_locator=mdates.AutoDateLocator(minticks=6, maxticks=9), 
                      minor_locator=mdates.AutoDateLocator(minticks=10, maxticks=28))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))
    axs[-1].xaxis.set_tick_params(rotation=0)
    axs[-1].set_xlabel(f"Time (UTC) / Date in {options.plot_start.year}", fontsize=15)
    axs[-1].set_xlim(options.plot_start, options.plot_end)
    fig.subplots_adjust(hspace=0.1)
    fig.patch.set_facecolor('white')
    fig.set_dpi(200)

    if options.spacecraft.value != "STEREO":
        print(f"Plotting {options.spacecraft.value} data for timerange {options.plot_start} - {options.plot_end}")
    else:
        print(f"Plotting STEREO {options.ster_sc.value} data for timerange {options.plot_start} - {options.plot_end}")

    return fig, axs

def add_line(time, ax, **kwargs):
    
    if isinstance(ax, np.ndarray):
        for axis in ax:
            axis.axvline(time, **kwargs)

    else:
        ax.axvline(time, **kwargs)
    
def add_shaded_area(start, end, ax, **kwargs):

    if isinstance(ax, np.ndarray):
        for axis in ax:
            axis.axvspan(start, end, **kwargs)

    else:
        ax.axvline(start, end, **kwargs)