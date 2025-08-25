import datetime as dt
import os
import pickle

# import sys
import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy

# from matplotlib.ticker import (MultipleLocator)
# import format_tick_labels
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from seppy.util import custom_warning

from anisotropy.anisotropy_functions_updated import (
    anisotropy_legendre_fit,
    anisotropy_prepare,
    anisotropy_weighted_sum,
    bootstrap_anisotropy,
)
from anisotropy.background_analysis_updated import (
    evaluate_background,
    evaluate_background_all,
    evaluate_background_binwise,
    run_background_analysis,
    run_background_analysis_all,
    run_background_analysis_all_binwise,
    run_background_analysis_all_nomag,
    run_background_analysis_binwise,
    run_background_analysis_equal_decay,
    run_background_analysis_equal_decay_binwise,
)
from anisotropy.solo_methods import (
    solo_download_and_prepare,  # , solo_download_intensities
)
from anisotropy.stereo_methods import stereo_download_and_prepare
from anisotropy.wind_methods import wind_download_and_prepare

plt.rcParams["font.size"] = 12
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams['mathtext.rm'] = 'Arial'
# plt.rcParams['mathtext.it'] = 'Arial:italic'
# plt.rcParams['mathtext.bf'] = 'Arial:bold'


def format_tick_labels(x):
    x = mpl.dates.num2date(x)
    aa = [xi.strftime('%H:%M:%S\n%b %d, %Y') for xi in x]
    date_str = [aa[0].split("\n")[1]]
    lbl = [date_str[0]]
    for m in range(1, len(aa)):
        if aa[m].split("\n")[1] in date_str:
            lbl.append("")
        else:
            lbl.append(aa[m].split("\n")[1])
            date_str.append(aa[m].split("\n")[1])
            if lbl[m-1] != "":
                lbl[m-1] = ""
    return lbl


def add_watermark(fig, scaling=0.15, alpha=0.5, zorder=-1, x=1.0, y=0.0):
    logo = Image.open(f'multi_sc_plots{os.sep}soler.png')
    new_size = (np.array(logo.size) * scaling).astype(int)
    logo_s = logo.resize(new_size, Image.Resampling.LANCZOS)
    # x_offset = int((fig.bbox.xmax - pad*logo_s.size[0]) * 1.0)
    # y_offset = int((fig.bbox.ymax - pad*logo_s.size[1]) * 0.0)
    x_offset = int(fig.bbox.xmax * x)
    y_offset = int(fig.bbox.ymax * y)

    fig.figimage(logo_s, x_offset, y_offset, alpha=alpha, zorder=zorder)


class SEPevent: 
  
    def __init__(self, event_id, path, spacecraft, instrument, species, channels, starttime, endtime, averaging, av_min, solo_ept_ion_contamination_correction, plot_folder=None):
        self.event_id = event_id
        self.path = path+os.sep
        if not plot_folder:
            plot_folder = os.getcwd()
        self.plot_folder = plot_folder+os.sep
        # create folders if not existing yet
        for p in [self.path, self.plot_folder]:
            if not os.path.isdir(p):
                os.makedirs(p)
        self.spacecraft = spacecraft
        self.instrument = instrument
        self.species = species
        self.channels = channels
        self.averaging = averaging
        self.av_min = av_min
        self.intensity_label = 'Intensity\n [1/(s$\\,$cm²$\\,$sr$\\,$MeV)]'

        if spacecraft == "Solar Orbiter":
            if (self.instrument == "EPT") and (self.species == "e"):
                self.solo_ept_ion_contamination_correction = solo_ept_ion_contamination_correction
            else:
                self.solo_ept_ion_contamination_correction = False

        self.check_start_end_format(starttime, endtime)
        
        if av_min is not None:
            # Do not average if the averaging window is equal or lower than the cadence.
            if ("STEREO" in spacecraft) and (instrument == "SEPT") and (av_min <= 1):
                print("Changing averaging from {} to None.".format(averaging))
                self.averaging = None
                self.av_min = None
            elif ("Wind" in spacecraft) and (instrument == "3DP") and (av_min < 0.5):
                print("Changing averaging from {} to None.".format(averaging))
                self.averaging = None
                self.av_min = None
        
    def check_start_end_format(self, starttime, endtime):
        if isinstance(starttime, dt.date):
            if not isinstance(starttime, dt.datetime):
                starttime = dt.datetime.combine(starttime, dt.time.min)
        if isinstance(endtime, dt.date):
            if not isinstance(endtime, dt.datetime):
                endtime = dt.datetime.combine(endtime, dt.time.max)
        self.start_time = starttime
        self.end_time = endtime
        
    def check_background_window(self, bg_start, bg_end, corr_window_end=None):
        # If invalid, use a nominal window of 5 h
        if (bg_start is None) or (bg_end is None):
            self.bg_start = self.start_time
            self.bg_end = self.start_time + pd.Timedelta(5, unit="h")
        elif (bg_start >= bg_end) or (bg_end < self.start_time) or (bg_start > self.end_time):
            print("Invalid background window. Setting to default.")
            self.bg_start = self.start_time
            self.bg_end = self.start_time + pd.Timedelta(5,unit="h")
        else:
            self.bg_start = bg_start
            self.bg_end = bg_end
        #Set the window where the background level is compared to data (if background is larger than data, decrease background).
        if corr_window_end is None:
            self.corr_window_end = self.bg_end + pd.Timedelta(hours=3)
        else:
            self.corr_window_end = corr_window_end

    def set_background_window(self,bg_start,bg_end,corr_window_end=None):
        if bg_start is None:
            self.bg_start = self.start_time
        else:
            self.bg_start = bg_start
        if bg_end is None:
            self.bg_end = self.start_time + pd.Timedelta(hours=5)
        else:
            self.bg_end = bg_end
        if corr_window_end is None:
            self.corr_window_end = self.bg_end + pd.Timedelta(hours=3)
        else:
            self.corr_window_end = corr_window_end
        self.bg_times = self.I_times[(self.I_times>=self.bg_start) & (self.I_times<=self.bg_end)]
        self.bg_I_data = self.I_data[(self.I_times>=self.bg_start) & (self.I_times<=self.bg_end),:]
        self.bg_mu_data = self.mu_data[(self.I_times>=self.bg_start) & (self.I_times<=self.bg_end),:]
        if self.I_unc is not None:
            self.bg_I_unc = self.I_unc[(self.I_times>=self.bg_start) & (self.I_times<=self.bg_end),:]
        
        print("Background window start: {}".format(self.bg_start))
        print("Background window end: {}".format(self.bg_end))
        print("Background correction window end: {}".format(self.corr_window_end))
    
    def download_and_prepare(self): 
        spacecraft = self.spacecraft
        if spacecraft == "Solar Orbiter":
            I_times, I_data, I_unc, sectors, en_channel_string, delta_E, count_str, mu_times, mu_data, mag_data, pol, phi_relative, pol_times, bg_times, bg_I_data, bg_I_unc, bg_mu_data, sp_str, ch_string, mag_data_coord, coverage, flux_arr, count_arr, t_arr, gf_arr, mag_sc, averaging, av_min = solo_download_and_prepare(self.instrument,self.start_time,self.end_time,self.path,self.averaging,self.species,self.channels,self.bg_start,self.bg_end,self.solo_ept_ion_contamination_correction)
            self.flux_arr = flux_arr
            self.count_arr = count_arr
            self.t_arr = t_arr
            self.gf_arr = gf_arr
            self.mag_sc = mag_sc
            self.averaging = averaging
            self.av_min = av_min
        elif "STEREO" in spacecraft:
            I_times, I_data, I_unc, sectors, en_channel_string, delta_E, count_str, mu_times, mu_data, mag_data, pol, phi_relative, pol_times, bg_times, bg_I_data, bg_I_unc, bg_mu_data, sp_str, ch_string, mag_data_coord, coverage, flux_arr, count_arr, t_arr, gf_arr, mag_sc = stereo_download_and_prepare(self.spacecraft,self.instrument,self.start_time,self.end_time,self.path,self.averaging,self.species,self.channels,self.bg_start,self.bg_end)
            self.flux_arr = flux_arr
            self.count_arr = count_arr
            self.t_arr = t_arr
            self.gf_arr = gf_arr
            self.mag_sc = mag_sc
        elif spacecraft == "Wind":
            I_times, I_data, I_unc, sectors, en_channel_string, delta_E, count_str, mu_times, mu_data, mag_data, pol, phi_relative, pol_times, bg_times, bg_I_data, bg_I_unc, bg_mu_data, sp_str, ch_string, mag_data_coord, coverage = wind_download_and_prepare(self.instrument,self.start_time,self.end_time,self.path,self.averaging,self.species,self.channels,self.bg_start,self.bg_end)

        self.I_times = I_times
        self.I_data = I_data
        self.I_unc = I_unc
        self.sectors = sectors
        self.en_channel_string = en_channel_string
        self.delta_E = delta_E
        self.count_str = count_str
        self.mu_times = mu_times
        self.mu_data = mu_data
        self.mag_data = mag_data
        self.pol = pol
        self.phi_relative = phi_relative
        self.pol_times = pol_times
        self.bg_times = bg_times
        self.bg_I_data = bg_I_data
        self.bg_I_unc = bg_I_unc
        self.bg_mu_data = bg_mu_data
        self.sp_str = sp_str
        self.ch_string = ch_string
        self.mag_data_coord = mag_data_coord
        self.coverage = coverage

    def pickle_event(self, save_folder):
        event_id = self.event_id
        species = self.species
        en_channel = self.channels
        instrument = self.instrument
        try:
            filename = f"{event_id}_{instrument}_{species}_ch{en_channel[0]}-{en_channel[1]}"
        except:
            filename = f"{event_id}_{instrument}_{species}_ch{en_channel}"

        with open(os.path.join(save_folder, filename), 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def wind_min_intensity(self):
        # Estimate intensity from 1 count, as that can be used
        # to estimate GF*dE*dT, which is needed for converting
        # intensities to counts.
        gf = 0.33  # cm^2 sr, for Wind 3DP SST foil for 25-400 keV electrons
        dE = np.sum(self.delta_E)
        cadence = np.median(np.diff(self.I_times))/np.timedelta64(1,"s")
        I0 = 1/(0.33*dE*cadence)
        I_data = self.I_data
        I_zero = np.min((I0, np.min(I_data[I_data > 0])))
        self.I_zero = I_zero

    def wind_peak_removal(self, factor=10, I_zero=None, n_lim=2):
        I_data = self.I_data
        log_I_data = np.log10(I_data)
        log_factor = np.log10(factor)
        if I_zero is None:
            self.wind_min_intensity()
            I_zero = self.I_zero
                
        log_I_data[I_data==0] = np.log10(I_zero)
        for i in range(np.shape(I_data)[1]):
            j = 0
            while j < np.shape(I_data)[0]-1:
                dlogI = log_I_data[j+1,i]-log_I_data[j,i]
                if dlogI >= log_factor:
                    j1 = j
                    while j1 < np.shape(I_data)[0]-2:
                        j1 += 1
                        dlogI = log_I_data[j1+1,i]-log_I_data[j1,i]
                        if dlogI < -log_factor:
                            break
                    if j1-j <= n_lim:
                        I_data[j+1:j1+1,i] = np.nan
                    j = j1 + 1
                else:
                    j += 1
                    
        self.I_data = I_data

    def overview_plot(self, end_str=None, plot_onset=False, savefig=False):
        font_size = plt.rcParams["font.size"]
        legend_font = plt.rcParams["font.size"] - 2
        fig, axes = plt.subplots(3, figsize=(8,5), sharex=True,gridspec_kw={'height_ratios': [1.6,1.6,2.2]},num=1,clear=True)
        plt.subplots_adjust(hspace=0.08)
        pad_norm = None

        event_id = self.event_id
        I_times = self.I_times
        I_data = self.I_data
        coverage = self.coverage
        pol = self.pol
        pol_times = self.pol_times
        phi_relative = self.phi_relative
        species = self.species
        en_channel = self.channels
        instrument = self.instrument
        spacecraft = self.spacecraft
        en_channel_string = self.en_channel_string
        plot_folder = self.plot_folder
        startdate = self.start_time
        enddate = self.end_time
        sectors = self.sectors
        ch_string = self.ch_string
        intensity_label = self.intensity_label

        if plot_onset:
            onset_time = self.onset_time

        if len(sectors) == 4:
            color = ['crimson','orange','darkslateblue','c']
        else:
            color = [f"C{i}" for i in range(len(sectors))]
                
        axnum = 0
        ax = axes[axnum]
        ax.set_title(spacecraft,pad=20,fontsize=font_size+2)
        for i, direction in enumerate(sectors): 
            col = color[i]
            ax.fill_between(coverage.index, coverage[direction]['min'], coverage[direction]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')
            ax.plot(coverage.index, coverage[direction]['center'], linewidth=1, label=direction, color=col, drawstyle='steps-mid')
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        if len(sectors) <= 4:
            leg = ax.legend(title=instrument,bbox_to_anchor=(1.003, 0.98), loc=2, borderaxespad=0.,labelspacing=0.3,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        else:
            leg = ax.legend(title=instrument,bbox_to_anchor=(1.003, 0.98), loc=2, ncol=2, columnspacing=0.4, borderaxespad=0.,labelspacing=0.3,handlelength=1,handletextpad=0.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        #ax.set_ylabel('Pitch angle [$\\degree$]',fontsize=font_size)
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, bottom=True, top=True, left=False, right=False)
        
        pol_ax = inset_axes(ax, height="10%", width="100%", loc=9, bbox_to_anchor=(0.,0.09,1,1.11), bbox_transform=ax.transAxes)
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_arr = np.zeros(len(pol))+1
        timestamp = pol_times[2] - pol_times[1]
        norm = mpl.colors.Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(pol_times[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(pol_times[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.text(1.01,0.1,"in",color="red",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        pol_ax.text(1.04,0.1,"out",color="blue",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        
        axnum += 1
        ax = axes[axnum]
        X, Y = np.meshgrid(coverage.index.values, np.arange(180)+1 )
        hist = np.zeros(np.shape(X))
        hist_counts = np.zeros(np.shape(X))
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ylabel = intensity_label
            nanind = np.where((np.isfinite((coverage[direction]['min'].values)==True) ) | (np.isfinite(coverage[direction]['max'].values)==True) )[0]
            pa_flux = np.ma.masked_where(nanind, av_flux.reshape((len(av_flux),)))
            pa_ind = np.where((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values))[0]
            new_hist = np.where(((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values)), pa_flux, 0)
            hist = hist + new_hist
            hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)
        
        hist = hist / hist_counts
        pad_norm_str = ''
        if pad_norm is not None:
            pad_norm_str = '_pad_normed-'+pad_norm
            if pad_norm == 'mean':
                hist_mean = np.nanmean(hist, axis=0)
            if pad_norm == 'median':
                hist_mean = np.nanmedian(hist, axis=0)
            if pad_norm == 'max':
                hist_mean = np.nanmax(hist, axis=0)
            hist_t = hist/hist_mean
            hist = hist_t#.transpose()
        
        cmap = cm.inferno.copy() #cm.jet  #cm.Spectral_r #
        even_limits = False
        hmin = -1
        hmax = -1
        cmap.set_bad('w',1.)
        cmap.set_under('white')
        hist_no_0  = np.copy(hist)
        hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
        hist_no_0[hist_no_0 < 0] = np.nan
        
        if pad_norm is None:
            if hmin == -1:
                hmin = np.nanmin(hist_no_0)
            if hmax == -1:
                hmax = np.nanmax(hist[np.isfinite(hist)==True])
        
            hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
            # finding even limits for the colorscale
            if even_limits:
                colmin = 10**(np.fix(np.log10(hmin)))
                colmax = 10**(1+np.fix(np.log10(hmax)))
            else:
                colmin = hmin
                colmax = hmax
        else:
            if pad_norm == 'max':
                hmin = colmin = 0.
                hmax = colmax = 1.
            if pad_norm in ['mean', 'median']:
                hmin = colmin = np.nanmin(hist_no_0)
                hmax = colmax = np.nanmax(hist[np.isfinite(hist)==True])
        
        # plot the color-coded PAD
        if pad_norm is None:
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm(vmin=colmin, vmax=colmax))
        if pad_norm == 'mean':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap) # without lognorm better
        if pad_norm == 'median':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm())
        if pad_norm == 'max':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap)
        
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.yaxis.set_ticklabels([0,45,90,135,""])
        ax.set_ylabel('Pitch angle [$\\degree$]',fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.xmax*1.005, bbox.ymin, bbox.height*0.1, bbox.height])
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical', aspect=40)#, ticks = LogLocator(subs=range(10)))
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
        if pad_norm is None:
            cax.set_ylabel(intensity_label,fontsize=legend_font)#, labelpad=-75)
        else: 
            cax.set_ylabel(f'Flux normalized\n({pad_norm})', fontsize=legend_font)
        
        axnum += 1
        ax = axes[axnum]
        for i, direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ax.plot(I_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid')
        
        ax.text(0.02, 0.92, en_channel_string, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label,fontsize=font_size)
        if len(sectors) <= 4:
            leg = ax.legend(title=instrument+' '+ch_string,bbox_to_anchor=(1.003, 0.85), loc=2, borderaxespad=0.,labelspacing=0.3,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        else:
            leg = ax.legend(title=instrument+' '+ch_string,bbox_to_anchor=(1.003, 0.85), loc=2, ncol=2,borderaxespad=0.,labelspacing=0.3,handlelength=1,handletextpad=0.5,columnspacing=0.4,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        
        ax.set_xlabel("Universal Time (UT)",fontsize=font_size,labelpad=13)
        ax.tick_params(axis='x', which='major', pad=5, direction="in")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))#\n%b %d, %Y'))
        
        for i, ax in enumerate(axes):
            if plot_onset:
                ax.axvline(onset_time,color="red",zorder=4)
            ax.tick_params(axis='y', which ='both', direction='in', right=True)
            ax.tick_params(axis='x', which ='both', direction='in', top=True)
            hours = mdates.HourLocator(interval=1)
            ax.xaxis.set_minor_locator(hours)
            ax.set_xlim([startdate, enddate])
        pol_ax.set_xlim([startdate, enddate])
        
        ax3 = ax.secondary_xaxis('bottom')
        x = ax.get_xticks()
        ax3.set_xticks(x)
        ax3.set_xticklabels(format_tick_labels(x))
        ax3.tick_params(axis='x', which='major', pad=15, direction="in")
        
        try:
            if end_str is None:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel[0]}-{en_channel[1]}.png"
            else:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel[0]}-{en_channel[1]}_{end_str}.png"
        except:
            if end_str is None:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel}.png"
            else:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel}_{end_str}.png"
        if not savefig:
            add_watermark(fig, scaling=0.15, alpha=0.5, zorder=-1, x=0.96)
        if savefig:
            fig.savefig(fname=os.path.join(plot_folder, filename), format='png', dpi=300, bbox_inches='tight')
        return fig, axes

    def en_channel_string_to_keV(self):
        string = self.en_channel_string
        if "MeV" in string:
            split_string = string.split(" - ")
            low = float(split_string[0])*1000
            high = float(split_string[1].split(" MeV")[0])*1000
            en_channel_string = "{:.1f} - {:.1f} keV".format(low,high)
            self.en_channel_string = en_channel_string

    def background_analysis_simpleaverage(self,c_perc=50):
        bg_times = self.bg_times
        bg_I_data = self.bg_I_data
        bg_I_unc = self.bg_I_unc
        I_times = self.I_times
        I_data = self.I_data
        self.c_perc = c_perc

        #uf uncertainties not available, use sqrt(intensities)
        #the relative weights will be correct but the absolute values of fit statistics (e.g., redchi) will not
        if bg_I_unc is None:
            bg_I_unc = np.sqrt(bg_I_data)
            bg_I_unc[bg_I_unc==0] = np.nanmin(bg_I_unc[bg_I_unc>0]) #prevent zero uncertainties as weights

        average = np.nanmean(bg_I_data.flatten())
        average_err = np.sqrt(np.nansum(bg_I_unc.flatten()**2))/len(bg_I_unc.flatten())

        bg_I_fit = average*np.ones(np.shape(I_data))
        bg_I_fit_err = average_err*np.ones(np.shape(I_data))

        I_data_bgsub = I_data - bg_I_fit
        a = I_data_bgsub[(I_times > self.bg_end) & (I_times <= self.corr_window_end),:]
        b = bg_I_fit[(I_times > self.bg_end) & (I_times <= self.corr_window_end),:]
        c = 0.0
        bg_corr = False
        for i in range(np.shape(I_data)[1]):
            while True:
                perc = np.nanpercentile(a[:,i]+c*b[:,i],c_perc)
                if perc<0:
                    c += 0.01
                    bg_corr = True
                else:
                    break
                    
        #Check also separately the part before onset and bg window if onset_time exists.
        try:
            a = I_data_bgsub[(I_times > self.bg_end) & (I_times <= self.onset_time),:]
            b = bg_I_fit[(I_times > self.bg_end) & (I_times <= self.onset_time),:]
            if len(a)>=20:
                if not bg_corr:
                    c = 0.0
                for i in range(np.shape(I_data)[1]):
                    while True:
                        perc = np.nanpercentile(a[:,i]+c*b[:,i],c_perc)
                        if perc<0:
                            c += 0.01
                            bg_corr = True
                        else:
                            break
            else:
                print("Passing separate BG correction check between BG end and onset due to too few observations.")
        except:
            pass
            
        if bg_corr:
            print(" ")
        self.bg_I_fit = np.maximum(bg_I_fit*(1-c),0*bg_I_fit)
        self.bg_I_fit_err = bg_I_fit_err
        self.bg_downcorrection_factor = c

    def background_analysis_all(self,minutes=None,c_perc=50):
        bg_times = self.bg_times
        bg_I_data = self.bg_I_data
        bg_I_unc = self.bg_I_unc
        I_times = self.I_times
        I_data = self.I_data
        self.bg_av_min = minutes
        self.c_perc = c_perc

        #if uncertainties not available, use sqrt(intensities)
        #the relative weights will be correct but the absolute values of fit statistics (e.g., redchi) will not
        if bg_I_unc is None:
            bg_I_unc = np.sqrt(bg_I_data)
            bg_I_unc[bg_I_unc==0] = np.nanmin(bg_I_unc[bg_I_unc>0]) #prevent zero uncertainties as weights

        x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
        x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()

        models = run_background_analysis_all_nomag(bg_times,bg_I_data,bg_I_unc,minutes=minutes)
        const_model = models[0]
        exp_model = models[1]
        n_varys1 = const_model.summary()["nvarys"]
        I_fit_bg, I_fit_err_bg = evaluate_background_all(bg_times,bg_I_data,const_model,x_start,x_end)
        redchi1 = np.sum((bg_I_data[~np.isnan(bg_I_data)]-I_fit_bg[~np.isnan(bg_I_data)])**2/(bg_I_unc[~np.isnan(bg_I_data)])**2)/(bg_I_data[~np.isnan(bg_I_data)].size-n_varys1)
        print("Reduced chi-squared (constant model): {:.2f}".format(redchi1))
        
        n_varys2 = exp_model.summary()["nvarys"]
        I_fit_bg, I_fit_err_bg = evaluate_background_all(bg_times,bg_I_data,exp_model,x_start,x_end)
        redchi2 = np.sum((bg_I_data[~np.isnan(bg_I_data)]-I_fit_bg[~np.isnan(bg_I_data)])**2/(bg_I_unc[~np.isnan(bg_I_data)])**2)/(bg_I_data[~np.isnan(bg_I_data)].size-n_varys2)
        print("Reduced chi-squared (exponential model): {:.2f}".format(redchi2))

        if redchi2 < redchi1:
            best_model = exp_model
        else:
            best_model = const_model

        bg_I_fit, bg_I_fit_err = evaluate_background_all(I_times,I_data,best_model,x_start,x_end)

        I_data_bgsub = I_data - bg_I_fit
        a = I_data_bgsub[(I_times > self.bg_end) & (I_times <= self.corr_window_end),:]
        b = bg_I_fit[(I_times > self.bg_end) & (I_times <= self.corr_window_end),:]
        c = 0.0
        bg_corr = False
        for i in range(np.shape(I_data)[1]):
            while True:
                perc = np.nanpercentile(a[:,i]+c*b[:,i],c_perc)
                if perc<0:
                    c += 0.01
                    bg_corr = True
                else:
                    break
                    
        #Check also separately the part before onset and bg window if onset_time exists.
        try:
            a = I_data_bgsub[(I_times > self.bg_end) & (I_times <= self.onset_time),:]
            b = bg_I_fit[(I_times > self.bg_end) & (I_times <= self.onset_time),:]
            if len(a)>=20:
                if not bg_corr:
                    c = 0.0
                for i in range(np.shape(I_data)[1]):
                    while True:
                        perc = np.nanpercentile(a[:,i]+c*b[:,i],c_perc)
                        if perc<0:
                            c += 0.01
                            bg_corr = True
                        else:
                            break
            else:
                print("Passing separate BG correction check between BG end and onset due to too few observations.")
        except:
            pass
                    
        if bg_corr:
            print(" ")
            # print("Background down-correction: {:2f}".format(c))
        self.bg_I_fit = np.maximum(bg_I_fit*(1-c),0*bg_I_fit)
        self.bg_I_fit_err = bg_I_fit_err
        self.bg_downcorrection_factor = c
    
    def background_analysis(self,n_groups=50,plot_results=False,plot_bins=False,minutes=None,choose_all=True,diff_slopes=False,c_perc=50,allow_extreme_slopes=False):
        bg_times = self.bg_times
        bg_I_data = self.bg_I_data
        bg_I_unc = self.bg_I_unc
        bg_mu_data = self.bg_mu_data
        I_times = self.I_times
        I_data = self.I_data
        mu_data = self.mu_data
        self.bg_av_min = minutes
        self.c_perc = c_perc

        bg_times = bg_times[~np.any(np.isnan(bg_I_data),axis=1)]
        bg_mu_data = bg_mu_data[~np.any(np.isnan(bg_I_data),axis=1)]
        bg_I_data = bg_I_data[~np.any(np.isnan(bg_I_data),axis=1)]

        #if uncertainties not available, use sqrt(intensities)
        #the relative weights will be correct but the absolute values of fit statistics (e.g., redchi) will not
        if bg_I_unc is None:
            bg_I_unc = np.sqrt(bg_I_data)
            bg_I_unc[bg_I_unc==0] = np.nanmin(bg_I_unc[bg_I_unc>0]) #prevent zero uncertainties as weights
        else:
            bg_I_unc = bg_I_unc[~np.any(np.isnan(bg_I_data),axis=1)]

        if (self.spacecraft == "Wind") and (self.instrument == "3DP"):
            self.bg_binning = "binwise"
        elif ("STEREO" in self.spacecraft) and (self.instrument == "LET"):
            self.bg_binning = "binwise"
        else:
            self.bg_binning = "mugroups"

        print("BACKGROUND SUBTRACTION")
        print("---------------------------")
        bad_flag = False
        if self.bg_binning == "mugroups":
            raw_data = bg_I_data.flatten()
            data = raw_data[~np.isnan(raw_data)]
            arr = [bg_times for i in range(np.shape(bg_I_data)[1])]
            bg_times_arr = np.column_stack(arr)
            datetimes = bg_times_arr.flatten()[~np.isnan(raw_data)]
            times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
            mu_vals = bg_mu_data.flatten()[~np.isnan(raw_data)]
            n_groups_orig = n_groups
            for j,n_groups in enumerate([n_groups_orig,50,40,33,25,20,16,13]):
                if (j > 0) and (n_groups == n_groups_orig):
                    continue
                bad_flag = False
                mu_std = 5/n_groups
                mu0 = 1-1/n_groups
                mu_groups = np.linspace(-mu0,mu0,n_groups)                
                for i in range(n_groups):
                    mu = mu_groups[i]
                    weights = scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)
                    weights += scipy.stats.norm.pdf(-2-mu_vals, loc=mu, scale=mu_std)
                    weights += scipy.stats.norm.pdf(2-mu_vals, loc=mu, scale=mu_std)
                    max_weight = scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std) + scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std) + scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std)
                    x = times[(weights>=0.01*max_weight) & (data>=0)]
                    w = weights[(weights>=0.01*max_weight) & (data>=0)]
                    x_norm = (x-np.min(times))/(np.max(times)-np.min(times))
                    if (x.size < 20) | (np.mean(w)<0.02*max_weight):
                        bad_flag = True
                        break
                    elif (np.sum(w[x_norm<0.25])<2*max_weight) | (np.sum(w[x_norm>0.75])<2*max_weight):
                        bad_flag = True
                        break
                if bad_flag:
                    continue
                best_models, decays = run_background_analysis(n_groups,bg_times,bg_I_data,bg_I_unc,bg_mu_data,plot_bins=plot_bins,plot_results=plot_results,plot_uncertainty=False,mu_std=mu_std,minutes=minutes)
                decays = np.array(decays)
                print("1/decay rates: mean {:.3f}, median {:.3f}".format(np.nanmean(1/decays),np.nanmedian(1/decays)))
                if allow_extreme_slopes:
                    break
                else:
                    if np.abs(np.nanmedian(1/decays)-np.nanmean(1/decays))>0.50:
                        #bad_flag = True
                        pass
                    else:
                        break
            
            if bad_flag:
                print("No pitch-angle dependent background calculations.")
            else:
                print("{} equally-spaced pitch-angle cosine groups with mu_std={:.2f}".format(n_groups,mu_std), end ="")
                print("; 1/decay rates: mean {:.3f}, median {:.3f}".format(np.nanmean(1/decays),np.nanmedian(1/decays)))
                self.bg_n_groups = n_groups
                best_models_const,best_models_exp = run_background_analysis_equal_decay(np.nanmean(decays),n_groups,bg_times,bg_I_data,bg_I_unc,bg_mu_data,plot_bins=False,plot_results=plot_results,mu_std=mu_std,minutes=minutes)
            redchi_all,aic_all,bic_all, best_model_all = run_background_analysis_all(n_groups,bg_times,bg_I_data,bg_I_unc,bg_mu_data,plot_bins=False,plot_results=plot_results,mu_std=mu_std,minutes=minutes)
        
        elif self.bg_binning == "binwise":
            best_models, decays = run_background_analysis_binwise(bg_times,bg_I_data,bg_I_unc,plot_results=plot_results,plot_uncertainty=False,minutes=minutes)
            best_models_const,best_models_exp = run_background_analysis_equal_decay_binwise(np.nanmean(decays),bg_times,bg_I_data,bg_I_unc,plot_results=plot_results,minutes=minutes)
            redchi_all,aic_all,bic_all, best_model_all = run_background_analysis_all_binwise(bg_times,bg_I_data,bg_I_unc,plot_results=plot_results,minutes=minutes)

        x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
        x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
        decays = np.array(decays)
        if not bad_flag:
            print("Reduced chi-squared values:")
            desc = ["Constant model", "Exponentially decaying model (equal 1/decay rates: mean {:.3f}, median {:.3f})".format(np.nanmean(1/decays),np.nanmedian(1/decays)), "Sector-averaged background fit","Exponentially decaying models (different decay rates)"]
            models_list = [best_models_const,best_models_exp,best_model_all,best_models]

            
            if self.bg_binning == "mugroups":
                n_varys = 0
                for mod in best_models:
                    n_varys += mod.summary()["nvarys"]
                n_params_list = [n_groups,n_groups+1,best_model_all.summary()["nvarys"],n_varys] #List containing the number of model parameters
                redchi_vals = []
                for m,models in zip(n_params_list,models_list):
                    I_fit_bg, I_fit_err_bg = evaluate_background(bg_times,bg_I_data,models,mu_groups,bg_mu_data,x_start,x_end)
                    redchi_vals.append(np.sum((bg_I_data-I_fit_bg)**2/(bg_I_unc)**2)/(bg_I_data.size-m))
            
            if self.bg_binning == "binwise":
                n_varys = 0
                n_bins = np.shape(I_data)[1]
                for mod in best_models:
                    n_varys += mod.summary()["nvarys"]
                n_params_list = [n_bins,n_bins+1,best_model_all.summary()["nvarys"],n_varys] #List containing the number of model parameters
                redchi_vals = []
                for m,models in zip(n_params_list,models_list):
                    I_fit_bg, I_fit_err_bg = evaluate_background_binwise(bg_times,bg_I_data,models,x_start,x_end)
                    redchi_vals.append(np.sum((bg_I_data-I_fit_bg)**2/(bg_I_unc)**2)/(bg_I_data.size-m))
            
            if choose_all:
                models = models_list[2]
                for i, b in enumerate(redchi_vals):
                    print("{:.3f} {}".format(b,desc[i]))
            else:
                if diff_slopes:
                    model_idx = np.nanargmin(redchi_vals)
                    models = models_list[model_idx]
                    for i, b in enumerate(redchi_vals):
                        print("{:.3f} {}".format(b,desc[i]))
                    print("Selected: {}".format(desc[model_idx]))
                    self.bg_method = desc[model_idx]
                else:
                    model_idx = np.nanargmin(redchi_vals[:-1])
                    models = models_list[model_idx]
                    for i, b in enumerate(redchi_vals[:-1]):
                        print("{:.3f} {}".format(b,desc[i]))
                    print("Selected: {}".format(desc[model_idx]))
                    self.bg_method = desc[model_idx]
        else:
            print("Reduced chi-squared values (bad flag):")
            models = best_model_all
            m = best_model_all.summary()["nvarys"] #List containing the number of model parameters
            I_fit_bg, I_fit_err_bg = evaluate_background(bg_times,bg_I_data,models,mu_groups,bg_mu_data,x_start,x_end)
            b = np.sum((bg_I_data-I_fit_bg)**2/(bg_I_unc)**2)/(bg_I_data.size-m)
            print("{:.3f} {}".format(b,"Sector-averaged background fit"))

        if self.bg_binning == "mugroups":
            bg_I_fit, bg_I_fit_err = evaluate_background(I_times,I_data,models,mu_groups,mu_data,x_start,x_end)
        elif self.bg_binning == "binwise":
            bg_I_fit, bg_I_fit_err = evaluate_background_binwise(I_times,I_data,models,x_start,x_end)

        I_data_bgsub = I_data - bg_I_fit
        a = I_data_bgsub[(I_times > self.bg_end) & (I_times <= self.corr_window_end),:]
        b = bg_I_fit[(I_times > self.bg_end) & (I_times <= self.corr_window_end),:]
        
        c = 0.0
        bg_corr = False
        for i in range(np.shape(I_data)[1]):
            while True:
                perc = np.nanpercentile(a[:,i]+c*b[:,i],c_perc)
                if perc<0:
                    c += 0.01
                    bg_corr = True
                else:
                    break

        #Check also separately the part before onset and bg window if onset_time exists.
        try:
            a = I_data_bgsub[(I_times > self.bg_end) & (I_times <= self.onset_time),:]
            b = bg_I_fit[(I_times > self.bg_end) & (I_times <= self.onset_time),:]
            if len(a)>=20:
                if not bg_corr:
                    c = 0.0
                for i in range(np.shape(I_data)[1]):
                    while True:
                        perc = np.nanpercentile(a[:,i]+c*b[:,i],c_perc)
                        if perc<0:
                            c += 0.01
                            bg_corr = True
                        else:
                            break
            else:
                print("Passing separate BG correction check between BG end and onset due to too few observations.")
        except:
            pass
            
        if bg_corr:
            print(" ")
        self.bg_I_fit = np.maximum(bg_I_fit*(1-c),0*bg_I_fit)
        self.bg_I_fit_err = bg_I_fit_err
        self.bg_downcorrection_factor = c

    def overview_plot_bgsub(self,plot_onset=False, savefig=False):
        font_size = plt.rcParams["font.size"]
        legend_font = plt.rcParams["font.size"] - 2
        fig, axes = plt.subplots(4, figsize=(8,8), sharex=True,gridspec_kw={'height_ratios': [1.6,1.6,2.2,2.2]},num=2,clear=True)
        plt.subplots_adjust(hspace=0.08)
        pad_norm = None

        event_id = self.event_id
        I_times = self.I_times
        I_data = self.I_data
        coverage = self.coverage
        pol = self.pol
        pol_times = self.pol_times
        phi_relative = self.phi_relative
        bg_start = self.bg_start
        bg_end = self.bg_end
        species = self.species
        en_channel = self.channels
        instrument = self.instrument
        spacecraft = self.spacecraft
        en_channel_string = self.en_channel_string
        plot_folder = self.plot_folder
        startdate = self.start_time
        enddate = self.end_time
        sectors = self.sectors
        ch_string = self.ch_string
        intensity_label = self.intensity_label
        bg_I_fit = self.bg_I_fit
        bg_I_fit_err = self.bg_I_fit_err

        if plot_onset:
            onset_time = self.onset_time

        if len(sectors) == 4:
            color = ['crimson','orange','darkslateblue','c']
        else:
            color = [f"C{i}" for i in range(len(sectors))]
                
        axnum = 0
        ax = axes[axnum]
        ax.set_title(spacecraft,pad=20,fontsize=font_size+2)
        for i, direction in enumerate(sectors): 
            col = color[i]
            ax.fill_between(coverage.index, coverage[direction]['min'], coverage[direction]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')
            ax.plot(coverage.index, coverage[direction]['center'], linewidth=1, label=direction, color=col, drawstyle='steps-mid')
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        if len(sectors) <= 4:
            leg = ax.legend(title=instrument,bbox_to_anchor=(1.003, 0.98), loc=2, borderaxespad=0.,labelspacing=0.3,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        else:
            leg = ax.legend(title=instrument,bbox_to_anchor=(1.003, 0.98), loc=2, ncol=2, columnspacing=0.4, borderaxespad=0.,labelspacing=0.3,handlelength=1,handletextpad=0.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        #ax.set_ylabel('Pitch angle [$\\degree$]',fontsize=font_size)
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, bottom=True, top=True, left=False, right=False)
        
        pol_ax = inset_axes(ax, height="10%", width="100%", loc=9, bbox_to_anchor=(0.,0.09,1,1.11), bbox_transform=ax.transAxes)
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_arr = np.zeros(len(pol))+1
        timestamp = pol_times[2] - pol_times[1]
        norm = mpl.colors.Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(pol_times[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(pol_times[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.text(1.01,0.1,"in",color="red",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        pol_ax.text(1.04,0.1,"out",color="blue",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        
        axnum += 1
        ax = axes[axnum]
        X, Y = np.meshgrid(coverage.index.values, np.arange(180)+1 )
        hist = np.zeros(np.shape(X))
        hist_counts = np.zeros(np.shape(X))
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ylabel = intensity_label
            nanind = np.where((np.isfinite((coverage[direction]['min'].values)==True) ) | (np.isfinite(coverage[direction]['max'].values)==True) )[0]
            pa_flux = np.ma.masked_where(nanind, av_flux.reshape((len(av_flux),)))
            pa_ind = np.where((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values))[0]
            new_hist = np.where(((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values)), pa_flux, 0)
            hist = hist + new_hist
            hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)
        
        hist = hist / hist_counts
        pad_norm_str = ''
        if pad_norm is not None:
            pad_norm_str = '_pad_normed-'+pad_norm
            if pad_norm == 'mean':
                hist_mean = np.nanmean(hist, axis=0)
            if pad_norm == 'median':
                hist_mean = np.nanmedian(hist, axis=0)
            if pad_norm == 'max':
                hist_mean = np.nanmax(hist, axis=0)
            hist_t = hist/hist_mean
            hist = hist_t#.transpose()
        
        cmap = cm.inferno.copy() #cm.jet  #cm.Spectral_r #
        even_limits = False
        hmin = -1
        hmax = -1
        cmap.set_bad('w',1.)
        cmap.set_under('white')
        hist_no_0  = np.copy(hist)
        hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
        hist_no_0[hist_no_0 < 0] = np.nan
        
        if pad_norm is None:
            if hmin == -1:
                hmin = np.nanmin(hist_no_0)
            if hmax == -1:
                hmax = np.nanmax(hist[np.isfinite(hist)==True])
        
            hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
            # finding even limits for the colorscale
            if even_limits:
                colmin = 10**(np.fix(np.log10(hmin)))
                colmax = 10**(1+np.fix(np.log10(hmax)))
            else:
                colmin = hmin
                colmax = hmax
        else:
            if pad_norm == 'max':
                hmin = colmin = 0.
                hmax = colmax = 1.
            if pad_norm in ['mean', 'median']:
                hmin = colmin = np.nanmin(hist_no_0)
                hmax = colmax = np.nanmax(hist[np.isfinite(hist)==True])
        
        # plot the color-coded PAD
        if pad_norm is None:
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm(vmin=colmin, vmax=colmax))
        if pad_norm == 'mean':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap) # without lognorm better
        if pad_norm == 'median':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm())
        if pad_norm == 'max':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap)
        
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.yaxis.set_ticklabels([0,45,90,135,""])
        ax.set_ylabel('Pitch angle [$\\degree$]',fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.xmax*1.005, bbox.ymin, bbox.height*0.1, bbox.height])
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical', aspect=40)  #, ticks = LogLocator(subs=range(10)))
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
        if pad_norm is None:
            cax.set_ylabel(intensity_label, fontsize=legend_font)#, labelpad=-75)
        else:
            cax.set_ylabel(f'Flux normalized\n({pad_norm})', fontsize=legend_font)
        
        axnum += 1
        ax = axes[axnum]
        for i, direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ax.plot(I_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid',alpha=0.7)

        ax.set_yscale('log')
        ylims = ax.get_ylim()

        for i, direction in enumerate(sectors):
            ax.plot(I_times,bg_I_fit[:,i],color=color[i],ls="--",lw=0.8, drawstyle='steps-mid',zorder=3)

        ax.set_ylim(ylims)
        
        ax.text(0.02, 0.92, en_channel_string, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size)
        if len(sectors) <= 4:
            leg = ax.legend(title=instrument+' '+ch_string,bbox_to_anchor=(1.003, 0.4), loc=2, borderaxespad=0.,labelspacing=0.3,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        else:
            leg = ax.legend(title=instrument+' '+ch_string,bbox_to_anchor=(1.003, 0.4), loc=2, ncol=2,borderaxespad=0.,labelspacing=0.3,handlelength=1,handletextpad=0.5,columnspacing=0.4,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"

        axnum += 1
        ax = axes[axnum]
        for i, direction in enumerate(sectors):
            av_flux = I_data[:,i]-bg_I_fit[:,i]
            ax.plot(I_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid')
        
        ax.text(0.02, 0.92, en_channel_string+"\nbackground subtracted", horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label,fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        ax.set_ylim(axes[axnum-1].get_ylim())
        
        ax.set_xlabel("Universal Time (UT)",fontsize=font_size,labelpad=13)
        ax.tick_params(axis='x', which='major', pad=5, direction="in")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))#\n%b %d, %Y'))
        
        for i, ax in enumerate(axes):
            if i>1:
                if (bg_start is not None) and (bg_end is not None):
                    ax.axvspan(bg_start, bg_end, color='gray', alpha=0.15,zorder=1)
            if plot_onset:
                ax.axvline(onset_time,color="red",zorder=4)
            ax.tick_params(axis='y', which ='both', direction='in', right=True)
            ax.tick_params(axis='x', which ='both', direction='in', top=True)
            #ax.xaxis.set_minor_locator(AutoMinorLocator(3))
            hours = mdates.HourLocator(interval=1)
            ax.xaxis.set_minor_locator(hours)
            ax.set_xlim([startdate, enddate])
        pol_ax.set_xlim([startdate, enddate])
        
        ax3 = ax.secondary_xaxis('bottom')
        x = ax.get_xticks()
        ax3.set_xticks(x)
        ax3.set_xticklabels(format_tick_labels(x))
        ax3.tick_params(axis='x', which='major', pad=15, direction="in")

        if not savefig:
            add_watermark(fig, scaling=0.15, alpha=0.5, zorder=-1, x=0.95)

        try:
            filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel[0]}-{en_channel[1]}_bgsub.png"
        except:
            filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel}_bgsub.png"
        if savefig: 
            fig.savefig(fname=os.path.join(plot_folder,filename),format='png',dpi=300, bbox_inches = 'tight')
        return fig, axes
    
    def calculate_anisotropy(self, ani_method='weighted_sum_bootstrap'):
        """_summary_

        Parameters
        ----------
        ani_method : str, optional
            based on this different methods are used to calculate the first-order anisotropies, 
            can be 'fit', 'weighted_sum', 'weighted_sum_bootstrap' 
            by default 'weighted_sum'
        """
        if (ani_method == 'weighted_sum_bootstrap') and (self.spacecraft == 'Wind'):
            ani_method = 'weighted_sum'
            custom_warning('No bootstrapping uncertainties available for Wind 3DP due to missing count data! Applying ani_method="weighted_sum" instead')
        # if self.spacecraft == "Wind":
        #     print("Error treatment for Wind only based on background subtraction, not on counting statistics (count data not available).")
        if ani_method == 'weighted_sum':
            self.anisotropy_weighted_sum()
            self.anisotropy_weighted_sum_bgsub()
        if ani_method == 'fit':
            self.anisotropy_fit_bgsub()
        if ani_method == 'weighted_sum_bootstrap': # or just 'bootstrap'.. then chose between fit and weighted sum. maybe make this then a separate keyword
            self.anisotropy_weighted_sum()
            self.anisotropy_weighted_sum_bootstrap()

    def anisotropy_weighted_sum(self):
        if not hasattr(self, 'mu_weights'):
            weights, max_ani, min_ani = anisotropy_prepare(self.coverage,self.I_data)
            self.mu_weights = weights
            self.max_ani_limit = max_ani
            self.min_ani_limit = min_ani
            
        if (self.spacecraft == "Wind") and (self.instrument == "3DP"):
            I_data = self.I_data.copy()
            I_dummy = I_data.copy()
            mu_data = self.mu_data.copy()
            mu_weights = self.mu_weights.copy()
            
            I_data[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),0] = 0
            I_data[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),-1] = 0
            mu_data[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),0] = 0
            mu_data[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),-1] = 0
            mu_weights[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),0] = 0
            mu_weights[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),-1] = 0
            self.ani_weighted_sum = anisotropy_weighted_sum(I_data,mu_data,mu_weights)
        else:
            self.ani_weighted_sum = anisotropy_weighted_sum(self.I_data,self.mu_data,self.mu_weights)

    def anisotropy_weighted_sum_bgsub(self):
        if not hasattr(self, 'mu_weights'):
            weights, max_ani, min_ani = anisotropy_prepare(self.coverage,self.I_data)
            self.mu_weights = weights
            self.max_ani_limit = max_ani
            self.min_ani_limit = min_ani
        I_bgsub = self.I_data-self.bg_I_fit
        I_bgsub[I_bgsub<0] = 0
        
        if (self.spacecraft == "Wind") and (self.instrument == "3DP"):
            I_dummy = I_bgsub.copy()
            mu_data = self.mu_data.copy()
            mu_weights = self.mu_weights.copy()
            
            I_bgsub[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),0] = 0
            I_bgsub[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),-1] = 0
            mu_data[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),0] = 0
            mu_data[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),-1] = 0
            mu_weights[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),0] = 0
            mu_weights[np.isnan(I_dummy[:,0]) | np.isnan(I_dummy[:,-1]),-1] = 0
            self.ani_weighted_sum_bgsub = anisotropy_weighted_sum(I_bgsub,mu_data,mu_weights)
        else:
            self.ani_weighted_sum_bgsub = anisotropy_weighted_sum(I_bgsub,self.mu_data,self.mu_weights)

    def anisotropy_fit(self):
        if not hasattr(self, 'mu_weights'):
            weights, max_ani, min_ani = anisotropy_prepare(self.coverage,self.I_data)
            self.mu_weights = weights
            self.max_ani_limit = max_ani
            self.min_ani_limit = min_ani
        ani_fit = np.nan*np.zeros(len(self.I_times))
        if self.I_unc is None:
            #Do not weighting if no uncertainties available.
            unc = np.ones(np.shape(self.I_data))
        else:
            unc = self.I_unc
        for i in range(len(ani_fit)):
            x = self.mu_data[i,:]
            y = self.I_data[i,:]
            y_err = unc[i,:]
            if len(y[~np.isnan(y)])>=4:
                model, ani_fit[i] = anisotropy_legendre_fit(y[~np.isnan(y)],x[~np.isnan(y)],y_err[~np.isnan(y)])
        self.ani_fit = ani_fit

    def anisotropy_fit_bgsub(self):
        if not hasattr(self, 'mu_weights'):
            weights, max_ani, min_ani = anisotropy_prepare(self.coverage,self.I_data)
            self.mu_weights = weights
            self.max_ani_limit = max_ani
            self.min_ani_limit = min_ani
        ani_fit = np.nan*np.zeros(len(self.I_times))
        ani_fit_bgsub = np.nan*np.zeros(len(self.I_times))
        I_bgsub = self.I_data-self.bg_I_fit
        if self.I_unc is None:
            #Do not weighting if no uncertainties available.
            unc = np.ones(np.shape(self.I_data))
            #Use background model uncertainties for fit weighting when using BG-subtracted data.
            unc_bgsub = self.bg_I_fit_err
        else:
            unc = self.I_unc
            unc_bgsub = np.sqrt(self.I_unc**2+self.bg_I_fit_err**2)
        for i in range(len(ani_fit)):
            x = self.mu_data[i,:]
            y = self.I_data[i,:]
            y_err = unc[i,:]
            y_bgsub = I_bgsub[i,:]
            y_err_bgsub = unc_bgsub[i,:]
            if len(y[~np.isnan(y)])>=4:
                model, ani_fit[i] = anisotropy_legendre_fit(y[~np.isnan(y)],x[~np.isnan(y)],y_err[~np.isnan(y)])
                if self.I_times[i]> self.bg_end:
                    model, ani_fit_bgsub[i] = anisotropy_legendre_fit(y_bgsub[~np.isnan(y)],x[~np.isnan(y)],y_err_bgsub[~np.isnan(y)])
        self.ani_fit = ani_fit
        self.ani_fit_bgsub = ani_fit_bgsub

    def anisotropy_weighted_sum_bootstrap(self,n_boot=1000):
        if not hasattr(self, 'mu_weights'):
            weights, max_ani, min_ani = anisotropy_prepare(self.coverage,self.I_data)
            self.mu_weights = weights
            self.max_ani_limit = max_ani
            self.min_ani_limit = min_ani
        I_data = self.I_data
        delta_E = self.delta_E
        bg_I_fit = self.bg_I_fit
        bg_I_fit_err = self.bg_I_fit_err
        mu_data = self.mu_data
        weights = self.mu_weights
        count_arr = self.count_arr
        t_arr = self.t_arr
        gf_arr = self.gf_arr
        Ani_bootres, Ani_bgsub_bootres = bootstrap_anisotropy(I_data,bg_I_fit,bg_I_fit_err,mu_data,weights,delta_E,count_arr,t_arr,gf_arr,n_boot=n_boot)
        self.ani_bootres = Ani_bootres
        self.ani_bgsub_bootres = Ani_bgsub_bootres
        self.n_boot = n_boot

    def anisotropy_bootstrap_plot(self,plot_peaks=False,plot_periods=False,plot_onset=False):
        font_size = plt.rcParams["font.size"]
        legend_font = plt.rcParams["font.size"] - 2

        event_id = self.event_id
        I_times = self.I_times
        I_data = self.I_data
        coverage = self.coverage
        pol = self.pol
        pol_times = self.pol_times
        phi_relative = self.phi_relative
        bg_start = self.bg_start
        bg_end = self.bg_end
        corr_window_end = self.corr_window_end
        species = self.species
        en_channel = self.channels
        instrument = self.instrument
        spacecraft = self.spacecraft
        en_channel_string = self.en_channel_string
        plot_folder = self.plot_folder
        startdate = self.start_time
        enddate = self.end_time
        sectors = self.sectors
        ch_string = self.ch_string
        intensity_label = self.intensity_label
        bg_I_fit = self.bg_I_fit
        bg_I_fit_err = self.bg_I_fit_err
        max_ani = self.max_ani_limit
        min_ani = self.min_ani_limit
        Ani_bootres = self.ani_bootres
        Ani_bgsub_bootres = self.ani_bgsub_bootres
        n_boot = self.n_boot
        if plot_onset:
            onset_time = self.onset_time
        
        if len(sectors) == 4:
            color = ['crimson','orange','darkslateblue','c']
        else:
            color = [f"C{i}" for i in range(len(sectors))]
            
        fig, axes = plt.subplots(5, figsize=(7,9.5), sharex=True,gridspec_kw={'height_ratios': [1.6,1.6,2.2,2.2,2.2]},num=3,clear=True)
        plt.subplots_adjust(hspace=0.08)
        pad_norm = None
        
        flux_times = I_times
                
        axnum = 0
        ax = axes[axnum]
        ax.set_title(spacecraft,pad=20,fontsize=font_size+2)
        for i,direction in enumerate(sectors):
            col = color[i]
            ax.fill_between(coverage.index, coverage[direction]['min'], coverage[direction]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')
            ax.plot(coverage.index, coverage[direction]['center'], linewidth=1, label=direction, color=col, drawstyle='steps-mid')
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        leg = ax.legend(title=instrument,bbox_to_anchor=(1.003, 1), loc=2, borderaxespad=0.,labelspacing=0.25,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, bottom=True, top=True, left=False, right=False)
        
        pol_ax = inset_axes(ax, height="10%", width="100%", loc=9, bbox_to_anchor=(0.,0.10,1,1.11), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_arr = np.zeros(len(pol))+1
        timestamp = pol_times[2] - pol_times[1]
        norm = mpl.colors.Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(pol_times[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(pol_times[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.text(1.01,0.1,"in",color="red",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        pol_ax.text(1.04,0.1,"out",color="blue",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        
        axnum += 1
        ax = axes[axnum]
        X, Y = np.meshgrid(coverage.index.values, np.arange(180)+1 )
        hist = np.zeros(np.shape(X))
        hist_counts = np.zeros(np.shape(X))
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ylabel = intensity_label
            nanind = np.where((np.isfinite((coverage[direction]['min'].values)==True) ) | (np.isfinite(coverage[direction]['max'].values)==True) )[0]
            pa_flux = np.ma.masked_where(nanind, av_flux.reshape((len(av_flux),)))
            pa_ind = np.where((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values))[0]
            new_hist = np.where(((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values)), pa_flux, 0)
            hist = hist + new_hist
            hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)
        
        hist = hist / hist_counts
        pad_norm_str = ''
        if pad_norm is not None:
            pad_norm_str = '_pad_normed-'+pad_norm
            if pad_norm == 'mean':
                hist_mean = np.nanmean(hist, axis=0)
            if pad_norm == 'median':
                hist_mean = np.nanmedian(hist, axis=0)
            if pad_norm == 'max':
                hist_mean = np.nanmax(hist, axis=0)
            hist_t = hist/hist_mean
            hist = hist_t#.transpose()
        
        cmap = cm.inferno.copy() #cm.jet  #cm.Spectral_r #
        even_limits = False
        hmin = -1
        hmax = -1
        cmap.set_bad('w',1.)
        cmap.set_under('white')
        hist_no_0  = np.copy(hist)
        hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
        hist_no_0[hist_no_0 < 0] = np.nan
        
        if pad_norm is None:
            if hmin == -1:
                hmin = np.nanmin(hist_no_0)
            if hmax == -1:
                hmax = np.nanmax(hist[np.isfinite(hist)==True])
        
            hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
            # finding even limits for the colorscale
            if even_limits:
                colmin = 10**(np.fix(np.log10(hmin)))
                colmax = 10**(1+np.fix(np.log10(hmax)))
            else:
                colmin = hmin
                colmax = hmax
        else:
            if pad_norm == 'max':
                hmin = colmin = 0.
                hmax = colmax = 1.
            if pad_norm in ['mean', 'median']:
                hmin = colmin = np.nanmin(hist_no_0)
                hmax = colmax = np.nanmax(hist[np.isfinite(hist)==True])
        
        # plot the color-coded PAD
        if pad_norm is None:
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm(vmin=colmin, vmax=colmax))
        if pad_norm == 'mean':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap) # without lognorm better
        if pad_norm == 'median':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm())
        if pad_norm == 'max':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap)
        
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.yaxis.set_ticklabels([0,45,90,135,""])
        ax.set_ylabel('Pitch angle [$\\degree$]',fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.xmax*1.005, bbox.ymin, bbox.height*0.15, bbox.height])
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical', aspect=40)#, ticks = LogLocator(subs=range(10)))
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
        if pad_norm is None:
            cax.set_ylabel(intensity_label, fontsize=legend_font)#, labelpad=-75)
        else:
            cax.set_ylabel(f'Flux normalized\n({pad_norm})', fontsize=legend_font)
        
        axnum += 1
        ax = axes[axnum]
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ax.plot(flux_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid',alpha=0.7)

        ax.set_yscale('log')
        ylims = ax.get_ylim()
        
        for i,direction in enumerate(sectors):
            ax.plot(I_times,bg_I_fit[:,i],color=color[i],zorder=3,ls="--",lw=0.8, drawstyle='steps-mid')

        ax.set_ylim(ylims)
        
        ax.text(0.02, 0.92, en_channel_string, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size-1)
        leg = ax.legend(title=instrument+' '+ch_string,bbox_to_anchor=(1.003, 0.4), loc=2, borderaxespad=0.,labelspacing=0.25,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        
        axnum += 1
        ax = axes[axnum]
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]-bg_I_fit[:,i]
            ax.plot(flux_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid')
        ax.text(0.02, 0.92, en_channel_string+"\nbackground subtracted", horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size-1)
        ax.set_ylabel(intensity_label,fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        ax.set_yscale('log')
        ax.set_ylim(axes[axnum-1].get_ylim())
        
        axnum += 1
        ax = axes[axnum]
        ax.fill_between(coverage.index,-3,max_ani,color="black",alpha=0.3,edgecolor=None)
        ax.fill_between(coverage.index,min_ani,3,color="black",alpha=0.3,edgecolor=None)
        ax.plot(I_times,Ani_bootres[:,0],label="w/o background substraction",color="black")
        ax.fill_between(I_times,Ani_bootres[:,2],Ani_bootres[:,3],alpha=0.2,zorder=1,edgecolor=None,facecolor="black")
        ani_bgsub = Ani_bgsub_bootres.copy()
        ax.plot(I_times[(I_times>=bg_end) & (I_times <= corr_window_end)],ani_bgsub[(I_times>=bg_end) & (I_times <= corr_window_end)],label="with background substraction",color="magenta")

        ax.fill_between(I_times[I_times>=bg_end],ani_bgsub[I_times>=bg_end,2],ani_bgsub[I_times>=bg_end,3],alpha=0.3,zorder=1,edgecolor=None,facecolor="magenta")
        ax.text(0.02, 0.92, "background subtracted", color="magenta",horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size-1)
        ax.set_ylabel("$A_1$")
        ax.axhline(0,ls=":",color="gray",zorder=1)
        ax.set_ylim(-3,3)

        if plot_periods:
            peak_df = self.peak_df
            ax.scatter(peak_df["peak_time"],peak_df["peak_ani"],zorder=4,color="blue",s=15)
            for st,et in zip(peak_df["start_time"],peak_df["end_time"]):
                ax.axvspan(st,et,alpha=0.2,color="blue")

        if plot_peaks:
            ax.scatter(I_times[self.ani_s2n_max_idx],Ani_bootres[self.ani_s2n_max_idx,0],zorder=4,color="black",s=10,edgecolors="white")
            ax.scatter(I_times[self.ani_bgsub_s2n_max_idx],ani_bgsub[self.ani_bgsub_s2n_max_idx,0],zorder=4,color="magenta",s=10,edgecolors="black")
        
        ax.set_xlabel("Universal Time (UT)",fontsize=font_size,labelpad=15)
        ax.tick_params(axis='x', which='major', pad=5, direction="in")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))#\n%b %d, %Y'))
        
        for i,ax in enumerate(axes):
            if i>1:
                ax.axvspan(bg_start, bg_end, color='gray', alpha=0.15,zorder=1)
            if plot_onset:
                ax.axvline(onset_time,color="red",zorder=4)
            ax.tick_params(axis='y', which ='both', direction='in', right=True)
            ax.tick_params(axis='x', which ='both', direction='in', top=True)
            hours = mdates.HourLocator(interval=1)
            ax.xaxis.set_minor_locator(hours)
            ax.set_xlim([startdate, enddate])
        pol_ax.set_xlim(ax.get_xlim())
        
        ax3 = ax.secondary_xaxis('bottom')
        x = ax.get_xticks()
        ax3.set_xticks(x)
        ax3.set_xticklabels(format_tick_labels(x))
        ax3.tick_params(axis='x', which='major', pad=15, direction="in")

        try:
            if plot_peaks:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel[0]}-{en_channel[1]}_ani_bgsub_bootstrap_peaks.png"
            else:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel[0]}-{en_channel[1]}_ani_bgsub_bootstrap.png"
        except:
            if plot_peaks:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel}_ani_bgsub_bootstrap_peaks.png"
            else:
                filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel}_ani_bgsub_bootstrap.png"
        
        fig.savefig(fname=os.path.join(plot_folder,filename),format='png',dpi=300, bbox_inches = 'tight')
        return fig, axes

    def anisotropy_plot(self, ani_method='weighted_sum_bootstrap', savefig=False):
        if (ani_method == 'weighted_sum_bootstrap') and (self.spacecraft == 'Wind'):
            ani_method = 'weighted_sum'

        font_size = plt.rcParams["font.size"]
        legend_font = plt.rcParams["font.size"] - 2

        event_id = self.event_id
        I_times = self.I_times
        I_data = self.I_data
        coverage = self.coverage
        pol = self.pol
        pol_times = self.pol_times
        phi_relative = self.phi_relative
        bg_start = self.bg_start
        bg_end = self.bg_end
        corr_window_end = self.corr_window_end
        species = self.species
        en_channel = self.channels
        instrument = self.instrument
        spacecraft = self.spacecraft
        en_channel_string = self.en_channel_string
        plot_folder = self.plot_folder
        startdate = self.start_time
        enddate = self.end_time
        sectors = self.sectors
        ch_string = self.ch_string
        intensity_label = self.intensity_label
        bg_I_fit = self.bg_I_fit
        bg_I_fit_err = self.bg_I_fit_err
        max_ani = self.max_ani_limit
        min_ani = self.min_ani_limit
        if ani_method == 'weighted_sum_bootstrap':
            Ani = self.ani_bootres
            Ani_bgsub = self.ani_bgsub_bootres       
            ani_method_str = '_weighted_sum_bootstrap'     
        if ani_method == 'weighted_sum':
            Ani = self.ani_weighted_sum
            Ani_bgsub = self.ani_weighted_sum_bgsub
            ani_method_str = '_weighted_sum'
        if ani_method == 'fit':
            Ani = self.ani_fit
            Ani_bgsub = self.ani_fit_bgsub
            ani_method_str = '_fit'

        if len(sectors) == 4:
            color = ['crimson','orange','darkslateblue','c']
        else:
            color = [f"C{i}" for i in range(len(sectors))]
            
        fig, axes = plt.subplots(5, figsize=(7,9.5), sharex=True,gridspec_kw={'height_ratios': [1.6,1.6,2.2,2.2,2.2]},num=3,clear=True)
        plt.subplots_adjust(hspace=0.08)
        pad_norm = None
        
        flux_times = I_times
                
        axnum = 0
        ax = axes[axnum]
        ax.set_title(spacecraft,pad=20,fontsize=font_size+2)
        for i,direction in enumerate(sectors):
            col = color[i]
            ax.fill_between(coverage.index, coverage[direction]['min'], coverage[direction]['max'], alpha=0.5, color=col, edgecolor=col, linewidth=0.0, step='mid')
            ax.plot(coverage.index, coverage[direction]['center'], linewidth=1, label=direction, color=col, drawstyle='steps-mid')
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        leg = ax.legend(title=instrument,bbox_to_anchor=(1.003, 1), loc=2, borderaxespad=0.,labelspacing=0.25,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.tick_params(axis="x",direction="in", which='both', pad=-15)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, bottom=True, top=True, left=False, right=False)
        
        pol_ax = inset_axes(ax, height="10%", width="100%", loc=9, bbox_to_anchor=(0.,0.10,1,1.11), bbox_transform=ax.transAxes) # center, you can check the different codes in plt.legend?
        pol_ax.get_xaxis().set_visible(False)
        pol_ax.get_yaxis().set_visible(False)
        pol_ax.set_ylim(0,1)
        pol_arr = np.zeros(len(pol))+1
        timestamp = pol_times[2] - pol_times[1]
        norm = mpl.colors.Normalize(vmin=0, vmax=180, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)
        pol_ax.bar(pol_times[(phi_relative>=0) & (phi_relative<180)],pol_arr[(phi_relative>=0) & (phi_relative<180)],color=mapper.to_rgba(phi_relative[(phi_relative>=0) & (phi_relative<180)]),width=timestamp)
        pol_ax.bar(pol_times[(phi_relative>=180) & (phi_relative<360)],pol_arr[(phi_relative>=180) & (phi_relative<360)],color=mapper.to_rgba(np.abs(360-phi_relative[(phi_relative>=180) & (phi_relative<360)])),width=timestamp)
        pol_ax.text(1.01,0.1,"in",color="red",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        pol_ax.text(1.04,0.1,"out",color="blue",transform=pol_ax.transAxes,fontsize=legend_font-0.5)
        
        axnum += 1
        ax = axes[axnum]
        X, Y = np.meshgrid(coverage.index.values, np.arange(180)+1 )
        hist = np.zeros(np.shape(X))
        hist_counts = np.zeros(np.shape(X))
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ylabel = intensity_label
            nanind = np.where((np.isfinite((coverage[direction]['min'].values)==True) ) | (np.isfinite(coverage[direction]['max'].values)==True) )[0]
            pa_flux = np.ma.masked_where(nanind, av_flux.reshape((len(av_flux),)))
            pa_ind = np.where((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values))[0]
            new_hist = np.where(((Y > coverage[direction]['min'].values) & (Y < coverage[direction]['max'].values)), pa_flux, 0)
            hist = hist + new_hist
            hist_counts = hist_counts + np.where(new_hist > 0, 1, 0)
        
        hist = hist / hist_counts
        pad_norm_str = ''
        if pad_norm is not None:
            pad_norm_str = '_pad_normed-'+pad_norm
            if pad_norm == 'mean':
                hist_mean = np.nanmean(hist, axis=0)
            if pad_norm == 'median':
                hist_mean = np.nanmedian(hist, axis=0)
            if pad_norm == 'max':
                hist_mean = np.nanmax(hist, axis=0)
            hist_t = hist/hist_mean
            hist = hist_t#.transpose()
        
        cmap = cm.inferno.copy() #cm.jet  #cm.Spectral_r #
        even_limits = False
        hmin = -1
        hmax = -1
        cmap.set_bad('w',1.)
        cmap.set_under('white')
        hist_no_0  = np.copy(hist)
        hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
        hist_no_0[hist_no_0 < 0] = np.nan
        
        if pad_norm is None:
            if hmin == -1:
                hmin = np.nanmin(hist_no_0)
            if hmax == -1:
                hmax = np.nanmax(hist[np.isfinite(hist)==True])
        
            hist_no_0[np.where(hist_no_0 == 0)[0]] = np.nan
            # finding even limits for the colorscale
            if even_limits:
                colmin = 10**(np.fix(np.log10(hmin)))
                colmax = 10**(1+np.fix(np.log10(hmax)))
            else:
                colmin = hmin
                colmax = hmax
        else:
            if pad_norm == 'max':
                hmin = colmin = 0.
                hmax = colmax = 1.
            if pad_norm in ['mean', 'median']:
                hmin = colmin = np.nanmin(hist_no_0)
                hmax = colmax = np.nanmax(hist[np.isfinite(hist)==True])
        
        # plot the color-coded PAD
        if pad_norm is None:
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm(vmin=colmin, vmax=colmax))
        if pad_norm == 'mean':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap) # without lognorm better
        if pad_norm == 'median':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap, norm=LogNorm())
        if pad_norm == 'max':
            pcm = ax.pcolormesh(X, Y, hist, cmap=cmap)
        
        ax.axhline(y=90, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=45, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(y=135, color='gray', linewidth=0.8, linestyle='--')
        ax.set_ylim([0, 180])
        ax.yaxis.set_ticks(np.arange(0, 180+45, 45))
        ax.yaxis.set_ticklabels([0,45,90,135,""])
        ax.set_ylabel('Pitch angle [$\\degree$]',fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.xmax*1.005, bbox.ymin, bbox.height*0.15, bbox.height])
        cbar = fig.colorbar(pcm, cax=cax, orientation='vertical', aspect=40)
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
        if pad_norm is None:
            cax.set_ylabel(intensity_label, fontsize=legend_font)
        else: 
            cax.set_ylabel(f'Flux normalized\n({pad_norm})', fontsize=legend_font)
        
        axnum += 1
        ax = axes[axnum]
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]
            ax.plot(flux_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid')

        ax.set_yscale('log')
        ylims = ax.get_ylim()
        
        for i,direction in enumerate(sectors):
            ax.plot(I_times,bg_I_fit[:,i],color=color[i],zorder=1,ls="--",lw=0.8, drawstyle='steps-mid')

        ax.set_ylim(ylims)
        
        ax.text(0.02, 0.92, en_channel_string, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size-1)
        leg = ax.legend(title=instrument+' '+ch_string,bbox_to_anchor=(1.003, 0.4), loc=2, borderaxespad=0.,labelspacing=0.25,handlelength=1.2,handletextpad=0.5,columnspacing=1.5,frameon=False,fontsize=legend_font,title_fontsize=legend_font)
        leg._legend_box.align = "left"
        
        axnum += 1
        ax = axes[axnum]
        for i,direction in enumerate(sectors):
            av_flux = I_data[:,i]-bg_I_fit[:,i]
            ax.plot(flux_times, av_flux, linewidth=1.2, label=direction, color=color[i], drawstyle='steps-mid')
        ax.text(0.02, 0.92, en_channel_string+"\nbackground subtracted", horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size-1)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label,fontsize=font_size)
        ax.yaxis.set_label_coords(-0.08,1.0)
        ax.set_ylim(axes[axnum-1].get_ylim())
        
        axnum += 1
        ax = axes[axnum]
        ax.fill_between(coverage.index,-3,max_ani,color="black",alpha=0.3,edgecolor=None)
        ax.fill_between(coverage.index,min_ani,3,color="black",alpha=0.3,edgecolor=None)
        if ani_method == 'weighted_sum_bootstrap':
            ax.plot(I_times, Ani[:,0], label="w/o background substraction", color="black", linewidth=1)
            ax.fill_between(I_times, Ani[:,2], Ani[:,3], alpha=0.2, zorder=1, edgecolor=None, facecolor="black")
            ind = (I_times >= bg_end) & (I_times <= corr_window_end)
            ax.fill_between(I_times[ind], Ani_bgsub[ind,2], Ani_bgsub[ind,3], alpha=0.3, zorder=1, edgecolor=None, facecolor="magenta")
            ax.plot(I_times[(I_times >= bg_end) & (I_times <= corr_window_end)], Ani_bgsub[(I_times >= bg_end) & (I_times <= corr_window_end), 0], label="with background substraction", color="magenta", linewidth=1)
        else:
            ax.plot(I_times, Ani, label="w/o background substraction", color="black", linewidth=1)
            ax.plot(I_times[(I_times >= bg_end) & (I_times <= corr_window_end)], Ani_bgsub[(I_times >= bg_end) & (I_times <= corr_window_end)], label="with background substraction", color="magenta", linewidth=1)

        ax.text(0.02, 0.92, "background subtracted", color="magenta",horizontalalignment='left', verticalalignment='top', transform = ax.transAxes,fontsize=font_size-1)
        ax.set_ylabel("$A_1$")
        ax.axhline(0,ls=":", color="gray", zorder=1)
        ax.set_ylim(-3, 3)

        ax.set_xlabel("Universal Time (UT)", fontsize=font_size, labelpad=15)
        ax.tick_params(axis='x', which='major', pad=5, direction="in")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))#\n%b %d, %Y'))

        for i,ax in enumerate(axes):
            if i>1:
                ax.axvspan(bg_start, bg_end, color='gray', alpha=0.15,zorder=1)
            ax.tick_params(axis='y', which ='both', direction='in', right=True)
            ax.tick_params(axis='x', which ='both', direction='in', top=True)
            hours = mdates.HourLocator(interval=1)
            ax.xaxis.set_minor_locator(hours)
            ax.set_xlim([startdate, enddate])
        pol_ax.set_xlim(ax.get_xlim())

        ax3 = ax.secondary_xaxis('bottom')
        x = ax.get_xticks()
        ax3.set_xticks(x)
        ax3.set_xticklabels(format_tick_labels(x))
        ax3.tick_params(axis='x', which='major', pad=15, direction="in")

        if not savefig:
            add_watermark(fig, scaling=0.15, alpha=0.5, zorder=-1, x=0.97)

        try:
            filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel[0]}-{en_channel[1]}_ani_bgsub{ani_method_str}.png"
        except:
            filename = f"{event_id}_{instrument}_{startdate.year}-{startdate.month}-{startdate.day}_{enddate.month}-{enddate.day}_{species}_ch{en_channel}_ani_bgsub{ani_method_str}.png"
        if savefig:
            fig.savefig(fname=os.path.join(plot_folder,filename),format='png',dpi=300, bbox_inches = 'tight')
        return fig, axes
