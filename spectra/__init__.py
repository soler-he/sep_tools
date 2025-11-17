# import datetime as dt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import sunpy
import warnings

from matplotlib import pyplot as plt
from solo_epd_loader import epd_load
from seppy.loader.soho import soho_load
from seppy.loader.psp import psp_isois_load
from seppy.loader.stereo import stereo_load
from seppy.loader.wind import wind3dp_load
from seppy.util import resample_df


# omit some warnings
warnings.simplefilter(action='once', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='No units provided for variable', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='astropy did not recognize units of', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='The variable "HET_', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message="Note that for the Dataframes containing the flow direction and SC coordinates timestamp position will not be adjusted by 'pos_timestamp'!", module='solo_epd_loader')
warnings.filterwarnings(action='once', message="Mean of empty slice", category=RuntimeWarning)


class Event:

    def __init__(self):
        pass

    def load_data(self, spacecraft, instrument, species, startdate, enddate, viewing='', resample=None, data_path=None):
        self.spacecraft = spacecraft
        self.instrument = instrument
        self.species = species
        self.startdate = startdate
        self.enddate = enddate
        self.viewing = viewing

        if self.spacecraft.lower() in ['wind']:
            self.viewing = 'omni'
            self.instrument = '3DP SST'
        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.instrument.lower() == 'het':
                self.viewing = ''
        if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
            self.instrument = 'EPI-Hi HET'
        if self.spacecraft.lower() == 'solar orbiter':
            self.spacecraft = 'solo'

        if not data_path:
            data_path = os.getcwd()+os.sep+'data'+os.sep

        if self.spacecraft.lower() == 'solo':
            if self.viewing == '':
                raise Exception("Solar Orbiter instruments require a defined 'viewing'!")
            df_protons, df_electrons, self.meta = epd_load(sensor=self.instrument, level='l2', startdate=self.startdate,
                                                           enddate=self.enddate, viewing=self.viewing, path=data_path,
                                                           autodownload=True)
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                self.df = df_electrons
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                self.df = df_protons
            if resample is not None:
                self.df = resample_df(self.df, resample)

        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.spacecraft.lower() in ['stereo a', 'stereo-a']:
                sc = 'ahead'
            if self.spacecraft.lower() in ['stereo b', 'stereo-b']:
                sc = 'behind'
            self.df, self.meta = stereo_load(instrument=self.instrument, startdate=self.startdate, enddate=self.enddate, spacecraft=sc,
                                             sept_species=self.species[0], sept_viewing=self.viewing, path=data_path, resample=resample)

        if self.spacecraft.lower() in ['wind']:  # !! only omni data implemented for now
            # !!! remove lowest energy channels (noisy)?
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                dataset = 'WI_SFSP_3DP'
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                dataset = 'WI_SOSP_3DP'
            self.df, self.meta = wind3dp_load(dataset=dataset, startdate=self.startdate, enddate=self.enddate, path=data_path, resample=resample)

        if self.spacecraft.lower() in ['soho']:
            self.viewing = ''
            self.erne_chstring = ['13-16 MeV', '16-20 MeV', '20-25 MeV', '25-32 MeV', '32-40 MeV', '40-50 MeV', '50-64 MeV', '64-80 MeV', '80-100 MeV', '100-130 MeV']
            self.df, self.meta = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=self.startdate, enddate=self.enddate, path=data_path, resample=resample, max_conn=1)


        if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                raise Exception("Parker Solar Probe does not provide intensity data for electrons as of now!")
            else:
                if self.viewing == '':
                    raise Exception("Parker Solar Probe HET requires a defined 'viewing'!")
                self.df, self.meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate=self.startdate, enddate=self.enddate,
                                                    path=data_path, resample=resample)
        # return df, meta

    def plot_flux(self, spec_start, spec_end, subtract_background=True, background_start=None, background_end=None, savefig=False, spec_type='integral'):
               
        fig, axs = plt.subplots(1, sharex=True, figsize=(9, 6), dpi=200)

        if self.spacecraft.lower() == 'solo':
            if self.instrument.lower() == 'ept':
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    flux_id = 'Electron_Flux'
                    energy_text = 'Electron_Bins_Text'
                    show_channels = [1, 4, 8, 12, 16, 20, 24, 28, 32]
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    flux_id = 'Ion_Flux'
                    energy_text = 'Ion_Bins_Text'
                    show_channels = [1, 4, 8, 12, 16, 20, 24, 28, 32]

            if self.instrument.lower() == 'het':
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    show_channels = [0, 1, 2, 3]
                    energy_text = "Electron_Bins_Text"
                    flux_id = 'Electron_Flux'
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    show_channels = [0, 4, 8, 12, 16, 20, 24, 28, 32]
                    energy_text = "H_Bins_Text"
                    flux_id = "H_Flux"

            
            # plotting
            for channel in show_channels:
                label = self.meta[energy_text][channel]
                axs.plot(self.df.index, self.df[flux_id][f'{flux_id}_{channel}'], label=label)
                
                if spec_type == 'peak':
                    ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not self.df[flux_id][f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = self.df[flux_id][f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = self.df[flux_id][f'{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.instrument.lower() == 'het':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    cols = self.df.filter(like='Proton_Flux').columns
                    flux_id = 'Proton_Flux'
                    energy_text = "Proton_Bins_Text"
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    cols = self.df.filter(like='Electron_Flux').columns
                    flux_id = 'Electron_Flux'
                    energy_text = "Electron_Bins_Text"

                show_channels = np.arange(len(cols))

            if self.instrument.lower() == 'sept':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    show_channels = np.arange(2, 31)
                energy_text = "Proton_Bins_Text"
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    show_channels = np.arange(2, 16)
                flux_id = 'ch'
                energy_text = 'ch_strings'

            # plotting
            for channel in show_channels:
                label = self.meta[energy_text][channel]
                axs.plot(self.df.index, self.df[f'{flux_id}_{channel}'], label=label)
                
                if spec_type == 'peak':
                    ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not self.df[f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = self.df[f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = self.df[f'{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if self.spacecraft.lower() in ['wind']:
            cols = self.df.filter(like='FLUX').columns
            flux_id = 'FLUX'
            show_channels = np.arange(len(cols))

            # plotting
            for channel in show_channels:
                label = self.meta['channels_dict_df']['Bins_Text'][f'ENERGY_{channel}']
                axs.plot(self.df.index, self.df[f'{flux_id}_{channel}']*1e6, label=label)

                if spec_type == 'peak':
                    ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not self.df[f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = self.df[f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = self.df[f'{flux_id}_{channel}'].iloc[ind].max()*1e6
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')
                    
        if self.spacecraft.lower() in ['soho']:
            flux_id = 'PH'
            show_channels = np.arange(len(self.erne_chstring))

            # plotting
            for channel in show_channels:
                label = self.erne_chstring[channel]
                axs.plot(self.df.index, self.df[f'{flux_id}_{channel}'], label=label)
                
                if spec_type == 'peak':
                    ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not self.df[f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = self.df[f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = self.df[f'{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')
                
       
        if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
            # if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:  # !!! no fluxes available for electrons
            #     print('!!! no intensity data available for PSP electrons!')
            #     energy_text = 'Electrons_ENERGY_LABL'
            #     flux_id = 'Electrons_Rate'
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                energy_text = 'H_ENERGY_LABL'
                flux_id = "H_Flux"
            show_channels = np.arange(len(self.meta[energy_text]))

            # plotting
            for channel in show_channels:
                label = self.meta[energy_text][channel]
                axs.plot(self.df.index, self.df[f'{self.viewing}_{flux_id}_{channel}'], label=label)
                
                if spec_type == 'peak':
                    ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not self.df[f'{self.viewing}_{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = self.df[f'{self.viewing}_{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = self.df[f'{self.viewing}_{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if subtract_background:
            axs.axvspan(background_start, background_end, color='pink', alpha=0.2, label='Background Period')
        axs.axvline(spec_start, color='red', label='Integration Start')
        axs.axvline(spec_end, color='green', label='Integration End')

        axs.set_yscale('log')
        axs.set_ylabel(r"Intensity [1/(cm$^2$ sr s MeV)]")

        axs.set_xlim(self.startdate, self.enddate)
        axs.xaxis.set_major_locator(mdates.AutoDateLocator())
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        axs.set_xlabel('Date / Time')

        axs.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', borderaxespad=0.0,
                   title=f'{self.species} {self.viewing}', fontsize=9, ncol=3, frameon=True,
                   facecolor='white', edgecolor='black', title_fontsize='11')

        fig.suptitle(f'{self.spacecraft.upper()} / {self.instrument.upper()} {self.viewing} (spectral type = {spec_type})', fontsize=14)

        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        # fig.autofmt_xdate()

        if savefig:
            filename = f'Time_series_{self.spacecraft.upper()}_{self.instrument.upper()}_{self.viewing}_{self.species}.png'
            plt.savefig(filename)
            print(f"Figure saved as {filename}")
        plt.show()
        return fig, axs

    def get_spec(self, spec_start, spec_end, spec_type='integral', subtract_background=True, background_start=None, background_end=None):
        I_spec = []
        unc_spec = []
        ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]
        if subtract_background:
            ind_bg = np.where((self.df.index >= background_start) & (self.df.index <= background_end))[0]

        if self.spacecraft.lower() in ['solo', 'solar orbiter']:
            if self.instrument.lower() == 'het':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    species_key = 'H'
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    species_key = 'Electron'

            if self.instrument.lower() == 'ept':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    species_key = 'Ion'
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    species_key = 'Electron'

            # cols = self.df.filter(like=species_key).columns
            flux_id = f'{species_key}_Flux_'
            unc_id = f'{species_key}_Uncertainty_'
            # fluxes = self.df[cols]

            low_E = np.array(self.meta[f'{species_key}_Bins_Low_Energy'])
            high_E = low_E + np.array(self.meta[f'{species_key}_Bins_Width'])
            self.spec_E = np.sqrt(low_E*high_E)

        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.instrument.lower() == 'het':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    # cols = self.df.filter(like='Proton').columns
                    flux_id = 'Proton_Flux_'
                    unc_id = 'Proton_Sigma_'
                    # fluxes = self.df[cols]
                    self.spec_E = np.array(self.meta['channels_dict_df_p'].mean_E)
                    # # num_channels = len(self.df.filter(like='Proton_Flux').columns)

                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    # cols = self.df.filter(like='Electron').columns
                    flux_id = 'Electron_Flux_'
                    unc_id = 'Electron_Sigma_'
                    # fluxes = self.df[cols]
                    self.spec_E = np.array(self.meta['channels_dict_df_e'].mean_E)
                    # num_channels = len(self.df.filter(like='Electron_Flux').columns)
            if self.instrument.lower() == 'sept':
                unc_id = 'err_ch_'
                self.spec_E = self.meta['mean_E'].values

        if self.spacecraft.lower() in ['wind']:
            # cols = self.df.filter(like='FLUX').columns
            flux_id = 'FLUX'
            # fluxes = self.df[cols]
            self.spec_E = self.meta['channels_dict_df']['mean_E'].values*1e-6

        if self.spacecraft.lower() in ['soho']:
            # cols = self.df.filter(like='PH').columns
            flux_id = 'PH_'
            # fluxes = self.df[cols]
            self.spec_E = self.meta['channels_dict_df_p']['mean_E']

        if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:  # !!! no fluxes available for electrons
                print('!!! no intensity data available for PSP electrons!')
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                flux_id = f"{self.viewing}_H_Flux"
                unc_id = f"{self.viewing}_H_Uncertainty"
                self.spec_E = self.meta['H_ENERGY']

        if self.instrument.lower() == 'sept':
            df_fluxes = self.df[self.df.columns.drop([i for i in self.df.columns if 'err' in i])]
        else:
            df_fluxes = self.df[self.df.filter(like=flux_id).columns]

        if spec_type == 'integral':
            I_spec = np.nansum(df_fluxes.iloc[ind], axis=0)
    
            if self.spacecraft.lower() == 'wind':
                I_spec = np.nansum(df_fluxes.iloc[ind], axis=0)*1e6
                unc_spec = np.zeros(len(I_spec))*np.nan
            elif self.spacecraft.lower() == 'soho':
                I_spec = np.nansum(df_fluxes.iloc[ind], axis=0)
                unc_spec = np.zeros(len(I_spec))*np.nan
            else:
                df_uncs = self.df[self.df.filter(like=unc_id).columns]
                # For PSP, remove asymmetric uncertainties like A_H_Uncertainty_Minus_0 & A_H_Uncertainty_Plus_0 for now
                if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
                    for col in ['Minus', 'Plus']:
                        try:
                            df_uncs = df_uncs[df_uncs.columns.drop([i for i in df_uncs.columns if col in i])]
                        except KeyError:
                            pass
                unc_spec = np.nansum(df_uncs.iloc[ind], axis=0)

        
        if spec_type == 'peak':
            I_spec = np.nanmax(df_fluxes.iloc[ind], axis=0)
            
            if self.spacecraft.lower() == 'wind':
                I_spec = np.nanmax(df_fluxes.iloc[ind], axis=0)*1e6
                unc_spec = np.zeros(len(I_spec))*np.nan
            elif self.spacecraft.lower() == 'soho':
                I_spec = np.nanmax(df_fluxes.iloc[ind], axis=0)
                unc_spec = np.zeros(len(I_spec))*np.nan
            else:
                df_uncs = self.df[self.df.filter(like=unc_id).columns]
                # For PSP, remove asymmetric uncertainties like A_H_Uncertainty_Minus_0 & A_H_Uncertainty_Plus_0 for now
                if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
                    for col in ['Minus', 'Plus']:
                        try:
                            df_uncs = df_uncs[df_uncs.columns.drop([i for i in df_uncs.columns if col in i])]
                        except KeyError:
                            pass
                unc_spec = np.nanmax(df_uncs.iloc[ind], axis=0)

        
        if subtract_background:
            print('subtracting background')
            bg_spec = np.nanmean(df_fluxes.iloc[ind_bg], axis=0)
            self.final_spec = I_spec - bg_spec
            if self.spacecraft.lower() in ['wind', 'soho']:
                self.final_unc = np.zeros(len(I_spec))*np.nan
            else:
                bg_unc_spec = np.nanmean(df_uncs.iloc[ind_bg], axis=0)
                self.final_unc = np.sqrt(bg_unc_spec**2 + unc_spec**2)
        else:
            self.final_spec = I_spec
            self.final_unc = unc_spec
        self.subtract_background = subtract_background
        self.spec_type = spec_type
    
    def plot_spectrum(self, savefig=None):
        fig, ax = plt.subplots(1, sharex=True, figsize=(5, 4), dpi=150)
        ax.errorbar(self.spec_E, self.final_spec, yerr=self.final_unc, fmt='o', markersize=8,
                    label=self.species, elinewidth=2, capsize=5, capthick=2)
        
        if self.spec_type == 'integral':
            spec_type_str = 'integral spectrum'
            ylabel_str = "Intensity (cm² sr MeV)⁻¹"
        if self.spec_type == 'peak':
            spec_type_str = 'peak spectrum'
            ylabel_str = "Intensity (cm² s sr MeV)⁻¹"
        if self.subtract_background:
            backsub_str = ', background subtracted'
        else:
            backsub_str = ''

        
        ax.set_title(f"{self.spacecraft.upper()} / {self.instrument.upper()} {self.viewing} ({spec_type_str}{backsub_str})")
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel(ylabel_str)
        ax.legend()
        fig.tight_layout()

        if savefig:
            filename = f'Spectrum_{self.spec_type}_{self.spacecraft.upper()}_{self.instrument.upper()}_{self.viewing}_{self.species}.png'
            plt.savefig(filename)
            print(f"Figure saved as {filename}")
        # fig.show()
        return fig, ax
