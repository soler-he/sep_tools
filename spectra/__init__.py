import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import sunpy
import warnings

from PIL import Image
from solo_epd_loader import epd_load
from seppy.loader.soho import soho_load
from seppy.loader.psp import psp_isois_load
from seppy.loader.stereo import stereo_load
from seppy.loader.wind import wind3dp_load
from seppy.util import resample_df, custom_warning, custom_notification, propagated_mean_uncertainty

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

    def load_data(self, spacecraft, instrument, species, startdate, enddate, viewing='', data_level='l2', data_path=None):
        self.spacecraft = spacecraft
        self.instrument = instrument
        self.species = species
        self.startdate = startdate
        self.enddate = enddate
        self.viewing = viewing
        self.data_level = data_level

        if self.spacecraft.lower() in ['wind']:
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
            df_protons, df_electrons, self.meta = epd_load(sensor=self.instrument, level=self.data_level, startdate=self.startdate,
                                                           enddate=self.enddate, viewing=self.viewing, path=data_path,
                                                           autodownload=True)
            # flatten multi-index columns
            df_electrons.columns = df_electrons.columns.droplevel(0)
            df_protons.columns = df_protons.columns.droplevel(0)
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                self.df = df_electrons
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                self.df = df_protons

        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.spacecraft.lower() in ['stereo a', 'stereo-a']:
                sc = 'ahead'
            if self.spacecraft.lower() in ['stereo b', 'stereo-b']:
                sc = 'behind'
            self.df, self.meta = stereo_load(instrument=self.instrument, startdate=self.startdate, enddate=self.enddate, spacecraft=sc,
                                             sept_species=self.species[0], sept_viewing=self.viewing, path=data_path)

        if self.spacecraft.lower() in ['wind']:
            # TODO: !!! remove lowest energy channels (noisy)?
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                if self.viewing == "omnidirectional":
                    dataset = 'WI_SFSP_3DP'
                elif self.viewing.lower().startswith('sector'):
                    dataset = 'WI_SFPD_3DP'
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                if self.viewing == "omnidirectional":
                    dataset = 'WI_SOSP_3DP'
                elif self.viewing.lower().startswith('sector'):
                    dataset = 'WI_SOPD_3DP'
            self.df, self.meta = wind3dp_load(dataset=dataset, startdate=self.startdate, enddate=self.enddate, path=data_path, multi_index=False)
            custom_notification('No intensity uncertainties available for Wind/3DP.')

        if self.spacecraft.lower() in ['soho']:
            if self.instrument.upper() == 'ERNE-HED':
                self.viewing = ''
                self.erne_chstring = ['13-16 MeV', '16-20 MeV', '20-25 MeV', '25-32 MeV', '32-40 MeV', '40-50 MeV', '50-64 MeV', '64-80 MeV', '80-100 MeV', '100-130 MeV']
                self.df, self.meta = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=self.startdate, enddate=self.enddate, path=data_path, max_conn=1)
                soho_fluxes = self.df[self.df.filter(like='PH_').columns]
                soho_counts = self.df[self.df.filter(like='PHC_').columns]
                for i in range(0, soho_fluxes.shape[1]):
                    soho_unc = np.divide(soho_fluxes[f'PH_{i}'], np.sqrt(np.where(soho_counts[f'PHC_{i}'] > 0, soho_counts[f'PHC_{i}'], np.nan)),  # avoid sqrt of NaN/0
                                         out=np.zeros_like(soho_fluxes[f'PH_{i}']),
                                         where=~((soho_counts[f'PHC_{i}'] == 0) & (soho_fluxes[f'PH_{i}'] == 0)))
                    soho_unc[np.isnan(soho_fluxes[f'PH_{i}'])] = np.nan
                    self.df[f'uncertainty_{i}'] = soho_unc

                custom_warning('No intensity uncertainties available for SOHO/ERNE. Calculating uncertainties as I/sqrt(counts).')

        if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                raise Exception("Parker Solar Probe does not provide intensity data for electrons as of now!")
            else:
                if self.viewing == '':
                    raise Exception("Parker Solar Probe HET requires a defined 'viewing'!")
                self.df, self.meta = psp_isois_load(dataset='PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate=self.startdate, enddate=self.enddate,
                                                    path=data_path)
        # return df, meta

    def plot_flux(self, spec_start, spec_end, subtract_background=True, background_start=None, background_end=None, savefig=False, spec_type='integral', resample=None):

        fig, axs = plt.subplots(1, sharex=True, figsize=(9, 6), dpi=200)
        if resample is not None:
            df_resampled = resample_df(self.df, resample, cols_unc='auto', verbose=False)
        else:
            df_resampled = self.df.copy()
        if self.spacecraft.lower() == 'solo':
            if self.instrument.lower() == 'ept':
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    flux_id = 'Electron_Flux'
                    energy_text = 'Electron_Bins_Text'
                    show_channels = np.arange(1, len(self.meta[energy_text]), 5)
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    flux_id = 'Ion_Flux'
                    energy_text = 'Ion_Bins_Text'
                    show_channels = np.arange(1, len(self.meta[energy_text]), 6)

            if self.instrument.lower() == 'het':
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    show_channels = [0, 1, 2, 3]
                    energy_text = "Electron_Bins_Text"
                    flux_id = 'Electron_Flux'
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    show_channels = [0, 4, 8, 12, 16, 20, 24, 28, 32]
                    energy_text = "H_Bins_Text"
                    flux_id = "H_Flux"
                    show_channels = np.arange(0, len(self.meta[energy_text]), 4)

            # plotting
            for channel in show_channels:
                label = self.meta[energy_text][channel]
                axs.plot(df_resampled.index, df_resampled[f'{flux_id}_{channel}'], label=label)

                if spec_type == 'peak':
                    ind = np.where((df_resampled.index >= spec_start) & (df_resampled.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not df_resampled[f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = df_resampled[f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = df_resampled[f'{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.instrument.lower() == 'het':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    cols = df_resampled.filter(like='Proton_Flux').columns
                    flux_id = 'Proton_Flux'
                    energy_text = "Proton_Bins_Text"
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    cols = df_resampled.filter(like='Electron_Flux').columns
                    flux_id = 'Electron_Flux'
                    energy_text = "Electron_Bins_Text"

                show_channels = np.arange(len(cols))
                energy_labels = self.meta[energy_text]

            if self.instrument.lower() == 'sept':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    show_channels = np.arange(2, 31)
                    energy_labels = self.meta['channels_dict_df_p']['ch_strings']
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    show_channels = np.arange(2, 16)
                    energy_labels = self.meta['channels_dict_df_e']['ch_strings']
                flux_id = 'ch'

            # plotting
            for channel in show_channels:
                label = energy_labels[channel]
                axs.plot(df_resampled.index, df_resampled[f'{flux_id}_{channel}'], label=label)

                if spec_type == 'peak':
                    ind = np.where((df_resampled.index >= spec_start) & (df_resampled.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not df_resampled[f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = df_resampled[f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = df_resampled[f'{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if self.spacecraft.lower() in ['wind']:
            if self.viewing == "omnidirectional":
                cols = df_resampled.filter(like='FLUX').columns
                flux_id = 'FLUX_'
                view_id = ''
            elif self.viewing.lower().startswith('sector'):
                cols = df_resampled.filter(like=f'_P{self.viewing[-1]}').columns  # ty: ignore[index-out-of-bounds]
                flux_id = 'FLUX_E'
                view_id = f'_P{self.viewing[-1]}'  # ty: ignore[index-out-of-bounds]
            show_channels = np.arange(len(cols))

            # plotting
            for channel in show_channels:
                # print(f'Plotting Wind/3DP {flux_id}{channel}{view_id}')
                label = self.meta['channels_dict_df']['Bins_Text'][f'ENERGY_{channel}']
                axs.plot(df_resampled.index, df_resampled[f'{flux_id}{channel}{view_id}']*1e6, label=label)

                if spec_type == 'peak':
                    ind = np.where((df_resampled.index >= spec_start) & (df_resampled.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not df_resampled[f'{flux_id}{channel}{view_id}'].iloc[ind].isnull().all():
                        peak_time = df_resampled[f'{flux_id}{channel}{view_id}'].iloc[ind].idxmax(skipna=True)
                        peak_val = df_resampled[f'{flux_id}{channel}{view_id}'].iloc[ind].max()*1e6
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if self.spacecraft.lower() in ['soho']:
            flux_id = 'PH'
            show_channels = np.arange(len(self.erne_chstring))

            # plotting
            for channel in show_channels:
                label = self.erne_chstring[channel]
                axs.plot(df_resampled.index, df_resampled[f'{flux_id}_{channel}'], label=label)

                if spec_type == 'peak':
                    ind = np.where((df_resampled.index >= spec_start) & (df_resampled.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not df_resampled[f'{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = df_resampled[f'{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = df_resampled[f'{flux_id}_{channel}'].iloc[ind].max()
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
                axs.plot(df_resampled.index, df_resampled[f'{self.viewing}_{flux_id}_{channel}'], label=label)

                if spec_type == 'peak':
                    ind = np.where((df_resampled.index >= spec_start) & (df_resampled.index <= spec_end))[0]
                    # only plot peak if there is at least one non-nan value in the interval
                    if not df_resampled[f'{self.viewing}_{flux_id}_{channel}'].iloc[ind].isnull().all():
                        peak_time = df_resampled[f'{self.viewing}_{flux_id}_{channel}'].iloc[ind].idxmax(skipna=True)
                        peak_val = df_resampled[f'{self.viewing}_{flux_id}_{channel}'].iloc[ind].max()
                        axs.plot(peak_time, peak_val, 'ko', markerfacecolor='none')

        if subtract_background:
            if background_start is None or background_end is None:
                raise ValueError("background_start and background_end must be defined when subtract_background=True")
            axs.axvspan(background_start, background_end, color='pink', alpha=0.2, label='Background Period')
        axs.axvline(spec_start, color='red', linestyle='--', label='Integration Start')
        axs.axvline(spec_end, color='green', linestyle='--', label='Integration End')

        axs.set_yscale('log')
        axs.set_ylabel(r"Intensity [1/(cm$^2$ sr s MeV)]")

        axs.set_xlim(self.startdate, self.enddate)
        axs.xaxis.set_major_locator(mdates.AutoDateLocator())
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        axs.set_xlabel('Date / Time')

        axs.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', borderaxespad=0.0,
                   title=f'{self.species} {self.viewing}', fontsize=9, ncol=4, frameon=True,
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

    def make_spec_gif(self, base_filename):
        # Get all PNG files (assuming they're named plot_0.png, plot_1.png, etc.)
        png_files = sorted(glob.glob(f'{base_filename}*.png'))
        # Define the output GIF filename
        self.gif_filename = f'{base_filename}_animation.gif'
        # write to animated gif; duration (in ms) defines how fast the animation is.
        frames = [Image.open(f) for f in png_files]
        frames[0].save(
            self.gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0
        )

    def plot_spec_slices(self, base_filename, spec_start, duration):
        # makes a plot of each spectrum slice based on the already saved csv files
        # taking all csv files, we determine a global y-range used in all plots

        csv_files = sorted(glob.glob(f'{base_filename}*.csv'))
        global_min = None
        global_max = None
        data_frames = []

        for file in csv_files:
            df = pd.read_csv(file)
            intensity = df['Intensity'].astype(float)
            if self.spacecraft == 'Wind':
                # i_err = np.zeros(len(intensity))
                i_err = intensity*0.1
            else:
                i_err = df['I_err'].astype(float)
            lower = intensity - i_err
            upper = intensity + i_err

            # Only consider non-negative lower bounds
            y_lower_nonneg = lower[lower > 0].dropna()
            if not y_lower_nonneg.empty:
                file_min = y_lower_nonneg.min()
                if global_min is None:
                    global_min = file_min
                else:
                    global_min = min(global_min, file_min)

            # For max, drop NAs just in case
            y_upper_nonneg = upper.dropna()
            if not y_upper_nonneg.empty:
                file_max = y_upper_nonneg.max()
                if global_max is None:
                    global_max = file_max
                else:
                    global_max = max(global_max, file_max)

            data_frames.append(df)

        # Optional: Check found range
        # print(global_min, global_max)
        print(f'Global y-range: {global_min:.2f} to {global_max:.2f}')
        # Plot each file with shared y-limits
        for idx, (df, file) in enumerate(zip(data_frames, csv_files)):
            t1 = spec_start + idx * duration
            t2 = spec_start + (idx+1) * duration

            fig, ax = plt.subplots(1, sharex=True, figsize=(5, 4), dpi=150)
            ax.errorbar(df['Energy'], df['Intensity'], yerr=df['I_err'], xerr=df['E_err'], fmt='o', markersize=8,
                        label=self.species, elinewidth=2, capsize=5, capthick=2, ecolor='lightgray')

            spec_type_str = 'integral spectrum'

            if self.subtract_background:
                backsub_str = ', backgr. subtr.'
            else:
                backsub_str = ''

            viewing = self.viewing
            if viewing == 'omnidirectional':
                viewing = 'omni'
            ax.set_title(f"{self.spacecraft.upper()} / {self.instrument.upper()} {viewing} ({spec_type_str}{backsub_str})")
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.set_ylim(global_min, global_max)

            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Intensity (cm² sr MeV)⁻¹")

            ax.text(0.95, 0.95, f'{t1}-{t2}', ha='right', transform=ax.transAxes)
            ax.legend(loc=3)

            filename = f'{base_filename}_{idx}.png'
            plt.savefig(filename)
            plt.close('all')

    def get_spec_slices(self, spec_start, spec_end, duration, subtract_background=True, background_start=None, background_end=None):
        # Determines spectra for each time slice
        # then makes plots for all spectra using a common y-range
        # then makes an animated gif out of all spectra plots

        num_steps = int((spec_end-spec_start) / duration)
        for i in np.arange(0, num_steps, 1):
            t1 = spec_start + i * duration
            t2 = spec_start + (i+1) * duration
            self.get_spec(t1, t2, spec_type='integral', subtract_background=subtract_background,
                          background_start=background_start, background_end=background_end)

            foldername = f'output_spectra{os.sep}'
            start_time = str(spec_start).replace(" ", "_")
            duration_min = (duration.total_seconds()/60)
            filename = f'{foldername}spectrum_slices_start_{start_time}_step_{duration_min}min_{self.spacecraft.upper()}_{self.instrument.upper()}_{self.viewing}_{self.species}_{i}'

            self.E_unc = self.DE/2.
            self.I_unc = self.final_unc

            spec_df = pd.DataFrame(dict(Energy=self.spec_E, Intensity=self.final_spec, E_err=self.E_unc, I_err=self.I_unc), index=range(len(self.spec_E)))
            spec_df.to_csv(filename+'.csv', index=False)

        # make  plots for each spec slice using common y-range:
        base_filename = filename = f'{foldername}spectrum_slices_start_{start_time}_step_{duration_min}min_{self.spacecraft.upper()}_{self.instrument.upper()}_{self.viewing}_{self.species}_'
        print(base_filename)
        self.plot_spec_slices(base_filename, spec_start, duration)

        self.make_spec_gif(base_filename)

    def get_spec(self,
                 spec_start: dt.datetime | pd.Timestamp | str,
                 spec_end: dt.datetime | pd.Timestamp | str,
                 spec_type: str = 'integral',
                 subtract_background: bool = True,
                 background_start: dt.datetime | pd.Timestamp | str | None = None,
                 background_end: dt.datetime | pd.Timestamp | str | None = None,
                 resample: str | None = None) -> None:
        """
        Compute an energy spectrum over a selected time interval and store the result on the instance.
        Depending on ``spec_type``, this method either integrates flux values over the interval
        (``"integral"``) or determines the peak spectrum from the maximum flux in the interval
        (``"peak"``). Instrument- and spacecraft-specific column names, energy bins, and channel
        widths are inferred from ``self.df`` and ``self.meta``.
        Optionally, a background interval can be used for background subtraction. For integral
        spectra, background subtraction is applied to the flux time series before integration.
        For peak spectra, the background is estimated from the original time series and subtracted
        from the peak spectrum afterward.

        Parameters
        ----------
        spec_start : dt.datetime or pd.Timestamp or str
            Start time of the spectrum interval.
        spec_end : dt.datetime or pd.Timestamp or str
            End time of the spectrum interval.
        spec_type : {'integral', 'peak'}, default 'integral'
            Type of spectrum to compute:
            - ``'integral'``: sum flux values over the selected interval and multiply by the
              characteristic cadence to obtain an integral spectrum.
            - ``'peak'``: take the maximum flux in each energy channel over the selected interval.
        subtract_background : bool, default True
            Whether to subtract a background spectrum estimated from ``background_start`` to
            ``background_end``.
        background_start : dt.datetime or pd.Timestamp or str, optional
            Start time of the background interval. Required when
            ``subtract_background=True``.
        background_end : dt.datetime or pd.Timestamp or str, optional
            End time of the background interval. Required when
            ``subtract_background=True``.
        resample : str or pandas offset alias, optional
            Resampling rule used only for ``spec_type='peak'``. If given, the time series is
            resampled before the peak spectrum is determined.

        Returns
        -------
        None
            Results are stored as instance attributes.

        Attributes Set
        --------------
        spec_E : numpy.ndarray
            Mean or representative energy of each channel.
        DE : numpy.ndarray
            Energy-bin widths.
        final_spec : numpy.ndarray
            Computed spectrum after optional background subtraction.
        final_unc : numpy.ndarray
            Uncertainty estimate associated with ``final_spec``.
        I_unc : numpy.ndarray
            Alias of ``final_unc``.
        E_unc : numpy.ndarray
            Half-width energy uncertainties, computed as ``DE / 2``.
        spec_df : pandas.DataFrame
            Table containing energy, intensity, and associated uncertainties with columns
            ``['Energy', 'Intensity', 'E_err', 'I_err']``.
        subtract_background : bool
            Copy of the input argument.
        spec_type : str
            Copy of the input argument.

        Notes
        -----
        - The method supports multiple spacecraft/instrument combinations and uses different
          metadata keys and column naming conventions accordingly.
        - For Wind, uncertainties are currently set to NaN.

          .. todo::
              Implement correct uncertainties for Wind/3DP.

        - For PSP, asymmetric uncertainty columns containing ``'Minus'`` and ``'Plus'`` are
          excluded where applicable.

          .. todo::
              Revisit the exclusion of asymmetric PSP uncertainty columns (``'Minus'`` and
              ``'Plus'``).

        - If the selected interval contains only data gaps, the resulting spectrum and
          uncertainties are filled with NaN.
        - Negative fluxes after background subtraction are kept intentionally: they reflect
          statistical noise around zero and zeroing/masking them would introduce a positive
          bias in the integral flux.
        - For integral spectra without background subtraction, the uncertainty is calculated as:

          .. math::

              \\delta F = \\Delta t \\cdot \\sqrt{\\sum_i \\delta J_i^2}

        - For background-subtracted integral spectra, the same background value is subtracted
          from every event bin, meaning the background uncertainty is correlated across all
          event bins. The uncertainty is therefore calculated as:

          .. math::

              \\delta F = \\Delta t \\cdot \\sqrt{\\sum_i \\delta J_i^2 + N_{event}^2 \\cdot \\delta J_{bg}^2}

          where :math:`\\delta J_{bg} = \\frac{1}{N_{bg}} \\sqrt{\\sum_k \\delta J_k^2}` is the
          uncertainty of the mean background per channel, and :math:`N_{event}` is the number
          of non-NaN event bins per channel.
        - For peak spectra, the uncertainty is taken from the same time step as the peak flux,
          not the time step with the maximum uncertainty.
        - For background-subtracted peak spectra, the peak uncertainty and background uncertainty
          are independent and are therefore combined in quadrature:

          .. math::

              \\delta I = \\sqrt{\\delta J_{peak}^2 + \\delta J_{bg}^2}
        """

        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"Expected self.df to be a pandas DataFrame, but got {type(self.df).__name__}. "
                            f"Please ensure that the data is loaded correctly before calling get_spec()."
                            )

        I_spec = []
        unc_spec = []
        ind = np.where((self.df.index >= spec_start) & (self.df.index <= spec_end))[0]

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

            flux_id = f'{species_key}_Flux_'
            unc_id = f'{species_key}_Uncertainty_'

            low_E = np.array(self.meta[f'{species_key}_Bins_Low_Energy'])
            high_E = low_E + np.array(self.meta[f'{species_key}_Bins_Width'])
            self.spec_E = np.sqrt(low_E*high_E)
            self.DE = self.meta[f'{species_key}_Bins_Width']

        if self.spacecraft.lower() in ['stereo a', 'stereo-a', 'stereo b', 'stereo-b']:
            if self.instrument.lower() == 'het':
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    # cols = self.df.filter(like='Proton').columns
                    flux_id = 'Proton_Flux_'
                    unc_id = 'Proton_Sigma_'
                    # fluxes = self.df[cols]
                    self.spec_E = np.array(self.meta['channels_dict_df_p'].mean_E)
                    self.DE = np.array(self.meta['channels_dict_df_p'].DE)
                    # # num_channels = len(self.df.filter(like='Proton_Flux').columns)

                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    # cols = self.df.filter(like='Electron').columns
                    flux_id = 'Electron_Flux_'
                    unc_id = 'Electron_Sigma_'
                    # fluxes = self.df[cols]
                    self.spec_E = np.array(self.meta['channels_dict_df_e'].mean_E)
                    self.DE = np.array(self.meta['channels_dict_df_e'].DE)
                    # num_channels = len(self.df.filter(like='Electron_Flux').columns)
            if self.instrument.lower() == 'sept':
                unc_id = 'err_ch_'
                if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                    self.spec_E = self.meta['channels_dict_df_p']['mean_E'].values
                    self.DE = self.meta['channels_dict_df_p']['DE'].values
                if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:
                    self.spec_E = self.meta['channels_dict_df_e']['mean_E'].values
                    self.DE = self.meta['channels_dict_df_e']['DE'].values

        if self.spacecraft.lower() in ['wind']:
            if self.viewing == "omnidirectional":
                flux_id = 'FLUX'
                # fluxes = self.df[cols]
            elif self.viewing.lower().startswith('sector'):
                flux_id = f'_P{self.viewing[-1]}'  # ty: ignore[index-out-of-bounds]
            self.spec_E = self.meta['channels_dict_df']['mean_E'].values
            self.DE = self.meta['channels_dict_df']['DE'].values

        if self.spacecraft.lower() in ['soho']:
            # cols = self.df.filter(like='PH').columns
            flux_id = 'PH_'
            unc_id = 'uncertainty_'
            self.spec_E = self.meta['channels_dict_df_p']['mean_E']
            self.DE = self.meta['channels_dict_df_p']['DE']
            # df_soho_counts = self.df[self.df.filter(like='PHC_').columns]

        if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
            if self.species.lower() in ['e', 'ele', 'electron', 'electrons']:  # !!! no fluxes available for electrons
                print('!!! no intensity data available for PSP electrons!')
            if self.species.lower() in ['p', 'ion', 'ions', 'protons']:
                flux_id = f"{self.viewing}_H_Flux"
                unc_id = f"{self.viewing}_H_Uncertainty"
                self.spec_E = self.meta['H_ENERGY']
                self.DE = self.meta['H_ENERGY_DELTAPLUS'] + self.meta['H_ENERGY_DELTAMINUS']

        if self.instrument.lower() == 'sept':
            df_fluxes = self.df[self.df.columns.drop([i for i in self.df.columns if 'err' in i])]
        else:
            df_fluxes = self.df[self.df.filter(like=flux_id).columns]

        if self.spacecraft.lower() != 'wind':  # all spacecraft except Wind
            df_uncs = self.df[self.df.filter(like=unc_id).columns]
            # For PSP, remove asymmetric uncertainties like A_H_Uncertainty_Minus_0 & A_H_Uncertainty_Plus_0 for now. TODO: update this?
            if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
                for col in ['Minus', 'Plus']:
                    try:
                        df_uncs = df_uncs[df_uncs.columns.drop([i for i in df_uncs.columns if col in i])]
                    except KeyError:
                        pass

        if subtract_background and spec_type == 'integral':
            ind_bg = np.where((self.df.index >= background_start) & (self.df.index <= background_end))[0]
            # bg_spec = np.nanmean(df_fluxes.iloc[ind_bg], axis=0)
            bg_spec = df_fluxes.iloc[ind_bg].mean()
            df_fluxes = df_fluxes-bg_spec
            # negative values after background subtraction are kept intentionally:
            # they reflect statistical noise around zero and zeroing/masking them
            # would introduce a positive bias in the integral flux

        if spec_type == 'integral':  # here we use the original (non-resamled) data
            df_fluxes_ind = df_fluxes.iloc[ind]
            if not df_fluxes_ind.empty:
                df_nan_test = df_fluxes_ind.dropna(axis=1, how="all")  # drop columns if they contain only NaNs
                nonan_points = df_nan_test.count()
                number_of_nan_entries = df_nan_test.isna().sum()

                nan_percent = np.nanmax(number_of_nan_entries/nonan_points) * 100
                if np.nanmax(number_of_nan_entries) > 0:
                    custom_warning(f'Data gaps in integration time interval! {np.nanmax(number_of_nan_entries)} out of {len(df_nan_test)} points ({nan_percent:.2f}%) of the intensity data points contributing to the integral spectrum are NaN. This may affect the resulting spectrum.')
                    if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:  # TODO: Jan: check note regarding PSP: is this understandable?
                        custom_warning('Note that for PSP there can be large datagaps which do not contain time-stamp data points. These can therefore not be considered in the NaN percentage above.')

                dt = df_fluxes_ind.index.to_series().diff()
                most_common_dt = dt.mode().iloc[0]
                integration_sec = most_common_dt.total_seconds()
                # nonan_points_per_channel = df_fluxes_ind.count()
                # total_integration_time_per_channel = integration_sec * nonan_points_per_channel.values

                if self.spacecraft.lower() == 'wind':
                    I_spec = np.nansum(df_fluxes.iloc[ind], axis=0)*1e6 * integration_sec
                    unc_spec = np.zeros(len(I_spec))*np.nan  # * integration_sec
                else:
                    I_spec = np.nansum(df_fluxes.iloc[ind], axis=0) * integration_sec

                    df_uncs_ind = df_uncs.iloc[ind]
                    # N_event is per-channel and NaN-aware, consistent with np.nansum above
                    N_event = df_uncs_ind.count().values

                    if subtract_background:
                        ind_bg = np.where((self.df.index >= background_start) & (self.df.index <= background_end))[0]
                        df_uncs_bg = df_uncs.iloc[ind_bg]

                        # uncertainty of mean background per channel:
                        # delta_J_bg = (1/N_bg) * sqrt(sum(delta_J_k^2))
                        delta_J_bg = propagated_mean_uncertainty(df_uncs_bg).values

                        # integral flux uncertainty with correlated background:
                        # delta_F = dt * N_event * sqrt(rms(delta_J_i)^2 + delta_J_bg^2)
                        # propagated_mean_uncertainty computes the uncertainty of a mean: sqrt(sum(delta_J_i^2)) / N_event.
                        # For the integral flux uncertainty we need the RSS sqrt(sum(delta_J_i^2)) instead,
                        # so we multiply back by N_event to undo the division.
                        unc_spec = N_event * np.sqrt(propagated_mean_uncertainty(df_uncs_ind).values**2 + delta_J_bg**2) * integration_sec
                    else:
                        # delta_F = dt * sqrt(sum(delta_J_i^2))
                        # propagated_mean_uncertainty computes the uncertainty of a mean: sqrt(sum(delta_J_i^2)) / N_event.
                        # For the integral flux uncertainty we need the RSS sqrt(sum(delta_J_i^2)) instead,
                        # so we multiply back by N_event to undo the division.
                        unc_spec = propagated_mean_uncertainty(df_uncs_ind).values * N_event * integration_sec

            else:  # if dataframe is empty (can happen for slices when bigger datagaps are present)
                I_spec = np.zeros(len(self.spec_E))*np.nan
                unc_spec = np.zeros(len(self.spec_E))*np.nan

            self.final_spec = I_spec
            self.final_unc = unc_spec

        if spec_type == 'peak':  # here we use the resamled data
            if resample is not None:
                # if self.spacecraft.lower() == 'wind':
                #     cols_unc = []
                # else:
                #     cols_unc = self.df.filter(like=unc_id).columns
                df_resampled = resample_df(self.df, resample, cols_unc='auto')
                print(f'Resampling used to determine this peak spectrum was {resample}')
            else:
                df_resampled = self.df.copy()
            ind_resampled = np.where((df_resampled.index >= spec_start) & (df_resampled.index <= spec_end))[0]

            if self.instrument.lower() == 'sept':
                df_fluxes_resampled = df_resampled[df_resampled.columns.drop([i for i in df_resampled.columns if 'err' in i])]
            else:
                df_fluxes_resampled = df_resampled[df_resampled.filter(like=flux_id).columns]

            def nanargmax_safe(col):
                """Return the index of the maximum value, or NaN if all values are NaN."""
                if col.isna().all():
                    return np.nan
                return np.nanargmax(col)

            # find the time index of peak flux per channel
            peak_idx = df_fluxes_resampled.iloc[ind_resampled].apply(nanargmax_safe, axis=0)

            # peak flux
            # The int() cast is needed because peak_idx values may be float (due to NaN being a float) when indexing with .iloc.
            I_spec = np.array([df_fluxes_resampled.iloc[ind_resampled].iloc[int(peak_idx[i]), i] if not np.isnan(peak_idx[i]) else np.nan for i in range(len(peak_idx))])

            if self.spacecraft.lower() == 'wind':
                I_spec = I_spec*1e6  # convert to 1/(cm² s sr MeV)
                unc_spec = np.zeros(len(I_spec))*np.nan  # no uncertainty data available for Wind/3DP
            else:
                df_uncs_resampled = df_resampled[df_resampled.filter(like=unc_id).columns]
                # For PSP, remove asymmetric uncertainties like A_H_Uncertainty_Minus_0 & A_H_Uncertainty_Plus_0 for now
                if self.spacecraft.lower() in ['parker', 'parker solar probe', 'psp']:
                    for col in ['Minus', 'Plus']:
                        try:
                            df_uncs_resampled = df_uncs_resampled[df_uncs_resampled.columns.drop([i for i in df_uncs_resampled.columns if col in i])]
                        except KeyError:
                            pass
                # unc_spec = np.nanmax(df_uncs_resampled.iloc[ind_resampled], axis=0)  # old implementation, which could end up at a different time step than the peak flux
                # look up uncertainty at the same time steps as the peak
                unc_spec = np.array([df_uncs_resampled.iloc[ind_resampled].iloc[int(peak_idx[i]), i] if not np.isnan(peak_idx[i]) else np.nan for i in range(len(peak_idx))])
            if subtract_background:    # here we use the original (non-resampled) data
                ind_bg = np.where((self.df.index >= background_start) & (self.df.index <= background_end))[0]
                bg_spec = np.nanmean(df_fluxes.iloc[ind_bg], axis=0)
                self.final_spec = I_spec - bg_spec
                if self.spacecraft.lower() in ['wind']:
                    self.final_unc = np.zeros(len(I_spec)) * np.nan  # TODO: implement correct uncerstainties for Wind/3DP
                else:
                    # bg_unc_spec = np.nanmean(df_uncs.iloc[ind_bg], axis=0)  # !!! check if implemented correctly
                    bg_unc_spec = propagated_mean_uncertainty(df_uncs.iloc[ind_bg])
                    # unc_spec = np.nanmax(df_uncs.iloc[ind_bg], axis=0)
                    self.final_unc = np.sqrt(bg_unc_spec**2 + unc_spec**2)
            else:
                self.final_spec = I_spec
                self.final_unc = unc_spec

        self.subtract_background = subtract_background
        self.spec_type = spec_type

        self.I_unc = self.final_unc
        self.E_unc = self.DE/2.

        self.spec_df = pd.DataFrame(dict(Energy=self.spec_E, Intensity=self.final_spec, E_err=self.E_unc, I_err=self.I_unc), index=range(len(self.spec_E)))

    # # moved to seppy.util. Make sure it works correctly: series vs dataframe (axis=0); len(series) includes NaNs, we want something like series.count()
    # def sqrt_sum_squares(self, series):
    #     # if isinstance(series, pd.DataFrame):
    #     #     custom_warning(f"Expected a pd.Series, got pd.DataFrame with shape {series.shape}")
    #     return np.sqrt(np.nansum(series**2, axis=0)) / len(series)

    def plot_spectrum(self, savefig=None, filename='', ylim=None):
        fig, ax = plt.subplots(1, sharex=True, figsize=(5, 4), dpi=150)
        ax.errorbar(self.spec_E, self.final_spec, xerr=self.E_unc, yerr=self.final_unc, fmt='o', markersize=8,
                    label=self.species, elinewidth=2, capsize=5, capthick=2, ecolor='lightgray')

        if self.spec_type == 'integral':
            spec_type_str = 'integral spec'
            ylabel_str = "Intensity (cm² sr MeV)⁻¹"
        if self.spec_type == 'peak':
            spec_type_str = 'peak spec'
            ylabel_str = "Intensity (cm² s sr MeV)⁻¹"
        if self.subtract_background:
            backsub_str = ', backgr. subtr.'
        else:
            backsub_str = ''
        viewing = self.viewing
        if viewing == 'omnidirectional':
            viewing = 'omni'
        ax.set_title(f"{self.spacecraft.upper()} / {self.instrument.upper()} {viewing} ({spec_type_str}{backsub_str})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel(ylabel_str)
        ax.legend()
        fig.tight_layout()

        if savefig:
            if filename == '':
                foldername = f'output_spectra{os.sep}'
                filename = f'{foldername}spectrum_{spec_type_str}_{self.spacecraft.upper()}_{self.instrument.upper()}_{self.viewing}_{self.species}.png'

            plt.savefig(filename)
            print(f"Figure saved as {filename}")
        # fig.show()
        return fig, ax
