import os
import warnings

import ipywidgets as w
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy
from IPython.display import display
from matplotlib.ticker import AutoMinorLocator
# from matplotlib.transforms import blended_transform_factory
from PIL import Image
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_PSP_EPILO, psp_isois_load
from seppy.loader.soho import calc_av_en_flux_ERNE, soho_load
from seppy.loader.stereo import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from seppy.loader.stereo import calc_av_en_flux_SEPT, stereo_load
from seppy.loader.wind import wind3dp_load
from seppy.util import custom_warning, resample_df
# from seppy.util import calc_av_en_flux_sixs  # bepi_sixs_load
try:
    from seppy.loader.bepi import bepi_sixsp_l3_loader
except ModuleNotFoundError:
    pass
from solo_epd_loader import combine_channels as calc_av_en_flux_EPD
from solo_epd_loader import epd_load, calc_ept_corrected_e
from sunpy.time import parse_time
# from sunpy.coordinates import frames, get_horizons_coord
# from tqdm import tqdm


# omit some warnings
warnings.simplefilter(action='once', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(action='ignore', message='No units provided for variable', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='astropy did not recognize units of', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message='The variable "HET_', category=sunpy.util.SunpyUserWarning, module='sunpy.io._cdf')
warnings.filterwarnings(action='ignore', message="Note that for the Dataframes containing the flow direction and SC coordinates timestamp position will not be adjusted by 'pos_timestamp'!", module='solo_epd_loader')


def add_watermark(fig, scaling=0.15, alpha=0.5, zorder=-1, x=1.0, y=0.0):
    logo = Image.open(f'multi_sc_plots{os.sep}soler.png')
    new_size = (np.array(logo.size) * scaling).astype(int)
    logo_s = logo.resize(new_size, Image.Resampling.LANCZOS)
    # x_offset = int((fig.bbox.xmax - pad*logo_s.size[0]) * 1.0)
    # y_offset = int((fig.bbox.ymax - pad*logo_s.size[1]) * 0.0)
    x_offset = int(fig.bbox.xmax * x)
    y_offset = int(fig.bbox.ymax * y)

    fig.figimage(logo_s, x_offset, y_offset, alpha=alpha, zorder=zorder)


class Event:

    def __init__(self):
        # manually define seaborn-colorblind colors
        seaborn_colorblind = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']  # blue, green, orange, magenta, yello, light blue
        # change some matplotlib plotting settings
        SIZE = 20
        plt.rc('font', size=SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
        plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SIZE)  # legend fontsize
        plt.rcParams['xtick.major.size'] = 10
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 5
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 10
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 5
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['lines.linewidth'] = 1.5

        self.e_instruments = ['BepiColombo/SIXS-P e', 'Parker Solar Probe/EPI-Hi HET e', 'Parker Solar Probe/EPI-Lo PE e', 'SOHO/EPHIN e', 'Solar Orbiter/EPT e', 'Solar Orbiter/HET e', 'STEREO-A/HET e', 'STEREO-A/SEPT e', 'WIND/3DP e']
        self.p_instruments = ['BepiColombo/SIXS-P p', 'Parker Solar Probe/EPI-Hi HET p', 'Parker Solar Probe/EPI-Lo IC p', 'SOHO/ERNE-HED p', 'Solar Orbiter/EPT p', 'Solar Orbiter/HET p', 'STEREO-A/SEPT p', 'STEREO-A/HET p', 'WIND/3DP p']

        # define default energy channels (lowest at the moment)
        self.channels_e = {}
        self.channels_e['BepiColombo/SIXS-P e'] = 1
        self.channels_e['Parker Solar Probe/EPI-Hi HET e'] = 0
        self.channels_e['Parker Solar Probe/EPI-Lo PE e'] = 0
        self.channels_e['SOHO/EPHIN e'] = 0
        self.channels_e['Solar Orbiter/EPT e'] = 0
        self.channels_e['Solar Orbiter/HET e'] = 0
        self.channels_e['STEREO-A/HET e'] = 0
        self.channels_e['STEREO-A/SEPT e'] = 2
        self.channels_e['WIND/3DP e'] = 0
        #
        self.channels_p = {}
        self.channels_p['BepiColombo/SIXS-P p'] = 1
        self.channels_p['Parker Solar Probe/EPI-Hi HET p'] = 0
        self.channels_p['Parker Solar Probe/EPI-Lo IC p'] = 0
        self.channels_p['SOHO/ERNE-HED p'] = 0
        self.channels_p['Solar Orbiter/EPT p'] = 0
        self.channels_p['Solar Orbiter/HET p'] = 0
        self.channels_p['STEREO-A/HET p'] = 0
        self.channels_p['STEREO-A/SEPT p'] = 2
        self.channels_p['WIND/3DP p'] = 0

        # define default plot colors
        self.plot_colors = {}
        self.plot_colors['BepiColombo/SIXS-P'] = 'orange'
        self.plot_colors['Parker Solar Probe/EPI-Hi HET'] = 'blueviolet'
        self.plot_colors['Parker Solar Probe/EPI-Lo PE'] = 'blueviolet'
        self.plot_colors['Parker Solar Probe/EPI-Lo IC'] = 'blueviolet'
        self.plot_colors['SOHO/EPHIN'] = 'k'
        self.plot_colors['SOHO/ERNE-HED'] = 'k'
        self.plot_colors['Solar Orbiter/EPT'] = seaborn_colorblind[5]
        self.plot_colors['Solar Orbiter/HET'] = seaborn_colorblind[0]
        self.plot_colors['STEREO-A/HET'] = 'orangered'
        # self.plot_colors['STEREO-A/LET'] = 'orangered'
        self.plot_colors['STEREO-A/SEPT'] = 'orangered'
        self.plot_colors['WIND/3DP'] = 'dimgrey'

        self.psp_epihi_e_scaling = 1  # 10
        self.psp_epilo_e_scaling = 1  # 100

        self.ept_data_product = 'l3'
        self.het_data_product = 'l2'

        # define default viewings and instrument channels
        self.viewing = {}
        self.viewing['BepiColombo/SIXS-P'] = 0
        self.viewing['Parker Solar Probe/EPI-Hi HET'] = 'A'
        self.viewing['Parker Solar Probe/EPI-Lo PE'] = 3
        self.viewing['Parker Solar Probe/EPI-Lo IC'] = 35  # 3x='sun', 7x='asun'
        self.viewing['Solar Orbiter/EPT'] = 'sun'
        self.viewing['Solar Orbiter/HET'] = 'sun'
        self.viewing['STEREO-A/SEPT'] = 'sun'
        # self.viewing['WIND/3DP'] = 'omni'

        self.psp_epilo_channel_e = 'F'
        self.psp_epilo_channel_p = 'P'  # 'P' or 'T'

    def instrument_selection(self):
        e_checkboxes = dict(zip(self.e_instruments, [w.Checkbox(value=True, description=option[:-1], indent=False) for option in self.e_instruments]))
        p_checkboxes = dict(zip(self.p_instruments, [w.Checkbox(value=True, description=option[:-1], indent=False) for option in self.p_instruments]))

        grid = w.GridspecLayout(max(len(self.e_instruments), len(self.p_instruments))+1, 2, width='50%')

        grid[0, 0] = w.HTML(value="<b>Electrons:</b>")
        grid[0, 1] = w.HTML(value="<b>Protons/Ions:</b>")
        for i, option in enumerate(self.e_instruments):
            grid[i+1, 0] = e_checkboxes[option]
        for i, option in enumerate(self.p_instruments):
            grid[i+1, 1] = p_checkboxes[option]
        display(grid)

        e_checkboxes.update(p_checkboxes)
        return e_checkboxes

    def load_data(self, startdate, enddate, dict_instruments, data_path=None):

        self.startdate = parse_time(startdate).to_datetime()
        self.enddate = parse_time(enddate).to_datetime()

        # create a list only containing spacecraft/instrument/particle combinations used
        self.instruments = []
        for key in dict_instruments:
            if dict_instruments[key].value:
                self.instruments.append(key)

        # create list containing all spacecraft used
        spacecraft = []
        for inst in self.instruments:
            spacecraft.append(inst.split('/')[0])
        self.spacecraft = [*{*spacecraft}]
        self.spacecraft.sort()

        if not data_path:
            data_path = os.getcwd()+os.sep+'data'+os.sep

        sixs_path = data_path
        soho_path = data_path
        solo_path = data_path
        stereo_path = data_path
        wind_path = data_path
        psp_path = data_path

        # catch alternative writings of 'asun'
        for key in self.viewing.keys():
            if type(self.viewing[key]) is str and self.viewing[key].lower() in ['antisun', 'anti-sun']:
                self.viewing[key] = 'asun'

        wind_3dp_threshold = None  # 1e3/1e6  # None
        psp_epilo_threshold_e = None  # 1e2  # None
        psp_epilo_threshold_p = None  # 1e2  # None

        self.psp_3600 = False  # don't change this!

        ##################################################################

        if 'WIND/3DP e' in self.instruments:
            # # print('loading wind/3dp e omni')
            self.wind3dp_e_df_org, self.wind3dp_e_meta = wind3dp_load(dataset="WI_SFSP_3DP", startdate=self.startdate, enddate=self.enddate, resample=None, multi_index=False, path=wind_path, threshold=wind_3dp_threshold)

        if 'WIND/3DP p' in self.instruments:
            # print('loading wind/3dp p omni')
            self.wind3dp_p_df_org, self.wind3dp_p_meta = wind3dp_load(dataset="WI_SOSP_3DP", startdate=self.startdate, enddate=self.enddate, resample=None, multi_index=False, path=wind_path)

        if 'STEREO-A/HET e' in self.instruments or 'STEREO-A/HET p' in self.instruments:
            # print('loading stereo/het')
            self.sta_het_e_labels = ['0.7-1.4 MeV', '1.4-2.8 MeV', '2.8-4.0 MeV']
            self.sta_het_p_labels = ['13.6-15.1 MeV', '14.9-17.1 MeV', '17.0-19.3 MeV', '20.8-23.8 MeV', '23.8-26.4 MeV', '26.3-29.7 MeV', '29.5-33.4 MeV', '33.4-35.8 MeV', '35.5-40.5 MeV', '40.0-60.0 MeV']
            self.sta_het_df_org, self.sta_het_meta = stereo_load(instrument='het', startdate=self.startdate, enddate=self.enddate, spacecraft='sta', resample=None, path=stereo_path, max_conn=1)

        # if 'STEREO-A/LET e' in self.instruments or 'STEREO-A/LET p' in self.instruments:
        #     # print('loading stereo/let')
        #     # for H and He4:
        #     self.let_chstring = ['1.8-2.2 MeV', '2.2-2.7 MeV', '2.7-3.2 MeV', '3.2-3.6 MeV', '3.6-4.0 MeV', '4.0-4.5 MeV', '4.5-5.0 MeV', '5.0-6.0 MeV', '6.0-8.0 MeV', '8.0-10.0 MeV', '10.0-12.0 MeV', '12.0-15.0 MeV']
        #     self.sta_let_df_org, self.sta_let_meta = stereo_load(instrument='let', startdate=self.startdate, enddate=self.enddate, spacecraft='sta', resample=sta_let_resample, path=stereo_path, max_conn=1)

        if 'STEREO-A/SEPT e' in self.instruments:
            # print('loading stereo/sept e')
            self.sta_sept_df_e_org, self.sta_sept_dict_e = stereo_load(instrument='sept', startdate=self.startdate, enddate=self.enddate, spacecraft='sta', sept_species='e', sept_viewing=self.viewing['STEREO-A/SEPT'], resample=None, path=stereo_path, max_conn=1)

        if 'STEREO-A/SEPT p' in self.instruments:
            # print('loading stereo/sept p')
            self.sta_sept_df_p_org, self.sta_sept_dict_p = stereo_load(instrument='sept', startdate=self.startdate, enddate=self.enddate, spacecraft='sta', sept_species='p', sept_viewing=self.viewing['STEREO-A/SEPT'], resample=None, path=stereo_path, max_conn=1)

        if 'SOHO/EPHIN e' in self.instruments or 'SOHO/EPHIN p' in self.instruments:
            # print('loading soho/ephin')
            try:
                self.soho_ephin_org, self.ephin_energies = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN",
                                                                    startdate=self.startdate,
                                                                    enddate=self.enddate,
                                                                    path=soho_path,
                                                                    resample=None,
                                                                    pos_timestamp='center')
            except UnboundLocalError:
                pass

        if 'SOHO/ERNE-HED p' in self.instruments:
            # print('loading soho/erne')
            self.erne_chstring = ['13-16 MeV', '16-20 MeV', '20-25 MeV', '25-32 MeV', '32-40 MeV', '40-50 MeV', '50-64 MeV', '64-80 MeV', '80-100 MeV', '100-130 MeV']
            self.soho_erne_org, self.erne_energies = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=self.startdate, enddate=self.enddate, path=soho_path, resample=None, max_conn=1)

        if 'Parker Solar Probe/EPI-Hi HET e' in self.instruments or 'Parker Solar Probe/EPI-Hi HET p' in self.instruments:
            # print('loading PSP/EPI-Hi HET data')
            self.psp_het, self.psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate=self.startdate, enddate=self.enddate, path=psp_path, resample=None)
            # psp_let1, psp_let1_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-LET1-RATES60', startdate, enddate, path=psp_path, resample=psp_resample)
            if len(self.psp_het) == 0:
                print(f'No PSP/EPI-Hi HET 60s data found for {self.startdate.date()} - {self.enddate.date()}. Trying 3600s data.')
                self.psp_het, self.psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES3600', startdate=self.startdate, enddate=self.enddate, path=psp_path, resample=None)
                self.psp_3600 = True
                # self.psp_het_resample = None

        if 'Parker Solar Probe/EPI-Lo PE e' in self.instruments:
            # print('loading PSP/EPI-Lo PE data')
            self.psp_epilo_e, self.psp_epilo_energies_e = psp_isois_load('PSP_ISOIS-EPILO_L2-PE',
                                                                     startdate=self.startdate, enddate=self.enddate,
                                                                     epilo_channel=self.psp_epilo_channel_e,
                                                                     epilo_threshold=psp_epilo_threshold_e,
                                                                     path=psp_path, resample=None)
            if len(self.psp_epilo_e) == 0:
                print(f'No PSP/EPI-Lo PE data for {self.startdate.date()} - {self.enddate.date()}')

        if 'Parker Solar Probe/EPI-Lo IC p' in self.instruments:
            custom_warning('Parker Solar Probe/EPI-Lo IC p is in beta mode! Please be cautious and report bugs.') 
        #     print('loading PSP/EPI-Lo IC proton data')
            self.psp_epilo_p, self.psp_epilo_energies_p = psp_isois_load('PSP_ISOIS-EPILO_L2-IC',
                                                               startdate=self.startdate, enddate=self.enddate,
                                                               epilo_channel=self.psp_epilo_channel_p,
                                                               epilo_threshold=psp_epilo_threshold_p,
                                                               path=psp_path, resample=None)
            if len(self.psp_epilo_p) == 0:
                print(f'No PSP/EPI-Lo IC data for {self.startdate.date()} - {self.enddate.date()}')

        if 'Solar Orbiter/EPT e' in self.instruments or 'Solar Orbiter/EPT p' in self.instruments:
            # print('loading solo/ept e & p')
            try:
                result = epd_load(sensor='EPT', viewing=self.viewing['Solar Orbiter/EPT'], level=self.ept_data_product, startdate=self.startdate, enddate=self.enddate, path=solo_path, autodownload=True)
                if self.ept_data_product == 'l2':
                    self.ept_p, self.ept_e, self.ept_energies = result
                if self.ept_data_product == 'l3':
                    self.ept, self.ept_rtn, self.ept_hci, self.ept_energies, self.ept_metadata = result
            except (Exception):
                print(f'No SOLO/EPT {self.ept_data_product} data for {self.startdate.date()} - {self.enddate.date()}')
                self.ept = []
                self.ept_e = []
                self.ept_p = []
        if 'Solar Orbiter/HET e' in self.instruments or 'Solar Orbiter/HET p' in self.instruments:
            # print('loading solo/het e & p')
            try:
                self.het_p, self.het_e, self.het_energies = epd_load(sensor='HET', viewing=self.viewing['Solar Orbiter/HET'], level=self.het_data_product, startdate=self.startdate, enddate=self.enddate, path=solo_path, autodownload=True)
            except (Exception):
                print(f'No SOLO/HET data for {self.startdate.date()} - {self.enddate.date()}')
                self.het_e = []
                self.het_p = []

        if 'BepiColombo/SIXS-P e' in self.instruments or 'BepiColombo/SIXS-P p' in self.instruments:
            # print('loading Bepi/SIXS')
            try:
                self.sixs_df, self.sixs_meta = bepi_sixsp_l3_loader(startdate=self.startdate, enddate=self.enddate, resample=None, path=sixs_path, pos_timestamp='center')
                if len(self.sixs_df) > 0:
                    self.sixs_df_p_org = self.sixs_df[[f"Side{self.viewing['BepiColombo/SIXS-P']}_P{i}" for i in range(1, 10)]]
                    self.sixs_df_e_org = self.sixs_df[[f"Side{self.viewing['BepiColombo/SIXS-P']}_E{i}" for i in range(1, 8)]]
            except NameError:
                custom_warning('BepiColombo/SIXS-P not yet supported.')
                self.sixs_df = []
                self.sixs_df_e_org = []
                self.sixs_df_p_org = []


    def print_energies(self):
        #
        particle = []
        for inst in self.instruments:
            particle.append(inst.split(' ')[-1])
        if particle.count('e') > 0:
            load_e = True
        else:
            load_e = False
        if particle.count('p') > 0:
            load_p = True
        else:
            load_p = False

        if load_e:
            self.energies_e = pd.DataFrame()
            if 'BepiColombo/SIXS-P e' in self.instruments:
                if len(self.sixs_df) > 0:
                    try:
                        t = {'E0': 'NaN'}
                        t.update(self.sixs_meta[f"Side{self.viewing['BepiColombo/SIXS-P']}_Electron_Bins_str"])
                        self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'BepiColombo/SIXS-P e': t.values()})], axis=1)
                    except NameError:
                        pass
            if 'Parker Solar Probe/EPI-Hi HET e' in self.instruments:
                try:
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'Parker Solar Probe/EPI-Hi HET e': self.psp_het_energies['Electrons_ENERGY_LABL']})], axis=1)
                except TypeError:
                    pass
            if 'Parker Solar Probe/EPI-Lo PE e' in self.instruments:
                try:
                    epilo_low = self.psp_epilo_energies_e[f'Electron_Chan{self.psp_epilo_channel_e}_Energy'].filter(like='_P0').values-self.psp_epilo_energies_e[f'Electron_Chan{self.psp_epilo_channel_e}_Energy_DELTAMINUS'].filter(like='_P0').values
                    epilo_high = self.psp_epilo_energies_e[f'Electron_Chan{self.psp_epilo_channel_e}_Energy'].filter(like='_P0').values+self.psp_epilo_energies_e[f'Electron_Chan{self.psp_epilo_channel_e}_Energy_DELTAPLUS'].filter(like='_P0').values
                    epilo_low = epilo_low[~np.isnan(epilo_low)]
                    epilo_high = epilo_high[~np.isnan(epilo_high)]
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame([f'{epilo_low[i]:.2f} - {epilo_high[i]:.2f} keV' for i in range(len(epilo_low))], columns=['Parker Solar Probe/EPI-Lo PE e'])], axis=1)
                except TypeError:
                    pass
            if 'SOHO/EPHIN e' in self.instruments:
                try:
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'SOHO/EPHIN e': [self.ephin_energies[chan] for chan in ['E150', 'E300', 'E1300', 'E3000']]})], axis=1)
                except AttributeError:
                    pass
            if 'Solar Orbiter/EPT e' in self.instruments:
                try:
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'Solar Orbiter/EPT e': self.ept_energies['Electron_Bins_Text']})], axis=1)
                except AttributeError:
                    pass
            if 'Solar Orbiter/HET e' in self.instruments:
                try:
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'Solar Orbiter/HET e': self.het_energies['Electron_Bins_Text']})], axis=1)
                except TypeError:
                    pass
            if 'STEREO-A/HET e' in self.instruments:
                if len(self.sta_het_df_org) > 0:
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'STEREO-A/HET e': self.sta_het_e_labels})], axis=1)
            # if 'STEREO-A/LET e' in self.instruments:
            #     print(self.let_chstring)
            if 'STEREO-A/SEPT e' in self.instruments:
                if len(self.sta_sept_df_e_org) > 0:
                    self.energies_e = pd.concat([self.energies_e, self.sta_sept_dict_e['ch_strings'].rename('STEREO-A/SEPT e')], axis=1)
            if 'WIND/3DP e' in self.instruments:
                try:
                    self.energies_e = pd.concat([self.energies_e, pd.DataFrame({'WIND/3DP e': np.array(self.wind3dp_e_meta['channels_dict_df']['Bins_Text'].values)})], axis=1)
                except TypeError:
                    pass

            self.energies_e.index.name = 'channel'

            print('Electron channels:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(self.energies_e)
            print('')

        if load_p:
            self.energies_p = pd.DataFrame()
            if 'BepiColombo/SIXS-P p' in self.instruments:
                if len(self.sixs_df) > 0:
                    try:
                        t = {'P0': 'NaN'}
                        t.update(self.sixs_meta[f"Side{self.viewing['BepiColombo/SIXS-P']}_Proton_Bins_str"])
                        self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'BepiColombo/SIXS-P p': t.values()})], axis=1)
                    except NameError:
                        pass
            if 'Parker Solar Probe/EPI-Hi HET p' in self.instruments:
                try:
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'Parker Solar Probe/EPI-Hi HET p': self.psp_het_energies['H_ENERGY_LABL']})], axis=1)
                except TypeError:
                    pass
            if 'Parker Solar Probe/EPI-Lo IC p' in self.instruments:
                try:
                    epilo_low = self.psp_epilo_energies_p[f'H_Chan{self.psp_epilo_channel_p}_Energy'].filter(like=f"_P{self.viewing['Parker Solar Probe/EPI-Lo IC']}").values-self.psp_epilo_energies_p[f'H_Chan{self.psp_epilo_channel_p}_Energy_DELTAMINUS'].filter(like=f"_P{self.viewing['Parker Solar Probe/EPI-Lo IC']}").values
                    epilo_high = self.psp_epilo_energies_p[f'H_Chan{self.psp_epilo_channel_p}_Energy'].filter(like=f"_P{self.viewing['Parker Solar Probe/EPI-Lo IC']}").values+self.psp_epilo_energies_p[f'H_Chan{self.psp_epilo_channel_p}_Energy_DELTAPLUS'].filter(like=f"_P{self.viewing['Parker Solar Probe/EPI-Lo IC']}").values
                    nan_mask = ~np.isnan(epilo_low) + ~np.isnan(epilo_high)
                    epilo_low = epilo_low[nan_mask]
                    epilo_high = epilo_high[nan_mask]
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame([f'{epilo_low[i]:.2f} - {epilo_high[i]:.2f} keV' for i in range(len(epilo_low))], columns=['Parker Solar Probe/EPI-Lo IC p'])], axis=1)
                except TypeError:
                    pass
            # if 'SOHO/EPHIN p' in self.instruments:
            #     print(self.ephin_energies)
            if 'SOHO/ERNE-HED p' in self.instruments:
                if len(self.soho_erne_org) > 0:
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'SOHO/ERNE-HED p': self.erne_chstring})], axis=1)
            if 'Solar Orbiter/EPT p' in self.instruments:
                try:
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'Solar Orbiter/EPT p': self.ept_energies['Ion_Bins_Text']})], axis=1)
                except AttributeError:
                    pass
            if 'Solar Orbiter/HET p' in self.instruments:
                try:
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'Solar Orbiter/HET p': self.het_energies['H_Bins_Text']})], axis=1)
                except TypeError:
                    pass
            if 'STEREO-A/HET p' in self.instruments:
                if len(self.sta_het_df_org) > 0:
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'STEREO-A/HET p': self.sta_het_p_labels})], axis=1)
            # if 'STEREO-A/LET p' in self.instruments:
            #     print(self.let_chstring)
            if 'STEREO-A/SEPT p' in self.instruments:
                if len(self.sta_sept_df_p_org) > 0:
                    self.energies_p = pd.concat([self.energies_p, self.sta_sept_dict_p['ch_strings'].rename('STEREO-A/SEPT p')], axis=1)
            if 'WIND/3DP p' in self.instruments:
                try:
                    # self.energies_p = pd.concat([self.energies_p, self.wind3dp_p_meta['channels_dict_df'].reset_index()['Bins_Text'].rename('WIND/3DP p')], axis=1)
                    self.energies_p = pd.concat([self.energies_p, pd.DataFrame({'WIND/3DP p': np.array(self.wind3dp_p_meta['channels_dict_df']['Bins_Text'].values)})], axis=1)
                except TypeError:
                    pass

            self.energies_p.index.name = 'channel'

            print('Proton/ion channels:')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(self.energies_p)

    def plot(self, averaging=None, plot_range=None, dict_plot_instruments=None):
        if not dict_plot_instruments:
            plot_instruments = self.instruments
        else:
            plot_instruments = []
            for key in dict_plot_instruments:
                if dict_plot_instruments[key].value:
                    plot_instruments.append(key)

        if not plot_range:
            plot_range = [self.startdate, self.enddate]

        particle = []
        for inst in plot_instruments:
            particle.append(inst.split(' ')[-1])
        if particle.count('e') > 0:
            plot_e = True
        else:
            plot_e = False
        if particle.count('p') > 0:
            plot_p = True
        else:
            plot_p = False

        ept_use_corr_e = False  # not included yet! But so far here level 3 data is used, which is already corrected.

        intensity_label = 'Flux\n/(s cm² sr MeV)'
        linewidth = plt.rcParams['lines.linewidth']

        sixs_resample = averaging
        soho_erne_resample = averaging
        soho_ephin_resample = averaging
        solo_ept_resample = averaging
        solo_het_resample = averaging
        sta_het_resample = averaging
        sta_sept_resample = averaging
        sta_let_resample = averaging
        wind_3dp_resample = averaging
        if self.psp_3600:
            self.psp_het_resample = None
        else:
            self.psp_het_resample = averaging
        psp_epilo_resample = averaging

        psp_het_viewing_dict = {'A': 'sun',
                                'B': 'asun'}

        ephine_e_channels_list = ['E150', 'E300', 'E1300', 'E3000']
        if self.channels_e['SOHO/EPHIN e'] in [0, 1, 2, 3]:
            ephine_e_channel = ephine_e_channels_list[self.channels_e['SOHO/EPHIN e']]
        else:
            print("Unsupported SOHO/EPHIN e channel detected, falling back to default channel (0)!")
            ephine_e_channel = ephine_e_channels_list[0]

        """
        ########## AVERAGE ENERGY CHANNELS & RESAMPLING ##########
        #############################################
        """
        if 'BepiColombo/SIXS-P e' in plot_instruments or 'BepiColombo/SIXS-P p' in plot_instruments:
            if len(self.sixs_df) > 0:
                if isinstance(sixs_resample, str):
                    self.sixs_df_e = resample_df(self.sixs_df_e_org, sixs_resample)
                    self.sixs_df_p = resample_df(self.sixs_df_p_org, sixs_resample)
                else:
                    self.sixs_df_e = self.sixs_df_e_org
                    self.sixs_df_p = self.sixs_df_p_org

        if 'Parker Solar Probe/EPI-Hi HET e' in plot_instruments or 'Parker Solar Probe/EPI-Hi HET p' in plot_instruments:
            if len(self.psp_het) > 0:
                if 'Parker Solar Probe/EPI-Hi HET e' in plot_instruments:
                    # print('calc_av_en_flux_PSP_EPIHI e 1 MeV')
                    self.df_psp_het_e, self.psp_het_chstring_e = calc_av_en_flux_PSP_EPIHI(self.psp_het, self.psp_het_energies, self.channels_e['Parker Solar Probe/EPI-Hi HET e'], 'e', 'het', self.viewing['Parker Solar Probe/EPI-Hi HET'])
                    if isinstance(self.psp_het_resample, str):
                        self.df_psp_het_e = resample_df(self.df_psp_het_e, self.psp_het_resample)
                if 'Parker Solar Probe/EPI-Hi HET p' in plot_instruments:
                    # print('calc_av_en_flux_PSP_EPIHI p')
                    self.df_psp_het_p, self.psp_het_chstring_p = calc_av_en_flux_PSP_EPIHI(self.psp_het, self.psp_het_energies, self.channels_p['Parker Solar Probe/EPI-Hi HET p'], 'p', 'het', self.viewing['Parker Solar Probe/EPI-Hi HET'])
                    if isinstance(self.psp_het_resample, str):
                        self.df_psp_het_p = resample_df(self.df_psp_het_p, self.psp_het_resample)

        if 'Parker Solar Probe/EPI-Lo PE e' in plot_instruments:
            if len(self.psp_epilo_e) > 0:
                # if plot_e_100:
                # print('calc_av_en_flux_PSP_EPI-Lo e 100 keV')
                self.df_psp_epilo_e, self.psp_epilo_chstring_e = calc_av_en_flux_PSP_EPILO(self.psp_epilo_e,
                                                                                           self.psp_epilo_energies_e,
                                                                                           self.channels_e['Parker Solar Probe/EPI-Lo PE e'],
                                                                                           species='e',
                                                                                           mode='pe',
                                                                                           chan=self.psp_epilo_channel_e,
                                                                                           viewing=self.viewing['Parker Solar Probe/EPI-Lo PE'])

                # select energy channel
                # TODO: introduce calc_av_en_flux_PSP_EPILO(). ATM, if list of channels, only first one is selected  # IS THIS OUTDATED???
                # if type(self.channels_e['Parker Solar Probe/EPI-Lo PE e']) is list:
                #     self.channels_e['Parker Solar Probe/EPI-Lo PE e'] = self.channels_e['Parker Solar Probe/EPI-Lo PE e'][0]
                # df_psp_epilo_e = df_psp_epilo_e.filter(like=f'_E{self.channels_e['Parker Solar Probe/EPI-Lo PE e']}_')

                # energy = en_dict[f'Electron_Chan{self.psp_epilo_channel_p}_Energy'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
                # energy_low = energy - en_dict[f'Electron_Chan{self.psp_epilo_channel_p}_Energy_DELTAMINUS'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
                # energy_high = energy + en_dict[f'Electron_Chan{self.psp_epilo_channel_p}_Energy_DELTAPLUS'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
                # chstring_e = np.round(energy_low,1).astype(str) + ' - ' + np.round(energy_high,1).astype(str) + ' keV'

                if isinstance(psp_epilo_resample, str):
                    self.df_psp_epilo_e = resample_df(self.df_psp_epilo_e, psp_epilo_resample)

        if 'Parker Solar Probe/EPI-Lo IC p' in plot_instruments:
            if len(self.psp_epilo_p) > 0:
                self.df_psp_epilo_p, self.psp_epilo_chstring_p = calc_av_en_flux_PSP_EPILO(self.psp_epilo_p,
                                                                                           self.psp_epilo_energies_p,
                                                                                           self.channels_p['Parker Solar Probe/EPI-Lo IC p'],
                                                                                           species='H',
                                                                                           mode='ic',
                                                                                           chan=self.psp_epilo_channel_p,
                                                                                           viewing=self.viewing['Parker Solar Probe/EPI-Lo IC'])

                if isinstance(psp_epilo_resample, str):
                    self.df_psp_epilo_p = resample_df(self.df_psp_epilo_p, psp_epilo_resample)

        if 'SOHO/EPHIN e' in plot_instruments or 'SOHO/EPHIN p' in plot_instruments:
            if hasattr(self, 'soho_ephin_org'):
                if isinstance(soho_ephin_resample, str):
                    self.soho_ephin = resample_df(self.soho_ephin_org, soho_ephin_resample)
                else:
                    self.soho_ephin = self.soho_ephin_org

        if 'SOHO/ERNE-HED p' in plot_instruments:
            if len(self.soho_erne_org) > 0:
                if isinstance(soho_erne_resample, str):
                    self.soho_erne = resample_df(self.soho_erne_org, soho_erne_resample)
                else:
                    self.soho_erne = self.soho_erne_org
                #
                if type(self.channels_p['SOHO/ERNE-HED p']) is list and len(self.soho_erne) > 0:
                    self.soho_erne_avg_p, self.soho_erne_chstring_p = calc_av_en_flux_ERNE(self.soho_erne.filter(like='PH_'),
                                                                                        self.erne_energies['channels_dict_df_p'],
                                                                                        self.channels_p['SOHO/ERNE-HED p'],
                                                                                        species='p',
                                                                                        sensor='HED')

        if 'Solar Orbiter/EPT e' in plot_instruments:
            if self.ept_data_product == 'l2':
                if len(self.ept_e) > 0:
                    self.df_ept_e = self.ept_e['Electron_Flux']
                    # ept_en_str_e = self.ept_energies['Electron_Bins_Text'][:]

                    if ept_use_corr_e:
                        ept_e2 = self.ept_e
                        ept_p2 = self.ept_p
                        print('correcting solo/ept e')
                        if isinstance(solo_ept_resample, str):
                            ept_e2 = resample_df(self.ept_e, solo_ept_resample)
                            ept_p2 = resample_df(self.ept_p, solo_ept_resample)
                        self.df_ept_e_corr = calc_ept_corrected_e(ept_e2, ept_p2)

                        # df_ept_e = df_ept_e[f'Electron_Flux_{self.channels_e['Solar Orbiter/EPT e'][0]}']
                        # ept_chstring_e = ept_energies['Electron_Bins_Text'][self.channels_e['Solar Orbiter/EPT e'][0]][0]

                        # TODO: calc_av_en_flux_EPD expects multi-index with df['Electron_Flux']['Electron_Flux_0'], and looks for the first 'Electron_Flux' to check whether this is electron or ion data...
                        self.df_ept_e_corr, self.ept_chstring_e_corr = calc_av_en_flux_EPD2(self.df_ept_e_corr, self.ept_energies, self.channels_e['Solar Orbiter/EPT e'], 'ept', particles='e')

                    self.df_ept_e, self.ept_chstring_e = calc_av_en_flux_EPD(self.ept_e, self.ept_energies, self.channels_e['Solar Orbiter/EPT e'], 'ept')
            elif self.ept_data_product == 'l3':
                if len(self.ept) > 0:
                    self.df_ept_e, self.ept_chstring_e = calc_av_en_flux_EPD(self.ept, self.ept_energies, self.channels_e['Solar Orbiter/EPT e'], 'ept', species='e', viewing=self.viewing['Solar Orbiter/EPT'])

            if hasattr(self, 'df_ept_e'):
                if isinstance(solo_ept_resample, str) and len(self.df_ept_e) > 0:
                    self.df_ept_e = resample_df(self.df_ept_e, solo_ept_resample)

        if 'Solar Orbiter/EPT p' in plot_instruments:
            if self.ept_data_product == 'l2':
                if len(self.ept_p) > 0:
                    self.df_ept_p = self.ept_p['Ion_Flux']
                    self.ept_en_str_p = self.ept_energies['Ion_Bins_Text'][:]
                    self.df_ept_p, self.ept_chstring_p = calc_av_en_flux_EPD(self.ept_p, self.ept_energies, self.channels_p['Solar Orbiter/EPT p'], 'ept')
            elif self.ept_data_product == 'l3':
                if len(self.ept) > 0:
                    self.df_ept_p, self.ept_chstring_p = calc_av_en_flux_EPD(self.ept, self.ept_energies, self.channels_p['Solar Orbiter/EPT p'], 'ept', species='p', viewing=self.viewing['Solar Orbiter/EPT'])

            if hasattr(self, 'df_ept_p'):
                if isinstance(solo_ept_resample, str) and len(self.df_ept_p) > 0:
                    self.df_ept_p = resample_df(self.df_ept_p, solo_ept_resample)

        if 'Solar Orbiter/HET e' in plot_instruments or 'Solar Orbiter/HET p' in plot_instruments:
            if len(self.het_e) > 0:
                if 'Solar Orbiter/HET e' in plot_instruments:
                    # print('calc_av_en_flux_HET e')
                    self.df_het_e, het_chstring_e = calc_av_en_flux_EPD(self.het_e, self.het_energies, self.channels_e['Solar Orbiter/HET e'], 'het')
                    if isinstance(solo_het_resample, str):
                        self.df_het_e = resample_df(self.df_het_e, solo_het_resample)
                if 'Solar Orbiter/HET p' in plot_instruments:
                    # print('calc_av_en_flux_HET p')
                    self.df_het_p, self.het_chstring_p = calc_av_en_flux_EPD(self.het_p, self.het_energies, self.channels_p['Solar Orbiter/HET p'], 'het')
                    if isinstance(solo_het_resample, str):
                        self.df_het_p = resample_df(self.df_het_p, solo_het_resample)

        if 'STEREO-A/HET e' in plot_instruments or 'STEREO-A/HET p' in plot_instruments:
            if len(self.sta_het_df_org) > 0:
                if isinstance(sta_het_resample, str):
                    self.sta_het_df = resample_df(self.sta_het_df_org, sta_het_resample)
                else:
                    self.sta_het_df = self.sta_het_df_org
            #
            if 'STEREO-A/HET e' in plot_instruments:
                # if type(self.channels_e['STEREO-A/HET e']) is list and len(self.sta_het_df) > 0:
                if hasattr(self, 'sta_het_df') and len(self.sta_het_df) > 0 and len(self.sta_het_meta) > 0:
                    self.sta_het_avg_e, self.st_het_chstring_e = calc_av_en_flux_ST_HET(self.sta_het_df.filter(like='Electron'),
                                                                                        self.sta_het_meta['channels_dict_df_e'],
                                                                                        self.channels_e['STEREO-A/HET e'], species='e')
                else:
                    self.sta_het_avg_e = []
                    self.st_het_chstring_e = ''
            #
            if 'STEREO-A/HET p' in plot_instruments:
                # if type(self.channels_p['STEREO-A/HET p']) is list and len(self.sta_het_df) > 0:
                if hasattr(self, 'sta_het_df') and len(self.sta_het_df) > 0 and len(self.sta_het_meta) > 0:
                    self.sta_het_avg_p, self.st_het_chstring_p = calc_av_en_flux_ST_HET(self.sta_het_df.filter(like='Proton'),
                                                                                        self.sta_het_meta['channels_dict_df_p'],
                                                                                        self.channels_p['STEREO-A/HET p'], species='p')
                else:
                    self.sta_het_avg_p = []
                    self.st_het_chstring_p = ''

        if 'STEREO-A/LET e' in plot_instruments or 'STEREO-A/LET p' in plot_instruments:
            if isinstance(sta_let_resample, str):
                self.sta_let_df = resample_df(self.sta_let_df_org, sta_let_resample)
            else:
                self.sta_let_df = self.sta_let_df_org

        if 'STEREO-A/SEPT e' in plot_instruments:
            if hasattr(self, 'sta_sept_df_e_org') and len(self.sta_sept_df_e_org) > 0:
                if isinstance(sta_sept_resample, str):
                    self.sta_sept_df_e = resample_df(self.sta_sept_df_e_org, sta_sept_resample)
                else:
                    self.sta_sept_df_e = self.sta_sept_df_e_org
            #
            if type(self.channels_e['STEREO-A/SEPT e']) is list and hasattr(self, 'sta_sept_df_e') and len(self.sta_sept_df_e) > 0:
                self.sta_sept_avg_e, self.sept_chstring_e = calc_av_en_flux_SEPT(self.sta_sept_df_e, self.sta_sept_dict_e, self.channels_e['STEREO-A/SEPT e'])
            else:
                self.sta_sept_avg_e = []
                self.sept_chstring_e = ''

        if 'STEREO-A/SEPT p' in plot_instruments:
            if hasattr(self, 'sta_sept_df_p_org') and len(self.sta_sept_df_p_org) > 0:
                if isinstance(sta_sept_resample, str):
                    self.sta_sept_df_p = resample_df(self.sta_sept_df_p_org, sta_sept_resample)
                else:
                    self.sta_sept_df_p = self.sta_sept_df_p_org
            #
            if type(self.channels_p['STEREO-A/SEPT p']) is list and hasattr(self, 'sta_sept_df_p') and len(self.sta_sept_df_p) > 0:
                self.sta_sept_avg_p, self.sept_chstring_p = calc_av_en_flux_SEPT(self.sta_sept_df_p, self.sta_sept_dict_p, self.channels_p['STEREO-A/SEPT p'])
            else:
                self.sta_sept_avg_p = []
                self.sept_chstring_p = ''

        if 'WIND/3DP e' in plot_instruments:
            if hasattr(self, 'wind3dp_e_df_org') and len(self.wind3dp_e_df_org) > 0:
                if isinstance(wind_3dp_resample, str):
                    self.wind3dp_e_df = resample_df(self.wind3dp_e_df_org, wind_3dp_resample)
                else:
                    self.wind3dp_e_df = self.wind3dp_e_df_org

        if 'WIND/3DP p' in plot_instruments:
            if hasattr(self, 'wind3dp_p_df_org') and len(self.wind3dp_p_df_org) > 0:
                if isinstance(wind_3dp_resample, str):
                    self.wind3dp_p_df = resample_df(self.wind3dp_p_df_org, wind_3dp_resample)
                else:
                    self.wind3dp_p_df = self.wind3dp_p_df_org
        ##########################################################################################

        panels = 0
        if plot_e:
            panels = panels + 1
        # if plot_e_100:
        #     panels = panels + 1
        if plot_p:
            panels = panels + 1

        fig, axes = plt.subplots(panels, figsize=(24, 6*panels), dpi=200, sharex=True)
        axnum = 0

        # ELECTRONS
        #################################################################
        if plot_e:
            if panels == 1:
                ax = axes
            else:
                ax = axes[axnum]
                # species_string = 'Electrons'

            if 'BepiColombo/SIXS-P e' in plot_instruments:
                if hasattr(self, 'sixs_df') and len(self.sixs_df) > 0:
                    ax.plot(self.sixs_df_e.index, self.sixs_df_e[f"Side{self.viewing['BepiColombo/SIXS-P']}_E{self.channels_e['BepiColombo/SIXS-P e']}"],
                            color=self.plot_colors['BepiColombo/SIXS-P'], linewidth=linewidth,
                            label=f"BepiColombo/SIXS-P (side{self.viewing['BepiColombo/SIXS-P']}) "+self.sixs_meta[f"Side{self.viewing['BepiColombo/SIXS-P']}_Electron_Bins_str"][f"E{self.channels_e['BepiColombo/SIXS-P e']}"], drawstyle='steps-mid')
            if 'Parker Solar Probe/EPI-Hi HET e' in plot_instruments:
                if hasattr(self, 'psp_het') and len(self.psp_het) > 0:
                    # ax.plot(psp_het.index, psp_het[f'A_Electrons_Rate_{self.channels_e['Parker Solar Probe/EPI-Hi HET e']}'], color=self.plot_colors['Parker Solar Probe/EPI-Hi HET'], linewidth=linewidth,
                    #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPI-Hi HET '+psp_het_energies['Electrons_ENERGY_LABL'][self.channels_e['Parker Solar Probe/EPI-Hi HET e']][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
                    #         drawstyle='steps-mid')
                    ax.plot(self.df_psp_het_e.index, self.df_psp_het_e*self.psp_epihi_e_scaling, color=self.plot_colors['Parker Solar Probe/EPI-Hi HET'], linewidth=linewidth,
                            label=f'PSP/ISOIS EPI-Hi HET {self.viewing["Parker Solar Probe/EPI-Hi HET"]} ({psp_het_viewing_dict[self.viewing["Parker Solar Probe/EPI-Hi HET"]]})\n'+self.psp_het_chstring_e+r" $\bf{(count\ rate)}$",  # +f' *{self.psp_epihi_e_scaling}',
                            drawstyle='steps-mid')
            if 'Parker Solar Probe/EPI-Lo PE e' in plot_instruments:
                if hasattr(self, 'psp_epilo_e') and len(self.psp_epilo_e) > 0:
                    ax.plot(self.df_psp_epilo_e.index, self.df_psp_epilo_e*self.psp_epilo_e_scaling, color=self.plot_colors['Parker Solar Probe/EPI-Lo PE'], linewidth=linewidth,
                            label=f'PSP/ISOIS EPI-Lo PE {self.psp_epilo_channel_e} (W{self.viewing["Parker Solar Probe/EPI-Lo PE"]})\n'+self.psp_epilo_chstring_e+r" $\bf{(count\ rate)}$",  # +f' *{self.psp_epilo_e_scaling}',
                            drawstyle='steps-mid')
            if 'SOHO/EPHIN e' in plot_instruments:
                # ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=self.plot_colors['SOHO/EPHIN'], linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}', drawstyle='steps-mid')
                if hasattr(self, 'soho_ephin') and len(self.soho_ephin) > 0:
                    ax.plot(self.soho_ephin.index, self.soho_ephin[ephine_e_channel], color=self.plot_colors['SOHO/EPHIN'], linewidth=linewidth, label='SOHO/EPHIN '+self.ephin_energies[ephine_e_channel], drawstyle='steps-mid')
            if 'Solar Orbiter/EPT e' in plot_instruments:
                if hasattr(self, 'df_ept_e') and len(self.df_ept_e) > 0:
                    flux_ept = self.df_ept_e.values
                    if ept_use_corr_e:
                        ax.plot(self.df_ept_e_corr.index.values, self.df_ept_e_corr.values, linewidth=linewidth, color=self.plot_colors['Solar Orbiter/EPT'], label=f'SOLO/EPT {self.viewing["Solar Orbiter/EPT"]} '+self.ept_chstring_e+'\n(corrected)', drawstyle='steps-mid')
                    else:
                        # if type(self.channels_e['Solar Orbiter/EPT e']) is list:
                        #     for ch in self.channels_e['Solar Orbiter/EPT e']:
                        #         ax.plot(self.df_ept_e.index.values, flux_ept[:, ch], linewidth=linewidth, color=self.plot_colors['Solar Orbiter/EPT'], label='SOLO\nEPT '+ept_en_str_e[ch, 0]+f'\n{self.viewing['Solar Orbiter/EPT']}', drawstyle='steps-mid')
                        # elif type(self.channels_e['Solar Orbiter/EPT e']) is int:
                        ax.plot(self.df_ept_e.index.values, flux_ept, linewidth=linewidth, color=self.plot_colors['Solar Orbiter/EPT'], label=f'SOLO/EPT {self.viewing["Solar Orbiter/EPT"]} '+self.ept_chstring_e, drawstyle='steps-mid')
            if 'Solar Orbiter/HET e' in plot_instruments:
                if hasattr(self, 'het_e') and len(self.het_e) > 0:
                    ax.plot(self.df_het_e.index.values, self.df_het_e.flux, linewidth=linewidth, color=self.plot_colors['Solar Orbiter/HET'], label=f'SOLO/HET {self.viewing["Solar Orbiter/HET"]} '+het_chstring_e+'', drawstyle='steps-mid')
            if 'STEREO-A/HET e' in plot_instruments:
                if hasattr(self, 'sta_het_avg_e') and len(self.sta_het_avg_e) > 0:
                    ax.plot(self.sta_het_avg_e.index, self.sta_het_avg_e, color=self.plot_colors['STEREO-A/HET'], linewidth=linewidth,
                            label='STEREO-A/HET '+self.st_het_chstring_e, drawstyle='steps-mid')
            if 'STEREO-A/SEPT e' in plot_instruments:
                if type(self.channels_e['STEREO-A/SEPT e']) is list and hasattr(self, 'sta_sept_avg_e') and len(self.sta_sept_avg_e) > 0:
                    ax.plot(self.sta_sept_avg_e.index, self.sta_sept_avg_e, color=self.plot_colors['STEREO-A/SEPT'], linewidth=linewidth,
                            label=f'STEREO-A/SEPT {self.viewing["STEREO-A/SEPT"]} '+self.sept_chstring_e, drawstyle='steps-mid')
                elif type(self.channels_e['STEREO-A/SEPT e']) is int and hasattr(self, 'sta_sept_df_e') and len(self.sta_sept_df_e) > 0:
                    ax.plot(self.sta_sept_df_e.index, self.sta_sept_df_e[f"ch_{self.channels_e['STEREO-A/SEPT e']}"], color=self.plot_colors['STEREO-A/SEPT'],
                            linewidth=linewidth, label=f'STEREO-A/SEPT {self.viewing["STEREO-A/SEPT"]} '+self.sta_sept_dict_e.loc[self.channels_e['STEREO-A/SEPT e']]['ch_strings'], drawstyle='steps-mid')
            if 'WIND/3DP e' in plot_instruments:
                if hasattr(self, 'wind3dp_e_df') and len(self.wind3dp_e_df) > 0:
                    # multiply by 1e6 to get per MeV
                    ax.plot(self.wind3dp_e_df.index, self.wind3dp_e_df[f"FLUX_{self.channels_e['WIND/3DP e']}"]*1e6, color=self.plot_colors['WIND/3DP'], linewidth=linewidth, label='Wind/3DP omni '+self.wind3dp_e_meta['channels_dict_df']['Bins_Text'].iloc[self.channels_e['WIND/3DP e']], drawstyle='steps-mid')

            ax.set_yscale('log')
            ax.set_ylabel(intensity_label)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Electrons')

            axnum = axnum + 1

        # PROTONS
        #################################################################
        if plot_p:
            if panels == 1:
                ax = axes
            else:
                ax = axes[axnum]
            if 'BepiColombo/SIXS-P p' in plot_instruments:
                if hasattr(self, 'sixs_df') and len(self.sixs_df) > 0:
                    ax.plot(self.sixs_df_p.index, self.sixs_df_p[f"Side{self.viewing['BepiColombo/SIXS-P']}_P{self.channels_p['BepiColombo/SIXS-P p']}"],
                            color=self.plot_colors['BepiColombo/SIXS-P'], linewidth=linewidth,
                            label=f"BepiColombo/SIXS-P (side{self.viewing['BepiColombo/SIXS-P']}) "+self.sixs_meta[f"Side{self.viewing['BepiColombo/SIXS-P']}_Proton_Bins_str"][f"P{self.channels_p['BepiColombo/SIXS-P p']}"], drawstyle='steps-mid')
            if 'Parker Solar Probe/EPI-Hi HET p' in plot_instruments:
                if hasattr(self, 'psp_het') and len(self.psp_het) > 0:
                    # ax.plot(psp_het.index, psp_het[f'A_H_Flux_{self.channels_p['Parker Solar Probe/EPI-Hi HET p']}'], color=self.plot_colors['Parker Solar Probe/EPI-Hi HET'], linewidth=linewidth,
                    #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPI-Hi HET '+psp_het_energies['H_ENERGY_LABL'][self.channels_p['Parker Solar Probe/EPI-Hi HET p']][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
                    #         drawstyle='steps-mid')
                    ax.plot(self.df_psp_het_p.index, self.df_psp_het_p, color=self.plot_colors['Parker Solar Probe/EPI-Hi HET'], linewidth=linewidth,
                            # label='PSP '+'\nISOIS-EPI-Hi HET '+psp_het_chstring_p+'\nA (sun)',
                            label=f"PSP/ISOIS EPI-Hi HET {self.viewing['Parker Solar Probe/EPI-Hi HET']} ({psp_het_viewing_dict[self.viewing['Parker Solar Probe/EPI-Hi HET']]})\n"+self.psp_het_chstring_p,
                            drawstyle='steps-mid')
            if 'Parker Solar Probe/EPI-Lo IC p' in plot_instruments:
                if type(self.viewing['Parker Solar Probe/EPI-Lo IC']) is list:
                    psp_epilo_viewstring_p = f"{self.viewing['Parker Solar Probe/EPI-Lo IC'][0]}-{self.viewing['Parker Solar Probe/EPI-Lo IC'][-1]}"
                elif type(self.viewing['Parker Solar Probe/EPI-Lo IC']) is int:
                    psp_epilo_viewstring_p = str(self.viewing['Parker Solar Probe/EPI-Lo IC'])
                #
                if hasattr(self, 'psp_epilo_p') and len(self.psp_epilo_p) > 0:
                    ax.plot(self.df_psp_epilo_p.index, self.df_psp_epilo_p, color=self.plot_colors['Parker Solar Probe/EPI-Lo IC'], linewidth=linewidth,
                            label=f'PSP/ISOIS EPI-Lo IC {self.psp_epilo_channel_p} (L{psp_epilo_viewstring_p})\n'+self.psp_epilo_chstring_p,
                            drawstyle='steps-mid')
            if 'SOHO/ERNE-HED p' in plot_instruments:
                if type(self.channels_p['SOHO/ERNE-HED p']) is list and hasattr(self, 'soho_erne') and len(self.soho_erne) > 0:
                    ax.plot(self.soho_erne_avg_p.index, self.soho_erne_avg_p, color=self.plot_colors['SOHO/ERNE-HED'], linewidth=linewidth, label='SOHO/ERNE/HED '+self.soho_erne_chstring_p, drawstyle='steps-mid')
                elif type(self.channels_p['SOHO/ERNE-HED p']) is int:
                    if hasattr(self, 'soho_erne') and len(self.soho_erne) > 0:
                        ax.plot(self.soho_erne.index, self.soho_erne[f"PH_{self.channels_p['SOHO/ERNE-HED p']}"], color=self.plot_colors['SOHO/ERNE-HED'], linewidth=linewidth, label='SOHO/ERNE/HED '+self.erne_chstring[self.channels_p['SOHO/ERNE-HED p']], drawstyle='steps-mid')
                # if ephin_p:
                #     ax.plot(ephin['date'], ephin[ephin_ch_p][0], color=self.plot_colors['SOHO/EPHIN'], linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_p][1], drawstyle='steps-mid')
            if 'Solar Orbiter/EPT p' in plot_instruments:
                if hasattr(self, 'df_ept_p') and (len(self.df_ept_p) > 0):
                    ax.plot(self.df_ept_p.index.values, self.df_ept_p.values, linewidth=linewidth, color=self.plot_colors['Solar Orbiter/EPT'], label=f"SOLO/EPT {self.viewing['Solar Orbiter/EPT']} "+self.ept_chstring_p, drawstyle='steps-mid')
            if 'Solar Orbiter/HET p' in plot_instruments:
                if hasattr(self, 'het_p') and (len(self.het_p) > 0):
                    ax.plot(self.df_het_p.index, self.df_het_p, linewidth=linewidth, color=self.plot_colors['Solar Orbiter/HET'], label=f"SOLO/HET {self.viewing['Solar Orbiter/HET']} "+self.het_chstring_p, drawstyle='steps-mid')
            if 'STEREO-A/HET p' in plot_instruments:
                if hasattr(self, 'sta_het_avg_p') and len(self.sta_het_avg_p) > 0:
                    ax.plot(self.sta_het_avg_p.index, self.sta_het_avg_p, color=self.plot_colors['STEREO-A/HET'],
                            linewidth=linewidth, label='STEREO-A/HET '+self.st_het_chstring_p, drawstyle='steps-mid')
            # if 'STEREO-A/LET p' in plot_instruments:
                # str_ch = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
                # ax.plot(self.sta_let_df.index, self.sta_let_df[f'H_unsec_flux_{let_ch}'], color=stereo_let_color, linewidth=linewidth, label='STEREO-A/LET '+self.let_chstring[let_ch], drawstyle='steps-mid')
            if 'STEREO-A/SEPT p' in plot_instruments:
                if type(self.channels_p['STEREO-A/SEPT p']) is list and hasattr(self, 'sta_sept_avg_p') and len(self.sta_sept_avg_p) > 0:
                    ax.plot(self.sta_sept_df_p.index, self.sta_sept_avg_p, color=self.plot_colors['STEREO-A/SEPT'], linewidth=linewidth, label=f"STEREO-A/SEPT {self.viewing['STEREO-A/SEPT']} "+self.sept_chstring_p, drawstyle='steps-mid')
                elif type(self.channels_p['STEREO-A/SEPT p']) is int and hasattr(self, 'sta_sept_df_p') and len(self.sta_sept_df_p) > 0:
                    ax.plot(self.sta_sept_df_p.index, self.sta_sept_df_p[f"ch_{self.channels_p['STEREO-A/SEPT p']}"], color=self.plot_colors['STEREO-A/SEPT'], linewidth=linewidth, label=f"STEREO-A/SEPT {self.viewing['STEREO-A/SEPT']} {self.sta_sept_dict_p.loc[self.channels_p['STEREO-A/SEPT p']]['ch_strings']}", drawstyle='steps-mid')
            if 'WIND/3DP p' in plot_instruments:
                if hasattr(self, 'wind3dp_p_df') and len(self.wind3dp_p_df) > 0:
                    # multiply by 1e6 to get per MeV
                    # ax.plot(self.wind3dp_p_df.index, self.wind3dp_p_df[f'FLUX_{self.channels_p['WIND/3DP p']}']*1e6, color=self.plot_colors['WIND/3DP'], linewidth=linewidth, label='Wind 3DP omni '+str(round(wind3dp_p_df[f'ENERGY_{self.channels_p['WIND/3DP p']}'].mean()/1000., 2)) + ' keV', drawstyle='steps-mid')
                    ax.plot(self.wind3dp_p_df.index, self.wind3dp_p_df[f"FLUX_{self.channels_p['WIND/3DP p']}"]*1e6, color=self.plot_colors['WIND/3DP'], linewidth=linewidth, label='Wind 3DP omni '+self.wind3dp_p_meta['channels_dict_df']['Bins_Text'].iloc[self.channels_p['WIND/3DP p']], drawstyle='steps-mid')

            ax.set_yscale('log')
            ax.set_ylabel(intensity_label)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Protons/Ions')

            axnum = axnum+1

        ax.set_xlim(plot_range[0], plot_range[1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%b %d'))  # -d linux, #d windows
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel(f"Time (UTC) / Date in {self.startdate.year}")
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.1)
        add_watermark(fig, scaling=0.7, alpha=0.5, zorder=-1, x=0.92)
        plt.show()

        if sum([plot_e, plot_p]) == 1:
            return fig, [axes]
        elif sum([plot_e, plot_p]) == 2:
            return fig, axes


def calc_av_en_flux_EPD2(df, energies, en_channel, sensor, particles):
    """
    Average the fluxes of several adjacent energy channels of one sensor into
    a combined energy channel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing electron or proton/ion data of the sensor
    energies : dict
        Energy/meta dictionary returned from epd_load (last returned object)
    en_channel : list of 2 integers
        Range of adjacent energy channels to be used, e.g. [3, 5] for
        combining 4th, 5th, and 6th channels (counting starts with 0).
    sensor : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out : contains channel-averaged flux
    string
        en_channel_string : describes the energry range of combined channel

    Raises
    ------
    Exception
        - Sensor 'step' not supported yet.
        - Lowest EPT channels not supported because of overlapping energies.

    Examples
    --------
    Load EPT sun viewing direction level 2 data for Aug 20 to Aug 21, 2020, and
    combine electron channels 9 to 12 (i.e., 10th to 13th).

    > df_p, df_e, meta = epd_load('ept', 20200820, 20200821, 'l2', 'sun')
    > df_new, chan_new = combine_channels(df_p, meta, [9, 12], 'ept')
    """
    if sensor.lower() == 'step':
        raise Exception('STEP data not supported yet!')
        return pd.DataFrame(), ''
    # if species.lower() in ['e', 'electrons']:
    if particles == 'e':
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Flux'
    # if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
    elif particles == 'p':
        if sensor.lower() == 'het':
            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Flux'
        if sensor.lower() == 'ept':
            en_str = energies['Ion_Bins_Text']
            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Flux'
    if type(en_channel) is list:
        energy_low = en_str[en_channel[0]][0].split('-')[0]
        energy_up = en_str[en_channel[-1]][0].split('-')[-1]
        en_channel_string = energy_low + '-' + energy_up

        if len(en_channel) > 2:
            raise Exception('en_channel must have 2 elements: start channel and end channel, e.g. [1,3]!')
        if len(en_channel) == 2:
            # catch overlapping EPT energy channels and cancel calculation:
            if sensor.lower() == 'ept' and 'Electron_Flux' in df.keys() and en_channel[0] < 4:
                raise Exception('Lowest 4 EPT e channels not supported because of overlapping energies!')
                return pd.DataFrame(), ''
            if sensor.lower() == 'ept' and 'Electron_Flux' not in df.keys() and en_channel[0] < 9:
                raise Exception('Lowest 9 EPT ion channels not supported because of overlapping energies!')
                return pd.DataFrame(), ''
            # try to convert multi-index dataframe to normal one. if this is already the case, just continue
            try:
                df = df[flux_key]
            except (AttributeError, KeyError):
                None
            DE = energies[bins_width]
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[f'{flux_key}_{bins}'] * DE[bins]
                else:
                    I_all = I_all + df[f'{flux_key}_{bins}'] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux': df[flux_key][f'{flux_key}_{en_channel}']}, index=df.index)
            en_channel_string = en_str[en_channel][0]
    else:
        flux_out = pd.DataFrame({'flux': df[flux_key][f'{flux_key}_{en_channel}']}, index=df.index)
        en_channel_string = en_str[en_channel][0]
    return flux_out, en_channel_string
