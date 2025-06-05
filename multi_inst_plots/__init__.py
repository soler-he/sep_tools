# TODO:
# - fontsize as options?
# - SOLO/RPW still under development
# - copy of fig and axs? to make modifying it a bit easier. Or just implement some tool to do shading automatically
#       (make it also so that the original figure isn't modified...)
# - colormapping stuff that Nina mentioned (plotly/PFSS notebook? dunno)
# - figure out how to check if older data persists over a date change (or reset dataframes after date change)


import datetime as dt
import ipywidgets as w
import os
import matplotlib as mpl

from IPython.display import display
import multi_inst_plots.stereo_tools as stereo
import multi_inst_plots.psp_tools as psp
import multi_inst_plots.l1_tools as l1
import multi_inst_plots.solo_tools as solo


_style = {'description_width' : '60%'} 

_common_attrs = ["spacecraft", "startdate", "enddate", "resample", "resample_mag", "resample_stixgoes",
                 "radio_cmap", "legends_inside"]

_variable_attrs = ['radio', 'mag', 'polarity', 'mag_angles', 
                   'Vsw', 'N', 'T', "p_dyn", 
                   "stix", "stix_ltc", "goes", "goes_man_select"]

_psp_attrs = ['psp_epilo_e', 'psp_epilo_p', 'psp_epihi_e',
              'psp_epihi_p', 'psp_het_viewing', 'psp_epilo_viewing',
              'psp_epilo_ic_viewing']

_stereo_attrs = ['ster_sc', 'ster_sept_e', 'ster_sept_p', 'ster_het_e',
                 'ster_het_p', 'ster_sept_viewing']

_l1_attrs = ['l1_wind_e', 'l1_wind_p', 'l1_ephin', 'l1_erne', 
             'l1_av_sep', 'l1_av_erne']

_solo_attrs = ['solo_ept_e', 'solo_ept_p', 'solo_het_e', 'solo_het_p', 'solo_viewing']


class Options:
    def __init__(self):

        self.spacecraft = w.Dropdown(value=None, description="Spacecraft", 
                                     options=["Parker Solar Probe", "Solar Orbiter", "L1 (Wind/SOHO)", "STEREO"], 
                                     style=_style)
        self.startdate = w.DatePicker(value=dt.date(2022, 3, 14), disabled=False, description="Start date", 
                                      style={'description_width': "40%"})   
        self.enddate = w.DatePicker(value=dt.date(2022, 3, 16), disabled=False, description="End date", 
                                    style={'description_width': "40%"})

        self.resample = w.IntText(value=10, step=1, description='Averaging (min)', disabled=False, 
                                         style=_style)
        self.resample_mag = w.IntText(value=5, step=1, description='MAG averaging (min)', 
                                             disabled=False, style=_style)
        self.resample_stixgoes = w.IntText(value=1, step=1, description="STIX/GOES averaging (min)", 
                                                  style=_style)
        
        self.radio_cmap = w.Dropdown(options=list(mpl.colormaps), value='jet', description='Radio colormap', style=_style)
        self.pos_timestamp = 'center' 
        self.legends_inside = w.Checkbox(value=False, description='Legends inside')
    
        self.radio = w.Checkbox(value=True, description="Radio")
        self.mag = w.Checkbox(value=True, description="MAG")
        self.mag_angles = w.Checkbox(value=True, description="MAG angles")
        self.polarity = w.Checkbox(value=True, description="MAG polarity")
        self.Vsw = w.Checkbox(value=True, description="V_sw")
        self.N = w.Checkbox(value=True, description="N")
        self.T = w.Checkbox(value=True, description="T")
        self.p_dyn = w.Checkbox(value=True, description="P_dyn")
        self.stix = w.Checkbox(value=True, description="SolO/STIX")
        self.stix_ltc = w.Checkbox(value=True, description="Correct STIX for light travel time")
        self.goes = w.Checkbox(value=True, description="GOES/XRS")
        self.goes_man_select = w.Checkbox(value=False, description="GOES: manual sat selection")
        
        self.path = f"{os.getcwd()}{os.sep}data"
        self.plot_start = None
        self.plot_end = None

        self.psp_epilo_e = w.Checkbox(description="EPI-Lo PE Electrons", value=True)
        self.psp_epilo_p = w.Checkbox(description="EPI-Lo IC Protons", value=True)
        self.psp_epihi_e = w.Checkbox(description="EPI-Hi/HET Electrons", value=True)
        self.psp_epihi_p = w.Checkbox(description="EPI-Hi/HET Protons", value=True)
        self.psp_epihi_p_combine_channels = w.Checkbox(description="Combine EPI-Hi/HET proton channels", value=True)
        self.psp_het_viewing = w.Dropdown(description="EPI-Hi/HET viewing", options=["A", "B"], style=_style)
        self.psp_epilo_viewing = w.Dropdown(description="EPI-Lo PE viewing", options=range(0,8), 
                                            style=_style, disabled=False, value=3)       
        self.psp_epilo_ic_viewing = w.Dropdown(description="EPI-Lo IC viewing", options=range(0,80), 
                                               style=_style, disabled=False, value=3)
        self.psp_epilo_channel = w.Dropdown(description="EPI-Lo PE channel", options=['F', 'E', 'G'], 
                                            style=_style, disabled=True, value='F')
        self.psp_epilo_ic_channel = w.Dropdown(description="EPI-Lo IC channel", options=['T', 'D', 'R', 'P', 'C'], 
                                               style=_style, disabled=True, value='T')
        self.psp_ch_het_e = w.SelectMultiple(description="EPI-Hi/HET Electrons", options=range(0,18+1), 
                                             value=tuple(range(0,18+1,3)), rows=10, style=_style)
        self.psp_ch_het_p = w.SelectMultiple(description="EPI-Hi/HET Protons", options=range(0,14+1), 
                                             value=tuple(range(0,14+1,2)), rows=10, style=_style)
        self.psp_ch_epilo_pe =  w.SelectMultiple(description="EPI-Lo PE Electrons", options=range(0,5+1), 
                                                 value=tuple(range(0,5+1,1)), rows=10, style=_style)
        self.psp_ch_epilo_ic = w.SelectMultiple(description="EPI-Lo IC Protons", options=range(0,31+1), 
                                                value=tuple(range(0,31+1,4)), rows=10, style=_style)
        
        self.solo_ept_e = w.Checkbox(value=True, description="EPD/EPT Electrons")
        self.solo_ept_p = w.Checkbox(value=True, description="EPD/EPT Protons")
        self.solo_het_e = w.Checkbox(value=True, description="EPD/HET Electrons")
        self.solo_het_p = w.Checkbox(value=True, description="EPD/HET Protons")
        self.solo_viewing = w.Dropdown(options=['sun', 'asun', 'north', 'south'], value='sun', style=_style, 
                                       description="HET+EPT viewing:")
        #self.solo_resample_particles = w.BoundedIntText(value=10, min=0, description="HET+EPT averaging:", style=_style)
        self.solo_ch_ept_e = w.SelectMultiple(description="EPT Electrons", options=range(0,16+1), 
                                              value=tuple(range(0,15+1,2)), rows=10, style=_style)
        self.solo_ch_het_e = w.SelectMultiple(description="HET Electrons", options=range(0,3+1), 
                                              value=tuple(range(0,3+1,1)), style=_style)
        self.solo_ch_ept_p = w.SelectMultiple(description="EPT Protons", options=range(0,31+1), 
                                              value=tuple(range(0,31+1,5)), rows=10, style=_style)
        self.solo_ch_het_p = w.SelectMultiple(description="HET Protons", options=range(0,35+1), 
                                              value=tuple(range(0,35+1,5)), rows=10, style=_style)

        self.l1_wind_e =  w.Checkbox(value=True, description="Wind/3DP Electrons")
        self.l1_wind_p = w.Checkbox(value=True, description="Wind/3DP Protons")
        self.l1_ephin = w.Checkbox(value=True, description="SOHO/COSTEP-EPHIN Electrons")
        self.l1_erne = w.Checkbox(value=True, description="SOHO/ERNE-HED Protons")
        self.l1_ch_erne_p = w.SelectMultiple(description="ERNE-HED Protons", options=range(0, 9+1, 1), 
                                             value=tuple(range(0,9+1,2)), rows=10, disabled=False, style=_style)
        self.l1_ch_ephin_e = w.SelectMultiple(description="EPHIN Electrons", options=range(0,4), 
                                              value=(0,2), rows=10, style=_style)
        self.l1_ch_wind_e = w.SelectMultiple(description="3DP Electrons", options=range(0,6), 
                                              value=tuple(range(0,6,1)), rows=10, style=_style)
        self.l1_ch_wind_p = w.SelectMultiple(description="3DP Protons", options=range(0,7), 
                                              value=tuple(range(0,7,1)), rows=10, style=_style)
        self.l1_av_sep = w.IntText(value=10, description="3DP+EPHIN averaging", style=_style)
        self.l1_av_erne = w.IntText(value=10, description="ERNE averaging", style=_style)
        
        self.ster_sc = w.Dropdown(description="STEREO A/B", options=["A", "B"], style=_style)
        self.ster_sept_e = w.Checkbox(description="SEPT Electrons", value=True)
        self.ster_sept_p = w.Checkbox(description="SEPT Protons", value=True)
        self.ster_het_e = w.Checkbox(description="HET Electrons", value=True)
        self.ster_het_p = w.Checkbox(description="HET Protons", value=True)
        self.ster_sept_viewing = w.Dropdown(description="SEPT viewing", options=['sun', 'asun', 'north', 'south'], 
                                            style=_style)
        
        self.ster_ch_sept_e = w.SelectMultiple(description="SEPT Electrons", options=range(0,14+1), 
                                               value=tuple(range(0,14+1, 2)), rows=10, style=_style)
        self.ster_ch_sept_p = w.SelectMultiple(description="SEPT Protons", options=range(0,29+1), 
                                               value=tuple(range(0,29+1,4)), rows=10, style=_style)
        self.ster_ch_het_p =  w.SelectMultiple(description="HET Protons", options=range(0,10+1), 
                                               value=tuple(range(0,10+1,2)), rows=10, style=_style)
        self.ster_ch_het_e = w.SelectMultiple(description="HET Electrons", options=(0, 1, 2), 
                                              value=(0, 1, 2), style=_style)

        self._psp_box = w.VBox([getattr(self, attr) for attr in _psp_attrs])
        self._solo_box = w.VBox([getattr(self, attr) for attr in _solo_attrs])
        self._l1_box = w.VBox([getattr(self, attr) for attr in _l1_attrs])
        self._stereo_box = w.VBox([getattr(self, attr) for attr in _stereo_attrs])
        

        def _change_sc(change):
            """
            Change spacecraft-specific options. 
            """
            
            self._out2.clear_output()
            with self._out2:
                if change.new == "Parker Solar Probe":
                    display(self._psp_box)
                    
                if change.new == "Solar Orbiter":
                    display(self._solo_box)    
                    
                if change.new == "L1 (Wind/SOHO)":
                    display(self._l1_box)
                    
                if change.new == "STEREO":
                    display(self._stereo_box)


        def _delete_previous_data(change):
            """
            Delete data related to previous selection.
            """

            # L1 
            dfs = ["df_wind_wav_rad2",
                    "df_wind_wav_rad1",
                    "df_solwind",
                    "mag_data",
                    "ephin_",
                    "erne_p_",
                    "edic_",
                    "pdic_",
                    "df_stix_",
                    "df_goes_",
                    "meta_ephin",
                    "meta_erne",
                    "meta_e",
                    "meta_p"
                    ]
            for df in dfs:
                try:
                    delattr(l1, df)
                except AttributeError:
                    pass

            # STEREO
            dfs = [
                "df_sept_electrons_orig",
                "df_sept_protons_orig",
                "df_het_orig",
                "df_waves_hfr",
                "df_waves_lfr",
                "df_stix_",
                "df_goes_",
                "df_mag_orig",
                "df_magplasma",
                "meta_magplas",
                "meta_mag",
                "meta_se",
                "meta_sp",
                "meta_het"
            ]
            for df in dfs:
                try:
                    delattr(stereo, df)
                except AttributeError:
                    pass
        
            # SolO
            dfs = [
                "df_ept_org",
                "electrons_het",
                "electrons_ept",
                "protons_ept",
                "protons_het",
                "df_stix_",
                "df_goes_",
                "swa_data",
                "mag_data_org",
                "energies_ept",
                "energies_het",
                "metadata_ept"
            ]
            for df in dfs:
                try:
                    delattr(solo, df)
                except AttributeError:
                    pass
            
            # PSP
            dfs = [
                "psp_rfs_lfr_psd",
                "psp_rfs_hfr_psd",
                "df_psp_spani",
                "df_psp_spc",
                "psp_mag",
                "psp_het_org",
                "psp_epilo_ic_org",
                "psp_epilo_org",
                "df_stix_",
                "df_goes_",
                "psp_het_energies",
                "psp_epilo_energies",
                "psp_epilo_ic_energies"
            ]
            for df in dfs:
                try:
                    delattr(psp, df)
                except AttributeError:
                    pass

            
        def _disable_checkbox(change):
            """
            Disable checkbox when options get update (e.g. MAG + MAG polarity).
            """
            if change.owner == self.mag:
                if change.new == False:
                    self.polarity.disabled = True

                elif change.new == True:
                    self.polarity.disabled = False

            if change.owner == self.stix:
                if change.new == False:
                    self.stix_ltc.disabled = True

                elif change.new == True:
                    self.stix_ltc.disabled = False

            if change.owner == self.goes:
                if change.new == False:
                    self.goes_man_select.disabled = True

                elif change.new == True:
                    self.goes_man_select.disabled = False


        def _no_negative_avg(change):
            if change.new < 0:
                change.owner.value = 0


        self.mag.observe(_disable_checkbox, names="value")
        self.stix.observe(_disable_checkbox, names="value")
        self.goes.observe(_disable_checkbox, names="value")

        self.resample.observe(_no_negative_avg, names="value")
        self.resample_mag.observe(_no_negative_avg, names="value")
        self.resample_stixgoes.observe(_no_negative_avg, names="value")
        self.l1_av_erne.observe(_no_negative_avg, names="value")
        self.l1_av_sep.observe(_no_negative_avg, names="value")
            
        self.spacecraft.observe(_change_sc, names="value")

        self.spacecraft.observe(_delete_previous_data, names="value")
        self.startdate.observe(_delete_previous_data, names="value")
        self.enddate.observe(_delete_previous_data, names="value")

        # Common attributes (dates, plot options etc.)
        self._commons = w.HBox([w.VBox([getattr(self, attr) for attr in _common_attrs]), 
                                w.VBox([getattr(self, attr) for attr in _variable_attrs])])
        
        # output widgets
        layout = w.Layout(width="auto")
        self._out1 = w.Output(layout=layout)  # for common attributes
        self._out2 = w.Output(layout=layout)  # for sc specific attributes
        self._txt_out = w.Output(layout=layout) # for printing additional info
        self._outs = w.HBox((w.VBox([self._out1, self._txt_out]), self._out2))         # side-by-side outputs

    def show(self):
        self.spacecraft.value = None
        self._out1.clear_output()
        self._out2.clear_output()
        self._txt_out.clear_output()
        display(self._outs)

        with self._out1:
            display(self._commons)
        

def load_data():

    options.startdt = dt.datetime(options.startdate.value.year, options.startdate.value.month,
                                  options.startdate.value.day, 0, 0)
    
    options.enddt = dt.datetime(options.enddate.value.year, options.enddate.value.month, 
                                options.enddate.value.day, 23, 59, 59)

    if options.startdt > options.enddt:
        print("End date cannot precede startdate!")
        return None, None
    
    if options.spacecraft.value is None:
        print("You must choose a spacecraft first!")
        return None, None
    
    if options.spacecraft.value == "STEREO":
        print(f"Loading {options.spacecraft.value} {options.ster_sc.value}",
              f"data for range: {options.startdt} - {options.enddt}")
    else:
        print(f"Loading {options.spacecraft.value}",
              f"data for range: {options.startdt} - {options.enddt}")

    if options.spacecraft.value == "Parker Solar Probe":
        if options.startdt >= dt.datetime(2018, 10, 2):
            data, metadata = psp.load_data(options)
            return data, metadata
        else:
            print("Parker Solar Probe: no data before 2 Oct 2018")

    if options.spacecraft.value == "Solar Orbiter":
        if options.startdt >= dt.datetime(2020, 2, 28):
            data, metadata = solo.load_data(options)
            return data, metadata
        else:
            print("Solar Orbiter: no data before 28 Feb 2020")

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        if options.startdt >= dt.datetime(1994, 11, 1):
            data, metadata = l1.load_data(options)
            return data, metadata
        else:
            print("L1: no data before 1 Nov 1994 (Wind) / 2 Dec 1995 (SOHO)")
        
    if options.spacecraft.value == "STEREO":
        if options.startdt >= dt.datetime(2006, 10, 26):
            if options.enddt >= dt.datetime(2014, 9, 29) and options.ster_sc.value == "B":
                print("STEREO B: no data after 29 Sep 2014")
            else:
                data, metadata = stereo.load_data(options)
                return data, metadata
        else:
            print("STEREO A/B: no data before 26 Oct 2006")
    
    
def energy_channel_selection():

    if options.spacecraft.value == "Parker Solar Probe":
        display(psp.energy_channel_selection(options))
        selection = [options.psp_ch_epilo_pe, options.psp_ch_epilo_ic, options.psp_ch_het_e, options.psp_ch_het_p]
        
    if options.spacecraft.value == "Solar Orbiter":
        display(solo.energy_channel_selection(options))
        selection = [options.solo_ch_ept_e, options.solo_ch_ept_p, options.solo_ch_het_e, options.solo_ch_het_p]
        
    if options.spacecraft.value == "L1 (Wind/SOHO)":
        display(l1.energy_channel_selection(options))
        selection = [options.l1_ch_wind_e, options.l1_ch_wind_p, options.l1_ch_ephin_e, options.l1_ch_erne_p]

    if options.spacecraft.value == "STEREO":
        display(stereo.energy_channel_selection(options))
        selection = [options.ster_ch_sept_e, options.ster_ch_sept_p, options.ster_ch_het_e, options.ster_ch_het_p]
        
    layout = w.Layout(width="auto")
    ch_box = w.HBox(selection, layout=layout)
    display(ch_box)

    
def range_selection(low_e_step=None, low_p_step=None, high_e_step=None, high_p_step=None):
    """
    Defines evenly spaced energy channel ranges. Defaults (i.e. calling without args) to nice, sensible ranges.
    """
    if low_e_step is None:
        options.psp_ch_epilo_pe.value = tuple(range(0,5+1,1))
        options.l1_ch_wind_e.value = tuple(range(0,6,1))
        options.ster_ch_sept_e.value = tuple(range(0,15,2))
        options.solo_ch_ept_e.value = tuple(range(0,17,2))
    else:
        options.psp_ch_epilo_pe.value = tuple(range(0,5+1,low_e_step))
        options.l1_ch_wind_e.value = tuple(range(0,6,low_e_step))
        options.ster_ch_sept_e.value = tuple(range(0,15,low_e_step))
        options.solo_ch_ept_e.value = tuple(range(0,17,low_e_step))

    if low_p_step is None:
        options.psp_ch_epilo_ic.value = tuple(range(0,31+1,6))
        options.l1_ch_wind_p.value = tuple(range(0,7,1))
        options.ster_ch_sept_p.value = tuple(range(0,30,4))
        options.solo_ch_ept_p.value = tuple(range(0,32,5))
    else:
        options.psp_ch_epilo_ic.value = tuple(range(0,31+1,low_p_step))
        options.l1_ch_wind_p.value = tuple(range(0,7,low_p_step))
        options.ster_ch_sept_p.value = tuple(range(0,30,low_p_step))
        options.solo_ch_ept_p.value = tuple(range(0,32,low_p_step))

    if high_e_step is None:
        options.psp_ch_het_e.value = tuple(range(0,18+1,4))
        options.l1_ch_ephin_e.value = (0,2)
        options.ster_ch_het_e.value = (0,1,2)
        options.solo_ch_het_e.value = tuple(range(0,4,1))
    else:
        options.psp_ch_het_e.value = tuple(range(0,18+1,high_e_step))
        options.l1_ch_ephin_e.value = tuple(range(0,4,high_e_step))
        options.ster_ch_het_e.value = tuple(range(0,3,high_e_step))
        options.solo_ch_het_e.value = tuple(range(0,4,high_e_step))
    
    if high_p_step is None:
        options.psp_ch_het_p.value = tuple(range(0,14+1,3))
        options.l1_ch_erne_p.value = tuple(range(0,10,2))
        options.ster_ch_het_p.value = tuple(range(0,11,2))
        options.solo_ch_het_p.value = tuple(range(0,36,5))
    else:
        options.psp_ch_het_p.value = tuple(range(0,14+1,high_p_step))
        options.l1_ch_erne_p.value = tuple(range(0,10,high_p_step))
        options.ster_ch_het_p.value = tuple(range(0,11,high_p_step))
        options.solo_ch_het_p.value = tuple(range(0,36,high_p_step))

    
def make_plot(show=True):
    options.showplot = show
    if options.spacecraft.value == "Parker Solar Probe":
        return psp.make_plot(options)
    
    if options.spacecraft.value == "Solar Orbiter":
        return solo.make_plot(options)

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        return l1.make_plot(options)
    
    if options.spacecraft.value == "STEREO":
        return stereo.make_plot(options)
    

options = Options()
