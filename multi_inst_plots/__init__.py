# TODO:
# - fontsize as options?
# - SOLO/RPW still under development
# - print energies?
# - WEEKLY PLOTS: 60 min particle avg, 15 min mag, 0 min STIX/GOES. Use only instruments at same location
# - go through datasets to figure out native cadences and fix issues related to them
# - replace resampling BoundedIntText boxes with IntText (max value capped was at 100)


import datetime as dt
import ipywidgets as w
import os

from IPython.display import display
import multi_inst_plots.stereo_tools as stereo
import multi_inst_plots.psp_tools as psp
import multi_inst_plots.l1_tools as l1
import multi_inst_plots.solo_tools as solo


_style = {'description_width' : '60%'} 

_common_attrs = ["spacecraft", "startdate", "enddate", "resample", "resample_mag", "resample_stixgoes",
                 "radio_cmap", "legends_inside"]

_variable_attrs = ['radio', 'mag', 'polarity', 'mag_angles', 
                   'Vsw', 'N', 'T', 
                   "stix", "stix_ltc", "goes", "goes_man_select"]

_psp_attrs = ["p_dyn", 'psp_epilo_e', 'psp_epilo_p', 'psp_epihi_e',
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
        self.startdate = w.DatePicker(value=dt.date(2023, 4, 21), disabled=False, description="Start date:", 
                                      style={'description_width': "40%"})   
        self.enddate = w.DatePicker(value=dt.date(2023, 4, 22), disabled=False, description="End date:", 
                                    style={'description_width': "40%"})

        self.resample = w.BoundedIntText(value=10, min=0, step=1, description='Averaging (min):', disabled=False, 
                                         style=_style)
        self.resample_mag = w.BoundedIntText(value=10, min=0, step=1, description='MAG averaging (min):', 
                                             disabled=False, style=_style)
        self.resample_stixgoes = w.BoundedIntText(value=10, min=0, step=1, description="STIX/GOES averaging (min):", 
                                                  style=_style)
        self.radio_cmap = w.Dropdown(options=['jet', 'plasma'], value='jet', description='Radio colormap', style=_style)
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
        self.psp_epilo_p = w.Checkbox(description="EPI-Lo IC Protons/Ions", value=True)
        self.psp_epihi_e = w.Checkbox(description="EPI-Hi/HET Electrons", value=True)
        self.psp_epihi_p = w.Checkbox(description="EPI-Hi/HET Protons/Ions", value=True)
        self.psp_epihi_p_combine_channels = w.Checkbox(description="Combine EPI-Hi/HET ion channels", value=True)
        self.psp_het_viewing = w.Dropdown(description="EPI-Hi/HET viewing", options=["A", "B"], style=_style)
        self.psp_epilo_viewing = w.Dropdown(description="EPI-Lo PE (e) viewing:", options=range(0,8), 
                                            style=_style, disabled=False, value=3)       
        self.psp_epilo_ic_viewing = w.Dropdown(description="EPI-Lo IC (p) viewing:", options=range(0,80), 
                                               style=_style, disabled=False, value=3)
        self.psp_epilo_channel = w.Dropdown(description="EPI-Lo PE channel", options=['F', 'E', 'G'], 
                                            style=_style, disabled=True, value='F')
        self.psp_epilo_ic_channel = w.Dropdown(description="EPI-Lo IC channel", options=['T', 'D', 'R', 'P', 'C'], 
                                               style=_style, disabled=True, value='T')
        self.psp_ch_het_e = w.SelectMultiple(description="EPI-Hi/HET Electrons", options=range(0,18+1), 
                                             value=tuple(range(0,18+1,3)), rows=10, style=_style)
        self.psp_ch_het_p = w.SelectMultiple(description="EPI-Hi/HET Protons/Ions", options=range(0,14+1), 
                                             value=tuple(range(0,14+1,2)), rows=10, style=_style)
        self.psp_ch_epilo_pe =  w.SelectMultiple(description="EPI-Lo PE Electrons", options=range(0,5+1), 
                                                 value=tuple(range(0,5+1,1)), rows=10, style=_style)
        self.psp_ch_epilo_ic = w.SelectMultiple(description="EPI-Lo IC Protons/Ions", options=range(0,31+1), 
                                                value=tuple(range(0,31+1,4)), rows=10, style=_style)
        
        self.solo_ept_e = w.Checkbox(value=True, description="EPD/EPT Electrons")
        self.solo_ept_p = w.Checkbox(value=True, description="EPD/EPT Ions")
        self.solo_het_e = w.Checkbox(value=True, description="EPD/HET Electrons")
        self.solo_het_p = w.Checkbox(value=True, description="EPD/HET Protons")
        self.solo_viewing = w.Dropdown(options=['sun', 'asun', 'north', 'south'], value='sun', style=_style, 
                                       description="HET+EPT viewing:")
        #self.solo_resample_particles = w.BoundedIntText(value=10, min=0, description="HET+EPT averaging:", style=_style)
        self.solo_ch_ept_e = w.SelectMultiple(description="EPT Electrons:", options=range(0,16+1), 
                                              value=tuple(range(0,15+1,2)), rows=10, style=_style)
        self.solo_ch_het_e = w.SelectMultiple(description="HET Electrons:", options=range(0,3+1), 
                                              value=tuple(range(0,3+1,1)), style=_style)
        self.solo_ch_ept_p = w.SelectMultiple(description="EPT Ions", options=range(0,31+1), 
                                              value=tuple(range(0,31+1,5)), rows=10, style=_style)
        self.solo_ch_het_p = w.SelectMultiple(description="HET Protons", options=range(0,35+1), 
                                              value=tuple(range(0,35+1,5)), rows=10, style=_style)

        self.l1_wind_e =  w.Checkbox(value=True, description="Wind/3DP Electrons")
        self.l1_wind_p = w.Checkbox(value=True, description="Wind/3DP Protons")
        self.l1_ephin = w.Checkbox(value=True, description="SOHO/COSTEP-EPHIN Electrons")
        self.l1_erne = w.Checkbox(value=True, description="SOHO/ERNE-HED Ions")
        self.l1_ch_erne_p = w.SelectMultiple(description="ERNE-HED Ions:", options=range(0, 9+1, 1), 
                                             value=tuple(range(0,9+1,2)), rows=10, disabled=False, style=_style)
        self.l1_ch_ephin_e = w.SelectMultiple(description="EPHIN Electrons:", options=range(0,4), 
                                              value=(0,2), rows=10, style=_style)
        self.l1_ch_wind_e = w.SelectMultiple(description="3DP Electrons:", options=range(1,7), 
                                              value=tuple(range(1,7,1)), rows=10, style=_style)
        self.l1_ch_wind_p = w.SelectMultiple(description="3DP Protons:", options=range(2,9), 
                                              value=tuple(range(2,9,1)), rows=10, style=_style)
        self.l1_av_sep = w.BoundedIntText(value=10, min=0, description="3DP+EPHIN averaging:", style=_style)
        self.l1_av_erne = w.BoundedIntText(value=10, min=0, description="ERNE averaging:", style=_style)
        
        self.ster_sc = w.Dropdown(description="STEREO A/B:", options=["A", "B"], style=_style)
        self.ster_sept_e = w.Checkbox(description="SEPT Electrons", value=True)
        self.ster_sept_p = w.Checkbox(description="SEPT Protons/Ions", value=True)
        self.ster_het_e = w.Checkbox(description="HET Electrons", value=True)
        self.ster_het_p = w.Checkbox(description="HET Protons/Ions", value=True)
        self.ster_sept_viewing = w.Dropdown(description="SEPT viewing", options=['sun', 'asun', 'north', 'south'], 
                                            style=_style)
        
        self.ster_ch_sept_e = w.SelectMultiple(description="SEPT Electrons", options=range(0,14+1), 
                                               value=tuple(range(0,14+1, 2)), rows=10, style=_style)
        self.ster_ch_sept_p = w.SelectMultiple(description="SEPT Protons/Ions", options=range(0,29+1), 
                                               value=tuple(range(0,29+1,4)), rows=10, style=_style)
        self.ster_ch_het_p =  w.SelectMultiple(description="HET Protons/Ions:", options=range(0,10+1), 
                                               value=tuple(range(0,10+1,2)), rows=10, style=_style)
        self.ster_ch_het_e = w.SelectMultiple(description="HET Electrons:", options=(0, 1, 2), 
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
                    display(self._solo_box)    # display(self.solo_vbox)

                if change.new == "L1 (Wind/SOHO)":
                    display(self._l1_box)

                if change.new == "STEREO":
                    display(self._stereo_box)

            options.plot_range = None

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


        self.mag.observe(_disable_checkbox, names="value")
        self.stix.observe(_disable_checkbox, names="value")
        self.goes.observe(_disable_checkbox, names="value")

            
        #Set observer to listen for changes in S/C dropdown menu
        self.spacecraft.observe(_change_sc, names="value")

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
                                options.enddate.value.day, 0, 0)

    if options.startdt > options.enddt:
        print("End date cannot precede startdate!")
        return None, None
    
    if options.spacecraft.value is None:
        print("You must choose a spacecraft first!")
        return None, None
    
    if options.spacecraft.value == "STEREO":
        print(f"Loading {options.spacecraft.value} {options.ster_sc.value}",
              f"data for range: {options.startdate.value} - {options.enddate.value}")
    else:
        print(f"Loading {options.spacecraft.value}",
              f"data for range: {options.startdate.value} - {options.enddate.value}")

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
        display(psp.energy_channel_selection())
        selection = [options.psp_ch_epilo_pe, options.psp_ch_epilo_ic, options.psp_ch_het_e, options.psp_ch_het_p]
        
    if options.spacecraft.value == "Solar Orbiter":
        display(solo.energy_channel_selection())
        selection = [options.solo_ch_ept_e, options.solo_ch_ept_p, options.solo_ch_het_e, options.solo_ch_het_p]
        
    if options.spacecraft.value == "L1 (Wind/SOHO)":
        display(l1.energy_channel_selection())
        selection = [options.l1_ch_wind_e, options.l1_ch_wind_p, options.l1_ch_ephin_e, options.l1_ch_erne_p]

    if options.spacecraft.value == "STEREO":
        display(stereo.energy_channel_selection())
        selection = [options.ster_ch_sept_e, options.ster_ch_sept_p, options.ster_ch_het_e, options.ster_ch_het_p]
        
    layout = w.Layout(width="auto")
    ch_box = w.HBox(selection, layout=layout)
    display(ch_box)
    
    
def make_plot():
    if options.spacecraft.value == "Parker Solar Probe":
        return psp.make_plot(options)
    
    if options.spacecraft.value == "Solar Orbiter":
        return solo.make_plot(options)

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        return l1.make_plot(options)
    
    if options.spacecraft.value == "STEREO":
        return stereo.make_plot(options)
    

options = Options()
