# TODO:
# - "choose all energy channels" checkbox (or choose every nth)
# - Empty plots and appropriate print output for time ranges with no data (right now crashing is pretty much guaranteed every time this happens) (L1 and PSP done)
# - legend overlapping with many energy channels
# - fix polarity axes on top of title
# - download retrying (making weekly plots is going to be a nightmare)
# - fontsize as options?
# - SOLO/RPW
# - GOES missing data handling


import datetime as dt
import pandas as pd
import ipywidgets as w
import os

from IPython.display import display
from multi_inst_plots import stereo_tools as stereo
from multi_inst_plots import psp_tools as psp
from multi_inst_plots import l1_tools as l1
from multi_inst_plots import solo_tools as solo


style = {'description_width' : '60%'} 

common_attrs = ["spacecraft", "startdate", "enddate", "starttime", "endtime", "resample", "resample_mag", "resample_stixgoes", "radio_cmap", "legends_inside"] # , "resample_pol"

variable_attrs = ['radio', 'mag', 'polarity', 'mag_angles', 'Vsw', 'N', 'T', 'p_dyn', "stix", "stix_ltc", "goes"] # ,'pad'

psp_attrs = ['psp_epilo_e', 'psp_epilo_p', 'psp_epihi_e',
             'psp_epihi_p', 'psp_het_viewing', 'psp_epilo_viewing',
             'psp_epilo_ic_viewing', 'psp_epilo_channel', 'psp_epilo_ic_channel', 
              "psp_ch_het_e", "psp_ch_het_p", "psp_ch_epilo_ic", "psp_ch_epilo_e"] # 

stereo_attrs = ['ster_sc', 'ster_sept_e', 'ster_sept_p', 'ster_het_e', 'ster_het_p', 'ster_sept_viewing', 'ster_ch_sept_e', 'ster_ch_sept_p',  'ster_ch_het_p'] #'ster_ch_het_e',

l1_attrs = ['l1_wind_e', 'l1_wind_p', 'l1_ephin', 'l1_erne', 'l1_ch_eph_e', 'l1_intercal', 'l1_av_sep', 'l1_av_erne']   # 'l1_ch_eph_p'

solo_attrs = ['solo_electrons', 'solo_protons', 'solo_viewing', 'solo_ch_ept_e', 'solo_ch_ept_p', 'solo_ch_het_e', 'solo_ch_het_p', 'solo_resample_particles']

class Options:
    def __init__(self):

        self.spacecraft = w.Dropdown(value="PSP", description="Spacecraft", options=["PSP", "SolO", "L1 (Wind/SOHO)", "STEREO"], style=style)
        self.startdate = w.DatePicker(value=dt.date(2021, 7, 27), disabled=False, description="Start date/time:", style={'description_width': "40%"})   
        self.enddate = w.DatePicker(value=dt.date(2021, 7, 28), disabled=False, description="End date/time:", style={'description_width': "40%"})
        self.starttime = w.TimePicker(description="Start time:", value=dt.time(0,0), step=60, style=style)
        self.endtime = w.TimePicker(description="End time:", value=dt.time(0,0), step=60, style=style)

        self.resample = w.BoundedIntText(value=15, min=0, max=30, step=1, description='Averaging (min):', disabled=False, style=style)
        self.resample_mag = w.BoundedIntText(value=5, min=0, max=30, step=1, description='MAG averaging (min):', disabled=False, style=style)
        self.resample_stixgoes = w.BoundedIntText(value=5, min=0, max=30, step=1, description="STIX/GOES averaging (min):", style=style)
        #self.resample_pol = w.BoundedIntText(value=1, min=0, max=30, step=1, description='Polarity resampling (min):', disabled=False, style=style)
        self.radio_cmap = w.Dropdown(options=['jet'], value='jet', description='Radio colormap', style=style)
        self.pos_timestamp = 'center' #w.Dropdown(options=['center', 'start', 'original'], description='Timestamp position', style=style)
        self.legends_inside = w.Checkbox(value=False, description='Legends inside')
    

        self.radio = w.Checkbox(value=True, description="Radio")
        #self.pad = w.Checkbox(value=True, description="Pitch angle distribution")    # TODO: remove disabled keyword after implementation
        self.mag = w.Checkbox(value=True, description="MAG")
        self.mag_angles = w.Checkbox(value=True, description="MAG angles")
        self.polarity = w.Checkbox(value=True, description="MAG polarity")
        self.Vsw = w.Checkbox(value=True, description="V_sw")
        self.N = w.Checkbox(value=True, description="N")
        self.T = w.Checkbox(value=True, description="T")
        self.p_dyn = w.Checkbox(value=True, description="p_dyn")
        self.stix = w.Checkbox(value=True, description="SolO/STIX")
        self.stix_ltc = w.Checkbox(value=True, description="Correct STIX for light travel time")
        self.goes = w.Checkbox(value=True, description="GOES/XRS")
        
        self.path = f"{os.getcwd()}{os.sep}data"
        self.plot_range = None

        self.psp_epilo_e = w.Checkbox(description="EPI-Lo electrons", value=True)
        self.psp_epilo_p = w.Checkbox(description="EPI-Lo protons", value=True)
        self.psp_epihi_e = w.Checkbox(description="EPI-Hi electrons", value=True)
        self.psp_epihi_p = w.Checkbox(description="EPI-Hi protons", value=True)
        self.psp_epihi_p_combined_pixels = w.Checkbox(description="EPI-Hi protons combined pixels", value=True)
        self.psp_het_viewing = w.Dropdown(description="HET viewing", options=["A", "B"], style=style)
        self.psp_epilo_viewing = w.Dropdown(description="EPI-Lo viewing:", options=["3"], style=style, disabled=True, value="3")          # TODO fill in correct channels and viewings
        self.psp_epilo_ic_viewing = w.Dropdown(description="EPI-Lo ic viewing:", options=["3"], style=style, disabled=True, value="3")
        self.psp_epilo_channel = w.Dropdown(description="EPI-Lo channel", options=['F'], style=style, disabled=True, value='F')
        self.psp_epilo_ic_channel = w.Dropdown(description="EPI-Lo ic channel", options=['T'], style=style, disabled=True, value='T')
        self.psp_ch_het_e = w.SelectMultiple(description="HET e channels", options=range(0,18+1), value=tuple(range(0,18+1,2)), rows=10, style=style)
        self.psp_ch_het_p = w.SelectMultiple(description="HET p channels", options=range(0,14+1), value=tuple(range(0,14+1,2)), rows=10, style=style)
        self.psp_ch_epilo_e =  w.SelectMultiple(description="EPI-Lo e channels", options=range(3,8+1), value=tuple(range(3,8+1,1)), rows=10, style=style)
        self.psp_ch_epilo_ic = w.SelectMultiple(description="EPI-Lo ic channels", options=range(0,31+1), value=tuple(range(0,31+1,4)), rows=10, style=style)
        
        self.solo_electrons = w.Checkbox(value=True, description="HET+EPT electrons")
        self.solo_protons = w.Checkbox(value=True, description="HET+EPT ions")
        self.solo_viewing = w.Dropdown(options=['sun', 'asun', 'north', 'south'], value='sun', style=style, description="HET+EPT viewing:")
        self.solo_resample_particles = w.BoundedIntText(value=5, min=0, max=30, description="HET+EPT averaging:", style=style)
        self.solo_ch_ept_e = w.SelectMultiple(description="EPT e channels", options=range(0,15+1), value=tuple(range(0,15+1,2)), rows=10, style=style)
        self.solo_ch_het_e = w.SelectMultiple(description="HET e channels", options=range(0,3+1), value=tuple(range(0,3+1,1)), rows=10, style=style)
        self.solo_ch_ept_p = w.SelectMultiple(description="EPT ion channels", options=range(0,30+1), value=tuple(range(0,30+1,5)), rows=10, style=style)
        self.solo_ch_het_p = w.SelectMultiple(description="HET ion channels", options=range(0,35+1), value=tuple(range(0,35+1,5)), rows=10, style=style)

        self.l1_wind_e =  w.Checkbox(value=True, description="Wind/3DP electrons")
        self.l1_wind_p = w.Checkbox(value=True, description="Wind/3DP protons")
        self.l1_ephin = w.Checkbox(value=True, description="SOHO/EPHIN electrons")
        self.l1_erne = w.Checkbox(value=True, description="SOHO/ERNE protons")
        self.l1_ch_eph_e = w.Dropdown(description="EPHIN e channel:", options=["E150", "E1300", "E3000"], value="E150", disabled=False, style=style)
        #self.l1_ch_eph_p = w.Dropdown(description="EPHIN p channel:", options=["P25"], value="P25", disabled=True, style=style)
        self.l1_intercal = w.BoundedIntText(value=1, min=1, max=14, description="Intercal", disabled=True, style=style)
        self.l1_av_sep = w.BoundedIntText(value=20, min=0, max=30, description="3DP+EPHIN averaging:", style=style)
        self.l1_av_erne = w.BoundedIntText(value=10, min=0, max=30, description="ERNE averaging:", style=style)
        
        self.ster_sc = w.Dropdown(description="STEREO A/B:", options=["A", "B"], style=style)
        self.ster_sept_e = w.Checkbox(description="SEPT electrons", value=True)
        self.ster_sept_p = w.Checkbox(description="SEPT protons", value=True)
        self.ster_het_e = w.Checkbox(description="HET electrons", value=True)
        self.ster_het_p = w.Checkbox(description="HET protons", value=True)
        self.ster_sept_viewing = w.Dropdown(description="SEPT viewing", options=['sun', 'asun', 'north', 'south'], style=style)
        
        self.ster_ch_sept_e = w.SelectMultiple(description="SEPT e channels", options=range(0,14+1), value=tuple(range(0,14+1, 2)), rows=10, style=style)
        self.ster_ch_sept_p = w.SelectMultiple(description="SEPT p channels", options=range(0,29+1), value=tuple(range(0,29+1,3)), rows=10, style=style)
        self.ster_ch_het_p =  w.SelectMultiple(description="HET p channels", options=range(0,10+1), value=tuple(range(0,10+1,1)), rows=10, style=style)
        self.ster_ch_het_e = w.SelectMultiple(description="HET e channels", options=(0, 1, 2), value=(0, 1, 2), disabled=True, style=style)


        self.psp_box = w.VBox([getattr(self, attr) for attr in psp_attrs])
        self.solo_box = w.VBox([getattr(self, attr) for attr in solo_attrs])
        self.l1_box = w.VBox([getattr(self, attr) for attr in l1_attrs])
        self.stereo_box = w.VBox([getattr(self, attr) for attr in stereo_attrs])

        ########### Define widget layout ###########
        
        def change_sc(change):
            """
            Change spacecraft-specific options.
            """
            self._out2.clear_output()

            with self._out2:
                if change.new == "PSP":
                    display(self.psp_box)

                if change.new == "SolO":
                    display(self.solo_box)    # display(self.solo_vbox)

                if change.new == "L1 (Wind/SOHO)":
                    display(self.l1_box)

                if change.new == "STEREO":
                    display(self.stereo_box)

            options.plot_range = None

        def disable_checkbox(change):
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

        # def limit_time_range(change):
        #     self._txt_out.clear_output()
        #     if self.enddate.value - change.new > dt.timedelta(days=7) or change.new - self.startdate.value > dt.timedelta(days=7):
        #         with self._txt_out:
        #             print("Data loading for more than 7 days is not supported!")


        self.mag.observe(disable_checkbox, names="value")
        self.stix.observe(disable_checkbox, names="value")
            
        # self.startdate.observe(limit_time_range, names="value")
        # self.enddate.observe(limit_time_range, names="value")
            
        #Set observer to listen for changes in S/C dropdown menu
        self.spacecraft.observe(change_sc, names="value")

        # Common attributes (dates, plot options etc.)
        self._commons = w.HBox([w.VBox([getattr(self, attr) for attr in common_attrs]), w.VBox([getattr(self, attr) for attr in variable_attrs])])
        
        # output widgets
        layout = w.Layout(width="auto")
        self._out1 = w.Output(layout=layout)  # for common attributes
        self._out2 = w.Output(layout=layout)  # for sc specific attributes
        self._txt_out = w.Output(layout=layout) # for printing additional info
        self._outs = w.HBox((w.VBox([self._out1, self._txt_out]), self._out2))         # side-by-side outputs

    def show(self):
        display(self._outs)

        with self._out1:
            display(self._commons)



def plot_range(startdate, enddate):
    """
    An HTML work-around to display datetimes without clipping on SelectionRangeSlider readouts.

    Author: Marcus Reaiche (https://github.com/jupyter-widgets/ipywidgets/issues/2855#issuecomment-966747483)
    """
    
    # dates = plot_range_interval(startdate=startdate, enddate=enddate)
    if startdate - enddate <= dt.timedelta(days=7):
        dates = pd.date_range(start=startdate, end=enddate, freq="1h")

        # First and last dates are selected by default
        initial_selection = (0, len(dates) - 1)

        # Define the date range slider: set readout to False
        date_range_selector = w.SelectionRangeSlider(
            options=dates,
            description="Plot range",
            index=initial_selection,
            continous_update=False,
            readout=False
        )

        # Define the display to substitute the readout
        date_range_display = w.HTML(
            value=(
                f"<b>{dates[initial_selection[0]]}" + 
                f" - {dates[initial_selection[1]]}</b>"))

        # Define the date range using the widgets.HBox
        date_range = w.HBox(
            (date_range_selector, date_range_display))

        # Callback function that updates the display
        def callback(dts):
            date_range_display.value = f"<b>{dts[0]} - {dts[1]}</b>"

        w.interactive_output(
            callback, 
            {"dts": date_range_selector})
        
        options.plot_range = date_range

        display(options.plot_range)
    
    else:
        print("Plotting for more than 7 days not supported!")
        return None
    



def load_data():
    options.startdt = dt.datetime(options.startdate.value.year,
                                    options.startdate.value.month,
                                    options.startdate.value.day,
                                    options.starttime.value.hour,
                                    options.starttime.value.minute)
    options.enddt = dt.datetime(options.enddate.value.year,
                                    options.enddate.value.month,
                                    options.enddate.value.day,
                                    options.endtime.value.hour,
                                    options.endtime.value.minute)
    
    
    if options.spacecraft.value is None:
        print("You must choose a spacecraft first!")
        return
    
    if options.spacecraft.value == "PSP":
        if options.startdt >= dt.datetime(2018, 10, 2):
            psp.load_data(options)
        else:
            print("PSP: no data before 2 Oct 2018")

    if options.spacecraft.value == "SolO":
        if options.startdt >= dt.datetime(2020, 2, 28):
            solo.load_data(options)
        else:
            print("SolO: no data before 28 Feb 2020")

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        if options.startdt >= dt.datetime(1994, 11, 1):
            l1.load_data(options)
        else:
            print("L1: no data before 1 Nov 1994 (Wind) / 2 Dec 1995 (SOHO)")
        
    if options.spacecraft.value == "STEREO":
        if options.startdt >= dt.datetime(2006, 10, 26):
            if options.enddt >= dt.datetime(2016, 9, 23) and options.ster_sc.value == "B":
                print("STEREO B: no data after 23 Sep 2016")
            else:
                stereo.load_data(options)
        else:
            print("STEREO A/B: no data before 26 Oct 2006")



def make_plot():
    if options.spacecraft.value == "PSP":
        return psp.make_plot(options)
    
    if options.spacecraft.value == "SolO":
        return solo.make_plot(options)

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        return l1.make_plot(options)
    
    if options.spacecraft.value == "STEREO":
        return stereo.make_plot(options)
    

options = Options()

