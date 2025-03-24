import datetime as dt
import ipywidgets as w


from multi_sc_plots import stereo_tools as stereo
from multi_sc_plots import psp_tools as psp
from multi_sc_plots import l1_tools as l1
#from multi_sc_plots import tools.solo_tools as solo


style = {'description_width' : '50%'} 

common_attrs = ["spacecraft", "startdate", "enddate", "resample", "resample_mag", "radio_cmap", "legends_inside"] # , "resample_pol"

variable_attrs = ['radio', 'mag', 'mag_angles', 'polarity', 'Vsw', 'N', 'T', 'p_dyn', 'pad', "stix"] 

psp_attrs = ['psp_epilo_e', 'psp_epilo_p', 'psp_epihi_e',
             'psp_epihi_p', 'psp_het_viewing', 'psp_epilo_viewing',
             'psp_epilo_ic_viewing', 'psp_epilo_channel', 'psp_epilo_ic_channel', 
              "psp_ch_het_e", "psp_ch_het_p", "psp_ch_epilo_ic", "psp_ch_epilo_e"] # 

stereo_attrs = ['ster_sc', 'ster_sept_e', 'ster_sept_p', 'ster_het_e', 'ster_het_p', 'ster_sept_viewing', 'ster_ch_sept_e', 'ster_ch_sept_p',  'ster_ch_het_p'] #'ster_ch_het_e',

l1_attrs = ['l1_wind_e', 'l1_wind_p', 'l1_ephin', 'l1_erne', 'l1_ch_eph_e', 'l1_ch_eph_p', 'l1_intercal', 'l1_av_sep', 'l1_av_erne']

solo_attrs = ['stix']

class Options:
    def __init__(self):
        # now = dt.datetime.now()
        # dt_now = dt.datetime(now.year, now.month, now.day, 0, 0)

        self.spacecraft = w.Dropdown(value=None, description="Spacecraft", options=["PSP", "SolO", "L1 (Wind/SOHO)", "STEREO"], style=style)
        self.startdate = w.NaiveDatetimePicker(value=dt.datetime(2022, 3, 14), disabled=False, description="Start date:")
        self.enddate = w.NaiveDatetimePicker(value=dt.datetime(2022, 3, 16), disabled=False, description="End date:")

        self.resample = w.BoundedIntText(value=15, min=0, max=30, step=1, description='Averaging (min):', disabled=False, style=style)
        self.resample_mag = w.BoundedIntText(value=5, min=0, max=30, step=1, description='MAG averaging (min):', disabled=False, style=style)
        #self.resample_pol = w.BoundedIntText(value=1, min=0, max=30, step=1, description='Polarity resampling (min):', disabled=False, style=style)
        self.radio_cmap = w.Dropdown(options=['jet'], value='jet', description='Radio colormap', style=style)
        self.pos_timestamp = 'center' #w.Dropdown(options=['center', 'start', 'original'], description='Timestamp position', style=style)
        self.legends_inside = w.Checkbox(value=False, description='Legends inside', disabled=True)  # 20.3.2025: L1 doesn't have this option

        self.radio = w.Checkbox(value=True, description="Radio")
        self.pad = w.Checkbox(value=False, description="Pitch angle distribution", disabled=True)    # TODO: remove disabled keyword after implementation
        self.mag = w.Checkbox(value=True, description="MAG")
        self.mag_angles = w.Checkbox(value=True, description="MAG angles")
        self.polarity = w.Checkbox(value=True, description="Polarity")
        self.Vsw = w.Checkbox(value=True, description="V_sw")
        self.N = w.Checkbox(value=True, description="N")
        self.T = w.Checkbox(value=True, description="T")
        self.p_dyn = w.Checkbox(value=False, description="p_dyn", disabled=True)
        self.stix = w.Checkbox(value=False, description="SolO/STIX", disabled=True)
        
        self.path = None
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
        

        self.l1_wind_e =  w.Checkbox(value=True, description="Wind/3DP electrons")
        self.l1_wind_p = w.Checkbox(value=True, description="Wind/3DP protons")
        self.l1_ephin = w.Checkbox(value=True, description="SOHO/EPHIN electrons")
        self.l1_erne = w.Checkbox(value=True, description="SOHO/ERNE protons")
        self.l1_ch_eph_e = w.Dropdown(description="EPHIN e channel:", options=["E150"], value="E150", disabled=True, style=style)
        self.l1_ch_eph_p = w.Dropdown(description="EPHIN p channel:", options=["P25"], value="P25", disabled=True, style=style)
        self.l1_intercal = w.BoundedIntText(value=1, min=1, max=14, description="Intercal", disabled=True, style=style)
        self.l1_av_sep = w.BoundedIntText(value=20, min=0, max=30, description="3DP+EPHIN averaging:", style=style)
        self.l1_av_erne = w.BoundedIntText(value=10, min=0, max=30, description="ERNE averaging:", style=style)
        

        self.ster_sc = w.Dropdown(description="STEREO A/B:", options=["A", "B"], style=style)
        self.ster_sept_e = w.Checkbox(description="SEPT electrons", value=True)
        self.ster_sept_p = w.Checkbox(description="SEPT protons", value=True)
        self.ster_het_e = w.Checkbox(description="HET electrons", value=True)
        self.ster_het_p = w.Checkbox(description="HET protons", value=True)
        self.ster_sept_viewing = w.Dropdown(description="SEPT viewing", options=['sun', 'asun', 'north', 'south'], style=style)
        
        # TODO: "choose all energy channels" checkbox
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
                    display(w.HTML("Work in progress!"))    # display(self.solo_vbox)

                if change.new == "L1 (Wind/SOHO)":
                    display(self.l1_box)

                if change.new == "STEREO":
                    display(self.stereo_box)

        # TODO figure out how to have plot range update based on changes in start and end date
        # def change_plot_range(change):
            
        #     if change.owner == self.startdate:
        #         print("Changed startdate!")
        #         dates = plot_range_interval(change.new, self.enddate.value)
        #         self.plot_range.children[0].options = dates
        #         self.plot_range.children[0].value[0] = change.new
        #         self.plot_range.children[0].value[1] = self.enddate.value
                
        #     elif change.owner == self.enddate:
        #         print("Changed enddate!")
        #         dates = plot_range_interval(self.startdate.value, change.new)
        #         self.plot_range.children[0].options = dates
        #         self.plot_range.children[0].value[0] = self.startdate.value
        #         self.plot_range.children[0].value[1] = change.new
                
                # dates = plot_range_interval(self.startdate.value, change.new)
            
            # self.plot_range.children[0].options = dates
            # self.plot_range.children[0].index = (0,s len(dates) - 1)
            # self.plot_range.children[1]
            
        #Set observer to listen for changes in S/C dropdown menu
        self.spacecraft.observe(change_sc, names="value")

        #Likewise for start/end dates for plot range
        # self.startdate.observe(change_plot_range, names="value")
        # self.enddate.observe(change_plot_range, names='value')

        # Common attributes (dates, plot options etc.)
        self._commons = w.HBox([w.VBox([getattr(self, attr) for attr in common_attrs]), w.VBox([getattr(self, attr) for attr in variable_attrs])])
        
        # output widgets
        self._out1 = w.Output(layout=w.Layout(width="auto"))  # for common attributes
        self._out2 = w.Output(layout=w.Layout(width="auto"))  # for sc specific attributes
        self._pr_out = w.Output(layout=w.Layout(width="auto"))
        self._outs = w.HBox((w.VBox([self._out1, self._pr_out]), self._out2))         # side-by-side outputs

        display(self._outs)

        with self._out1:
            display(self._commons)

        # with self._pr_out:
        #     display(self.plot_range)
            
       

def plot_range_interval(startdate, enddate):
    timestamps = []
    date_iter = startdate
    while date_iter <= enddate:
        timestamps.append(date_iter)
        date_iter = date_iter + dt.timedelta(hours=1)
    return timestamps


def plot_range(startdate, enddate):
    """
    An HTML work-around to display datetimes without clipping on SelectionRangeSlider readouts.

    Author: Marcus Reaiche (https://github.com/jupyter-widgets/ipywidgets/issues/2855#issuecomment-966747483)
    """
    if not isinstance(startdate, dt.datetime) or not isinstance(enddate, dt.datetime):
        raise ValueError("Start and end dates have to be valid datetime objects")
    
    dates = plot_range_interval(startdate=startdate, enddate=enddate)

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

    return date_range


def load_data():
    global load_flag
    load_flag = False

    if options.spacecraft.value is None:
        print("You must choose a spacecraft first!")
        return
    
    if options.spacecraft.value == "PSP":
        psp.load_data(options)
        load_flag = True

    # if options.spacecraft.value == "SolO":
    #     solo.load_data(options)
    if options.spacecraft.value == "L1 (Wind/SOHO)":
        l1.load_data(options)
        load_flag = True

    if options.spacecraft.value == "STEREO":
        stereo.load_data(options)
        load_flag = True



def make_plot():
    if not load_flag:
        print("You must run load_data() first!")
        return (None, None)
    
    if options.spacecraft.value == "PSP":
        return psp.make_plot(options)
    
    # if options.spacecraft.value == "SolO":
    #     return solo.make_plot(options)

    if options.spacecraft.value == "L1 (Wind/SOHO)":
        return l1.make_plot(options)
    
    if options.spacecraft.value == "STEREO":
        return stereo.make_plot(options)
    

options = Options()
