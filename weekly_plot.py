#!/home/osant/conda/envs/sep_tools/bin/python3

# Usage instructions (6.11.2025): call signature is "bash weeklyplot.sh <spacecraft> <date_iso> <week_number>",
# where spacecraft is either "l1", "sta", "psp" or "solo", date_iso is the first Monday of the year (or whichever)
# start point you prefer and week_number is the number of weeks counted up from date_iso. So if the whole thing
# crashes midway (which it will 100% at some point), then you can continue from where it crashed.
# 
# The reason for the bash script workaround I believe was that the memory would get a bit clogged up using one
# Python interpreter. So for every week just throw the interpreter out and start a new one


import os
import sys
import datetime as dt
import multi_inst_plots as m
import matplotlib.pyplot as plt

local_data_path = "/Users/jagies/data/"  # TODO: update with your local path!

m.options.resample.value = 60
m.options.resample_mag.value = 15
m.options.resample_stixgoes.value = 0
m.options.l1_av_erne.value = 60
m.options.l1_av_sep.value = 60
m.options.Vsw.value = True
m.options.T.value = True
m.options.N.value = True

# Weekly plot options
print(sys.argv)

if sys.argv[1] == "l1":
    m.options.path = f"{local_data_path}/l1/"
    m.options.spacecraft.value = "L1 (Wind/SOHO)"
    m.options.stix.value = False
    m.options.goes.value = True

elif sys.argv[1] == "sta":
    m.options.path = f"{local_data_path}/stereo/"
    m.options.spacecraft.value = "STEREO"
    m.options.ster_sc.value = "A"
    m.options.stix.value = False
    m.options.goes.value = False

elif sys.argv[1] == "psp":
    m.options.path = f"{local_data_path}/psp/"
    m.options.spacecraft.value = "Parker Solar Probe"
    m.options.stix.value = False
    m.options.goes.value = False
    m.options.psp_epilo_p.value = False
    print(m.options.psp_epilo_p.value)

elif sys.argv[1] == "solo":
    m.options.path = f"{local_data_path}/solo/"
    m.options.spacecraft.value = "Solar Orbiter"
    m.options.stix.value = True
    m.options.goes.value = False


startdate = dt.date.fromisoformat(sys.argv[2])
week = int(sys.argv[3])
fig_save_path = f"{os.getcwd()}{os.sep}plots/{sys.argv[1]}/{startdate.year}/{sys.argv[1]}_{startdate.year}_w{week + 1:02d}"

m.options.startdate.value = dt.date(startdate.year,startdate.month,startdate.day) + dt.timedelta(weeks=week)
m.options.enddate.value = dt.date(startdate.year,startdate.month,startdate.day) + dt.timedelta(days=6) + dt.timedelta(weeks=week)
data, metadata = m.load_data()
fig, axs = m.make_plot(show=False)
fig.savefig(fig_save_path, bbox_inches="tight")
plt.close(fig)
print("Complete!")

    
    

