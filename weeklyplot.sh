#!/bin/bash

# Usage instructions (6.11.2025): call signature is "bash weeklyplot.sh <spacecraft> <date_iso> <week_number>",
# where spacecraft is either "l1", "sta", "psp" or "solo", date_iso is the first Monday of the year (or whichever)
# start point you prefer and week_number is the number of weeks counted up from date_iso. So if the whole thing
# crashes midway (which it will 100% at some point), then you can continue from where it crashed.
# 
# The reason for the bash script workaround I believe was that the memory would get a bit clogged up using one
# Python interpreter. So for every week just throw the interpreter out and start a new one

for i in $(seq $3 51)
do
    echo Week $i
    python3 weekly_plot.py $1 $2 $i
done


