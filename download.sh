#! /bin/bash

# Download select data files for the whole year to speed up the weekly plotting process.
# Call sig: bash download.sh "spacecraft" "year"

if [ $1 = "l1" ]
then
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/3dp/3dp_sfsp/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/3dp/3dp_sosp/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/3dp/3dp_k0/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h3-rtn/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/soho/erne/hed_l2-1min/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/waves/rad1_l2/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/wind/waves/rad2_l2/$2/"
fi

if [ $1 = "stereo" ]
then
    wget -r -nv --show-progress --tries=10 -w 0.5 -nc -nH -np -nd -P "data" -A "*_sun_*.dat" "http://www2.physik.uni-kiel.de/STEREO/data/sept/level2/ahead/1min/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/ahead/l3/waves/lfr/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/ahead/l3/waves/hfr/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/ahead/l1/impact/het/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/ahead/l1/impact/rtn/mag/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/stereo/ahead/l2/impact/magplasma/1min/$2/"
fi

if [ $1 = "psp" ]
then
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/psp/isois/epilo/l2/ic/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/psp/isois/epilo/l2/pe/$2/"
    wget -r -nv --show-progress --tries=10 -nc -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/psp/isois/epihi/l2/het_rates1min/$2/"
fi

if [ $1 = "solo" ]
then
    wget -r -nv --show-progress --tries=10 -nc -w 0.2 -nH -np -nd -P "data/l2/epd/het" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/epd/science/l2/het/sun-rates/$2/"
    #wget -r -nv --show-progress --tries=10 -nc -w 0.2 -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/mag/science/l2/rtn-normal/$2/"
    #wget -r -nv --show-progress --tries=10 -nc -w 0.2 -nH -np -nd -P "data" -A "*.cdf" "https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/swa/science/l2/pas-grnd-mom/$2/"
fi
