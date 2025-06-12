import numpy as np
import os
import pandas as pd
from seppy.loader.wind import wind3dp_load
# from pandas.tseries.frequencies import to_offset
# from numba import njit, prange
# import pyspedas
# import datetime as dt
from anisotropy.polarity_plotting import polarity_gse  # wind_polarity_preparation
from sunpy.coordinates import get_horizons_coord, HeliographicStonyhurst


def wind_download_and_prepare(instrument, startdate, enddate, path, averaging, species, en_channel, bg_start, bg_end):
    if species == "p":
        dataset1 = 'WI_SOSP_3DP'  # Proton omnidirectional fluxes 70 keV - 6.8 MeV, often at 24 sec
        dataset2 = 'WI_SOPD_3DP'  # Proton energy-angle distributions 70 keV - 6.8 MeV, often at 24 sec
        ch_string = 'Protons'
        sp_str = "Proton"
    elif species == "e":
        dataset1 = 'WI_SFSP_3DP'  # Electron omnidirectional fluxes 27 keV - 520 keV, often at 24 sec
        dataset2 = 'WI_SFPD_3DP'  # Electron energy-angle distributions 27 keV to 520 keV, often at 24 sec
        ch_string = 'Electrons'
        sp_str = "Electron"

    df_omni, meta_omni = wind3dp_load(dataset1, startdate, enddate, resample=averaging, path=path, multi_index=False)
    df_angle, meta_angle = wind3dp_load(dataset2, startdate, enddate, resample=averaging, path=path)
   
    en_ch_df = pd.DataFrame({'energy':meta_angle['channels_dict_df']['Bins_Text'].values})
    en_ch_df.index.names = ['channel']
    en_ch_df.to_csv(f'anisotropy/channels_Wind_{instrument}_{species}.csv')

    df_omni = df_omni.loc[(df_omni.index >= startdate) & (df_omni.index <= enddate)]
    df_angle = df_angle.loc[(df_angle.index >= startdate) & (df_angle.index <= enddate)]

    mag_gse = wind_mag_gse(startdate, enddate, path)
    pol, phi_relative, pol_times = wind_polarity_preparation(mag_gse, V=400)
    mag_gse["b"] = np.linalg.norm(mag_gse[["bx_gse", "by_gse", "bz_gse"]].values, axis=1)
    mag_data_coord = "GSE"

    coverage, mu_data = calc_mu_coverage(df_angle)
    coverage = coverage.rename(columns={'PANGLE_0': "P0", 'PANGLE_1': "P1", 'PANGLE_2': "P2", 'PANGLE_3': "P3",
                                        'PANGLE_4': "P4", 'PANGLE_5': "P5", 'PANGLE_6': "P6", 'PANGLE_7': "P7"}, level=0)
    sectors = coverage.columns.levels[0].values.tolist()
    mu_times = mu_data.index

    I_data = df_angle[f"FLUX_E{en_channel}"]*1e6
    I_times = df_angle.index
    en_channel_string = meta_angle['channels_dict_df']["Bins_Text"][f"ENERGY_{en_channel}"]
    delta_E = meta_angle['channels_dict_df']["DE"][f"ENERGY_{en_channel}"]*1e-6  # MeV

    bg_I_data, bg_mu_data = get_background_data(I_data, mu_data, bg_start, bg_end)
    bg_times = bg_I_data.index

    # no uncertainties available
    count_str = None
    I_unc = None
    bg_I_unc = None

    return I_times, I_data.values, I_unc, sectors, en_channel_string, delta_E, count_str, mu_times, mu_data.values, mag_gse, pol, phi_relative, pol_times, bg_times, bg_I_data.values, bg_I_unc, bg_mu_data.values, sp_str, ch_string, mag_data_coord, coverage


def get_background_data(I_data, mu_data, backsub_start, backsub_end):
    bg_I_data = I_data.loc[(I_data.index >= backsub_start) & (I_data.index <= backsub_end)]
    bg_mu_data = mu_data.loc[(mu_data.index >= backsub_start) & (mu_data.index <= backsub_end)]
    return bg_I_data, bg_mu_data


def calc_mu_coverage(df_angle):
    opening = 22.5
    i = 0
    angle_bins = []
    col_names = []
    while True:
        if "PANGLE_{}".format(i) in df_angle.columns:
            angle_bins.append(i)
            col_names.append("PANGLE_{}".format(i))
            i += 1
        else:
            break
    df_arr = []
    mu_arr = []
    for col in col_names:
        pa = df_angle[col].values.flatten()
        pa_min = pa - opening/2
        pa_max = pa + opening/2
        cov = pd.DataFrame({'min': pa_min, 'center': pa, 'max': pa_max}, index=df_angle.index)
        mu = np.cos(np.deg2rad(pa))
        df_arr.append(cov.copy())
        mu_arr.append(mu.copy())
    keys = col_names
    coverage = pd.concat(df_arr, keys=keys, axis=1)
    coverage[coverage > 180] = 180
    coverage[coverage < 0] = 0
    mu = pd.DataFrame(data=np.column_stack(mu_arr), index=df_angle.index, columns=keys)
    return coverage, mu


def wind_mag_gse(startdate, enddate, path=None):
    if path:
        os.environ['WIND_DATA_DIR'] = path
        # pyspedas.wind.config.CONFIG['local_data_dir'] = path
    import pyspedas
    mfi_vars = pyspedas.wind.mfi(trange=[startdate.strftime('%Y-%m-%d %H:%M:%S.%f'), enddate.strftime('%Y-%m-%d %H:%M:%S.%f')], datatype="h0", notplot=True, time_clip=True)
    mag_gse = pd.DataFrame(data=mfi_vars["BGSE"]["y"], index=mfi_vars["BGSE"]["x"], columns=["bx_gse", "by_gse", "bz_gse"])
    return mag_gse


def wind_specieschannels(specieschannel):
    if specieschannel == "p 25 MeV":
        raise Exception("Not included for Wind 3DP: {}.".format(specieschannel))
    elif specieschannel == "e- 100 keV":
        species = "e"
        instrument = "3DP"
        channels = [3]
    elif specieschannel == "e- 1 MeV":
        raise Exception("Not included for Wind 3DP: {}.".format(specieschannel))
    else:
        raise ValueError("Invalid species and energy string: {}.".format(specieschannel))
    return instrument, species, channels


def wind_polarity_preparation(mag_gse, V=400):
    pos = get_horizons_coord('Wind', time={'start': mag_gse.index[0]-pd.Timedelta(minutes=15), 'stop': mag_gse.index[-1]+pd.Timedelta(minutes=15), 'step': "1min"})  # (lon, lat, radius) in (deg, deg, AU)
    pos = pos.transform_to(HeliographicStonyhurst())
    # Interpolate position data to magnetic field data cadence
    r = np.interp([t.timestamp() for t in mag_gse.index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.radius.value)
    pol, phi_relative = polarity_gse(mag_gse["bx_gse"].values, mag_gse["by_gse"].values, r, V=V)
    pol_times = mag_gse.index.values
    return pol, phi_relative, pol_times
