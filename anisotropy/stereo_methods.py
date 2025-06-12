import numpy as np
import pandas as pd
from seppy.loader.stereo import stereo_load, stereo_sept_loader
from pandas.tseries.frequencies import to_offset
from numba import njit, prange
import datetime as dt
from anisotropy.polarity_plotting import polarity_rtn  # stereo_polarity_preparation
# from pandas.tseries.frequencies import to_offset
# from my_func_py3 import angle_between
from sunpy.coordinates import get_horizons_coord, HeliographicStonyhurst
from tqdm.auto import tqdm


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def stereo_download_and_prepare(sc, instrument, startdate, enddate, path, averaging, species, en_channel, bg_start, bg_end):
    if sc == "STEREO A":
        sc = "A"
    elif sc == "STEREO B":
        sc = "B"

    if instrument == "SEPT":
        data_product = "l2"
        df_sun, df_asun, df_north, df_south, energies = sept_load_data(species, sc, data_product, startdate, enddate, path)
        df_sun = df_sun.loc[(df_sun.index >= startdate) & (df_sun.index <= enddate)]
        df_asun = df_asun.loc[(df_asun.index >= startdate) & (df_asun.index <= enddate)]
        df_north = df_north.loc[(df_north.index >= startdate) & (df_north.index <= enddate)]
        df_south = df_south.loc[(df_south.index >= startdate) & (df_south.index <= enddate)]
        df_dict = sept_prepare(df_sun, df_asun, df_north, df_south, species, averaging=averaging)
        df_dict_counts = sept_prepare_counts(df_sun, df_asun, df_north, df_south, species, averaging=averaging)
    else:
        raise Exception("Only SEPT currently implemented.")

    en_ch_df = pd.DataFrame({'energy':energies['ch_strings']})
    en_ch_df.index.names = ['channel']
    en_ch_df.to_csv(f'anisotropy/channels_STEREO_{instrument}_{species}.csv')

    mag_sc, mag_rtn = stereo_mag_preparation(startdate, enddate, sc, path)
    mag_sc = resample_mag_to_sept(df_dict["sun"].index, mag_sc, averaging=averaging)
    mag_rtn = resample_mag_to_sept(df_dict["sun"].index, mag_rtn, averaging=averaging)
    pol, phi_relative, pol_times = stereo_polarity_preparation(sc, mag_rtn, V=400)
    mag_data = mag_rtn.rename(columns={'Br': 'b_rtn_0', 'Bt': 'b_rtn_1', 'Bn': 'b_rtn_2'})
    mag_data["b"] = np.linalg.norm(mag_data[["b_rtn_0", "b_rtn_1", "b_rtn_2"]].to_numpy(), axis=1)
    mag_data_coord = "RTN"

    if instrument == "SEPT":
        coverage = calc_sept_pa_coverage(sc, species, mag_sc)
    mu_times, mu_data = mu_from_coverage(coverage)

    if species == "p":
        sp_str = "Ion"
        ch_string = 'Ions'
    elif species == "e":
        sp_str = "Electron"
        ch_string = 'Electrons'
    I_times, I_data, I_unc, sectors, en_channel_string = sept_combine_intensities(df_dict, species, en_channel, energies)
    I_unc[I_unc == 0] = np.nanmin(I_unc[I_unc > 0])  # prevent zero uncertainties as weights
    delta_E = delta_E_array(energies, en_channel)
    delta_E = delta_E.flatten()
    count_str = f"{sp_str}_Rate"
    flux_str = f"{sp_str}_Flux"

    bg_times, bg_I_data, bg_I_unc, bg_mu_data = get_background_data(I_times, I_data, I_unc, mu_times, mu_data, bg_start, bg_end)

    t_str = "seconds"
    count_arr = np.nan*np.zeros((np.shape(I_data)[0], np.shape(I_data)[1], delta_E.size))
    flux_arr = count_arr.copy()
    t_arr = count_arr.copy()
    if isinstance(en_channel, int):
        bins = en_channel
        for j, view in enumerate(["sun", "asun", "north", "south"]):
            count_arr[:, j, 0] = df_dict_counts[view][count_str][f"%s_%i" % (count_str, bins)].to_numpy()
            flux_arr[:, j, 0] = df_dict[view][flux_str][f"%s_%i" % (flux_str, bins)].to_numpy()
            t_arr[:, j, 0] = df_dict_counts[view][t_str][t_str].to_numpy()
    else:
        for i, bins in enumerate(np.arange(en_channel[0], en_channel[-1]+1)):
            for j, view in enumerate(["sun", "asun", "north", "south"]):
                count_arr[:, j, i] = df_dict_counts[view][count_str][f"%s_%i" % (count_str, bins)].to_numpy()
                flux_arr[:, j, i] = df_dict[view][flux_str][f"%s_%i" % (flux_str, bins)].to_numpy()
                t_arr[:, j, i] = df_dict_counts[view][t_str][t_str].to_numpy()
    gf_arr = np.nanmedian(count_arr/flux_arr/delta_E/t_arr, axis=0)

    mag_sc_arr = mag_sc[["Bx", "By", "Bz"]].to_numpy()

    return I_times, I_data, I_unc, sectors, en_channel_string, delta_E, count_str, mu_times, mu_data, mag_data, pol, phi_relative, pol_times, bg_times, bg_I_data, bg_I_unc, bg_mu_data, sp_str, ch_string, mag_data_coord, coverage, flux_arr, count_arr, t_arr, gf_arr, mag_sc_arr


def delta_E_array(energies, en_channel):
    if type(en_channel) is list:
        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies["DE"]
            delta_E = []
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                delta_E.append(DE[energies.index == bins])
            delta_E = np.array(delta_E)
        if len(en_channel) == 1:
            delta_E = energies["DE"][energies.index == en_channel[0]]
    else:
        delta_E = energies["DE"][energies.index == en_channel]
    return np.array(delta_E)


def get_background_data(I_times, I_data, I_unc, mu_times, mu_data, backsub_start, backsub_end):
    if len(mu_times) == len(I_times) + 1:
        mu_times = mu_times[1:]
        mu_data = mu_data[1:, :]
    bg_sun = I_data[(I_times >= backsub_start) & (I_times <= backsub_end), 0]
    bg_asun = I_data[(I_times >= backsub_start) & (I_times <= backsub_end), 1]
    bg_north = I_data[(I_times >= backsub_start) & (I_times <= backsub_end), 2]
    bg_south = I_data[(I_times >= backsub_start) & (I_times <= backsub_end), 3]
    bg_unc_sun = I_unc[(I_times >= backsub_start) & (I_times <= backsub_end), 0]
    bg_unc_asun = I_unc[(I_times >= backsub_start) & (I_times <= backsub_end), 1]
    bg_unc_north = I_unc[(I_times >= backsub_start) & (I_times <= backsub_end), 2]
    bg_unc_south = I_unc[(I_times >= backsub_start) & (I_times <= backsub_end), 3]
    bg_mu_sun = mu_data[(mu_times >= backsub_start) & (mu_times <= backsub_end), 0]
    bg_mu_asun = mu_data[(mu_times >= backsub_start) & (mu_times <= backsub_end), 1]
    bg_mu_north = mu_data[(mu_times >= backsub_start) & (mu_times <= backsub_end), 2]
    bg_mu_south = mu_data[(mu_times >= backsub_start) & (mu_times <= backsub_end), 3]
    bg_I_data = np.column_stack((bg_sun, bg_asun, bg_north, bg_south))
    bg_I_unc = np.column_stack((bg_unc_sun, bg_unc_asun, bg_unc_north, bg_unc_south))
    bg_mu_data = np.column_stack((bg_mu_sun, bg_mu_asun, bg_mu_north, bg_mu_south))
    bg_times = I_times[(I_times >= backsub_start) & (I_times <= backsub_end)]
    return bg_times, bg_I_data, bg_I_unc, bg_mu_data


def sept_combine_intensities(df_dict, species, en_channel, energies):
    df_sun = df_dict["sun"]
    df_asun = df_dict["asun"]
    df_north = df_dict["north"]
    df_south = df_dict["south"]

    if species in ['e', 'electrons']:
        en_str = energies['ch_strings']
        ch_string = 'Electrons'

    if species in ['p', 'i', 'protons', 'ions']:
        en_str = energies['ch_strings']
        ch_string = 'Ions'

    I_sun, en_channel_string = calc_av_en_flux_SEPT_new(df_sun, energies, en_channel, species)
    I_asun, en_channel_string = calc_av_en_flux_SEPT_new(df_asun, energies, en_channel, species)
    I_north, en_channel_string = calc_av_en_flux_SEPT_new(df_north, energies, en_channel, species)
    I_south, en_channel_string = calc_av_en_flux_SEPT_new(df_south, energies, en_channel, species)

    I_sun_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_SEPT_new(df_sun, energies, en_channel, species)
    I_asun_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_SEPT_new(df_asun, energies, en_channel, species)
    I_north_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_SEPT_new(df_north, energies, en_channel, species)
    I_south_uncertainty, en_channel_string = calc_av_en_flux_uncertainty_SEPT_new(df_south, energies, en_channel, species)

    try:
        I_data = np.column_stack((I_sun, I_asun, I_north, I_south))
        I_unc = np.column_stack((I_sun_uncertainty, I_asun_uncertainty, I_north_uncertainty, I_south_uncertainty))
        I_times = I_sun.index
    except:
        print("Unequal timestamps.")
        freq = "{:d}ms".format(int(10))
        I_sun.index = I_sun.index.round(freq=freq)
        I_asun.index = I_asun.index.round(freq=freq)
        I_south.index = I_south.index.round(freq=freq)
        I_north.index = I_north.index.round(freq=freq)
        N_arr = np.array([len(I_sun), len(I_asun), len(I_north), len(I_south)])
        df_arr = [I_sun, I_asun, I_north, I_south]
        idx = np.argsort(N_arr)
        I_df = pd.merge(df_arr[0], df_arr[1], how='outer', left_index=True, right_index=True, suffixes=('_a', '_b'))
        I_df = pd.merge(I_df, df_arr[2], how='outer', left_index=True, right_index=True, suffixes=('_c', '_d'))
        I_df = pd.merge(I_df, df_arr[3], how='outer', left_index=True, right_index=True, suffixes=('_e', '_f'))
        I_data = I_df.to_numpy()
        I_times = I_df.index
        df_arr = [I_sun_uncertainty, I_asun_uncertainty, I_north_uncertainty, I_south_uncertainty]
        I_df = pd.merge(df_arr[0], df_arr[1], how='outer', left_index=True, right_index=True, suffixes=('_a', '_b'))
        I_df = pd.merge(I_df, df_arr[2], how='outer', left_index=True, right_index=True, suffixes=('_c', '_d'))
        I_df = pd.merge(I_df, df_arr[3], how='outer', left_index=True, right_index=True, suffixes=('_e', '_f'))
        I_unc = I_df.to_numpy()
    sectors = ["sun", "asun", "north", "south"]
    return I_times, I_data, I_unc, sectors, en_channel_string


def calc_av_en_flux_SEPT_new(df, energies, en_channel, species):  # original from Nina Slack Feb 9, 2022, rewritten Jan Apr 8, 2022
    """This function averages the flux of several energy channels of HET into a combined energy channel
    channel numbers counted from 0

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """
    if species.lower() in ['e', 'electrons']:
        en_str = energies['ch_strings']
        bins_width = 'DE'
        flux_key = 'Electron_Flux'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        en_str = energies['ch_strings']
        bins_width = 'DE'
        flux_key = 'Ion_Flux'

    try:
        if type(en_channel) is list:
            energy_low = en_str[en_channel[0]].split('-')[0]
            energy_up = en_str[en_channel[-1]].split('-')[-1]

            en_channel_string = energy_low + ' - ' + energy_up

            if len(en_channel) > 2:
                raise Exception('en_channel must have len 2 or less!')
            if len(en_channel) == 2:
                DE = energies[bins_width]
                try:
                    df = df[flux_key]
                except (AttributeError, KeyError):
                    None
                for bins in np.arange(en_channel[0], en_channel[-1]+1):
                    if bins == en_channel[0]:
                        I_all = df[f'ch_{bins}'] * DE.loc[bins]
                    else:
                        I_all = I_all + df[f'ch_{bins}'] * DE.loc[bins]
                DE_total = np.sum(DE.loc[en_channel[0]:en_channel[-1]])
                flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
            else:
                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'ch_{en_channel}']}, index=df.index)
                en_channel_string = en_str[en_channel]
        else:
            en_channel_string = en_str[en_channel]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'ch_{en_channel}']}, index=df.index)
    except:
        if type(en_channel) is list:
            energy_low = en_str[en_channel[0]].split('-')[0]
            energy_up = en_str[en_channel[-1]].split('-')[-1]

            en_channel_string = energy_low + ' - ' + energy_up

            if len(en_channel) > 2:
                raise Exception('en_channel must have len 2 or less!')
            if len(en_channel) == 2:
                DE = energies[bins_width]
                try:
                    df = df[flux_key]
                except (AttributeError, KeyError):
                    None
                for bins in np.arange(en_channel[0], en_channel[-1]+1):
                    if bins == en_channel[0]:
                        I_all = df[f'{flux_key}_{bins}'] * DE.loc[bins]
                    else:
                        I_all = I_all + df[f'{flux_key}_{bins}'] * DE.loc[bins]
                DE_total = np.sum(DE.loc[en_channel[0]:en_channel[-1]])
                flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
            else:
                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
                en_channel_string = en_str[en_channel]
        else:
            en_channel_string = en_str[en_channel]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
    return flux_out, en_channel_string


def calc_av_en_flux_uncertainty_SEPT_new(df, energies, en_channel, species):  # Laura edited from the function above
    """This function averages the uncertainty of several energy channels of HET into a combined energy channel
    channel numbers counted from 0

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """
    if species.lower() in ['e', 'electrons']:
        en_str = energies['ch_strings']
        bins_width = 'DE'
        flux_key = 'Electron_Uncertainty'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        en_str = energies['ch_strings']
        bins_width = 'DE'
        flux_key = 'Ion_Uncertainty'

    try:
        if type(en_channel) is list:
            energy_low = en_str[en_channel[0]].split('-')[0]
            energy_up = en_str[en_channel[-1]].split('-')[-1]

            en_channel_string = energy_low + ' - ' + energy_up

            if len(en_channel) > 2:
                raise Exception('en_channel must have len 2 or less!')
            if len(en_channel) == 2:
                DE = energies[bins_width]
                try:
                    df = df[flux_key]
                except (AttributeError, KeyError):
                    None
                for bins in np.arange(en_channel[0], en_channel[-1]+1):
                    if bins == en_channel[0]:
                        I_all = (df[f'err_ch_{bins}'] * DE.loc[bins])**2
                    else:
                        I_all = I_all + (df[f'err_ch_{bins}'] * DE.loc[bins])**2
                DE_total = np.sum(DE.loc[en_channel[0]:en_channel[-1]])
                flux_out = pd.DataFrame({'flux': np.sqrt(I_all)/DE_total}, index=df.index)
            else:
                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'err_ch_{en_channel}']}, index=df.index)
                en_channel_string = en_str[en_channel]
        else:
            en_channel_string = en_str[en_channel]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'err_ch_{en_channel}']}, index=df.index)
    except:
        if type(en_channel) is list:
            energy_low = en_str[en_channel[0]].split('-')[0]

            energy_up = en_str[en_channel[-1]].split('-')[-1]

            en_channel_string = energy_low + ' - ' + energy_up

            if len(en_channel) > 2:
                raise Exception('en_channel must have len 2 or less!')
            if len(en_channel) == 2:
                DE = energies[bins_width]
                try:
                    df = df[flux_key]
                except (AttributeError, KeyError):
                    None
                for bins in np.arange(en_channel[0], en_channel[-1]+1):
                    if bins == en_channel[0]:
                        I_all = (df[f'{flux_key}_{bins}'] * DE.loc[bins])**2
                    else:
                        I_all = I_all + (df[f'{flux_key}_{bins}'] * DE.loc[bins])**2
                DE_total = np.sum(DE.loc[en_channel[0]:en_channel[-1]])
                flux_out = pd.DataFrame({'flux': np.sqrt(I_all)/DE_total}, index=df.index)
            else:
                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
                en_channel_string = en_str[en_channel]
        else:
            en_channel_string = en_str[en_channel]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}'][f'{flux_key}_{en_channel}']}, index=df.index)
    return flux_out, en_channel_string


def mu_from_coverage(coverage):
    mu_times = coverage.index
    mu_sun = np.cos(np.deg2rad(coverage['sun'].center.to_numpy()))
    mu_asun = np.cos(np.deg2rad(coverage['asun'].center.to_numpy()))
    mu_north = np.cos(np.deg2rad(coverage['north'].center.to_numpy()))
    mu_south = np.cos(np.deg2rad(coverage['south'].center.to_numpy()))
    mu_data = np.column_stack((mu_sun, mu_asun, mu_north, mu_south))
    return mu_times, mu_data


def telescope_pointing(sc, instrument):
    if instrument.lower() == "sept":
        if sc == "A":
            pointing_sun = np.array([-0.70710678, 0., -0.70710678])
            pointing_asun = np.array([0.70710678, 0., 0.70710678])
            pointing_north = np.array([0., -1., 0.])
            pointing_south = np.array([0., 1., 0.])
        elif sc == "B":
            pointing_sun = np.array([-0.70710678, 0., 0.70710678])
            pointing_asun = np.array([0.70710678, 0., -0.70710678])
            pointing_north = np.array([0., 1., 0.])
            pointing_south = np.array([0., -1., 0.])
        else:
            print("Input proper string for spacecraft: A or B.")
        pointing_arr = np.vstack((pointing_sun, pointing_asun, pointing_north, pointing_south))
    else:
        raise Exception("Only SEPT implemented.")
    return pointing_arr


def calc_sept_pa_coverage(sc, species, mag_data):
    print('Calculating PA coverage.')
    if species in ['e', 'ele', 'electron']:
        opening = 52.8
    elif species in ['ion', 'p', 'i', 'proton']:
        opening = 52.
    else:
        print("Input proper string for species: e/ele/electron for electrons or ion for ions.")

    # Make sure input data is in SC coordinates
    mag_vec = np.array([mag_data.Bx.to_numpy(), mag_data.By.to_numpy(), mag_data.Bz.to_numpy()])

    # pointing directions of SEPT in XYZ/SC coordinates (!) (arrows point into the sensor)
    pointing_arr = telescope_pointing(sc, "sept")
    pointing_sun = pointing_arr[0, :]
    pointing_asun = pointing_arr[1, :]
    pointing_north = pointing_arr[2, :]
    pointing_south = pointing_arr[3, :]

    pa_sun = np.ones(len(mag_data.Bx.to_numpy())) * np.nan
    pa_asun = np.ones(len(mag_data.Bx.to_numpy())) * np.nan
    pa_north = np.ones(len(mag_data.Bx.to_numpy())) * np.nan
    pa_south = np.ones(len(mag_data.Bx.to_numpy())) * np.nan

    for i in tqdm(range(len(mag_data.Bx.to_numpy()))):
        pa_sun[i] = np.rad2deg(angle_between(pointing_sun, mag_vec[:, i]))
        pa_asun[i] = np.rad2deg(angle_between(pointing_asun, mag_vec[:, i]))
        pa_north[i] = np.rad2deg(angle_between(pointing_north, mag_vec[:, i]))
        pa_south[i] = np.rad2deg(angle_between(pointing_south, mag_vec[:, i]))

    sun_min = pa_sun - opening/2
    sun_max = pa_sun + opening/2
    asun_min = pa_asun - opening/2
    asun_max = pa_asun + opening/2
    north_min = pa_north - opening/2
    north_max = pa_north + opening/2
    south_min = pa_south - opening/2
    south_max = pa_south + opening/2
    cov_sun = pd.DataFrame({'min': sun_min, 'center': pa_sun, 'max': sun_max}, index=mag_data.index)
    cov_asun = pd.DataFrame({'min': asun_min, 'center': pa_asun, 'max': asun_max}, index=mag_data.index)
    cov_north = pd.DataFrame({'min': north_min, 'center': pa_north, 'max': north_max}, index=mag_data.index)
    cov_south = pd.DataFrame({'min': south_min, 'center': pa_south, 'max': south_max}, index=mag_data.index)
    keys = [('sun'), ('asun'), ('north'), ('south')]
    coverage = pd.concat([cov_sun, cov_asun, cov_north, cov_south], keys=keys, axis=1)

    coverage[coverage > 180] = 180
    coverage[coverage < 0] = 0
    return coverage


def stereo_mag_preparation(startdate, enddate, sc, path, averaging=None):
    # downloadpath = f'{path}l1/mag/'

    # MAG for PA coverage calculations
    print('Loading MAG...')
    msdate = dt.datetime.combine(startdate.date(), dt.time.min)
    medate = dt.datetime.combine(enddate.date() + dt.timedelta(days=1), dt.time.min)
    enddate_sept = dt.datetime.combine(enddate.date(), dt.time.max)
    if sc == "A":
        sc_str = "ahead"
    elif sc == "B":
        sc_str = "behind"

    if averaging is None:
        averaging = "1s"  # make faster
    mag_sc, metadata_sc = stereo_load('MAG', msdate, medate, mag_coord='SC', spacecraft=sc_str, path=path, resample=averaging)
    mag_sc = mag_sc[['BFIELD_0', 'BFIELD_1', 'BFIELD_2', 'BFIELD_3']].rename({'BFIELD_0': 'Bx', 'BFIELD_1': 'By', 'BFIELD_2': 'Bz', 'BFIELD_3': 'B'}, axis=1)

    # L1 MAG RTN data set
    # downloadpath = f'{path}l1/mag/'
    mag_rtn, metadata_rtn = stereo_load('MAG', msdate, medate, mag_coord='RTN', spacecraft=sc_str, path=path, resample=averaging)
    mag_rtn = mag_rtn.rename({'BFIELD_0': 'Br', 'BFIELD_1': 'Bt', 'BFIELD_2': 'Bn', 'BFIELD_3': 'B'}, axis=1)
    return mag_sc, mag_rtn


@njit
def calc_resample_mag_to_sept(timestamps, mag_times, mag_data_arr, tlimits):
    arr = np.nan*np.zeros((len(timestamps), np.shape(mag_data_arr)[1]))
    for i in prange(len(timestamps)):
        for j in prange(np.shape(mag_data_arr)[1]):
            arr[i, j] = np.nanmean(mag_data_arr[(mag_times >= tlimits[i, 0]) & (mag_times < tlimits[i, 1]), j])
    return arr


def resample_mag_to_sept(datetimes, df_mag, averaging, pos_timestamp="center"):
    if averaging is None:
        # For STEREO SEPT cadence is 1 minute
        averaging = "1min"
    timestamps = np.array([t.timestamp() for t in datetimes])
    if pos_timestamp == "center":
        tlimits = np.column_stack([[t.timestamp() for t in datetimes-pd.Timedelta(averaging)/2], [t.timestamp() for t in datetimes+pd.Timedelta(averaging)/2]])
    elif pos_timestamp == "left":
        tlimits = np.column_stack([[t.timestamp() for t in datetimes], [t.timestamp() for t in datetimes+pd.Timedelta(averaging)]])
    elif pos_timestamp == "right":
        tlimits = np.column_stack([[t.timestamp() for t in datetimes-pd.Timedelta(averaging)], [t.timestamp() for t in datetimes]])
    mag_times = np.array([t.timestamp() for t in df_mag.index])
    arr = calc_resample_mag_to_sept(timestamps, mag_times, df_mag.to_numpy(), tlimits)
    df_mag_resampled = pd.DataFrame(index=datetimes, data=arr, columns=df_mag.columns)
    return df_mag_resampled


def stereo_specieschannels(specieschannel):
    if specieschannel == "p 25 MeV":
        species = "p"
        instrument = "HET"
        channels = [5, 8]
    elif specieschannel == "e- 100 keV":
        species = "e"
        instrument = "EPT"
        channels = [6, 7]
    elif specieschannel == "e- 1 MeV":
        species = "e"
        instrument = "HET"
        channels = [0, 1]
    else:
        raise ValueError("Invalid species and energy string: {}.".format(specieschannel))
    return instrument, species, channels


def sept_load_data(species, sc, data_product, startdate, enddate, downloadpath):
    instrument = "SEPT"
    if sc.upper() == "A":
        sc_str = "ahead"
    elif sc.upper() == "B":
        sc_str = "behind"

    df_sun, energies = stereo_sept_loader(startdate=startdate,
                                          enddate=enddate+pd.Timedelta(1, unit="day"),
                                          spacecraft=sc_str,
                                          species=species,
                                          viewing="sun",
                                          path=downloadpath,
                                          all_columns=True,
                                          pos_timestamp="center")
    df_sun = df_sun.drop(columns=['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec'])
    df_asun, energies = stereo_sept_loader(startdate=startdate,
                                           enddate=enddate+pd.Timedelta(1, unit="day"),
                                           spacecraft=sc_str,
                                           species=species,
                                           viewing="asun",
                                           path=downloadpath,
                                           all_columns=True,
                                           pos_timestamp="center")
    df_asun = df_asun.drop(columns=['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec'])
    df_north, energies = stereo_sept_loader(startdate=startdate,
                                            enddate=enddate+pd.Timedelta(1, unit="day"),
                                            spacecraft=sc_str,
                                            species=species,
                                            viewing="north",
                                            path=downloadpath,
                                            all_columns=True,
                                            pos_timestamp="center")
    df_north = df_north.drop(columns=['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec'])
    df_south, energies = stereo_sept_loader(startdate=startdate,
                                            enddate=enddate+pd.Timedelta(1, unit="day"),
                                            spacecraft=sc_str,
                                            species=species,
                                            viewing="south",
                                            path=downloadpath,
                                            all_columns=True,
                                            pos_timestamp="center")
    df_south = df_south.drop(columns=['julian_date', 'year', 'frac_doy', 'hour', 'min', 'sec'])
    return df_sun, df_asun, df_north, df_south, energies


def sept_prepare_counts(df_sun, df_asun, df_north, df_south, species, averaging=None):
    df_sun_orig = df_sun.copy()
    df_asun_orig = df_asun.copy()
    df_north_orig = df_north.copy()
    df_south_orig = df_south.copy()

    df_sun = df_sun.drop(columns=['integration_time'])
    df_asun = df_asun.drop(columns=['integration_time'])
    df_north = df_north.drop(columns=['integration_time'])
    df_south = df_south.drop(columns=['integration_time'])

    df_sun_new = change_count_df_format(df_sun, species)
    df_asun_new = change_count_df_format(df_asun, species)
    df_north_new = change_count_df_format(df_north, species)
    df_south_new = change_count_df_format(df_south, species)

    df_sun = change_flux_df_format(df_sun, species)
    df_asun = change_flux_df_format(df_asun, species)
    df_north = change_flux_df_format(df_north, species)
    df_south = change_flux_df_format(df_south, species)

    df_sun = pd.merge(df_sun, df_sun_new, left_index=True, right_index=True, how='outer')
    df_asun = pd.merge(df_asun, df_asun_new, left_index=True, right_index=True, how='outer')
    df_north = pd.merge(df_north, df_north_new, left_index=True, right_index=True, how='outer')
    df_south = pd.merge(df_south, df_south_new, left_index=True, right_index=True, how='outer')

    df_sun = pd.merge(df_sun, pd.DataFrame(data=df_sun_orig['integration_time'].to_numpy(), columns=pd.MultiIndex.from_product([['seconds'], ['seconds']]), index=df_sun.index), left_index=True, right_index=True, how='outer')
    df_asun = pd.merge(df_asun, pd.DataFrame(data=df_asun_orig['integration_time'].to_numpy(), columns=pd.MultiIndex.from_product([['seconds'], ['seconds']]), index=df_asun.index), left_index=True, right_index=True, how='outer')
    df_north = pd.merge(df_north, pd.DataFrame(data=df_north_orig['integration_time'].to_numpy(), columns=pd.MultiIndex.from_product([['seconds'], ['seconds']]), index=df_north.index), left_index=True, right_index=True, how='outer')
    df_south = pd.merge(df_south, pd.DataFrame(data=df_south_orig['integration_time'].to_numpy(), columns=pd.MultiIndex.from_product([['seconds'], ['seconds']]), index=df_south.index), left_index=True, right_index=True, how='outer')

    __echannels__ = {'bins': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                     'ch_strings': ['45.0-55.0 keV', '55.0-65.0 keV',  '65.0-75.0 keV', '75.0-85.0 keV',
                                    '85.0-105.0 keV', '105.0-125.0 keV', '125.0-145.0 keV',
                                    '145.0-165.0 keV', '165.0-195.0 keV', '195.0-225.0 keV',
                                    '225.0-255.0 keV', '255.0-295.0 keV', '295.0-335.0 keV',
                                    '335.0-375.0 keV', '375.0-425.0 keV'],
                     'DE': [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02,
                            0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05],
                     'gf': [0.089, 0.095, 0.101, 0.101, 0.106, 0.108, 0.113, 0.109,
                            0.110, 0.114, 0.112, 0.113, 0.095, 0.074, 0.054]}
    # specific constants for the magnet/ion telescope; "DE" in MeV; "gf" in cm^2 sr
    __pchannels__ = {'bins': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                     'ch_strings': ['84.1-92.7 keV', '92.7-101.3 keV', '101.3-110.0 keV',
                                    '110.0-118.6 keV', '118.6-137.0 keV', '137.0-155.8 keV',
                                    '155.8-174.6 keV', '174.6-192.6 keV', '192.6-219.5 keV',
                                    '219.5-246.4 keV', '246.4-273.4 keV', ' 273.4-312.0 keV',
                                    '312.0-350.7 keV', '350.7-389.5 keV', '389.5-438.1 keV',
                                    '438.1-496.4 keV', '496.4-554.8 keV', ' 554.8-622.9 keV',
                                    '622.9-700.7 keV', '700.7-788.3 keV', '788.3-875.8 keV',
                                    '875.8- 982.8 keV', '982.8-1111.9 keV', '1111.9-1250.8 keV',
                                    '1250.8-1399.7 keV', '1399.7-1578.4 keV', '1578.4-1767.0 keV',
                                    '1767.0-1985.3 keV', '1985.3-2223.6 keV', '2223.6-6500.0 keV'],
                     'DE': [0.0086, 0.0086, 0.0087, 0.0086, 0.0184, 0.0188, 0.0188, 0.018, 0.0269, 0.0269, 0.027,
                            0.0386, 0.0387, 0.0388, 0.0486, 0.0583, 0.0584, 0.0681, 0.0778, 0.0876, 0.0875, 0.107,
                            0.1291, 0.1389, 0.1489, 0.1787, 0.1886, 0.2183, 0.2383, 4.2764],
                     'gf': [0.17]*len(range(2, 32))}

    # counts
    if species in ["p", "i"]:
        gf = np.array(__pchannels__["gf"])
        deltaE = np.array(__pchannels__["DE"])
        df_sun["Ion_Rate"] = df_sun["Ion_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_sun["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        df_asun["Ion_Rate"] = df_asun["Ion_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_asun["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        df_north["Ion_Rate"] = df_north["Ion_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_north["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        df_south["Ion_Rate"] = df_south["Ion_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_south["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        # df_sun["Ion_Rate"] = (df_sun["Ion_Flux"].to_numpy()/df_sun["Ion_Uncertainty"].to_numpy())**2
        # df_sun.loc[df_sun["Ion_Uncertainty"]==0, "Ion_Rate"] = 0
        # df_asun["Ion_Rate"] = (df_asun["Ion_Flux"].to_numpy()/df_asun["Ion_Uncertainty"].to_numpy())**2
        # df_asun.loc[df_asun["Ion_Uncertainty"]==0, "Ion_Rate"] = 0
        # df_south["Ion_Rate"] = (df_south["Ion_Flux"].to_numpy()/df_south["Ion_Uncertainty"].to_numpy())**2
        # df_south.loc[df_south["Ion_Uncertainty"]==0, "Ion_Rate"] = 0
        # df_north["Ion_Rate"] = (df_north["Ion_Flux"].to_numpy()/df_north["Ion_Uncertainty"].to_numpy())**2
        # df_north.loc[df_north["Ion_Uncertainty"]==0, "Ion_Rate"] = 0

    elif species == "e":
        gf = np.array(__echannels__["gf"])
        deltaE = np.array(__echannels__["DE"])
        df_sun["Electron_Rate"] = df_sun["Electron_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_sun["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        df_asun["Electron_Rate"] = df_asun["Electron_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_asun["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        df_north["Electron_Rate"] = df_north["Electron_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_north["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        df_south["Electron_Rate"] = df_south["Electron_Rate"].copy().to_numpy()*gf[np.newaxis, :]*deltaE[np.newaxis, :]*df_south["seconds"]["seconds"].to_numpy()[:, np.newaxis]
        # df_sun["Electron_Rate"] = (df_sun["Electron_Flux"].to_numpy()/df_sun["Electron_Uncertainty"].to_numpy())**2
        # df_sun.loc[df_sun["Electron_Uncertainty"]==0, "Electron_Rate"] = 0
        # df_asun["Electron_Rate"] = (df_asun["Electron_Flux"].to_numpy()/df_asun["Electron_Uncertainty"].to_numpy())**2
        # df_asun.loc[df_asun["Electron_Uncertainty"]==0, "Electron_Rate"] = 0
        # df_south["Electron_Rate"] = (df_south["Electron_Flux"].to_numpy()/df_south["Electron_Uncertainty"].to_numpy())**2
        # df_south.loc[df_south["Electron_Uncertainty"]==0, "Electron_Rate"] = 0
        # df_north["Electron_Rate"] = (df_north["Electron_Flux"].to_numpy()/df_north["Electron_Uncertainty"].to_numpy())**2
        # df_north.loc[df_north["Electron_Uncertainty"]==0, "Electron_Rate"] = 0

    if averaging is not None:
        split_strings = averaging.split("min")
        if len(split_strings) == 1:
            split_strings = averaging.split("s")
            if len(split_strings) == 2:
                av_min = float(split_strings[0])/60
            else:
                print("Invalid averaging.")
                exit()
        elif len(split_strings) == 2:
            av_min = float(split_strings[0])
        else:
            print("Invalid averaging.")
            exit()
        df_sun = df_sun.resample(averaging, label='left').sum()
        df_sun.index = df_sun.index + to_offset(pd.Timedelta(averaging)/2)
        df_asun = df_asun.resample(averaging, label='left').sum()
        df_north = df_north.resample(averaging, label='left').sum()
        df_south = df_south.resample(averaging, label='left').sum()
        df_asun.index = df_asun.index + to_offset(pd.Timedelta(averaging)/2)
        df_north.index = df_north.index + to_offset(pd.Timedelta(averaging)/2)
        df_south.index = df_south.index + to_offset(pd.Timedelta(averaging)/2)
    return {"sun": df_sun, "asun": df_asun, "north": df_north, "south": df_south}


def sept_prepare(df_sun, df_asun, df_north, df_south, species, averaging=None):
    df_sun = change_flux_df_format(df_sun, species)
    df_asun = change_flux_df_format(df_asun, species)
    df_north = change_flux_df_format(df_north, species)
    df_south = change_flux_df_format(df_south, species)

    if averaging is not None:
        split_strings = averaging.split("min")
        if len(split_strings) == 1:
            split_strings = averaging.split("s")
            if len(split_strings) == 2:
                av_min = float(split_strings[0])/60
            else:
                print("Invalid averaging.")
                exit()
        elif len(split_strings) == 2:
            av_min = float(split_strings[0])
        else:
            print("Invalid averaging.")
            exit()
        if species in ["i", "p"]:
            ion_str = "Ion"
        elif species == "e":
            ion_str = "Electron"
        else:
            print("Invalid species. Choose p/e.")
        # Prepare uncertainties for correct resampling
        delta_t = np.mean(np.diff(df_sun.index))/np.timedelta64(1, 's')
        N = np.round(av_min*60/delta_t)
        df_sun[f"{ion_str}_Uncertainty"] = df_sun[f"{ion_str}_Uncertainty"].copy().to_numpy()**2
        df_asun[f"{ion_str}_Uncertainty"] = df_asun[f"{ion_str}_Uncertainty"].copy().to_numpy()**2
        df_north[f"{ion_str}_Uncertainty"] = df_north[f"{ion_str}_Uncertainty"].copy().to_numpy()**2
        df_south[f"{ion_str}_Uncertainty"] = df_south[f"{ion_str}_Uncertainty"].copy().to_numpy()**2

        df_sun = df_sun.resample(averaging, label='left').mean()
        df_sun.index = df_sun.index + to_offset(pd.Timedelta(averaging)/2)
        df_asun = df_asun.resample(averaging, label='left').mean()
        df_north = df_north.resample(averaging, label='left').mean()
        df_south = df_south.resample(averaging, label='left').mean()
        df_asun.index = df_asun.index + to_offset(pd.Timedelta(averaging)/2)
        df_north.index = df_north.index + to_offset(pd.Timedelta(averaging)/2)
        df_south_index = df_south.index + to_offset(pd.Timedelta(averaging)/2)

        df_sun[f"{ion_str}_Uncertainty"] = np.sqrt(df_sun[f"{ion_str}_Uncertainty"].copy().to_numpy()/N)
        df_asun[f"{ion_str}_Uncertainty"] = np.sqrt(df_asun[f"{ion_str}_Uncertainty"].copy().to_numpy()/N)
        df_north[f"{ion_str}_Uncertainty"]= np.sqrt(df_north[f"{ion_str}_Uncertainty"].copy().to_numpy()/N)
        df_south[f"{ion_str}_Uncertainty"] = np.sqrt(df_south[f"{ion_str}_Uncertainty"].copy().to_numpy()/N)

    return {"sun": df_sun, "asun": df_asun, "north": df_north, "south": df_south}


def change_count_df_format(df, species):
    if species in ["i", "p"]:
        Rate_sigma_p_channels = [col for col in df.columns if 'err' in col]
        Rate_p_channels = [col for col in df.columns if 'err' not in col]
        df_new = pd.concat(
                    [df[Rate_p_channels], df[Rate_sigma_p_channels]],
                    axis=1, keys=['Ion_Rate', 'Ion_Uncertainty'])
        for str in df_new.columns:
            if str[0] == "Ion_Rate":
                df_new = df_new.rename({str[1]: str[1].replace("ch", "Ion_Rate")}, axis=1)
            elif str[0] == "Ion_Uncertainty":
                df_new = df_new.rename({str[1]: str[1].replace("err_ch", "Ion_Uncertainty")}, axis=1)
        df_new = df_new.drop(columns="Ion_Uncertainty", level=0)
    elif species == "e":
        Rate_sigma_e_channels = [col for col in df.columns if 'err' in col]
        Rate_e_channels = [col for col in df.columns if 'err' not in col]
        df_new = pd.concat(
                    [df[Rate_e_channels], df[Rate_sigma_e_channels]],
                    axis=1, keys=['Electron_Rate', 'Electron_Uncertainty'])
        for str in df_new.columns:
            if str[0] == "Electron_Rate":
                df_new = df_new.rename({str[1]: str[1].replace("ch", "Electron_Rate")}, axis=1)
            elif str[0] == "Electron_Uncertainty":
                df_new = df_new.rename({str[1]: str[1].replace("err_ch", "Electron_Uncertainty")}, axis=1)
        df_new = df_new.drop(columns="Electron_Uncertainty", level=0)
    return df_new


def change_flux_df_format(df, species):
    if species in ["i", "p"]:
        flux_sigma_p_channels = [col for col in df.columns if 'err' in col]
        flux_p_channels = [col for col in df.columns if 'err' not in col]
        df_new = pd.concat(
                    [df[flux_p_channels], df[flux_sigma_p_channels]],
                    axis=1, keys=['Ion_Flux', 'Ion_Uncertainty'])
        for str in df_new.columns:
            if str[0] == "Ion_Flux":
                df_new = df_new.rename({str[1]: str[1].replace("ch", "Ion_Flux")}, axis=1)
            elif str[0] == "Ion_Uncertainty":
                df_new = df_new.rename({str[1]: str[1].replace("err_ch", "Ion_Uncertainty")}, axis=1)
    elif species == "e":
        flux_sigma_e_channels = [col for col in df.columns if 'err' in col]
        flux_e_channels = [col for col in df.columns if 'err' not in col]
        df_new = pd.concat(
                    [df[flux_e_channels], df[flux_sigma_e_channels]],
                    axis=1, keys=['Electron_Flux', 'Electron_Uncertainty'])
        for str in df_new.columns:
            if str[0] == "Electron_Flux":
                df_new = df_new.rename({str[1]: str[1].replace("ch", "Electron_Flux")}, axis=1)
            elif str[0] == "Electron_Uncertainty":
                df_new = df_new.rename({str[1]: str[1].replace("err_ch", "Electron_Uncertainty")}, axis=1)
    return df_new


def stereo_polarity_preparation(sc, mag_rtn, V=400):
    pos = get_horizons_coord('STEREO-'+sc, time={'start': mag_rtn.index[0]-pd.Timedelta(minutes=15), 'stop': mag_rtn.index[-1]+pd.Timedelta(minutes=15), 'step': "1min"})  # (lon, lat, radius) in (deg, deg, AU)
    pos = pos.transform_to(HeliographicStonyhurst())
    # Interpolate position data to magnetic field data cadence
    r = np.interp([t.timestamp() for t in mag_rtn.index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.radius.value)
    lat = np.interp([t.timestamp() for t in mag_rtn.index], [t.timestamp() for t in pd.to_datetime(pos.obstime.value)], pos.lat.value)
    pol, phi_relative = polarity_rtn(mag_rtn.Br.values, mag_rtn.Bt.values, mag_rtn.Bn.values, r, lat, V=V)
    pol_times = mag_rtn.index.values
    return pol, phi_relative, pol_times
