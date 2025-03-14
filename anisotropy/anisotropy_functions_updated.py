import numpy as np
import pandas as pd
from scipy.stats import poisson
from lmfit import Model, conf_interval2d
# from numba import njit


def bootstrap_anisotropy_adjacent(I_data,bg_fit,bg_fit_unc,mu_data,weights,delta_E,count_arr,t_arr,gf_arr,coverage,mag_sc,n_boot=1000,n_adj=1):
    rng = np.random.default_rng()
    n_times = np.shape(I_data)[0]
    n_tele = np.shape(I_data)[1]
    Ani_bootres = np.nan*np.zeros((n_times,5))
    Ani_bgsub_bootres = np.nan*np.zeros((n_times,5))
    new_coverage = coverage.copy()
    n_adj_arr = np.zeros((n_times,2))
    for i in range(n_times):
        b0_vec = mag_sc[i,:]
        I_boot_arr = np.nan*np.zeros((n_tele*(1+2*n_adj),n_boot))
        bg_boot_arr = np.nan*np.zeros((n_tele*(1+2*n_adj),n_boot))
        mu_arr = np.nan*np.zeros((1+2*n_adj)*n_tele)
        I_arr = np.nan*np.zeros((1+2*n_adj)*n_tele)
        w_arr = np.nan*np.zeros((1+2*n_adj)*n_tele)
        co = 0
        R_arr = rng.uniform(0,1,((n_boot,delta_E.size,n_tele*(1+2*n_adj))))
        for k in range(0,n_adj+1):
            if i+k >= np.shape(mu_data)[0]:
                continue
            if np.dot(b0_vec, mag_sc[i+k,:])>0:
                for j in range(n_tele):
                    I_boot_arr[co,:] = bootstrap_intensities(R_arr[:,:,co],count_arr[i+k,j,:],t_arr[i+k,j,:],delta_E,gf_arr[j,:])
                    bg_boot_arr[co,:] = bg_fit[i+k,j] + bg_fit_unc[i+k,j]*rng.standard_normal(n_boot)
                    mu_arr[co] = mu_data[i+k,j]
                    I_arr[co] = I_data[i+k,j]
                    w_arr[co] = weights[i+k,j]
                    co += 1
                n_adj_arr[i,1] = k
        for k in range(1,n_adj+1):
            if i-k < 0:
                continue
            if np.dot(b0_vec, mag_sc[i-k,:])>0:
                for j in range(n_tele):
                    I_boot_arr[co,:] = bootstrap_intensities(R_arr[:,:,co],count_arr[i-k,j,:],t_arr[i-k,j,:],delta_E,gf_arr[j,:])
                    bg_boot_arr[co,:] = bg_fit[i-k,j] + bg_fit_unc[i-k,j]*rng.standard_normal(n_boot)
                    mu_arr[co] = mu_data[i-k,j]
                    I_arr[co] = I_data[i-k,j]
                    w_arr[co] = weights[i-k,j]
                    co += 1
                n_adj_arr[i,0] = k
        if np.isnan(mu_arr).all():
            continue
        I_boot_arr = I_boot_arr[0:co,:]
        bg_boot_arr = bg_boot_arr[0:co,:]
        I_boot_sub_arr = I_boot_arr - bg_boot_arr
        I_boot_sub_arr[I_boot_sub_arr<0] = 0  # not doing this keeps the relative intensities ok
        try:
            mu_arr = mu_arr[0:co]
            I_arr = I_arr[0:co]
            w_arr = w_arr[0:co]
            Ani_boot = 3*(np.sum(I_boot_arr*mu_arr[:,np.newaxis]*w_arr[:,np.newaxis], axis=0)*1./np.sum(I_boot_arr*w_arr[:,np.newaxis], axis=0))
            Ani_boot[np.all(I_boot_arr==0, axis=0)] = 0
            Ani_boot = check_mu_sum(I_arr,mu_arr,Ani_boot)
            Ani_bootres[i,0] = np.mean(Ani_boot)
            Ani_bootres[i,1] = np.median(Ani_boot)
            Ani_bootres[i,2] = np.percentile(Ani_boot,2.5)
            Ani_bootres[i,3] = np.percentile(Ani_boot,97.5)
            Ani_bootres[i,4] = np.std(Ani_boot)
            Ani_bgsub_boot = 3*(np.sum(I_boot_sub_arr*mu_arr[:,np.newaxis]*w_arr[:,np.newaxis], axis=0)*1./np.sum(I_boot_sub_arr*w_arr[:,np.newaxis], axis=0))
            Ani_bgsub_boot[np.all(I_boot_sub_arr==0, axis=0)] = 0
            Ani_bgsub_boot = check_mu_sum(I_arr,mu_arr,Ani_bgsub_boot)
            Ani_bgsub_bootres[i,0] = np.mean(Ani_bgsub_boot)
            Ani_bgsub_bootres[i,1] = np.median(Ani_bgsub_boot)
            Ani_bgsub_bootres[i,2] = np.percentile(Ani_bgsub_boot,2.5)
            Ani_bgsub_bootres[i,3] = np.percentile(Ani_bgsub_boot,97.5)
            Ani_bgsub_bootres[i,4] = np.std(Ani_bgsub_boot)
        except:
            continue
    return Ani_bootres, Ani_bgsub_bootres, n_adj_arr


def bootstrap_intensities(R_arr,count_arr,t_arr,delta_E,gf_arr):
    new_I_arr = np.nan*R_arr
    for i in range(delta_E.size):
        if delta_E.size == 1:
            dE = delta_E
        else:
            dE = delta_E[i]
        count = count_arr[i]
        new_I_arr[:,i] = poisson.ppf(R_arr[:,i],mu=count)/t_arr[i]/gf_arr[i]/dE
    I_all = np.zeros(len(R_arr[:,0]))
    if delta_E.size == 1:
        I_comb = new_I_arr.flatten()
    else:
        for j in range(delta_E.size):
            I_all += new_I_arr[:,j]*delta_E[j]
        DE_total = np.sum(delta_E)
        I_comb = I_all/DE_total
    return I_comb


def bootstrap_anisotropy(I_data,bg_fit,bg_fit_unc,mu_data,weights,delta_E,count_arr,t_arr,gf_arr,n_boot=1000):
    rng = np.random.default_rng()
    n_times = np.shape(I_data)[0]
    n_tele = np.shape(I_data)[1]
    Ani_bootres = np.nan*np.zeros((n_times,5))
    Ani_bgsub_bootres = np.nan*np.zeros((n_times,5))
    for i in range(n_times):
        I_boot_arr = np.nan*np.zeros((n_tele,n_boot))
        bg_boot_arr = np.nan*np.zeros((n_tele,n_boot))
        for j in range(n_tele):
            R_arr = rng.uniform(0,1,((n_boot,delta_E.size)))
            I_boot_arr[j,:] = bootstrap_intensities(R_arr,count_arr[i,j,:],t_arr[i,j,:],delta_E,gf_arr[j,:])
            bg_boot_arr[j,:] = bg_fit[i,j] + bg_fit_unc[i,j]*rng.standard_normal(n_boot)
        I_boot_sub_arr = I_boot_arr - bg_boot_arr
        I_boot_sub_arr[I_boot_sub_arr<0] = 0  # not doing this keeps the relative intensities ok
        Ani_boot = 3*(np.sum(I_boot_arr*mu_data[i,:][:,np.newaxis]*weights[i,:][:,np.newaxis], axis=0)*1./np.sum(I_boot_arr*weights[i,:][:,np.newaxis], axis=0))
        Ani_boot[np.all(I_boot_arr==0, axis=0)] = 0
        Ani_boot = check_mu_sum(I_data[i,:],mu_data[i,:],Ani_boot)
        Ani_bootres[i,0] = np.mean(Ani_boot)
        Ani_bootres[i,1] = np.median(Ani_boot)
        Ani_bootres[i,2] = np.percentile(Ani_boot,2.5)
        Ani_bootres[i,3] = np.percentile(Ani_boot,97.5)
        Ani_bootres[i,4] = np.std(Ani_boot)
        Ani_bgsub_boot = 3*(np.sum(I_boot_sub_arr*mu_data[i,:][:,np.newaxis]*weights[i,:][:,np.newaxis], axis=0)*1./np.sum(I_boot_sub_arr*weights[i,:][:,np.newaxis], axis=0))
        Ani_bgsub_boot[np.all(I_boot_sub_arr==0, axis=0)] = 0
        Ani_bgsub_boot = check_mu_sum(I_data[i,:],mu_data[i,:],Ani_bgsub_boot)
        Ani_bgsub_bootres[i,0] = np.mean(Ani_bgsub_boot)
        Ani_bgsub_bootres[i,1] = np.median(Ani_bgsub_boot)
        Ani_bgsub_bootres[i,2] = np.percentile(Ani_bgsub_boot,2.5)
        Ani_bgsub_bootres[i,3] = np.percentile(Ani_bgsub_boot,97.5)
        Ani_bgsub_bootres[i,4] = np.std(Ani_bgsub_boot)
    return Ani_bootres, Ani_bgsub_bootres


def check_mu_sum(I_vals,mu_vals,ani):
        mu_data_copy = mu_vals.copy()
        mu_data_copy[np.isnan(I_vals)] = np.nan
        if mu_data_copy.ndim == 1:
            mu_sum = np.nansum(mu_data_copy)
            if np.abs(mu_sum)>=0.1:
                ani = np.nan*ani
        elif mu_data_copy.ndim == 2:
            mu_sum = np.nansum(mu_data_copy,axis=1)
            ani[np.abs(mu_sum)>=0.1] = np.nan
        return ani


def anisotropy_weighted_sum(I_data,mu_data,weights):
    ani = 3*(np.sum(I_data*mu_data*weights, axis=1)*1./np.sum(I_data*weights, axis=1))
    ani = check_mu_sum(I_data,mu_data,ani)
    return ani


def anisotropy_prepare(coverage,I_data):
    cov_arr = []
    centers = []
    for i, name in enumerate(coverage.columns.levels[0]):
        cov_arr.append(np.abs(np.cos(np.deg2rad(coverage[name]['max']))-np.cos(np.deg2rad(coverage[name]['min']))))
        centers.append(coverage[name]['center'])
    weights = np.column_stack(cov_arr)
    centers_arr = np.column_stack(centers)
    centers_arr[np.isnan(I_data)] = np.nan
    max_center = np.nanmax(np.column_stack(centers), axis=1)
    min_center = np.nanmin(np.column_stack(centers), axis=1)
    max_pa = np.cos(np.deg2rad(max_center))
    min_pa = np.cos(np.deg2rad(min_center))
    max_ani = 3*max_pa
    min_ani = 3*min_pa
    return weights, max_ani, min_ani


def anisotropy_legendre_fit(y,x,y_err = None):
    #Legendre polynomial fit up to 6th degree.
    #This is the maximum degree for 8 data points.
    if y_err is None:
        weights = np.ones(len(y))
    else:
        weights = 1/y_err
    l1 = Model(legendre1,nan_policy="omit")
    l2 = Model(legendre2,nan_policy="omit")
    l3 = Model(legendre3,nan_policy="omit")
    l4 = Model(legendre4,nan_policy="omit")
    l5 = Model(legendre5,nan_policy="omit")
    l6 = Model(legendre6,nan_policy="omit")
    models = [l1,l2,l3,l4,l5,l6]
    results = []
    anis = []
    bics = []
    param_dict = {'a': np.nanmean(y), 'b': 1.0, 'c': 0.5, 'd': 0.1, 'e': 0.1, 'f': 0.1, 'g': 0.1}
    degree = 1
    while (degree <= len(y)-2) & (degree<=6):
        model = models[degree-1]
        model.set_param_hint('a', vary=True,value=param_dict["a"],min=0)
        model.set_param_hint('b', vary=True,value=param_dict["b"])
        if degree>=2:
            model.set_param_hint('c', vary=True,value=param_dict["c"])
        if degree>=3:
            model.set_param_hint('d', vary=True,value=param_dict["d"])
        if degree>=4:
            model.set_param_hint('e', vary=True,value=param_dict["e"])
        if degree>=5:
            model.set_param_hint('f', vary=True,value=param_dict["f"])
        if degree>=6:
            model.set_param_hint('g', vary=True,value=param_dict["g"])
        params = model.make_params()
        res = model.fit(y, params, x=x,weights=weights)
        results.append(res)
        bics.append(res.bic)
        anis.append(res.params["b"].value/res.params["a"].value)
        degree += 1
    idx = np.nanargmin(bics)
    return results[idx], anis[idx]


def anisotropy_fit_cdf(y,x,y_err=None):
    best_model,ani = anisotropy_legendre_fit(y,x,y_err)
    a_limits = (np.max([best_model.params["a"].value-3*best_model.params["a"].stderr,0]),best_model.params["a"].value+3*best_model.params["a"].stderr)
    b_limits = (best_model.params["b"].value-3*best_model.params["b"].stderr,best_model.params["b"].value+3*best_model.params["b"].stderr)
    a,b,gr = conf_interval2d(best_model,best_model,"a","b",nx=15,ny=15,limits=(a_limits,b_limits))
    A,B = np.meshgrid(a,b)
    ani_arr = B/A
    df_ani = pd.DataFrame({"ani": ani_arr[(A>0) & (np.abs(ani_arr)<=3)].flatten(), "weights": gr[(A>0) & (np.abs(ani_arr)<=3)].flatten()})
    df_ani = df_ani.sort_values(by="ani")
    df_ani["cum_weights"] = df_ani["weights"].cumsum()
    df_ani["cdf"] = df_ani["cum_weights"]/df_ani["cum_weights"].iloc[-1]
    vals = np.interp(np.array([0.025,0.5,0.975]), df_ani["cdf"].values, df_ani["ani"].values)
    low = vals[0]
    median = vals[1]
    high = vals[2]
    mean = np.average(df_ani["ani"].values,weights=df_ani["weights"].values)
    return [mean, median, low, high, ani]


def anisotropy_fit_bootstrap(R,y,x,y_err=None):
    best_model,ani = anisotropy_legendre_fit(y,x,y_err)
    best_model.conf_interval(sigmas=[1],p_names=["a","b"])
    g0_center = best_model.params["a"].value
    g0_std = best_model.ci_out["a"][2][1]-g0_center
    g1_center = best_model.params["b"].value
    g1_std = best_model.ci_out["b"][2][1]-g1_center
    g0_boot = g0_center+g0_std*R[0,:]
    g1_boot = g1_center+g1_std*R[1,:]
    ani_arr = g1_boot/g0_boot
    ani_valid = ani_arr[np.abs(ani_arr)<=3]
    return [np.mean(ani_valid),np.median(ani_valid),np.percentile(ani_valid,2.5),np.percentile(ani_valid,97.5)]


def legendre1(x, a, b):
    return np.polynomial.legendre.legval(x, [a,b],tensor=False)


def legendre2(x, a, b, c):
    return np.polynomial.legendre.legval(x, [a,b,c],tensor=False)


def legendre3(x, a, b, c, d):
    return np.polynomial.legendre.legval(x, [a,b,c,d],tensor=False)


def legendre4(x, a, b, c, d, e):
    return np.polynomial.legendre.legval(x, [a,b,c,d,e],tensor=False)


def legendre5(x, a, b, c, d, e, f):
    return np.polynomial.legendre.legval(x, [a,b,c,d,e,f],tensor=False)


def legendre6(x, a, b, c, d, e, f, g):
    return np.polynomial.legendre.legval(x, [a,b,c,d,e,f,g],tensor=False)
