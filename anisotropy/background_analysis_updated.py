# from scipy.stats import linregress
from lmfit.models import ConstantModel, LinearModel, ExponentialModel
import lmfit.model
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
# from collections import Counter

def evaluate_background_all(I_times,I_data,res,x_start=None,x_end=None):
    times = np.nan*np.zeros(len(I_times))
    bg_I_fit = np.nan*np.zeros(np.shape(I_data))
    bg_I_fit_err = np.nan*np.zeros(np.shape(I_data))
    for i in range(len(times)):
        datetime = pd.to_datetime(I_times[i])
        times[i] = datetime.timestamp()
            
    t = times.copy()
    if res.model.name == "Model(exponential)":
        t = norm_x(t,x_start,x_end)
    y = res.eval(x=t)
    y_err = res.eval_uncertainty(x=t,sigma=1)
    bg_I_fit = np.ones(np.shape(I_data))*y[:,np.newaxis]
    bg_I_fit_err = np.ones(np.shape(I_data))*y_err[:,np.newaxis]
    return bg_I_fit, bg_I_fit_err

def run_background_analysis_all_nomag(bg_times,bg_I_data,bg_I_unc_data,minutes=None):
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    arr = [bg_times for i in range(np.shape(bg_I_data)[1])]
    bg_times_arr = np.column_stack(arr)
    raw_data = bg_I_data.flatten()
    raw_unc_data = bg_I_unc_data.flatten()
    data = raw_data[~np.isnan(raw_data)]
    unc = raw_unc_data[~np.isnan(raw_data)]
    datetimes = bg_times_arr.flatten()[~np.isnan(raw_data)]
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    x = times[data>=0]
    y = data[data>=0]
    y_err = unc[data>=0]
    if minutes is not None:
        x, y, y_err = average_background_with_uncertainty(x,y,y_err,1/y_err,minutes=minutes)
        res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
    else:
        res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
    stats = []
    models = [res1,res3]
    return models

def evaluate_background_binwise(I_times,I_data,models,x_start=None,x_end=None):
    times = np.nan*np.zeros(len(I_times))
    bg_I_fit = np.nan*np.zeros(np.shape(I_data))
    bg_I_fit_err = np.nan*np.zeros(np.shape(I_data))
    for i in range(len(times)):
        datetime = pd.to_datetime(I_times[i])
        times[i] = datetime.timestamp()
            
    if isinstance(models,lmfit.model.ModelResult):
        res = models
        t = times.copy()
        if res.model.name == "Model(exponential)":
            t = norm_x(t,x_start,x_end)
        y = res.eval(x=t)
        y_err = res.eval_uncertainty(x=t,sigma=1)
        bg_I_fit = np.ones(np.shape(I_data))*y[:,np.newaxis]
        bg_I_fit_err = np.ones(np.shape(I_data))*y_err[:,np.newaxis]
    else:
        for i,res in enumerate(models):
            t = times.copy()
            if res.model.name == "Model(exponential)":
                t = norm_x(t,x_start,x_end)
            y = res.eval(x=t)
            y_err = res.eval_uncertainty(x=t,sigma=1)
            bg_I_fit[:,i] = y
            bg_I_fit_err[:,i] = y_err
    return bg_I_fit, bg_I_fit_err
    
def run_background_analysis_all_binwise(bg_times,bg_I_data,bg_I_unc_data,plot_results=False,minutes=None):
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    n_groups = np.shape(bg_I_data)[1]
    arr = [bg_times for i in range(np.shape(bg_I_data)[1])]
    bg_times_arr = np.column_stack(arr) 
    if plot_results:
        nrow = int(np.ceil(n_groups/3))
        fig, axarr = plt.subplots(nrow,3,figsize=[10,5], constrained_layout=True)
        for ax in axarr.flatten()[n_groups:]: ax.set_axis_off()
    raw_data = bg_I_data.flatten()
    raw_unc_data = bg_I_unc_data.flatten()
    data = raw_data[~np.isnan(raw_data)]
    unc = raw_unc_data[~np.isnan(raw_data)]
    datetimes = bg_times_arr.flatten()[~np.isnan(raw_data)]
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    x = times[data>=0]
    y = data[data>=0]
    y_err = unc[data>=0]
    if minutes is not None:
        x, y, y_err = average_background_with_uncertainty(x,y,y_err,1/y_err,minutes=minutes)
        res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
    else:
        res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
    stats = []
    models = [res1,res3]
    for k, res in enumerate(models):
        stats.append(res.bic)
    best_model_all = models[np.argmin(stats)]
    if best_model_all.model.name == "Model(constant)":
        mean_c = best_model_all.params["c"].value
    elif best_model_all.model.name == "Model(exponential)":
        mean_amp = best_model_all.params["amplitude"].value
        mean_decay = best_model_all.params["decay"].value
    redchis = []
    aics = []
    bics = []
    data = bg_I_data.copy()
    unc = bg_I_unc_data.copy()
    datetimes = bg_times.copy()
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    for i in range(n_groups):
        y = data[:,i]
        x = times[y>=0]
        y_err = unc[:,i]
        y_err = y_err[y>=0]
        y = y[y>=0]
        if best_model_all.model.name == "Model(constant)":
            model1 = ConstantModel(nan_policy="omit")
            model1.set_param_hint('c', value=mean_c,vary=False)
            params = model1.make_params()
            res = model1.fit(y,params,x=x,weights=1/y_err)
        elif best_model_all.model.name == "Model(exponential)":
            model3 = ExponentialModel(nan_policy="omit")
            x_norm = norm_x(x,x_start,x_end)
            model3.set_param_hint('decay', value=mean_decay,vary=False)
            model3.set_param_hint('amplitude', value=mean_amp,vary=False)
            params = model3.make_params()
            res = model3.fit(y,params,x=x_norm,weights=1/y_err)
        if plot_results:
            ax=axarr.flatten()[i]
            ax.scatter(x,y,s=2)
            if res.model.name == "Model(exponential)":
                t = norm_x(x,x_start,x_end)
            else:
                t = x.copy()
            ax.plot(x,res.eval(x=t),color="C1",lw=1.3)
            ax.set_ylabel("$I$")
            ax.set_xlabel("Unix time (s)")
            ax.set_yscale("log")
            ax.set_title(label=f"{res.model.name}: $\\chi^2_\\nu$ = {res.redchi:.2f}", fontdict={'fontsize': 9.5}, color=f"C{str(i)}")
        redchis.append(res.redchi)
        aics.append(res.aic)
        bics.append(res.bic)
    return redchis,aics,bics,best_model_all

def run_background_analysis_equal_decay_binwise(mean_decay,bg_times,bg_I_data,bg_I_unc_data,plot_results=False,minutes=None):
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    best_models_exp = []
    best_models_const = []
    n_groups = np.shape(bg_I_data)[1]
    if plot_results:
        nrow = int(np.ceil(n_groups/3))
        fig, axarr = plt.subplots(nrow,3,figsize=[10,5], constrained_layout=True)
        for ax in axarr.flatten()[n_groups:]: ax.set_axis_off()
    data = bg_I_data.copy()
    unc = bg_I_unc_data.copy()
    datetimes = bg_times.copy()
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    for i in range(n_groups):
        y = data[:,i]
        x = times[y>=0]
        y_err = unc[:,i]
        y_err = y_err[y>=0]
        y = y[y>=0]
        if minutes is not None:
            x, y, y_err = average_background_with_uncertainty(x,y,y_err,1/y_err,minutes=minutes)
            res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
        else:
            res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
        
        model1 = ConstantModel(nan_policy="omit")
        model1.set_param_hint('c',vary=True,min=0)
        params = model1.make_params()
        res1 = model1.fit(y,params,x=x,weights=1/y_err)
        best_models_const.append(res1)
        
        model3 = ExponentialModel(nan_policy="omit")
        x_norm = norm_x(x,x_start,x_end)
        model3.set_param_hint('decay', value=mean_decay,vary=False)
        model3.set_param_hint('amplitude',vary=True,min=0)
        params = model3.make_params()
        res3 = model3.fit(y,params,x=x_norm,weights=1/y_err)
        best_models_exp.append(res3)
        if plot_results:
            ax=axarr.flatten()[i]
            ax.scatter(x,y,s=2)
            t = norm_x(x,x_start,x_end)
            ax.plot(x,res1.eval(x=x),color="C1",lw=1.3)
            ax.plot(x,res3.eval(x=t),color="C1",lw=1.3)
            ax.set_ylabel("$I$")
            ax.set_xlabel("Unix time (s)")
            ax.set_yscale("log")
            ax.set_title(label=f"{res1.model.name}: $\\chi^2_\\nu$ = {res1.redchi:.2f}\n{res3.model.name}: $\\chi^2_\\nu$ = {res3.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
    return best_models_const,best_models_exp

def run_background_analysis_binwise(bg_times,bg_I_data,bg_I_unc_data,plot_results=False,plot_uncertainty=False,minutes=None):
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    n_groups = np.shape(bg_I_data)[1]
    best_models = []
    if plot_results:
        nrow = int(np.ceil(n_groups/3))
        fig, axarr = plt.subplots(nrow,3,figsize=[10,5], constrained_layout=True)
        for ax in axarr.flatten()[n_groups:]: ax.set_axis_off()
    if plot_uncertainty:
        nrow = int(np.ceil(n_groups/3))
        fig2, axarr2 = plt.subplots(nrow,3,figsize=[10,5], constrained_layout=True)
        for ax in axarr2.flatten()[n_groups:]: ax.set_axis_off()
    data = bg_I_data.copy()
    unc = bg_I_unc_data.copy()
    datetimes = bg_times.copy()
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    decays = []
    for i in range(n_groups):
        y = data[:,i]
        x = times[y>=0]
        y_err = unc[:,i]
        y_err = y_err[y>=0]
        y = y[y>=0]
        if minutes is not None:
            x, y, y_err = average_background_with_uncertainty(x,y,y_err,1/y_err,minutes=minutes)
            res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
        else:
            res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
        stats = []
        models = [res1,res3]
        decays.append(res3.params["decay"].value)
        for k, res in enumerate(models):
            stats.append(res.bic)
        best_model = models[np.argmin(stats)]
        best_models.append(best_model)
        if plot_results:
            ax=axarr.flatten()[i]
            ax.scatter(x,y,s=2)
            if best_model.model.name == "Model(exponential)":
                t = norm_x(x,x_start,x_end)
            else:
                t = x.copy()
            ax.plot(x,best_model.eval(x=t),color="C1",lw=1.3)
            ax.set_ylabel("$I$")
            ax.set_yscale("log")
            ax.set_xlabel("Unix time (s)")
            ax.set_title(label=f"{best_model.model.name}: $\\chi^2_\\nu$ = {best_model.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
        if plot_uncertainty:
            ax=axarr2.flatten()[i]
            ax.scatter(x,y,s=2)
            if best_model.model.name == "Model(exponential)":
                t = norm_x(x,x_start,x_end)
            else:
                t = x.copy()
            ax.fill_between(x,best_model.eval(x=t)-best_model.eval_uncertainty(x=t,sigma=2),best_model.eval(x=t)+best_model.eval_uncertainty(x=t,sigma=2),alpha=0.2,color="C1")
            ax.set_ylabel("$I$")
            ax.set_xlabel("Unix time (s)")
            ax.set_yscale("log")
            ax.set_title(label=f"{best_model.model.name}: $\\chi^2_\\nu^2=${best_model.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
    return best_models, decays

def evaluate_background(I_times,I_data,models,mu_groups,mu_data,x_start=None,x_end=None):
    I_times_arr = np.column_stack((I_times,I_times,I_times,I_times))
    times = np.nan*np.zeros(np.shape(I_times_arr))
    bg_I_fit = np.nan*np.zeros(np.shape(I_data))
    bg_I_fit_err = np.nan*np.zeros(np.shape(I_data))
    for i in range(np.shape(I_times)[0]):
        for j in range(np.shape(times)[1]):
            datetime = pd.to_datetime(I_times_arr[i,j])
            times[i,j] = datetime.timestamp()
            
    if isinstance(models,lmfit.model.ModelResult):
        res = models
        t = times[:,0]
        if res.model.name == "Model(exponential)":
            t = norm_x(t,x_start,x_end)
        y = res.eval(x=t)
        y_err = res.eval_uncertainty(x=t,sigma=1)
        bg_I_fit = np.ones(np.shape(I_data))*y[:,np.newaxis]
        bg_I_fit_err = np.ones(np.shape(I_data))*y_err[:,np.newaxis]
    else:
        diff = np.abs(np.diff(mu_groups)/2)[0]
        for i,res in enumerate(models):
            if i == 0:
                idx = np.where((mu_data>=-1) & (mu_data<=mu_groups[i]+diff))
            elif i == len(mu_groups)-1:
                idx = np.where((mu_data>=mu_groups[i]-diff) & (mu_data<=1))
            else:
                idx = np.where((mu_data>=mu_groups[i]-diff) & (mu_data<=mu_groups[i]+diff))
            t = times[idx]
            if res.model.name == "Model(exponential)":
                t = norm_x(t,x_start,x_end)
            y = res.eval(x=t)
            y_err = res.eval_uncertainty(x=t,sigma=1)
            bg_I_fit[idx] = y
            bg_I_fit_err[idx] = y_err
    return bg_I_fit, bg_I_fit_err
        
def run_background_analysis(n_groups,bg_times,bg_I_data,bg_I_unc_data,bg_mu_data,plot_bins=False,plot_results=False,plot_uncertainty=False,mu_std = 0.1,minutes=None):
    mu0 = 1-1/n_groups
    mu_groups = np.linspace(-mu0,mu0,n_groups)
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    best_models = []
    arr = [bg_times for i in range(np.shape(bg_I_data)[1])]
    bg_times_arr = np.column_stack(arr)
    if plot_bins: 
        fig, axarr = plt.subplots(1,2,figsize=[7,1.5],sharex=True,constrained_layout=True,num=10,clear=True)
        ax = axarr[0]; ax.set_xlabel("$\\mu$"); ax.set_ylabel("Weight"); ax.set_yscale("log");ax.set_ylim(ymin=1e-3);ax = axarr[1]; ax.set_ylabel("Weight");ax.set_xlim(-1,1)
        for mu in mu_groups: mu_vals = np.linspace(-1,1,100);axarr[0].plot(mu_vals,scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)); axarr[1].plot(mu_vals,scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)); 
    if plot_results:
        nrow = int(np.ceil(n_groups/3))
        fig, axarr = plt.subplots(nrow,3,figsize=[10,20], constrained_layout=True,num=11,clear=True)
        for ax in axarr.flatten()[n_groups:]: ax.set_axis_off()
    if plot_uncertainty:
        nrow = int(np.ceil(n_groups/3))
        fig2, axarr2 = plt.subplots(nrow,3,figsize=[10,20], constrained_layout=True,num=12,clear=True)
        for ax in axarr2.flatten()[n_groups:]: ax.set_axis_off()
    raw_data = bg_I_data.flatten()
    raw_unc_data = bg_I_unc_data.flatten()
    data = raw_data[~np.isnan(raw_data)]
    unc = raw_unc_data[~np.isnan(raw_data)]
    datetimes = bg_times_arr.flatten()[~np.isnan(raw_data)]
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    mu_vals = bg_mu_data.flatten()[~np.isnan(raw_data)]
    decays = []
    for i in range(n_groups):
        mu = mu_groups[i]
        weights = scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)
        weights += scipy.stats.norm.pdf(-2-mu_vals, loc=mu, scale=mu_std)
        weights += scipy.stats.norm.pdf(2-mu_vals, loc=mu, scale=mu_std)
        max_weight = scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std) + scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std) + scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std)
        x = times[(weights>=0.01*max_weight) & (data>=0)]
        if len(x)==0:
            decays.append(np.nan)
            best_models.append(np.nan)
            continue
        y = data[(weights>=0.01*max_weight) & (data>=0)]
        y_err = unc[(weights>=0.01*max_weight) & (data>=0)]
        weights = weights[(weights>=0.01*max_weight) & (data>=0)]
        weights = weights/np.nansum(weights)*len(weights)
        if minutes is not None:
            x, y, y_err = average_background_with_uncertainty(x,y,y_err/weights,weights/y_err,minutes=minutes)
            res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
        else:
            res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,weights/y_err)
        stats = []
        models = [res1,res3]
        decays.append(res3.params["decay"].value)
        for k, res in enumerate(models):
            stats.append(res.bic)
            #VAIHDA TÄHÄN REDCHI
        best_model = models[np.argmin(stats)]
        best_models.append(best_model)
        if plot_results:
            ax=axarr.flatten()[i]
            ax.scatter(x,y,s=2,c=weights,cmap="Blues",norm="log")
            if best_model.model.name == "Model(exponential)":
                t = norm_x(x,x_start,x_end)
            else:
                t = x.copy()
            ax.plot(x,best_model.eval(x=t),color="C1",lw=1.3)
            ax.set_ylabel("$I$")
            ax.set_yscale("log")
            ax.set_xlabel("Unix time (s)")
            ax.set_title(label=f"{best_model.model.name}: $\\chi^2_\\nu$ = {best_model.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
        if plot_uncertainty:
            ax=axarr2.flatten()[i]
            ax.scatter(x,y,s=2)
            if best_model.model.name == "Model(exponential)":
                t = norm_x(x,x_start,x_end)
            else:
                t = x.copy()
            ax.fill_between(x,best_model.eval(x=t)-best_model.eval_uncertainty(x=t,sigma=2),best_model.eval(x=t)+best_model.eval_uncertainty(x=t,sigma=2),alpha=0.2,color="C1")
            ax.set_ylabel("$I$")
            ax.set_xlabel("Unix time (s)")
            ax.set_yscale("log")
            ax.set_title(label=f"{best_model.model.name}: $\\chi^2_\\nu^2=${best_model.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
    return best_models, decays

def run_background_analysis_equal_decay(mean_decay,n_groups,bg_times,bg_I_data,bg_I_unc_data,bg_mu_data,plot_bins=False,plot_results=False,mu_std = 0.1,minutes=None):
    mu0 = 1-1/n_groups
    mu_groups = np.linspace(-mu0,mu0,n_groups)
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    best_models_exp = []
    best_models_const = []
    arr = [bg_times for i in range(np.shape(bg_I_data)[1])]
    bg_times_arr = np.column_stack(arr)
    if plot_bins: 
        fig, axarr = plt.subplots(1,2,figsize=[7,1.5],sharex=True,constrained_layout=True,num=13,clear=True)
        ax = axarr[0]; ax.set_xlabel("$\\mu$"); ax.set_ylabel("Weight"); ax.set_yscale("log");ax.set_ylim(ymin=1e-3);ax = axarr[1]; ax.set_ylabel("Weight");ax.set_xlim(-1,1)
        for mu in mu_groups: mu_vals = np.linspace(-1,1,100);axarr[0].plot(mu_vals,scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)); axarr[1].plot(mu_vals,scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)); 
    if plot_results:
        nrow = int(np.ceil(n_groups/3))
        fig, axarr = plt.subplots(nrow,3,figsize=[10,25], constrained_layout=True,num=14,clear=True)
        for ax in axarr.flatten()[n_groups:]: ax.set_axis_off()
    raw_data = bg_I_data.flatten()
    raw_unc_data = bg_I_unc_data.flatten()
    data = raw_data[~np.isnan(raw_data)]
    unc = raw_unc_data[~np.isnan(raw_data)]
    datetimes = bg_times_arr.flatten()[~np.isnan(raw_data)]
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    mu_vals = bg_mu_data.flatten()[~np.isnan(raw_data)]
    for i in range(n_groups):
        mu = mu_groups[i]
        weights = scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)
        weights += scipy.stats.norm.pdf(-2-mu_vals, loc=mu, scale=mu_std)
        weights += scipy.stats.norm.pdf(2-mu_vals, loc=mu, scale=mu_std)
        max_weight = scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std) + scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std) + scipy.stats.norm.pdf(mu, loc=mu, scale=mu_std)
        x = times[(weights>=0.01*max_weight) & (data>=0)]
        if len(x)==0:
            best_models_const.append(np.nan)
            best_models_exp.append(np.nan)
            continue
        y = data[(weights>=0.01*max_weight) & (data>=0)]
        y_err = unc[(weights>=0.01*max_weight) & (data>=0)]
        weights = weights[(weights>=0.01*max_weight) & (data>=0)]
        weights = weights/np.nansum(weights)*len(weights)
        if minutes is not None:
            x, y, y_err = average_background_with_uncertainty(x,y,y_err/weights,weights/y_err,minutes=minutes)
        
        model1 = ConstantModel(nan_policy="omit")
        model1.set_param_hint('c',vary=True,min=0)
        params = model1.make_params()
        if minutes is not None:
            res1 = model1.fit(y,params,x=x,weights=1/y_err)
        else:
            res1 = model1.fit(y,params,x=x,weights=weights/y_err)
        best_models_const.append(res1)
        
        model3 = ExponentialModel(nan_policy="omit")
        x_norm = norm_x(x,x_start,x_end)
        model3.set_param_hint('decay', value=mean_decay,vary=False)
        model3.set_param_hint('amplitude',vary=True,min=0)
        params = model3.make_params()
        if minutes is not None:
            res3 = model3.fit(y,params,x=x_norm,weights=1/y_err)
        else:
            res3 = model3.fit(y,params,x=x_norm,weights=weights/y_err)
        best_models_exp.append(res3)
        if plot_results:
            ax=axarr.flatten()[i]
            ax.scatter(x,y,s=2,c=weights,cmap="Blues",norm="log")
            t = norm_x(x,x_start,x_end)
            ax.plot(x,res1.eval(x=x),color="C1",lw=1.3)
            ax.plot(x,res3.eval(x=t),color="C1",lw=1.3)
            ax.set_ylabel("$I$")
            ax.set_xlabel("Unix time (s)")
            ax.set_yscale("log")
            ax.set_title(label=f"{res1.model.name}: $\\chi^2_\\nu$ = {res1.redchi:.2f}\n{res3.model.name}: $\\chi^2_\\nu$ = {res3.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
    return best_models_const,best_models_exp

def run_background_analysis_all(n_groups,bg_times,bg_I_data,bg_I_unc_data,bg_mu_data,plot_bins=False,plot_results=False,mu_std = 0.1,minutes=None):
    mu0 = 1-1/n_groups
    mu_groups = np.linspace(-mu0,mu0,n_groups)
    x_start = pd.to_datetime(np.nanmin(bg_times)).timestamp()
    x_end = pd.to_datetime(np.nanmax(bg_times)).timestamp()
    arr = [bg_times for i in range(np.shape(bg_I_data)[1])]
    bg_times_arr = np.column_stack(arr)
    if plot_bins: 
        fig, axarr = plt.subplots(1,2,figsize=[7,1.5],sharex=True,constrained_layout=True,num=15,clear=True)
        ax = axarr[0]; ax.set_xlabel("$\\mu$"); ax.set_ylabel("Weight"); ax.set_yscale("log");ax.set_ylim(ymin=1e-3);ax = axarr[1]; ax.set_ylabel("Weight");ax.set_xlim(-1,1)
        for mu in mu_groups: mu_vals = np.linspace(-1,1,100);axarr[0].plot(mu_vals,scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)); axarr[1].plot(mu_vals,scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)); 
    if plot_results:
        nrow = int(np.ceil(n_groups/3))
        fig, axarr = plt.subplots(nrow,3,figsize=[10,20], constrained_layout=True,num=16,clear=True)
        for ax in axarr.flatten()[n_groups:]: ax.set_axis_off()
    raw_data = bg_I_data.flatten()
    raw_unc_data = bg_I_unc_data.flatten()
    data = raw_data[~np.isnan(raw_data)]
    unc = raw_unc_data[~np.isnan(raw_data)]
    datetimes = bg_times_arr.flatten()[~np.isnan(raw_data)]
    times = np.array([t.timestamp() for t in pd.to_datetime(datetimes)])
    mu_vals = bg_mu_data.flatten()[~np.isnan(raw_data)]
    x = times[data>=0]
    y = data[data>=0]
    y_err = unc[data>=0]
    if minutes is not None:
        x, y, y_err = average_background_with_uncertainty(x,y,y_err,1/y_err,minutes=minutes)
        res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
    else:
        res1,res2,res3 = analyze_background_weighted(x,y,x_start,x_end,1/y_err)
    stats = []
    models = [res1,res3]
    for k, res in enumerate(models):
        stats.append(res.bic)
    best_model_all = models[np.argmin(stats)]
    if best_model_all.model.name == "Model(constant)":
        mean_c = best_model_all.params["c"].value
    elif best_model_all.model.name == "Model(exponential)":
        mean_amp = best_model_all.params["amplitude"].value
        mean_decay = best_model_all.params["decay"].value
    redchis = []
    aics = []
    bics = []
    for i in range(n_groups):
        mu = mu_groups[i]
        weights = scipy.stats.norm.pdf(mu_vals, loc=mu, scale=mu_std)
        weights += scipy.stats.norm.pdf(-2-mu_vals, loc=mu, scale=mu_std)
        weights += scipy.stats.norm.pdf(2-mu_vals, loc=mu, scale=mu_std)
        max_weight = np.nanmax(weights)
        x = times[(weights>=0.01*max_weight) & (data>=0)]
        if len(x)==0:
            redchis.append(np.nan)
            aics.append(np.nan)
            bics.append(np.nan)
        y = data[(weights>=0.01*max_weight) & (data>=0)]
        y_err = unc[(weights>=0.01*max_weight) & (data>=0)]
        weights = weights[(weights>=0.01*max_weight) & (data>=0)]
        weights = weights/np.nansum(weights)*len(weights)
        if minutes is not None:
            x, y, y_err = average_background_with_uncertainty(x,y,y_err/weights,weights/y_err,minutes=minutes)
        if best_model_all.model.name == "Model(constant)":
            model1 = ConstantModel(nan_policy="omit")
            model1.set_param_hint('c', value=mean_c,vary=False)
            params = model1.make_params()
            if minutes is not None:
                res = model1.fit(y,params,x=x,weights=1/y_err)
            else:
                res = model1.fit(y,params,x=x,weights=weights/y_err)
        elif best_model_all.model.name == "Model(exponential)":
            model3 = ExponentialModel(nan_policy="omit")
            x_norm = norm_x(x,x_start,x_end)
            model3.set_param_hint('decay', value=mean_decay,vary=False)
            model3.set_param_hint('amplitude', value=mean_amp,vary=False)
            params = model3.make_params()
            if minutes is not None:
                res = model3.fit(y,params,x=x_norm,weights=1/y_err)
            else:
                res = model3.fit(y,params,x=x_norm,weights=weights/y_err)
        if plot_results:
            ax=axarr.flatten()[i]
            ax.scatter(x,y,s=2,c=weights,cmap="Blues",norm="log")
            if res.model.name == "Model(exponential)":
                t = norm_x(x,x_start,x_end)
            else:
                t = x.copy()
            ax.plot(x,res.eval(x=t),color="C1",lw=1.3)
            ax.set_ylabel("$I$")
            ax.set_xlabel("Unix time (s)")
            ax.set_yscale("log")
            ax.set_title(label=f"{res.model.name}: $\\chi^2_\\nu$ = {res.redchi:.2f}", fontdict={'fontsize': 9.5},color=f"C{str(i)}")
        redchis.append(res.redchi)
        aics.append(res.aic)
        bics.append(res.bic)
    return redchis,aics,bics,best_model_all

def average_background_with_uncertainty(x,y,y_err,weights,minutes=10):
    x_edges = np.arange(np.min(x),np.max(x),minutes*60)
    x_edges = np.append(x_edges,np.max(x)*1.01)
    x_mean = []
    y_mean = []
    y_mean_err = []
    for i in range(len(x_edges)-1):
        w = weights[(x>=x_edges[i]) & (x<x_edges[i+1])]
        if w.size == 0:
            continue
        elif np.sum(w)==0:
            continue
        x_mean.append(np.mean(x[(x>=x_edges[i]) & (x<x_edges[i+1])]))
        y_mean.append(np.average(y[(x>=x_edges[i]) & (x<x_edges[i+1])],weights=w))
        y_mean_err.append(np.sqrt(np.sum((w*y_err[(x>=x_edges[i]) & (x<x_edges[i+1])])**2))/(np.sum(w)))
    return np.array(x_mean), np.array(y_mean), np.array(y_mean_err)

def average_background(x,y,weights,minutes=10):
    x_edges = np.arange(np.min(x),np.max(x),minutes*60)
    x_edges = np.append(x_edges,np.max(x)*1.01)
    x_mean = []
    y_mean = []
    for i in range(len(x_edges)-1):
        w = weights[(x>=x_edges[i]) & (x<x_edges[i+1])]
        if w.size == 0:
            continue
        elif np.sum(w)==0:
            continue
        x_mean.append(np.mean(x[(x>=x_edges[i]) & (x<x_edges[i+1])]))
        y_mean.append(np.average(y[(x>=x_edges[i]) & (x<x_edges[i+1])],weights=w))
    return np.array(x_mean), np.array(y_mean)

def norm_x(new_x,x_start,x_end):
    x_norm = (new_x - x_start)/(x_end - x_start)
    return x_norm

# def norm_x(new_x,x_start,x_end):
#     x_norm = (new_x - x_start)/(60*60) #scaled to hours
#     return x_norm

def analyze_background_noexp(x,y):
    model1 = ConstantModel(nan_policy="omit")
    params = model1.guess(y,x=x)
    res1 = model1.fit(y,params,x=x)
    model2 = LinearModel(nan_policy="omit")
    model2.set_param_hint('slope', vary=True,max=0)
    model2.set_param_hint('intercept', vary=True,min=0)
    params = model2.make_params()
    res2 = model2.fit(y,params,x=x)
    return res1,res2

def analyze_background(x,y,x_start,x_end):
    model1 = ConstantModel(nan_policy="omit")
    params = model1.guess(y,x=x)
    res1 = model1.fit(y,params,x=x)
    model2 = LinearModel(nan_policy="omit")
    model2.set_param_hint('slope', vary=True,max=0)
    model2.set_param_hint('intercept', vary=True,min=0)
    params = model2.make_params()
    res2 = model2.fit(y,params,x=x)
    model3 = ExponentialModel(nan_policy="omit")
    x_norm = norm_x(x,x_start,x_end)
    model3.set_param_hint('decay', vary=True,min=0)
    model3.set_param_hint('amplitude', vary=True,min=0)
    params = model3.make_params()
    res3 = model3.fit(y,params,x=x_norm)
    return res1,res2,res3

def analyze_background_weighted_noexp(x,y,weights):
    model1 = ConstantModel(nan_policy="omit")
    params = model1.guess(y,x=x,weights=weights)
    res1 = model1.fit(y,params,x=x,weights=weights)
    model2 = LinearModel(nan_policy="omit")
    model2.set_param_hint('slope', vary=True,max=0)
    model2.set_param_hint('intercept', vary=True,min=0)
    params = model2.make_params()
    res2 = model2.fit(y,params,x=x,weights=weights)
    return res1,res2

def analyze_background_weighted(x,y,x_start,x_end,weights):
    model1 = ConstantModel(nan_policy="omit")
    params = model1.guess(y,x=x,weights=weights)
    res1 = model1.fit(y,params,x=x,weights=weights)
    model2 = LinearModel(nan_policy="omit")
    model2.set_param_hint('slope', vary=True,max=0)
    model2.set_param_hint('intercept', vary=True,min=0)
    params = model2.make_params()
    res2 = model2.fit(y,params,x=x,weights=weights)
    model3 = ExponentialModel(nan_policy="omit")
    x_norm = norm_x(x,x_start,x_end)
    model3.set_param_hint('decay', vary=True,min=0)
    model3.set_param_hint('amplitude', vary=True,min=0)
    params = model3.make_params()
    res3 = model3.fit(y,params,x=x_norm,weights=weights)
    return res1,res2,res3

def analyze_background_with_uncertainty(x,y,y_err,x_start,x_end):
    model1 = ConstantModel(nan_policy="omit")
    params = model1.guess(y,x=x)
    res1 = model1.fit(y,params,x=x,weights=1/y_err)
    model2 = LinearModel(nan_policy="omit")
    model2.set_param_hint('slope', vary=True,max=0)
    model2.set_param_hint('intercept', vary=True,min=0)
    params = model2.make_params()
    res2 = model2.fit(y,params,x=x,weights=1/y_err)
    model3 = ExponentialModel(nan_policy="omit")
    x_norm = norm_x(x,x_start,x_end)
    model3.set_param_hint('decay', vary=True,min=0)
    model3.set_param_hint('amplitude', vary=True,min=0)
    params = model3.make_params()
    res3 = model3.fit(y,params,x=x_norm,weights=1/y_err)
    return res1,res2,res3