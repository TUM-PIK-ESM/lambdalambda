#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:48:52 2024

@author: bathiany
"""

import numpy as np
import numpy.ma as ma
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

##### Building on Taylor Smith's code,
## this code helps investigate the effect of gaps and of shifting the start value on the CSD estimators lambda_AC and lambda_Var and their ratio
## applied to a single time series.

Npoints=15

### what is the lower limit of lambda_Var according to Andreas formula?
# i.e. when end points are 0
def calc_lambda_var_lower(N, lambda_AC_Andreas):
    lambda_Var_Andreas=1/2*np.log(1-N/(N-1)*(1-np.exp(2*lambda_AC_Andreas)))
    return lambda_Var_Andreas

def calc_ar1(x):
    #return np.corrcoef(x[:-1], x[1:])[0,1]
    return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0,1]

def compute_lam(x, dt=1):
    dx = (x[1:] - x[:-1]) / dt
    x0 = x[:-1]
    mask = ~np.isnan(x0) & ~np.isnan(dx)
    return st.linregress(x0[mask], dx[mask])[0]

def compute_sigma(x, dt=1):
    dx = (x[1:] - x[:-1]) / dt
    lamb = compute_lam(x, dt)
    diff = dx - lamb * x[:-1]
    return np.nanstd(diff) * np.sqrt(dt)

def proc_ts(rint, ts, res, res0):   # start index, orig ts, residual ts

    outdf = pd.DataFrame()
    outdf['rint'] = [rint]
    outdf['fullmn'] = [np.nanmean(ts.values)]
    
    car_raw = calc_ar1(ts.values)
    car_detrend = calc_ar1(res.values)
    car_detrend0 = calc_ar1(res0.values)

    if np.isnan(car_raw):
        car_raw = np.nan
    if np.isnan(car_detrend):
        car_detrend = np.nan
    outdf['ar1_raw'] = [car_raw]
    outdf['ar1_resid'] = [car_detrend]
    outdf['ar1_resid_0'] = [car_detrend0]

    lg = np.log(car_detrend)
    if np.isnan(lg):
        lg = np.nan
    outdf['lambda_ar1'] = [lg]

    lg0 = np.log(car_detrend0)
    if np.isnan(lg0):
        lg0 = np.nan
    outdf['lambda_ar1_0'] = [lg0]

    outdf['lambda_var_lower'] = calc_lambda_var_lower(505-rint, lg)
    outdf['lambda_var_lower_0'] = calc_lambda_var_lower(505-rint, lg0)    
    
    outdf['variance_raw'] = [np.nanvar(ts.values)]
    var = np.nanvar(res.values)
    outdf['variance_resid'] = [var]

    var0 = np.nanvar(res0.values)
    outdf['variance_resid_0'] = [var0]

    
    sigma = compute_sigma(res.values)
    outdf['sigma_lamb'] = [sigma]
    #rr_var = -sigma**2 / (2 * var)
    rr_var = 0.5 * np.log(1-sigma**2 / var)
    if np.isnan(rr_var):
        rr_var = np.nan
    outdf['lambda_variance'] = [rr_var]

    sigma0 = compute_sigma(res0.values)
    outdf['sigma_lamb_0'] = [sigma0]
    rr_var0 = 0.5 * np.log(1-sigma0**2 / var0)
    if np.isnan(rr_var0):
        rr_var0 = np.nan
    outdf['lambda_variance_0'] = [rr_var0]

     
    return outdf

df = pd.read_csv('Raw_kndvi.csv')
s1 = pd.Series(df.values[:,1].astype(float), index=[pd.Timestamp(x) for x in df.values[:,0]]) # orig ts
df = pd.read_csv('DS_kndvi.csv')
res = pd.Series(df.values[:,1].astype(float), index=[pd.Timestamp(x) for x in df.values[:,0]]) # residual ts


#res0 = res.fillna(0)    # like res but filled nans with 0
#res0 = res.fillna(0.5)  # like res but filled nans with value 0.3
#res0 = res.fillna(method="ffill") # use the most recent non-nan value

#res0 = res.interpolate(method='linear')
res0 = res.interpolate(method='spline', order=3)
## Randomly pick a value from the non-NaN values - no change!
#res0=res
#for idx in res[res.isna()].index:
#    random_value = np.random.choice(res.dropna().values)
#    res0.loc[idx] = random_value



l_dist = []
for x0 in range(0,Npoints):
    d = proc_ts(x0, s1[x0:], res[x0:], res0[x0:])
    lamb = d[['rint','ar1_resid','ar1_resid_0','lambda_ar1', 'sigma_lamb','sigma_lamb_0','variance_resid','variance_resid_0','lambda_variance','lambda_var_lower','lambda_var_lower_0','lambda_ar1_0', 'lambda_variance_0']]
    l_dist.append(lamb)

cc = pd.concat(l_dist)



plt.close('all')

### entire ts

### raw ts
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(s1)
year_starts = s1.resample("AS").first().index  # 'AS' stands for 'Annual Start'
# Add a vertical dashed line at each year's start
for date in year_starts:
    ax.axvline(x=date, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel('year')
ax.set_ylabel('kNDVI raw')
plt.savefig('ts_raw.png')

#deseasoned ts, first N points
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(res)
year_starts = res.resample("AS").first().index  # 'AS' stands for 'Annual Start'
# Add a vertical dashed line at each year's start
for date in year_starts:
    ax.axvline(x=date, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel('year')
ax.set_ylabel('kNDVI deseasoned')
plt.savefig('ts_deseasoned.png')


####


### raw ts
#f, ax = plt.subplots(1, figsize=(8, 5))
#ax.plot(s1)
#ax.set_xlabel('year')
#ax.set_ylabel('kNDVI raw')

#deseasoned ts, first N points
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(res[:Npoints])
ax.set_xlabel('year')
ax.set_ylabel('kNDVI deseasoned')
plt.savefig('ts_deseasoned_firstN.png')

#f, ax = plt.subplots(1, figsize=(8, 5))
#ax.plot(res0[:Npoints])
#ax.set_xlabel('year')
#ax.set_ylabel('kNDVI deseasoned, padded')


## AR    ## the outliers decrease AR1
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(cc.rint, cc.ar1_resid, label='with gaps')
ax.plot(cc.rint, cc.ar1_resid_0, label='gaps filled')
ax.legend()
ax.set_xlabel('Points from start dropped (series length 500)')
ax.set_ylabel('AR1')
plt.savefig('ts_deseasoned_AR1.png')


### lambda AR  ## the outliers decreases lambdaAR1 toward less negative values
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(cc.rint, cc.lambda_ar1, label='with gaps')
ax.plot(cc.rint, cc.lambda_ar1_0, label='gaps filled')
ax.legend()
ax.set_xlabel('Points from start dropped (series length 500)')
ax.set_ylabel('Lambda AR1')
plt.savefig('ts_deseasoned_Lambda_AR1.png')


#### sigma   ## the outliers increase sigma 
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(cc.rint, cc.sigma_lamb, label='with gaps')
ax.plot(cc.rint, cc.sigma_lamb_0, label='gaps filled')
ax.legend()
ax.set_xlabel('Points from start dropped (series length 500)')
ax.set_ylabel('Sigma')
plt.savefig('ts_deseasoned_Sigma.png')


## Variance ## the outliers increase Var
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(cc.rint, cc.variance_resid, label='with gaps')
ax.plot(cc.rint, cc.variance_resid_0, label='gaps filled')
ax.legend()
ax.set_xlabel('Points from start dropped (series length 500)')
ax.set_ylabel('Var')
plt.savefig('ts_deseasoned_Var.png')





### lambda Var   ## the outliers change lambda_Var:
## if first data point is an outlier, lambda_var is high
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(cc.rint, cc.lambda_variance, 'b-', label='with gaps')
ax.plot(cc.rint, cc.lambda_var_lower, 'b--', label='with gaps, lower bound')
ax.plot(cc.rint, cc.lambda_variance_0, color='orange', linestyle='-', label='gaps filled')
ax.plot(cc.rint, cc.lambda_var_lower_0, color='orange', linestyle='--', label='gaps filled, lower bound')
ax.legend()
ax.set_xlabel('Points from start dropped (series length 500)')
ax.set_ylabel('Lambda Var')
plt.savefig('ts_deseasoned_Lambda_Var.png')


#
#### lambda-lambda ratio
f, ax = plt.subplots(1, figsize=(8, 5))
ax.plot(cc.rint, cc.lambda_ar1 / cc.lambda_variance, 'b-', label='with gaps')
ax.plot(cc.rint, cc.lambda_ar1 / cc.lambda_var_lower, 'b--', label='with gaps, lower bound')
ax.plot(cc.rint, cc.lambda_ar1_0 / cc.lambda_variance_0, color='orange', linestyle='-', label='gaps filled')
ax.plot(cc.rint, cc.lambda_ar1_0 / cc.lambda_var_lower_0, color='orange', linestyle='--', label='gaps filled, lower bound')

ax.legend()
ax.set_xlabel('Points from start dropped (series length 500)')
ax.set_ylabel('Lambda AR1/Lambda Var')
plt.savefig('ts_deseasoned_lambdaratio.png')




