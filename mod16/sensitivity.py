'''
'''

import json
import os
import numpy as np
import h5py
import mod16
from tqdm import tqdm
from mod16 import MOD16
from mod16.utils import restore_bplut, pft_dominant
from mod17.science import nash_sutcliffe
from SALib.sample import saltelli
from SALib.analyze import sobol

OUTPUT_TPL = '/home/arthur/Workspace/NTSG/projects/Y2021_MODIS-VIIRS/data/MOD16_sensitivity_%s_analysis.json'
MOD16_DIR = os.path.dirname(mod16.__file__)
with open(os.path.join(MOD16_DIR, 'data/MOD16_calibration_config.json'), 'r') as file:
    CONFIG = json.load(file)
BOUNDS = {
    "tmin_close": [-35, 0],
    "tmin_open": [0, 25],
    "vpd_open": [0, 1500],
    "vpd_close": [1500, 8000],
    "gl_sh": [0.01, 0.12],
    "gl_wv": [0.01, 0.12],
    "g_cuticular": [1e-6, 1e-3],
    "csl": [0.001, 0.1],
    "rbl_min": [10, 80],
    "rbl_max": [80, 150],
    "beta": [0, 1000]
}


def main(pft = None):
    # Stratify the data using the validation mask so that an equal number of
    #   samples from each PFT are used
    drivers, tower_obs = load_data(pft = pft, validation_mask_only = pft is None)
    params = MOD16.required_parameters
    problem = {
        'num_vars': len(params),
        'names': params,
        'bounds': [
            BOUNDS[p]
            for p in params
        ]
    }
    # NOTE: Number of samples must be a power of 2
    param_sweep = saltelli.sample(problem, 256 if pft is None else 128)
    Y = np.zeros([param_sweep.shape[0]])
    for i, X in enumerate(tqdm(param_sweep)):
        yhat = MOD16._et(X, *drivers)
        Y[i] = nash_sutcliffe(yhat, tower_obs, norm = True)
    metrics = sobol.analyze(problem, Y)
    filename = OUTPUT_TPL % 'ET'
    if pft is not None:
        filename = OUTPUT_TPL % f'ET-PFT{pft}'
    with open(filename, 'w') as file:
        json.dump(dict([(k, v.tolist()) for k, v in metrics.items()]), file)


def load_data(pft, validation_mask_only = False):
    print('Loading driver datasets...')
    with h5py.File(CONFIG['data']['file'], 'r') as hdf:
        if pft is not None:
            site_list = hdf['FLUXNET/site_id'][:].tolist()
            if hasattr(site_list[0], 'decode'):
                site_list = [s.decode('utf-8') for s in site_list]
            sites = pft_dominant(hdf['state/PFT'][:], site_list = site_list)
            sites = sites == pft
        else:
            shp = hdf['MERRA2/Tmin'].shape
            sites = np.ones(shp[1]).astype(bool)
        lw_net_day = hdf['MERRA2/LWGNT_daytime'][:][:,sites]
        lw_net_night = hdf['MERRA2/LWGNT_nighttime'][:][:,sites]
        sw_albedo = hdf['MODIS/MCD43A3_black_sky_sw_albedo'][:][:,sites]
        sw_rad_day = hdf['MERRA2/SWGDN_daytime'][:][:,sites]
        sw_rad_night = hdf['MERRA2/SWGDN_nighttime'][:][:,sites]
        temp_day = hdf['MERRA2/T10M_daytime'][:][:,sites]
        temp_night = hdf['MERRA2/T10M_nighttime'][:][:,sites]
        tmin = hdf['MERRA2/Tmin'][:][:,sites]
        # As long as the time series is balanced w.r.t. years (i.e., same
        #   number of records per year), the overall mean is the annual mean
        temp_annual = hdf['MERRA2/T10M'][:][:,sites].mean(axis = 0)[None,:]\
            .repeat(tmin.shape[0], axis = 0)
        vpd_day = MOD16.vpd(
            hdf['MERRA2/QV10M_daytime'][:][:,sites],
            hdf['MERRA2/PS_daytime'][:][:,sites],
            temp_day)
        vpd_night = MOD16.vpd(
            hdf['MERRA2/QV10M_nighttime'][:][:,sites],
            hdf['MERRA2/PS_nighttime'][:][:,sites],
            temp_night)
        pressure = hdf['MERRA2/PS'][:][:,sites]
        # Read in fPAR, LAI, and convert from (%) to [0,1]
        fpar = np.nanmean(
            hdf['MODIS/MOD15A2HGF_fPAR_interp'][:][:,sites], axis = -1)
        lai = np.nanmean(
            hdf['MODIS/MOD15A2HGF_LAI_interp'][:][:,sites], axis = -1)
        # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
        fpar /= 100
        lai /= 10
        tower_obs = hdf['FLUXNET/latent_heat'][:][:,sites]
        if pft is None:
            is_test = hdf['FLUXNET/validation_mask'][:].sum(axis = 0).astype(bool)
    # Compile driver datasets
    drivers = [
        lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
        temp_day, temp_night, temp_annual, tmin, vpd_day, vpd_night,
        pressure, fpar, lai
    ]
    # Speed things up by focusing only on data points where valid data exist
    mask = ~np.isnan(tower_obs)
    if pft is None and validation_mask_only:
        # Stratify the data using the validation mask so that an equal number
        #   of samples from each PFT are used
        mask = np.logical_and(is_test, mask)
    drivers = [d[mask] for d in drivers]
    return (drivers, tower_obs[mask])


if __name__ == '__main__':
    import fire
    fire.Fire(main)
