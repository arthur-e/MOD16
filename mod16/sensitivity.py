'''
If a parameter sensitivity analysis is requested, `analysis="parameters"`,
then the "model output" that is analyzed is the Nash-Sutcliffe efficiency
of the current model (with the parameter sweep), based on the observed
tower data.

If a sensitivity analysis of the *driver data* is requested,
`analysis="drivers"`, instead, then the model output is the predicted value
using an average value for the parameters.
'''

import json
import os
import yaml
import warnings
import numpy as np
import h5py
import mod16
from collections import OrderedDict
from tqdm import tqdm
from mod16 import MOD16
from mod16.utils import restore_bplut, pft_dominant
from mod17.science import nash_sutcliffe
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze import sobol

OUTPUT_TPL = '/home/arthur/Workspace/NTSG/projects/Y2021_MODIS-VIIRS/data/MOD16_sensitivity_%s_analysis.json'
MOD16_DIR = os.path.dirname(mod16.__file__)
with open(os.path.join(MOD16_DIR, 'data/MOD16_calibration_config.yaml'), 'r') as file:
    CONFIG = yaml.safe_load(file)
BOUNDS = OrderedDict({ # Based on [2, 98] percentiles of Cal-Val data
    'lw_net_day': [-100, 0],
    'lw_net_night': [-50, 0],
    'sw_rad_day': [0, 360],
    'sw_rad_night': [0, 0.00001],
    'sw_albedo': [0.1, 0.22],
    'temp_day': [255, 305],
    'temp_night': [250, 300],
    'temp_annual': [265, 300],
    'tmin': [252, 298],
    'vpd_day': [10, 3400],
    'vpd_night': [10, 2400],
    'pressure': [70000, 101340],
    'fpar': [0.02, 0.89],
    'lai': [0.13, 5.34]
})
PARAM_BOUNDS = {
    "tmin_close": [-35, 0],
    "tmin_open": [0, 25],
    "vpd_open": [0, 1000],
    "vpd_close": [1000, 8000],
    "gl_sh": [0.001, 0.2],
    "gl_wv": [0.001, 0.2],
    "g_cuticular": [1e-7, 1e-3],
    "csl": [0.0001, 0.2],
    "rbl_min": [10, 1000],
    "rbl_max": [100, 2000],
    "beta": [0, 2000]
}


def main(pft = None, analysis = 'parameters'):
    assert analysis == 'parameters' or pft is None,\
        'Cannot do a PFT-level analysis of the sensitivity when --analysis="drivers"'
    # Stratify the data using the validation mask so that an equal number of
    #   samples from each PFT are used
    drivers, tower_obs = load_data(pft = pft, validation_mask_only = pft is None)

    # Generate a vectorized set of (default) parameters
    if analysis == 'drivers':
        bplut = restore_bplut(CONFIG['BPLUT']['ET'])
        bplut['beta'] = [250] * bplut['beta'].size
        params_vector = []
        # NOTE: Calculate the average of the parameters
        for key in MOD16.required_parameters:
            params_vector.append(np.nanmean(bplut[key]))

    # For a sensitivity analysis of the parameters
    if analysis == 'parameters':
        filename = OUTPUT_TPL % 'ET'
        if pft is not None:
            filename = OUTPUT_TPL % f'ET-PFT{pft}'
        params = MOD16.required_parameters
        problem = {
            'num_vars': len(params),
            'names': params,
            'bounds': [
                PARAM_BOUNDS[p] for p in params
            ]
        }
        # NOTE: Number of samples must be a power of 2
        param_sweep = sobol_sample(problem, 512 if pft is None else 128)
        Y = np.zeros([param_sweep.shape[0]])
        for i, X in enumerate(tqdm(param_sweep)):
            yhat = MOD16._et(X, *drivers)
            Y[i] = nash_sutcliffe(yhat, tower_obs, norm = True)

    elif analysis == 'drivers':
        filename = OUTPUT_TPL % 'ET-drivers'
        params = BOUNDS.keys()
        problem = {
            'num_vars': len(params),
            'names': params,
            'bounds': list(BOUNDS.values())
        }
        # NOTE: Number of samples must be a power of 2
        param_sweep = sobol_sample(problem, 2048)
        Y = np.zeros([param_sweep.shape[0]])
        # Exclude warnings, because some driver data combinations will lead
        #   to physically implausible situations
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i, X in enumerate(tqdm(param_sweep)):
                Y[i] = MOD16._et(params_vector, *X)

    metrics = sobol.analyze(problem, Y)
    with open(filename, 'w') as file:
        json.dump(dict([(k, v.tolist()) for k, v in metrics.items()]), file)


def load_data(pft, validation_mask_only = False):
    print('Loading driver datasets...')
    lookup = CONFIG['data']['datasets']
    with h5py.File(CONFIG['data']['file'], 'r') as hdf:
        nsteps = hdf['time'].shape[0]
        if pft is not None:
            site_list = hdf['FLUXNET/site_id'][:].tolist()
            if hasattr(site_list[0], 'decode'):
                site_list = [s.decode('utf-8') for s in site_list]
            sites = pft_dominant(hdf['state/PFT'][:], site_list = site_list)
            sites = sites == pft
        else:
            shp = hdf[lookup['Tmin']].shape
            sites = np.ones(shp[1]).astype(bool)
        lw_net_day = hdf[lookup['LWGNT'][0]][:][:,sites]
        lw_net_night = hdf[lookup['LWGNT'][1]][:][:,sites]
        sw_albedo = np.nanmean(
            hdf[lookup['albedo']][:][:,sites], axis = -1)
        sw_rad_day = hdf[lookup['SWGDN'][0]][:][:,sites]
        sw_rad_night = np.zeros(sw_rad_day.shape)
        temp_day = hdf[lookup['T10M'][0]][:][:,sites]
        temp_night = hdf[lookup['T10M'][1]][:][:,sites]
        tmin = hdf[lookup['Tmin']][:][:,sites]
        temp_annual = hdf[lookup['MAT']][:][:,sites]
        if 'VPD' in lookup.keys():
            vpd_day = hdf[lookup['VPD'][0]][:][:,sites]
            vpd_night = hdf[lookup['VPD'][1]][:][:,sites]
        else:
            vpd_day = MOD16.vpd(
                hdf[lookup['QV10M_daytime']][:][:,sites],
                hdf[lookup['PS_daytime']][:][:,sites],
                temp_day)
            vpd_night = MOD16.vpd(
                hdf[lookup['QV10M_nighttime']][:][:,sites],
                hdf[lookup['PS_nighttime']][:][:,sites],
                temp_night)
        # After VPD is calculated, air pressure is based solely
        #   on elevation
        elevation = hdf[lookup['elevation']][:]
        elevation = elevation[np.newaxis,:]\
            .repeat(nsteps, axis = 0)[:,sites]
        pressure = MOD16.air_pressure(elevation.mean(axis = -1))
        # Read in fPAR, LAI, and convert from (%) to [0,1]
        fpar = np.nanmean(hdf[lookup['fPAR']][:][:,sites], axis = -1)
        lai = np.nanmean(hdf[lookup['LAI']][:][:,sites], axis = -1)
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
