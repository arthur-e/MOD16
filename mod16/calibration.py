'''
Calibration of MOD16 against a representative, global eddy covariance (EC)
flux tower network. The model calibration is based on Markov-Chain Monte
Carlo (MCMC). Example use:

    # For a single run with the configured number of chains
    python calibration.py tune --pft=1 --config="my_config.yaml"

    # For a 3-folds cross-validation
    python calibration.py tune --pft=1 --config="my_config.yaml" --k-folds=3

The general calibration protocol used here involves:

1. Check how well the chain(s) are mixing by running short, e.g., only 5000
draws from the posterior:
`python calibration.py tune 1 --draws=5000`
2. If any chain is "sticky," run a short chain while tuning the jump scale:
`python calibration.py tune 1 --tune=scaling --draws=5000`
3. Using the trace plot from Step (2) as a reference, experiment with
different jump scales to try and achieve the same (optimal) mixing when
tuning on `lambda` (default) instead, e.g.:
`python calibration.py tune 1 --scaling=1e-2 --draws=5000`
4. When the right jump scale is found, run a chain at the desired length.

Instead of changing `draws` and `scaling` at the command line, as above, you
could change these parameters in the configuration file.

Once a good mixture is obtained, it is necessary to prune the samples to
eliminate autocorrelation, e.g., from the command line:

    python calibration.py plot-autocorr --pft=1 --burn=1000 --thin=10

Or in Python (use `--ipdb` to get an interactive session):

    # sampler is already available if you used --ipdb, otherwise:
    sampler = MOD16StochasticSampler(...)
    sampler.plot_autocorr(burn = 1000, thin = 10)
    trace = sampler.get_trace(burn = 1000, thin = 10)

A thinned posterior can be exported from the command line, e.g.:

    python calibration.py export-posterior ET <parameter_name>
        output.h5 --burn=1000 --thin=10

NOTE: If using k-folds cross-validation, add the following option to any
command, where K is the number of folds:

    --k-folds=K

**The Cal-Val dataset** is a single HDF5 file that contains all the input
variables necessary to drive MOD16 as well as the observed latent heat fluxes
against which the model is calibrated. The HDF5 file specification is as
follows, where the shape of multidimensional arrays is given in terms of
T, the number of time steps (days); N, the number of tower sites; L, the
number of land-cover types (PFTs); and P, a sub-grid of MODIS pixels
surrounding a tower:


    FLUXNET/
      SEB               -- (T x N) Surface energy balance, from tower data
      air_temperature   -- (T x N) Air temperatures reported at the tower
      *latent_heat      -- (T x N) Observed latent heat flux [W m-2]
      site_id           -- (N) Unique identifier for each site, e.g., "US-BZS"
      validation_mask   -- (L x T x N) Indicates what site-days are reserved

    *MERRA2/
      LWGNT             -- (T x N) Net long-wave radiation, 24-hr mean [W m-2]
      LWGNT_daytime     -- (T x N) ... for daytime hours only
      LWGNT_nighttime   -- (T x N) ... for nighttime hours only
      PS                -- (T x N) Surface air pressure [Pa]
      PS_daytime        -- (T x N) ... for daytime hours only
      PS_nighttime      -- (T x N) ... for nighttime hours only
      QV10M             -- (T x N) Water vapor mixing ratio at 10-meter height
      QV10M_daytime     -- (T x N) ... for daytime hours only
      QV10M_nighttime   -- (T x N) ... for nighttime hours only
      SWGDN             -- (T x N) Down-welling short-wave radiation [W m-2]
      SWGDN_daytime     -- (T x N) ... for daytime hours only
      SWGDN_nighttime   -- (T x N) ... for nighttime hours only
      T10M              -- (T x N) Air temperature at 10-meter height [deg K]
      T10M_daytime      -- (T x N) ... for daytime hours only
      T10M_nighttime    -- (T x N) ... for nighttime hours only
      Tmin              -- (T x N) Daily minimum air temperature [deg K]

    *MODIS/
      *MCD43GF_black_sky_sw_albedo
          -- (T x N x P) Short-wave albedo under black-sky conditions
      *MOD15A2HGF_LAI
          -- (T x N x P) Leaf area index in scaled units (10 * [m3 m-3])
      *MOD15A2HGF_fPAR
          -- (T x N x P) Fraction of photosynthetically active radiation [%]

    *IMERG/
      *mean_annual_precip
          -- (Y x N) Mean annual precipitation in each year (Y years)

    coordinates/
      lng_lat       -- (2 x N) Longitude, latitude coordinates of each tower

    state/
      *PFT          -- (N x P) The plant functional type (PFT) of each pixel
      elevation_m   -- (N) The elevation in meters above sea level

    time            -- (T x 3) The Year, Month, Day of each daily time step
    weights         -- (N) A number between 0 and 1 used to down-weight towers


NOTE: A star, `*`, indicates that this dataset or group's name can be changed
in the configuration file. All others are currently required to match this
specification exactly.

NOTE: For now, constraints (like an annual precipitation constraint) cannot be
used unless `classes_are_dynamic = True` in the configuration file. This is
because the only supported constraint (annual precipitation) requires a
`(T x N)` data structure.

NOTE: The biggest improvement needed here is a way for users to specify not
just the values of the prior and which parameters are fixed but also the
functional form of the prior; currently, this is hard-coded into the
`compile_et()` function.
'''

import datetime
import yaml
import os
import numpy as np
import h5py
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import mod16
from collections import OrderedDict
from functools import partial
from statistics import mode
from pathlib import Path
from typing import Sequence
from scipy import signal
from matplotlib import pyplot
from mod16 import MOD16, latent_heat_vaporization
from mod16.utils import restore_bplut, pft_dominant
from mod17.calibration import BlackBoxLikelihood, StochasticSampler

MOD16_DIR = os.path.dirname(mod16.__file__)
DRIVER_NAMES = ('lw_net_day', 'lw_net_night', 'sw_rad_day',
    'sw_rad_night', 'sw_albedo', 'temp_day', 'temp_night',
    'temp_annual', 'tmin', 'vpd_day', 'vpd_night',
    'pressure', 'fpar', 'lai')


class MOD16StochasticSampler(StochasticSampler):
    '''
    A Markov Chain-Monte Carlo (MCMC) sampler for MOD16. The specific sampler
    used is the Differential Evolution (DE) MCMC algorithm described by
    Ter Braak (2008), though the implementation is specific to the PyMC3
    library.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters
    model : Callable
        The function to call (with driver data and parameters); this function
        should take driver data as positional arguments and the model
        parameters as a `*Sequence`; it should require no external state.
    observed : Sequence
        Sequence of observed values that will be used to calibrate the model;
        i.e., model is scored by how close its predicted values are to the
        observed values
    params_dict : dict or None
        Dictionary of model parameters, to be used as initial values and as
        the basis for constructing a new dictionary of optimized parameters
    backend : str or None
        Path to a NetCDF4 file backend (Default: None)
    weights : Sequence or None
        Optional sequence of weights applied to the model residuals (as in
        weighted least squares)
    '''
    required_parameters = {
        'ET': MOD16.required_parameters
    }
    required_drivers = {
        'ET': [
            'lw_net_day', 'lw_net_night', 'sw_rad_day', 'sw_rad_night',
            'sw_albedo', 'temp_day', 'temp_night', 'temp_annual', 'tmin',
            'vpd_day', 'vpd_night', 'pressure', 'fpar', 'lai'
        ]
    }

    def compile_et_model(
            self, observed: Sequence, drivers: Sequence) -> pm.Model:
        '''
        Creates a new ET model based on the prior distribution. Model can be
        re-compiled multiple times, e.g., for cross validation.

        There are two attributes that are set on the sampler when it is
        initialized that could be helpful here:

            self.priors
            self.bounds

        `self.priors` is a dict with a key for each parameter that has
        informative priors. For parameters with a non-informative (Uniform)
        prior, `self.bounds` is a similar dict (with a key for each parameter)
        that describes the lower and upper bounds of the Uniform prior, but
        this is deprecated.

        Parameters
        ----------
        observed : Sequence
            Sequence of observed values that will be used to calibrate the model;
            i.e., model is scored by how close its predicted values are to the
            observed values
        drivers : list or tuple
            Sequence of driver datasets to be supplied, in order, to the
            model's run function

        Returns
        -------
        pm.Model
        '''
        # Define the objective/ likelihood function
        log_likelihood = BlackBoxLikelihood(
            self.model, observed, x = drivers, weights = self.weights,
            objective = self.config['optimization']['objective'],
            constraints = self.constraints)
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # NOTE: Parameters shared with MOD17 are fixed based on MOD17
            #   re-calibration
            tmin_close = self.params['tmin_close']
            tmin_open = self.params['tmin_open']
            vpd_open = self.params['vpd_open']
            vpd_close =   pm.Uniform('vpd_close', **self.prior['vpd_close'])
            gl_sh =       pm.LogNormal('gl_sh', **self.prior['gl_sh'])
            gl_wv =       pm.LogNormal('gl_wv', **self.prior['gl_wv'])
            g_cuticular = pm.LogNormal(
                'g_cuticular', **self.prior['g_cuticular'])
            csl =         pm.LogNormal('csl', **self.prior['csl'])
            rbl_min =     pm.Triangular('rbl_min', **self.prior['rbl_min'])
            rbl_max =     pm.Triangular('rbl_max', **self.prior['rbl_max'])
            beta =        pm.Uniform('beta', **self.prior['beta'])
            # (Stochstic) Priors for unknown model parameters
            params_list = [
                tmin_close, tmin_open, vpd_open, vpd_close, gl_sh, gl_wv,
                g_cuticular, csl, rbl_min, rbl_max, beta
            ]
            # Convert model parameters to a tensor vector
            params = pt.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        # If the value for this parameter (and this PFT) is fixed...
        fixed = dict()
        for i, name in enumerate(self.required_parameters['ET']):
            if self.fixed is not None:
                if name in self.fixed.keys():
                    if self.fixed[name] is not None:
                        # e.g., {beta: fixed_value}
                        fixed[getattr(model, name)] = self.fixed[name]
        if len(fixed) > 0:
            # i.e., Return "a distinct PyMC model with the relevant variables
            #   replaced by the intervention expressions; all remaining
            #   variables are cloned"
            return pm.do(model, fixed)
        return model


class CalibrationAPI(object):
    '''
    Convenience class for calibrating the MOD16 ET model. Meant to be used
    at the command line, in combination with the option to specify a
    configuration file:

        --config=my_configuration.yaml

    For example, to run calibration for PFT 1, you would write:

        python calibration.py tune --pft=1 --config=my_configuration.yaml

    If `--config` is not provided, the default configuration file,
    `mod16/MOD16_calibration_config.yaml` will be used.
    '''

    def __init__(self, config = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                MOD16_DIR, 'data/MOD16_calibration_config.yaml')
        print(f'Using configuration file: "{config_file}"')
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.hdf5 = self.config['data']['file']

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def _load_data(self, pft: int):
        'Read in driver datasets from the HDF5 file'
        with h5py.File(self.hdf5, 'r') as hdf:
            sites = hdf['FLUXNET/site_id'][:].tolist()
            if hasattr(sites[0], 'decode'):
                sites = [s.decode('utf-8') for s in sites]
            # Number of time steps
            nsteps = hdf['time'].shape[0]
            # In case some tower sites should not be used
            blacklist = self.config['data']['sites_blacklisted']
            # Get dominant PFT across a potentially heterogenous sub-grid,
            #   centered on the eddy covariance flux tower, UNLESS we have a
            #   dynamic land-cover map, in which case it is assumed
            #   (required) that there is only one PFT value per site
            pft_array = hdf[self.config['data']['class_map']][:]
            # For a static PFT map, sub-site land-cover heterogeneity is
            #   allowed; get the dominant (single) PFT at each site
            pft_map = pft_dominant(pft_array, site_list = sites)
            # But do create a (T x N) selection mask
            pft_map = pft_map[np.newaxis,:].repeat(nsteps, axis = 0)

            # Get a binary mask that indicates which tower-days should be used
            #   to calibrate the current PFT class
            if blacklist is not None:
                pft_mask = np.logical_and(
                    pft_map == pft, ~np.in1d(sites, blacklist))
            else:
                pft_mask = pft_map == pft

            # Get tower weights, for when towers are too close together
            weights = hdf['weights'][:]
            # If only a single value is given for each site, repeat the weight
            #   along the time axis
            if weights.ndim == 1:
                weights = weights[None,:].repeat(nsteps, axis = 0)[pft_mask]

            # Read in tower observations
            tower_obs = hdf['FLUXNET/latent_heat'][:]
            # Clean the tower observations (smoothing)
            tower_obs = self.clean_observed(tower_obs)

            # Read the validation mask; mask out observations that are
            #   reserved for validation
            print('Masking out validation data...')
            mask = hdf['FLUXNET/validation_mask'][pft]
            if np.isnan(tower_obs[~mask]).all():
                print(f'WARNING: Validation mask for PFT={pft} leaves no valid data for training; ignoring mask')
            else:
                tower_obs[mask] = np.nan
            # Subset to the PFT of interest
            tower_obs = tower_obs[pft_mask]

            # Read in driver datasets
            print('Loading driver datasets...')
            lookup = self.config['data']['datasets']
            lw_net_day = hdf[lookup['LWGNT'][0]][:][pft_mask]
            lw_net_night = hdf[lookup['LWGNT'][1]][:][pft_mask]
            sw_albedo = hdf[lookup['albedo']][:][pft_mask]
            sw_rad_day = hdf[lookup['SWGDN'][0]][:][pft_mask]
            sw_rad_night = np.zeros(sw_rad_day.shape, dtype = np.int16)
            temp_day = hdf[lookup['T10M'][0]][:][pft_mask]
            temp_night = hdf[lookup['T10M'][1]][:][pft_mask]
            tmin = hdf[lookup['Tmin']][:][pft_mask]

            # As long as the time series is balanced w.r.t. years (i.e., same
            #   number of records per year), the overall mean is the annual mean
            temp_annual = hdf[lookup['MAT']][:][pft_mask].mean(axis = 0)
            if 'VPD' in lookup.keys():
                vpd_day = hdf[lookup['VPD'][0]][:][pft_mask]
                vpd_night = hdf[lookup['VPD'][1]][:][pft_mask]
            else:
                vpd_day = MOD16.vpd(
                    hdf[lookup['QV10M'][0]][:][pft_mask],
                    hdf[lookup['PS'][0]][:][pft_mask], temp_day)
                vpd_night = MOD16.vpd(
                    hdf[lookup['QV10M'][1]][:][pft_mask],
                    hdf[lookup['PS'][1]][:][pft_mask], temp_night)
            vpd_night = np.where(vpd_night < 0, 0, vpd_night)

            # After VPD is calculated, air pressure is based solely
            #   on elevation
            elevation = hdf[lookup['elevation']][:]
            elevation = elevation[np.newaxis,:]\
                .repeat(nsteps, axis = 0)[pft_mask]
            pressure = MOD16.air_pressure(elevation.mean(axis = -1))

            # Read in fPAR, LAI, and convert from (%) to [0,1]
            fpar = hdf[lookup['fPAR']][:][pft_mask]
            lai = hdf[lookup['LAI']][:][pft_mask]

            # If a heterogeneous sub-grid is used at each tower (i.e., there
            #   is a third axis to these datasets), then average over that
            #   sub-grid
            if sw_albedo.ndim == 2 and fpar.ndim == 2 and lai.ndim == 2:
                sw_albedo = np.nanmean(sw_albedo, axis = -1)
                fpar = np.nanmean(fpar, axis = -1)
                lai = np.nanmean(lai, axis = -1)
            # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
            fpar /= 100
            lai /= 10

        # Compile driver datasets
        drivers = dict(zip(DRIVER_NAMES, [
            lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
            temp_day, temp_night, temp_annual, tmin, vpd_day, vpd_night,
            pressure, fpar, lai]))
        return (tower_obs, drivers, weights)

    def _load_data_annual(self, pft: int):
        'Read in driver datasets from the HDF5 file, structured by years'
        constraints = dict()
        with h5py.File(self.hdf5, 'r') as hdf:
            sites = hdf['FLUXNET/site_id'][:].tolist()
            if hasattr(sites[0], 'decode'):
                sites = [s.decode('utf-8') for s in sites]
            # Number of time steps
            nsteps = hdf['time'].shape[0]
            # In case some tower sites should not be used
            blacklist = self.config['data']['sites_blacklisted']
            pft_map = hdf[self.config['data']['class_map']][:]
            # Also, ensure the blacklist matches the shape of this mask;
            #   i.e., blacklisted sites should NEVER be used
            if blacklist is not None:
                if len(blacklist) > 0:
                    blacklist = np.array(blacklist)
                    blacklist = blacklist[None,:].repeat(pft_map.shape[0], axis = 0)

            years = np.array([
                datetime.date(*ymd).year for ymd in hdf['time'][:].tolist()
            ])
            if pft_map.shape[0] != years.size:
                # NOTE: This is a generalization of a very specific case,
                #   PFT = 3 (Deciduous Needleleaf), which is not a dominant
                #   land-cover class anywhere
                if not (pft_map == pft).any():
                    # Assume the problem will be fixed by special-case
                    #   handling, i.e., assigning PFT to certain sites
                    pft_map = pft_dominant(pft_map, sites)
                print('WARNING: First axis of class_map does not match the length of the "time" vector; reshaping to fit')
                pft_map = pft_map[np.newaxis,...].repeat(years.size, axis = 0)
            # If any days of this year correspond to the current PFT, select
            #   that year for calibration
            year_masks = []
            valid = self.config['data']['classes'] # Valid PFT classes
            for y in np.unique(years):
                # NOTE: This is where we select the current year
                pft_map_now = pft_map[years == y].swapaxes(0,1)
                pft_map_now = pft_map_now.reshape(
                    (pft_map_now.shape[0], pft_map_now.size // pft_map_now.shape[0]))
                # Update the PFT map (this year) to handle invalid PFT classes
                for i in range(pft_map_now.shape[0]):
                    if not np.in1d(pft_map_now[i], valid).all():
                        # Replace invalid PFTs (this year) with most common
                        #   valid PFT
                        potential = pft_map_now[i,np.in1d(pft_map_now[i], valid)]
                        if potential.size == 0:
                            continue # Skip this for now
                        pft_map_now[i,~np.in1d(pft_map_now[i], valid)] =\
                            mode(potential)
                # Now, with (mostly) good PFTs (this year), find the dominant
                pft_map_now = np.apply_along_axis(mode, 1, pft_map_now)
                year_masks.append(pft_map_now == pft)

            year_masks = np.stack(year_masks, axis = 0)
            # "Average" over the subgrid, then select years where a site has
            #   that PFT for any part of the year; to verify that this has
            #   resulted in an array that indicates which (full) years a PFT
            #   occurs at each site, try:
            # np.apply_along_axis(lambda x: x.sum(), 0, year_masks)
            pft_mask = year_masks[years - years.min()]
            # Set as False any tower-days where the tower is blacklisted
            if blacklist is not None:
                pft_mask[:,np.in1d(sites, blacklist)] = False
            # Finally, get an (N,) selection mask for the towers we want
            site_mask = pft_mask.any(axis = 0)

            # Get tower weights, for when towers are too close together
            weights = hdf['weights'][:]
            # If only a single value is given for each site, repeat the weight
            #   along the time axis
            if weights.ndim == 1:
                weights = weights[None,...].repeat(nsteps, axis = 0)
            weights = weights[:,site_mask]

            # Read in tower observations; we select obs of interest in three
            #   steps because we want *only* matching tower-day observations
            #   but we'll want driver data for a full year if that year
            #   contains *any* matching tower-day observations
            print('Masking out validation data...')
            tower_obs = hdf['FLUXNET/latent_heat'][:]
            tower_obs[~pft_mask] = np.nan # Step 1: Mask out invalid days
            # Step 2: Mask out validation samples
            tower_obs[hdf['FLUXNET/validation_mask'][pft]] = np.nan
            tower_obs = tower_obs[:,site_mask] # Step 3: Select matching sites

            # Read in driver datasets
            print('Loading driver datasets...')
            lookup = self.config['data']['datasets']
            lw_net_day = hdf[lookup['LWGNT'][0]][:][:,site_mask]
            lw_net_night = hdf[lookup['LWGNT'][1]][:][:,site_mask]
            sw_albedo = hdf[lookup['albedo']][:][:,site_mask]
            sw_rad_day = hdf[lookup['SWGDN'][0]][:][:,site_mask]
            sw_rad_night = np.zeros(sw_rad_day.shape, dtype = np.int16)
            temp_day = hdf[lookup['T10M'][0]][:][:,site_mask]
            temp_night = hdf[lookup['T10M'][1]][:][:,site_mask]
            tmin = hdf[lookup['Tmin']][:][:,site_mask]

            if self.config['constraints']['annual_precipitation']:
                # Convert mean annual precip (Y x N) to a (T x N) array,
                #   then subset
                ann_precip = hdf[lookup['annual_precip']][:]
                constraints['annual_precipitation'] = (years, ann_precip[:,site_mask])

            # Assumed that annual mean is (T x N) array, with the same
            #   annual value copied to each time step if time steps are
            #   more frequent than one year
            temp_annual = hdf[lookup['MAT']][:][:,site_mask]
            if 'VPD' in lookup.keys():
                vpd_day = hdf[lookup['VPD'][0]][:][:,site_mask]
                vpd_night = hdf[lookup['VPD'][1]][:][:,site_mask]
            else:
                vpd_day = MOD16.vpd(
                    hdf[lookup['QV10M'][0]][:][:,site_mask],
                    hdf[lookup['PS'][0]][:][:,site_mask], temp_day)
                vpd_night = MOD16.vpd(
                    hdf[lookup['QV10M'][1]][:][:,site_mask],
                    hdf[lookup['PS'][1]][:][:,site_mask], temp_night)
            vpd_night = np.where(vpd_night < 0, 0, vpd_night)

            # After VPD is calculated, air pressure is based solely
            #   on elevation
            elevation = hdf[lookup['elevation']][:]
            elevation = elevation[np.newaxis,:]\
                .repeat(nsteps, axis = 0)[:,site_mask]
            # Assumed to be (T x N) or (T x N x ...)
            if elevation.ndim == 3:
                # If there is a site sub-grid...
                elevation = elevation.mean(axis = -1)
            pressure = MOD16.air_pressure(elevation)

            # Read in fPAR, LAI, and convert from (%) to [0,1]
            fpar = hdf[lookup['fPAR']][:][:,site_mask]
            lai = hdf[lookup['LAI']][:][:,site_mask]

            # If a heterogeneous sub-grid is used at each tower (i.e., there
            #   is a third axis to these datasets), then average over that
            #   sub-grid
            if sw_albedo.ndim == 3 and fpar.ndim == 3 and lai.ndim == 3:
                sw_albedo = np.nanmean(sw_albedo, axis = -1)
                fpar = np.nanmean(fpar, axis = -1)
                lai = np.nanmean(lai, axis = -1)
            # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
            fpar /= 100
            lai /= 10

        drivers = dict(zip(DRIVER_NAMES, [
            lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
            temp_day, temp_night, temp_annual, tmin, vpd_day, vpd_night,
            pressure, fpar, lai]))
        # Clean the tower observations
        tower_obs = self.clean_observed(tower_obs)
        return (tower_obs, drivers, weights, constraints)

    def clean_observed(
            self, raw: Sequence, filter_length: int = 2) -> Sequence:
        '''
        Cleans observed tower flux data according to a prescribed protocol.
        NOT intended to be called from the command line.

        Parameters
        ----------
        raw : Sequence
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data

        Returns
        -------
        Sequence
        '''
        # Read in the observed data and apply smoothing filter; then mask out
        #   negative latent heat observations
        assert raw.ndim == 2, 'Expected 2D input raw observations'
        obs = self._filter(raw, filter_length)
        # Filtering out negative latent heat fluxes, as these generally arise
        #   during stable conditions, when EC tower measurements are less
        #   reliable (see https://doi.org/10.5194/bg-21-2051-2024)
        return np.where(obs < 0, np.nan, obs)

    def export_posterior(
            self, model: str, param: str, output_path: str, thin: int = 10,
            burn: int = 1000, k_folds: int = 1):
        '''
        Exports posterior distribution for a parameter, for each PFT to HDF5.

        Parameters
        ----------
        model : str
            The name of the model (only "ET" is supported)
        param : str
            The model parameter to export
        output_path : str
            The output HDF5 file path
        thin : int
            Thinning rate
        burn : int
            The burn-in (i.e., first N samples to discard)
        k_folds : int
            The number of k-folds used in cross-calibration/validation;
            if more than one (default), the folds for each PFT will be
            combined into a single HDF5 file
        '''
        params_dict = restore_bplut(self.config['BPLUT'][model])
        bplut = params_dict.copy()
        # Filter the parameters to just those for the PFT of interest
        post = []
        for pft in self.config['data']['classes']:
            params = dict([(k, v[pft]) for k, v in params_dict.items()])
            post_by_fold = []
            for fold in range(1, k_folds + 1):
                backend = self.config['optimization']['backend_template'] %\
                    (model, pft)
                if k_folds > 1:
                    backend = backend[:backend.rfind('.')] + f'-k{fold}' + backend[backend.rfind('.'):]
                # NOTE: This value was hard-coded in the extant version of MOD16
                if 'beta' not in params:
                    params['beta'] = 250
                sampler = MOD16StochasticSampler(
                    self.config, getattr(MOD16, '_%s' % model.lower()), params,
                    backend = backend)
                trace = sampler.get_trace()
                fit = trace.sel(draw = slice(burn, None, thin))['posterior']
                if param in fit:
                    post_by_fold.append(
                        az.extract_dataset(fit, combined = True)[param].values)
                else:
                    # In case there is, e.g., a parameter that takes on a
                    #   constant value for a specific PFT
                    if k_folds > 1:
                        post_by_fold.append(
                            np.ones((1, post[-1].shape[-1])) * np.nan)
                    else:
                        a_key = list(fit.keys())[0]
                        post_by_fold.append(
                            np.ones(fit[a_key].values.shape) * np.nan)
            if k_folds > 1:
                post.append(np.vstack(post_by_fold))
            else:
                post.extend(post_by_fold)
        # If not every PFT's posterior has the same number of samples (e.g.,
        #   when one set of chains was run longer than another)...
        if not all([p.shape == post[0].shape for p in post]):
            max_len = max([p.shape for p in post])[0]
            # ...Reshape all posteriors to match the greatest sample size
            post = [
                np.pad(
                    p.astype(np.float32), (0, max_len - p.size),
                    mode = 'constant', constant_values = (np.nan,))
                for p in post
            ]
        with h5py.File(output_path, 'a') as hdf:
            post = np.stack(post)
            ts = datetime.date.today().strftime('%Y-%m-%d') # Today's date
            dataset = hdf.create_dataset(
                f'{param}_posterior', post.shape, np.float32, post)
            dataset.attrs['description'] = 'CalibrationAPI.export_posterior() on {ts}'

    def plot_autocorr(self, pft: int, k_folds: int = 1, **kwargs):
        '''
        Plot the autocorrelation in the trace for each parameter.

        Parameters
        ----------
        pft : int
            The numeric PFT code
        '''
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut(self.config['BPLUT']['ET'])
        params_dict = dict([(k, v[pft]) for k, v in params_dict.items()])
        backend = self.config['optimization']['backend_template'] % ('ET', pft)
        # Use a different naming scheme for the backend
        if k_folds > 1:
            for fold in range(1, k_folds + 1):
                sampler = MOD16StochasticSampler(
                    self.config, MOD16._et, params_dict,
                    backend = backend[:backend.rfind('.')] + f'-k{fold}' + backend[backend.rfind('.'):])
                sampler.plot_autocorr(**kwargs, title = f'Fold {fold} of {k_folds}')
        else:
            sampler = MOD16StochasticSampler(
                self.config, MOD16._et, params_dict, backend = backend)
            sampler.plot_autocorr(**kwargs)

    def tune(
            self, pft: int, plot_trace: bool = False, k_folds: int = 1,
            ipdb: bool = False, save_fig: bool = False, **kwargs):
        '''
        Run the MOD16 ET calibration. If k-folds cross-validation is used,
        the model is calibrated on $k$ random subsets of the data and a
        series of file is created, e.g., as:

            MOD16_ET_calibration_PFT1.h5
            MOD16_ET_calibration_PFT1-k1.nc4
            MOD16_ET_calibration_PFT1-k2.nc4
            ...

        Where each `.nc4` file is a standard `arviz` backend and the `.h5`
        indicates which indices from the observations vector, after removing
        NaNs, were excluded (i.e., the indices of the test data).

        Parameters
        ----------
        pft : int
            The Plant Functional Type (PFT) to calibrate
        plot_trace : bool
            True to plot the trace for a previous calibration run; this will
            also NOT start a new calibration (Default: False)
        k_folds : int
            Number of folds to use in k-folds cross-validation; defaults to
            k=1, i.e., no cross-validation is performed.
        ipdb : bool
            True to drop the user into an ipdb prompt, prior to and instead of
            running calibration
        save_fig : bool
            True to save figures to files instead of showing them
            (Default: False)
        **kwargs
            Additional keyword arguments passed to
            `MOD16StochasticSampler.run()`

        NOTE that `MOD16StochasticSampler` inherits methods from the `mod17`
        module, including [run()](https://arthur-e.github.io/MOD17/calibration.html#mod17.calibration.StochasticSampler).
        '''
        def constrain_by_map(pred_le, years, lhv, annual_precip):
            # Constrain the results by (mean) annual precipitation
            # pred_le : np.ndarray - (T,N) array of predicted latent heat flux
            # years : np.ndarray - (T,) array indicating year, out of Y years
            # lhv : np.ndarray - (T,N) array of latent heat of vaporization
            # annual_precip : np.ndarray -
            #   (Y,N) array of the annual precipitation at the site
            mass_rate = (pred_le * 60 * 60 * 24) / lhv # Convert [W m-2] to [mm day-1]
            mass_rate[mass_rate < 0] = 0
            annual_mass_rate = []
            for y in np.unique(years):
                a = np.apply_along_axis(
                    lambda x: x.sum(), 0, mass_rate[years == y])
                annual_mass_rate.append(a)
            pred_precip = np.stack(annual_mass_rate, axis = 0)
            diff = pred_precip - annual_precip
            diff = np.where(diff < 0, 0, diff)
            # Return the (negative) normalized RMSD; it's negative because
            #   we are maximizing the objective function
            nrmsd = 100 * ((diff**2).mean() / annual_precip.sum())
            return -nrmsd

        assert pft in self.config['data']['classes'], f'Invalid PFT: {pft}'
        # Pass configuration parameters to MOD16StochasticSampler.run()
        for key in ('chains', 'draws', 'tune', 'scaling'):
            if key in self.config['optimization'].keys():
                kwargs[key] = self.config['optimization'][key]
        # Filter the parameters to just those for the PFT of interest
        params_dict = restore_bplut(self.config['BPLUT']['ET'])
        params_dict = dict([(k, v[pft]) for k, v in params_dict.items()])
        # NOTE: This value was hard-coded in the extant version of MOD16
        if np.isnan(params_dict['beta']):
            params_dict['beta'] = 250

        # There may be additional constraints when dynamic classes are used
        if self.config['data']['classes_are_dynamic']:
            tower_obs, drivers, weights, constr = self._load_data_annual(pft)
        else:
            tower_obs, drivers, weights = self._load_data(pft)

        # TODO The current k-folds code assumes that the observations
        #   and driver datasets are 1D vectors; however, the functions
        #   _load_data() and _load_data_annual() return datasets with
        #   different dimensions; the former returns 1D vectors but
        #   the latter returns 2D arrays
        if self.config['data']['classes_are_dynamic'] and k_folds > 1:
            tower_obs = tower_obs.ravel()
            weights = weights.ravel()
            for key in drivers.keys():
                drivers[key] = drivers[key].ravel()

        constraints = None
        if len(self.config['constraints']) > 0:
            # Constraints may be defined in the config file but actually
            #   set to false; if none are set true, there are no constraints
            if any([
                self.config['constraints'][k]
                for k in self.config['constraints'].keys()
            ]):
                constraints = []
            if self.config['constraints']['annual_precipitation']:
                # Get the mean daily temperature, to derive LHV
                air_t_day = drivers['temp_day']
                air_t_night = drivers['temp_night']
                air_t = np.stack(
                    [air_t_day, air_t_night], axis = 0).mean(axis = 0)
                lhv = latent_heat_vaporization(air_t)
                # Unpack sequence of years (e.g., 2000, 2001, ...), precip
                #   data (mm year-1)
                years, ann_precip = constr['annual_precipitation']
                # Add a version of constrain_by_map() function as a constraint
                constraints.append(partial(
                    constrain_by_map, years = years, lhv = lhv,
                    annual_precip = ann_precip))

        if k_folds > 1:
            print(f'NOTE: Using k-folds CV with k={k_folds}...')
            # Back-up the original (complete) datasets; we do this so we can
            #   simply mask out the test samples (1/k), after restoring the
            #   original datasets
            _drivers = dict([(k, d.copy()) for k, d in drivers.items()])
            _tower_obs = tower_obs.copy()
            _weights = weights.copy()
            # Randomize the indices of the NPP data
            indices = np.arange(0, tower_obs.size)
            np.random.shuffle(indices)
            # Get the starting and ending index of each fold
            fold_idx = np.array([indices.size // k_folds] * k_folds) * np.arange(0, k_folds)
            fold_idx = list(map(list, zip(fold_idx, fold_idx + indices.size // k_folds)))
            # Ensure that the entire dataset is used; i.e., if each fold takes
            #   slices of the indices from A to B, ensure that the last fold's
            #   B is the final (maximum) index of the sequence
            fold_idx[-1][-1] = indices.max()
            idx_test = [indices[start:end] for start, end in fold_idx]

        # Loop over each fold (or the entire dataset, if num. folds == 1)
        for k, fold in enumerate(range(1, k_folds + 1)):
            backend = self.config['optimization']['backend_template'] % ('ET', pft)
            if k_folds > 1 and fold == 1:
                # Create an HDF5 file with the same name as the (original)
                #   netCDF4 back-end, store the test indices
                with h5py.File(backend.replace('nc4', 'h5'), 'w') as hdf:
                    out = list(idx_test)
                    size = indices.size // k_folds
                    try:
                        out = np.stack(out)
                    except ValueError:
                        size = max((o.size for o in out))
                        for i in range(0, len(out)):
                            out[i] = np.concatenate((out[i], [np.nan] * (size - out[i].size)))
                    hdf.create_dataset(
                        'test_indices', (k_folds, size), np.uint32, np.stack(out))
                # Restore the original tower dataset
                if fold > 1:
                    tower_obs = _tower_obs.copy()
                    weights = _weights.copy()
                # Set to NaN all the test indices
                idx = idx_test[k]
                tower_obs[idx] = np.nan
                # Same for drivers, after restoring from the original
                drivers = dict([
                    (k, d.copy()[~np.isnan(tower_obs)]) if d.ndim > 0 else (k, d.copy())
                    for k, d in _drivers.items()
                ])
                weights = weights[~np.isnan(tower_obs)] # NOTE: Do first
                tower_obs = tower_obs[~np.isnan(tower_obs)]
            # Use a different naming scheme for the backend
            if k_folds > 1:
                backend = backend[:backend.rfind('.')] + f'-k{fold}' + backend[backend.rfind('.'):]

            print('Initializing sampler...')
            sampler = MOD16StochasticSampler(
                self.config, MOD16._et, params_dict, backend = backend,
                weights = weights, constraints = constraints)

            # Either: Enter diagnostic mode or run the sampler
            if plot_trace or ipdb:
                # This matplotlib setting prevents labels from overplotting
                pyplot.rcParams['figure.constrained_layout.use'] = True
                trace = sampler.get_trace()
                if ipdb:
                    import ipdb
                    ipdb.set_trace()#FIXME
                az.plot_trace(trace, var_names = MOD16.required_parameters)
                pyplot.show()
                return

            # Get (informative) priors for just those parameters that have them
            with open(self.config['optimization']['prior'], 'r') as file:
                prior = yaml.safe_load(file)
            prior_params = list(filter(
                lambda p: p in prior.keys(), sampler.required_parameters['ET']))
            prior = dict([
                (p, dict([(k, v[pft]) for k, v in prior[p].items()]))
                for p in prior_params
            ])

            # Determine whether any parameters are fixed
            fixed = []
            for name in MOD16.required_parameters:
                if self.config['optimization']['fixed'] is None:
                    break
                if name in self.config['optimization']['fixed'].keys():
                    fixed.append(
                        (name, self.config['optimization']['fixed'][name][pft]))
            fixed = dict(fixed)

            # Set var_names to tell ArviZ to plot only the free parameters; i.e.,
            #   those with priors and which are not fixed
            var_names = list(filter(
                lambda x: x in prior.keys(), MOD16.required_parameters))
            # Remove any random variables that have fixed values from the list
            #   of variables to be plotted
            for key in fixed.keys():
                if fixed[key] is not None and key in var_names:
                    var_names.remove(key)
            kwargs.update({'var_names': var_names})

            # TODO Someday, MOD17 will be updated to allow "drivers" to be a
            #   dictionary instead of a sequence; until then: drivers.values()
            sampler.run( # Only show the trace plot if not using k-folds
                tower_obs, drivers.values(), prior = prior, fixed = fixed,
                save_fig = save_fig, show_fig = (k_folds == 1), **kwargs)


if __name__ == '__main__':
    import fire
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
