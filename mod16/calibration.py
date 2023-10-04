'''
Calibration of MOD16 against a representative, global eddy covariance (EC)
flux tower network. The model calibration is based on Markov-Chain Monte
Carlo (MCMC). Example use:

    python calibration.py tune --pft=1
    python calibration.py tune --pft=1 --config=MOD16_calibration_config.json

The default configuration file provided in the repository,
`mod16/data/MOD16_calibration_config.json`, should be copied and modified to
suit your needs.

The general calibration protocol used here involves:

1. Check how well the chain(s) are mixing by running short:
`python calibration.py tune 1 --draws=5000`
2. If any chain is "sticky," run a short chain while tuning the jump scale:
`python calibration.py tune 1 --tune=scaling --draws=5000`
3. Using the trace plot from Step (2) as a reference, experiment with
different jump scales to try and achieve the same (optimal) mixing when
tuning on `lambda` (default) instead, e.g.:
`python calibration.py tune 1 --scaling=1e-2 --draws=5000`
4. When the right jump scale is found, run a chain at the desired length.

Once a good mixture is obtained, it is necessary to prune the samples to
eliminate autocorrelation, e.g., in Python:

    sampler = MOD16StochasticSampler(...)
    sampler.plot_autocorr(burn = 1000, thin = 10)
    trace = sampler.get_trace(burn = 1000, thin = 10)

A thinned posterior can be exported from the command line:

    python calibration.py export-bplut output.csv --burn=1000 --thin=10
'''

import datetime
import json
import os
import numpy as np
import h5py
import pymc as pm
import aesara.tensor as at
import arviz as az
import mod16
from multiprocessing import get_context, set_start_method
from pathlib import Path
from typing import Sequence
from scipy import signal
from matplotlib import pyplot
from mod16 import MOD16
from mod16.utils import restore_bplut, pft_dominant
from mod17 import PFT_VALID
from mod17.calibration import BlackBoxLikelihood, StochasticSampler

MOD16_DIR = os.path.dirname(mod16.__file__)


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
            self.model, observed, x = drivers, weights = self.weights)
        # With this context manager, "all PyMC3 objects introduced in the indented
        #   code block...are added to the model behind the scenes."
        with pm.Model() as model:
            # NOTE: Parameters shared with MOD17 are fixed based on MOD17
            #   re-calibration
            tmin_close = self.params['tmin_close']
            tmin_open = self.params['tmin_open']
            vpd_open = self.params['vpd_open']
            vpd_close = self.params['vpd_close']
            gl_sh =       pm.Triangular('gl_sh', **self.prior['gl_sh'])
            gl_wv =       pm.Triangular('gl_wv', **self.prior['gl_wv'])
            g_cuticular = pm.LogNormal(
                'g_cuticular', **self.prior['g_cuticular'])
            csl =         pm.LogNormal('csl', **self.prior['csl'])
            rbl_min =     pm.Triangular('rbl_min', **self.prior['rbl_min'])
            rbl_max =     pm.Triangular('rbl_max', **self.prior['rbl_max'])
            beta =        pm.Triangular('beta', **self.prior['beta'])
            # (Stochstic) Priors for unknown model parameters
            # Convert model parameters to a tensor vector
            params_list = [
                tmin_close, tmin_open, vpd_open, vpd_close, gl_sh, gl_wv,
                g_cuticular, csl, rbl_min, rbl_max, beta
            ]
            params = at.as_tensor_variable(params_list)
            # Key step: Define the log-likelihood as an added potential
            pm.Potential('likelihood', log_likelihood(params))
        return model


class CalibrationAPI(object):
    '''
    Convenience class for calibrating the MOD17 GPP and NPP models. Meant to
    be used with `fire.Fire()`.
    '''

    def __init__(self, config = None):
        config_file = config
        if config_file is None:
            config_file = os.path.join(
                MOD16_DIR, 'data/MOD16_calibration_config.json')
        with open(config_file, 'r') as file:
            self.config = json.load(file)
        self.hdf5 = self.config['data']['file']

    def _filter(self, raw: Sequence, size: int):
        'Apply a smoothing filter with zero phase offset'
        if size > 1:
            window = np.ones(size) / size
            return np.apply_along_axis(
                lambda x: signal.filtfilt(window, np.ones(1), x), 0, raw)
        return raw # Or, revert to the raw data

    def clean_observed(
            self, raw: Sequence, drivers: Sequence, protocol: str = 'ET',
            filter_length: int = 2) -> Sequence:
        '''
        Cleans observed tower flux data according to a prescribed protocol.

        Parameters
        ----------
        raw : Sequence
        drivers : Sequence
        protocol : str
        filter_length : int
            The window size for the smoothing filter, applied to the observed
            data

        Returns
        -------
        Sequence
        '''
        # Read in the observed data and apply smoothing filter; then mask out
        #   negative latent heat observations
        obs = self._filter(raw, filter_length)
        return np.where(obs < 0, np.nan, obs)

    def export_posterior(
            self, model: str, param: str, output_path: str, thin: int = 10,
            burn: int = 1000, k_folds: int = 1):
        '''
        Exports posterior distribution for a parameter, for each PFT to HDF5.

        Parameters
        ----------
        model : str
            The name of the model ("GPP" or "NPP")
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
        for pft in PFT_VALID:
            params = dict([(k, v[pft]) for k, v in params_dict.items()])
            backend = self.config['optimization']['backend_template'] %\
                (model, pft)
            post_by_fold = []
            for fold in range(1, k_folds + 1):
                if k_folds > 1:
                    backend = self.config['optimization']['backend_template'] %\
                        (f'{model}-k{fold}', pft)
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
            import ipdb
            ipdb.set_trace()#FIXME
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

    def tune(
            self, pft: int, plot_trace: bool = False, ipdb: bool = False,
            save_fig: bool = False, **kwargs):
        '''
        Run the MOD16 ET calibration.

        Parameters
        ----------
        pft : int
            The Plant Functional Type (PFT) to calibrate
        plot_trace : bool
            True to plot the trace for a previous calibration run; this will
            also NOT start a new calibration (Default: False)
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
        assert pft in PFT_VALID, f'Invalid PFT: {pft}'
        # Set var_names to tell ArviZ to plot only the free parameters
        kwargs.update({'var_names': MOD16.required_parameters[4:]})
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
        model = MOD16(params_dict)
        with h5py.File(self.hdf5, 'r') as hdf:
            sites = hdf['FLUXNET/site_id'][:].tolist()
            if hasattr(sites[0], 'decode'):
                sites = [s.decode('utf-8') for s in sites]
            # Get dominant PFT
            pft_map = pft_dominant(hdf['state/PFT'][:], site_list = sites)
            # Blacklist various sites
            blacklist = self.config['data']['sites_blacklisted']
            pft_mask = np.logical_and(pft_map == pft, ~np.in1d(sites, blacklist))
            weights = hdf['weights'][pft_mask]
            # Read in tower observations
            tower_obs = hdf['FLUXNET/latent_heat'][:][:,pft_mask]
            # Read the validation mask; mask out observations that are
            #   reserved for validation
            print('Masking out validation data...')
            mask = hdf['FLUXNET/validation_mask'][pft]
            tower_obs[mask] = np.nan
            # Read start and end dates and mask data appropriately
            timestamps = [
                 f'{y}-{str(m).zfill(2)}-{str(d).zfill(2)}'
                 for y, m, d in hdf['time'][:].tolist()
            ]
            start = self.config['data']['dates']['start']
            end = self.config['data']['dates']['end']
            t0 = timestamps.index(start)
            t1 = timestamps.index(end) + 1
            tower_obs = tower_obs[t0:t1]
            # Read in driver datasets
            print('Loading driver datasets...')
            lw_net_day = hdf['MERRA2/LWGNT_daytime'][:][t0:t1,pft_mask]
            lw_net_night = hdf['MERRA2/LWGNT_nighttime'][:][t0:t1,pft_mask]
            if self.config['optimization']['platform'] == 'VIIRS':
                sw_albedo = hdf['VIIRS/VNP43MA3_black_sky_sw_albedo'][:][t0:t1,pft_mask]
            else:
                sw_albedo = hdf['MODIS/MCD43GF_black_sky_sw_albedo'][:][t0:t1,pft_mask]
            sw_albedo = np.nanmean(sw_albedo, axis = -1)
            sw_rad_day = hdf['MERRA2/SWGDN_daytime'][:][t0:t1,pft_mask]
            sw_rad_night = hdf['MERRA2/SWGDN_nighttime'][:][t0:t1,pft_mask]
            temp_day = hdf['MERRA2/T10M_daytime'][:][t0:t1,pft_mask]
            temp_night = hdf['MERRA2/T10M_nighttime'][:][t0:t1,pft_mask]
            tmin = hdf['MERRA2/Tmin'][:][t0:t1,pft_mask]
            # As long as the time series is balanced w.r.t. years (i.e., same
            #   number of records per year), the overall mean is the annual mean
            temp_annual = hdf['MERRA2/T10M'][:][t0:t1,pft_mask].mean(axis = 0)
            vpd_day = MOD16.vpd(
                hdf['MERRA2/QV10M_daytime'][:][t0:t1,pft_mask],
                hdf['MERRA2/PS_daytime'][:][t0:t1,pft_mask],
                temp_day)
            vpd_night = MOD16.vpd(
                hdf['MERRA2/QV10M_nighttime'][:][t0:t1,pft_mask],
                hdf['MERRA2/PS_nighttime'][:][t0:t1,pft_mask],
                temp_night)
            pressure = hdf['MERRA2/PS'][:][t0:t1,pft_mask]
            # Read in fPAR, LAI, and convert from (%) to [0,1]
            prefix = 'MODIS/MOD'
            if self.config['optimization']['platform'] == 'VIIRS':
                prefix = 'VIIRS/VNP'
            fpar = np.nanmean(
                hdf[f'{prefix}15A2HGF_fPAR_interp'][:][t0:t1,pft_mask], axis = -1)
            lai = np.nanmean(
                hdf[f'{prefix}15A2HGF_LAI_interp'][:][t0:t1,pft_mask], axis = -1)
            # Convert fPAR from (%) to [0,1] and re-scale LAI; reshape fPAR and LAI
            fpar /= 100
            lai /= 10
        # Compile driver datasets
        drivers = [
            lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
            temp_day, temp_night, temp_annual, tmin, vpd_day, vpd_night,
            pressure, fpar, lai
        ]
        print('Initializing sampler...')
        backend = self.config['optimization']['backend_template'] % ('ET', pft)
        sampler = MOD16StochasticSampler(
            self.config, MOD16._et, params_dict, backend = backend,
            weights = weights)
        if plot_trace or ipdb:
            # This matplotlib setting prevents labels from overplotting
            pyplot.rcParams['figure.constrained_layout.use'] = True
            trace = sampler.get_trace()
            if ipdb:
                import ipdb
                ipdb.set_trace()
            az.plot_trace(trace, var_names = MOD16.required_parameters)
            pyplot.show()
            return
        tower_obs = self.clean_observed(tower_obs, drivers)
        # Get (informative) priors for just those parameters that have them
        with open(self.config['optimization']['prior'], 'r') as file:
            prior = json.load(file)
        prior_params = list(filter(
            lambda p: p in prior.keys(), sampler.required_parameters['ET']))
        prior = dict([
            (p, dict([(k, v[pft]) for k, v in prior[p].items()]))
            for p in prior_params
        ])
        sampler.run(
            tower_obs, drivers, prior = prior, save_fig = save_fig, **kwargs)


if __name__ == '__main__':
    import fire
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fire.Fire(CalibrationAPI)
