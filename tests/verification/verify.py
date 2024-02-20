'''
Compares the MOD16 (Python implementation) predictions against those of the
MOD16 product (from NASA AppEEARS) for some FLUXNET sites.

    >>> np.nanpercentile(obs, (1, 10, 50, 90, 99)).round(1)
    array([ 1.9,  3.1,  6.6, 13.5, 19.4])

    >>> np.nanpercentile(
        pred_night * (8 * 24 * 60 * 60), (1, 10, 50, 90, 99)).round(1)
    array([ 0.2,  1.1,  8.6, 18.7, 25.7])

    >>> np.nanpercentile(
        pred_day * (8 * 24 * 60 * 60), (1, 10, 50, 90, 99)).round(1)
    array([ 1.7,  3.1, 11.4, 22.7, 32.3])
'''

import datetime
import os
import numpy as np
import h5py
import mod16
from matplotlib import pyplot
from mod16 import MOD16, latent_heat_vaporization, svp
from mod16.utils import pft_dominant, restore_bplut

BPLUT = os.path.join(os.path.dirname(mod16.__file__), 'data/MOD16_BPLUT_C5.1_05deg_MCD43B_Albedo_MERRA_GMAO.csv')
FLUXNET_HDF5 = '/anx_lagr4/MODIS_VIIRS/calibration/VIIRS_MOD16_tower_site_latent_heat_and_drivers_v5.h5'
L4C_HDF5 = '/home/arthur.endsley/Remotes/arthur.endsley/DATA/L4_C_tower_site_drivers_NRv8-3_for_356_sites.h5'
SITE_ID = 'AR-SLu' # Focus on one site for now

def main():
    ''
    # Get a list of the sites in our calibration dataset (because this extract
    #   from AppEEARS is a superset of our cal sites)
    with h5py.File(FLUXNET_HDF5) as hdf:
        sites = hdf['FLUXNET/site_id'][:].tolist()
        if hasattr(sites[0], 'decode'):
            sites = [s.decode('utf-8') for s in sites]
        idx = sites.index(SITE_ID)
        pft = int(pft_dominant(hdf['state/PFT'][:], site_list = sites)[idx])
    # Get the model parameters associated with this PFT
    bplut = restore_bplut(BPLUT)
    params = dict([(k, v[pft]) for k, v in bplut.items()])
    # NOTE: In the current version of MOD16, beta is a fixed parameter
    params['beta'] = 250
    model = MOD16(params)
    ##################################################
    # Begin defining quantities to test against C code
    LAI = np.array((0.3, 0.6, 1.0))
    Tday = np.array((286.20189, 292.52667, 298.3286))
    Tnight = np.array((284.20189, 290.52667, 292.3286))
    pressure = np.array((92753.47, 92753.47, 92753.47))
    VPD_day = np.array((710.9, 1249.4, 1979.))
    AIR_DENSITY_day = np.array((1.12791642, 1.10072136, 1.07518633))
    SVP_day = np.array((1502.86379395, 2249.56542969, 3201.63623047))
    Tmin = np.array((278.92, 284.43, 289.88))
    day_length = 1
    A = 255.27261
    A_day = np.array((255.27261, 255.27261, 255.27261))
    A_night = np.array((255.27261, 255.27261, 0))
    G = 0.0
    Fc = 0.35839
    Fwet_day = np.array((0, 0.4, 0.8))
    rad_soil = (1 - Fc) * A - G
    Tavg_ann = 289.74402 # Annual mean temperature
    lamda = latent_heat_vaporization(Tday)
    print('Lambda:', lamda)
    print('RH:', (SVP_day - VPD_day) / SVP_day)
    lw_net_day = -117
    sw_rad_day = 419
    sw_albedo = 0.116
    result = model.evapotranspiration(
        lw_net_day, -65, sw_rad_day, 0, sw_albedo, Tday, Tnight,
        Tavg_ann, Tmin, VPD_day, 512, pressure, Fc, LAI,
        f_wet = Fwet_day, separate = True)
    # i.e., only Daytime results
    e_canopy, e_soil, trans = result[0]
    print('Transpiration..: ', list(map(lambda x: '%.6f' % x, trans)))
    print('Canopy Evap....: ', list(map(lambda x: '%.6f' % x, e_canopy)))
    print('Soil Evapo.....: ', list(map(lambda x: '%.6f' % x, e_soil)))
    print('TOTAL ET.......: ', list(map(lambda x: '%.6f' % x, trans + e_canopy + e_soil)))
    print('-- Using streamlined code:')
    result = model._evapotranspiration(
        [params[p] for p in MOD16.required_parameters],
        lw_net_day, -65, sw_rad_day, 0, sw_albedo, Tday, Tnight,
        Tavg_ann, Tmin, VPD_day, 512, pressure, Fc, LAI,
        f_wet = Fwet_day)
    # Divide by the latent heat of vaporization (J kg-1) to obtain mass
    #   flux (kg m-2 s-1)
    print('TOTAL ET [W/m2]: ', list(map(lambda x: '%.2f' % x, result[0])))
    print('TOTAL ET.......: ', list(map(lambda x: '%.6f' % x, result[0] / lamda)))


if __name__ == '__main__':
    main()
