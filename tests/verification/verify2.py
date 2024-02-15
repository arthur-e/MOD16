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

import csv
import datetime
import os
import numpy as np
import h5py
import mod16
from matplotlib import pyplot
from mod16 import MOD16
from mod16.utils import pft_dominant, restore_bplut
from suntransit import sunrise_sunset
from tqdm import tqdm

BPLUT = os.path.join(os.path.dirname(mod16.__file__), 'data/MOD16_BPLUT_C5.1_05deg_MCD43B_Albedo_MERRA_GMAO.csv')
VIIRS_MODIS_HDF5 = '/anx_lagr4/MODIS_VIIRS/calibration/VIIRS_MOD16_tower_site_latent_heat_and_drivers_v4.h5'
APPEEARS_CSV = '/home/arthur/Workspace/NTSG/projects/Y2021_MODIS-VIIRS/data/NASA_AppEEARS_MOD16A2_ET_and_LE_at_356_tower_sites.csv'
OUTPUT_DIR = '/home/arthur/Downloads/MOD16_plots'

def main(site_id):
    ''
    # Get a list of the sites in our calibration dataset (because this extract
    #   from AppEEARS is a superset of our cal sites)
    with h5py.File(VIIRS_MODIS_HDF5, 'r') as hdf:
        sites = hdf['FLUXNET/site_id'][:].tolist()
        if hasattr(sites[0], 'decode'):
            sites = [s.decode('utf-8') for s in sites]
        dates = [
            datetime.date(*d)
            for d in hdf['time'][:].tolist()
        ]
        idx = sites.index(site_id)
        # Get coordinates in (lat, lng) order for sunrise_sunset()
        try:
            coords = hdf['coordinates/lng_lat'][idx,:][::-1]
        except:
            import ipdb
            ipdb.set_trace()#FIXME
        pft = int(pft_dominant(hdf['state/PFT'][:], site_list = sites)[idx])
        lw_net_day = hdf['MERRA2/LWGNT_daytime'][:,idx]
        lw_net_night = hdf['MERRA2/LWGNT_nighttime'][:,idx]
        sw_albedo = hdf['MODIS/MCD43A3_black_sky_sw_albedo'][:,idx]
        sw_rad_day = hdf['MERRA2/SWGDN_daytime'][:,idx]
        sw_rad_night = hdf['MERRA2/SWGDN_nighttime'][:,idx]
        temp_day = hdf['MERRA2/T10M_daytime'][:,idx]
        temp_night = hdf['MERRA2/T10M_nighttime'][:,idx]
        tmin = hdf['MERRA2/Tmin'][:,idx]
        # As long as the time series is balanced w.r.t. years (i.e., same
        #   number of records per year), the overall mean is the annual mean
        temp_annual = hdf['MERRA2/T10M'][:,idx].mean(axis = 0)
        vpd_day = MOD16.vpd(
            hdf['MERRA2/QV10M_daytime'][:,idx],
            hdf['MERRA2/PS_daytime'][:,idx],
            temp_day)
        vpd_night = MOD16.vpd(
            hdf['MERRA2/QV10M_nighttime'][:,idx],
            hdf['MERRA2/PS_nighttime'][:,idx],
            temp_night)
        pressure = hdf['MERRA2/PS'][:,idx]
        # Read in fPAR, LAI, and convert from (%) to [0,1]
        fpar = np.nanmean(hdf['MODIS/MOD15A2HGF_fPAR_interp'][:,idx], axis = -1)
        lai = np.nanmean(hdf['MODIS/MOD15A2HGF_LAI_interp'][:,idx], axis = -1)
        fpar /= 100
        lai /= 100
    # Read in the ET and LE data
    with open(APPEEARS_CSV, 'r') as file:
        reader = csv.reader(file)
        dates_8day = []
        obs = []
        for line in reader:
            if reader.line_num == 1:
                continue # Skip first line
            _, this_site_id, date, et, le = line
            if this_site_id != site_id:
                continue # Skip sites not in our cal dataset
            dates_8day.append(date)
            obs.append(float(et))
    obs = np.array(obs)
    obs = np.where(obs >= 32e3, np.nan, obs)
    # Get the model parameters associated with this PFT
    bplut = restore_bplut(BPLUT)
    params = dict([(k, v[pft]) for k, v in bplut.items()])
    # NOTE: In the current version of MOD16, beta is a fixed parameter
    params['beta'] = 250
    model = MOD16(params)
    # Predictions
    pred_day, pred_night = model.evapotranspiration(
        lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
        temp_day, temp_night, temp_annual, tmin, vpd_day, vpd_night, pressure,
        fpar, lai)
    # Get day length in hours, then multiply by number of days (8) and number
    #   of seconds in an hour (60 * 60)
    updown = np.array([
        down - up if down > 0 and up > 0 else np.nan
        for up, down in [
            sunrise_sunset(coords, d) for d in dates
        ]
    ])
    # Convert to kg m-2 (8 days)-1 to match MOD16 product
    pred = ((pred_day * updown * 8 * 60 * 60) +\
        (pred_night * (24 - updown) * 8 * 60 * 60))
    pred_filt = [
        pred[i] if date in dates_8day else np.nan
        for i, date in enumerate([
            d.strftime('%Y-%m-%d') for d in dates
        ])
    ]
    while np.nan in pred_filt:
        pred_filt.remove(np.nan)
    fig = pyplot.figure(figsize = (6, 6))
    pyplot.plot(obs, pred_filt, 'k.')
    ax = fig.get_axes()[0]
    if len(pred_filt) == 0:
        return
    _max = np.nanmax(np.concatenate((obs, pred_filt)))
    pyplot.xlim((0, _max))
    pyplot.ylim((0, _max))
    pyplot.xlabel('MOD16A2 ET from AppEEARS')
    pyplot.ylabel('Predicted ET - Python Implementation')
    pyplot.title(site_id)
    ax.plot([0, 1], [0, 1], transform = ax.transAxes, linestyle = 'dashed', color = 'black')
    pyplot.savefig(f'{OUTPUT_DIR}/{site_id}.png')
    pyplot.close()


if __name__ == '__main__':
    with h5py.File(VIIRS_MODIS_HDF5, 'r') as hdf:
        sites = hdf['FLUXNET/site_id'][:].tolist()
        if hasattr(sites[0], 'decode'):
            sites = [s.decode('utf-8') for s in sites]
    for site_id in tqdm(reversed(sites)):
        main(site_id)
