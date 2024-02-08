'''
Utilities related to the MOD16 algorithm.
'''

from __future__ import annotations
import csv
import os
import numpy as np
import pandas as pd
import mod16
from collections import Counter
from typing import Callable, Sequence
from pandas._typing import FilePath, ReadCsvBuffer

BPLUT_FIELD_LOOKUP = {
    'Tmin_min(C)':      'tmin_close',
    'Tmin_max(C)':      'tmin_open',
    'VPD_min(Pa)':      'vpd_open',
    'VPD_max(Pa)':      'vpd_close',
    'gl_sh((m/s)':      'gl_sh',
    'gl_e_wv(m/s)':     'gl_wv',
    'g_cuticular(m/s)': 'g_cuticular',
    'Cl(m/s)':          'csl',
    'RBL_MIN(s/m)':     'rbl_min',
    'RBL_MAX(s/m)':     'rbl_max',
    'beta':             'beta'
}

def pft_dominant(
        pft_map: np.ndarray, site_list: list = None,
        valid_pft: list = (1,2,3,4,5,6,7,8,9,10,12)):
    '''
    Returns the dominant PFT type, based on the PFT mode among the sub-grid
    for each site. Note that this is specific to the MOD17 calibration/
    validation (Cal/Val) protocol, i.e., three sites are always classified as
    Deciduous Needleleaf (PFT 3):

        CA-SF2
        CA-SF3
        US-NGC

    One other site, US-A10, is black-listed because it's PI-reported PFT is
    not compatible with our PFT classification.

    Parameters
    ----------
    pft_map : numpy.ndarray
        (N x M) array of PFT classes, where N is the number of sites and
        M is the number of sub-grid cells (N PFT classes are returned)
    site_list : list
        (Optional) List of the site names; must be provided to get PFT
        classes that accurately match the Cal/Val protocol
    valid_pft : list
        (Optional) List of valid PFT classes (Default: `range(1, 13)` but
        without 11)

    Returns
    -------
    numpy.ndarray
        An (N,) array of the dominant PFT classes
    '''
    pft_dom = np.zeros(pft_map.shape[0], np.float32)
    for i in range(0, pft_map.shape[0]):
        try:
            pft_dom[i] = Counter(
                list(filter(lambda x: x in valid_pft, pft_map[i])))\
                .most_common()[0][0]
        except:
            # Skip those sites that have no valid PFTs
            continue
    if site_list is not None:
        if 'US-A10' in site_list:
            pft_dom[site_list.index('US-A10')] = 0
        # For PFT==3, DNF, use pre-determined locations
        for sid in ('CA-SF2', 'CA-SF3', 'US-NGC'):
            if sid in site_list:
                pft_dom[site_list.index(sid)] = 3
    return pft_dom


def restore_bplut(
        path_or_buffer: FilePath | ReadCsvBuffer | str,
        nrows: int = 11) -> dict:
    '''
    NOTE: I manually exported Maosheng's fixed-width version (fixed-width
    files are a crime) to CSV for easier handling.

    Parameters
    ----------
    path_or_buffer : str or buffer
        File path or buffer representing the BPLUT to be read
    nrows : int
        Number of rows to read

    Returns
    -------
    dict
    '''
    # Remaps Maosheng's PFT order to the actual PFT code from MCD12Q1
    #   LC_Type2
    pft_lookup = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    data = pd.read_csv(path_or_buffer, nrows = nrows)
    # Create a dictionary with an array for every key
    output = dict([
        (k, np.full((13,), np.nan))
        for k in BPLUT_FIELD_LOOKUP.values()
    ])
    # Assumes the first column indexes the parameter/ field names
    field_index = data.columns[0]
    pft_index = list(data.columns)
    pft_index.remove(field_index)
    for k, key in enumerate(data[field_index]):
        if key not in data[data.columns[0]].values:
            continue # e.g., "beta" not included in Collection 5.x
        values = data.loc[data[field_index] == key, pft_index].values.ravel()
        output[BPLUT_FIELD_LOOKUP[key]][pft_lookup] = values
    return output


def write_bplut(params_dict: dict, output_path: str):
    '''
    Writes a BPLUT parameters dictionary to an output CSV file.

    Parameters
    ----------
    params_dict : dict
    output_path : str
        The output CSV file path
    '''
    template = os.path.join(
        os.path.dirname(mod16.__file__), 'data/MOD16_BPLUT_C5.1_05deg_MCD43B_Albedo_MERRA_GMAO.csv')
    with open(template, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if reader.line_num > 1:
                break
            header = line
    with open(output_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for name, key in BPLUT_FIELD_LOOKUP.items():
            values = []
            for pft in mod16.PFT_VALID:
                values.append(params_dict[key][pft])
            writer.writerow((name, *values))
