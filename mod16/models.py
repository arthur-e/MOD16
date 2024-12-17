'''
This module makes it easier to stand-up MOD16 model variants; i.e., to
initialize MOD16 with certain parameters.
'''

import os
import mod16
import numpy as np
from mod16 import PFT_VALID, MOD16
from mod16.utils import restore_bplut

MOD16_DIR = os.path.dirname(mod16.__file__)
PFT_ALL = {
    'Evergreen Needleleaf Forest (ENF)': 1,
    'Evergreen Broadleaf Forest (EBF)': 2,
    'Deciduous Needleleaf Forest (DNF)': 3,
    'Deciduous Broadleaf Forest (DBF)': 4,
    'Mixed Forest (MF) ': 5,
    'Closed Shrublands (CSH)': 6,
    'Open Shrublands (OSH)': 7,
    'Woody Savannas (WSV)': 8,
    'Savannas (SAV)': 9,
    'Grasslands (GRS)': 10,
    'Croplands (CRO)': 12
}

class MOD16Collection61(MOD16):
    '''
    The MOD16 Collection 6.1 model.

    Parameters
    ----------
    pft : int
        The numeric code of the Plant Functional Type of interest.
    '''
    def __init__(self, pft):
        assert pft in PFT_VALID,\
            f'Not a recognized numeric PFT code; should be one of: {",".join(PFT_VALID)}'
        # Get the path to the Collection 6.1 BPLUT
        file_path = os.path.join(
            MOD16_DIR, 'data/MOD16_BPLUT_C5.1_05deg_MCD43B_Albedo_MERRA_GMAO.csv')
        # Read in the file as a dictionary of parameters for each Plant
        #   Functional Type (PFT)
        params_dict = restore_bplut(file_path)
        params = dict([
            (key, params_dict[key][pft]) for key in params_dict.keys()
        ])
        # Set "beta" if not already set
        if np.isnan(params['beta']):
            params['beta'] = 250
        super().__init__(params = params)
