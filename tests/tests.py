'''
Unit tests for the `mod16` Python utilities library.
'''

import os
import unittest
import numpy as np
from mod16 import MOD16, psychrometric_constant, radiation_net, svp_slope

# MOD17_BPLUT = os.path.join(
#     os.path.dirname(mod17.__file__), 'data/MOD17_BPLUT_C5.1_MERRA_NASA.csv')

class CanopyEvaporation(unittest.TestCase):
    '''
    Suite of test cases related to wet canopy evaporation.
    '''

    @classmethod
    def setUp(cls):
        cls.params = dict().fromkeys(MOD16.required_parameters, None)
        cls.params.update({
            'gl_sh': 0.01,
            'gl_wv': 0.01,
            'g_cuticular': 1e-5,
            'tmin_close': -8, # deg C
            'tmin_open': 8, # deg C
            'vpd_open': 650,
            'vpd_close': 3000,
            'rbl_min': 60,
            'rbl_max': 90,
            'csl': 2.4e-3,
            'beta': 250 # Original (hard-coded constant) value
        })
        cls.pressure = 100e3
        cls.temp_k = 273.15 + 30
        cls.tmin = 273.15 + 20
        cls.vpd = 1000
        cls.lai = 1.5
        cls.fpar = 0.5
        cls.rad_canopy = 5000
        cls.rad_soil = 5000
        cls.f_wet = 0.5
        cls.r_corr = (101300 / cls.pressure) * (cls.temp_k / 293.15)**1.75
        # 5 levels of each
        cls._pressure = np.arange(98e3, 103e3, 1e3)
        cls._temp_k = 273.15 + np.array([0, 10, 20, 30, 40])
        cls._vpd = np.arange(0, 5000, 1000)
        cls._lai = np.arange(0.5, 3, 0.5)
        cls._fpar = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        cls._rad_canopy = np.arange(3e3, 8e3, 1e3)
        cls._f_wet = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def test_evaporation_soil(self):
        'Should accurately calculate evaporation from bare soil'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_soil(
            self.pressure, self.temp_k, self.vpd, self.fpar, self.rad_soil,
            self.r_corr)
        self.assertEqual(evap.round(3), 3.102)

    def test_evaporation_soil_by_fpar(self):
        'Should accurately calc. evap. from bare soil across fPAR gradient'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_soil(
            self.pressure, self.temp_k, self.vpd, self._fpar, self.rad_soil,
            self.r_corr)
        # <KAE> 2021-12-08 Maosheng indicated that the User Guide has an error
        #   w.r.t. to the scaling of soil heat flux by (1 - fPAR); this has
        #   been removed from MOD16.soil_heat_flux()
        self.assertTrue(np.equal(evap.round(3), np.array([
            3.128, 3.115, 3.102, 3.089, 3.076
        ])).all())

    def test_transpiration_daytime(self):
        'Should accurately calculate transpiration during daytime'
        model = MOD16(self.params)
        trans = 60 * 60 * model.transpiration(
            self.pressure, self.temp_k, self.vpd, self.lai, self.fpar,
            self.rad_canopy, self.tmin, self.r_corr, daytime = True)
        # <KAE> 2021-12-08 The Tmin ramp function parameters had been
        #   inadvertently switched when the tests were first written
        self.assertEqual(trans.round(3), 1.248) # kg m-2 hr-1

    def test_transpiration_nighttime(self):
        'Should accurately calculate transpiration during nighttime'
        model = MOD16(self.params)
        trans = 60 * 60 * model.transpiration(
            self.pressure, self.temp_k, self.vpd, self.lai, self.fpar,
            self.rad_canopy, self.tmin, self.r_corr, daytime = False)
        # <KAE> 2021-12-08 The Tmin ramp function parameters had been
        #   inadvertently switched when the tests were first written
        self.assertEqual(trans.round(3), 3.854) # kg m-2 hr-1

    def test_wet_canopy_evaporation(self):
        'Should accurately calculate wet canopy evaporation (kg m-2 s-1)'
        model = MOD16(self.params)
        evap = model.evaporation_wet_canopy(
            self.pressure, self.temp_k, self.vpd, self.lai, self.fpar,
            self.rad_canopy).round(6)
        # <KAE> 2021-12-08 There was a bug in the r_corr definition
        self.assertEqual(evap, 4.49e-4)

    def test_wet_canopy_evaporation_by_pressure(self):
        'Accurately calculate wet canopy evaporation by pressure gradient'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_wet_canopy(
            self._pressure, self.temp_k, self.vpd, self.lai, self.fpar,
            self.rad_canopy)
        self.assertTrue(np.equal(evap.round(3), np.array([
            1.623, 1.62 , 1.618, 1.615, 1.612 # kg m-2 hr-1
        ])).all())

    def test_wet_canopy_evaporation_by_temperature(self):
        'Accurately calculate wet canopy evaporation by temperature gradient'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_wet_canopy(
            self.pressure, self._temp_k, self.vpd, self.lai, self.fpar,
            self.rad_canopy)
        self.assertTrue(np.equal(evap.round(3), np.array([
            0., 0., 0., 1.618, 3.222 # kg m-2 hr-1
        ])).all())

    def test_wet_canopy_evaporation_by_vpd(self):
        'Accurately calculate wet canopy evaporation across VPD gradient'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_wet_canopy(
            self.pressure, self.temp_k, self._vpd, self.lai, self.fpar,
            self.rad_canopy)
        self.assertTrue(np.equal(evap.round(3), np.array([
            5.382, 1.618, 0, 0, 0 # kg m-2 hr-1
        ])).all())

    def test_wet_canopy_evaporation_by_lai(self):
        'Accurately calculate wet canopy evaporation across LAI gradient'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_wet_canopy(
            self.pressure, self.temp_k, self.vpd, self._lai, self.fpar,
            self.rad_canopy)
        self.assertTrue(np.equal(evap.round(3), np.array([
            1.174, 1.478, 1.618, 1.699, 1.752 # kg m-2 hr-1
        ])).all())

    def test_wet_canopy_evaporation_by_fpar(self):
        'Accurately calculate wet canopy evaporation across fPAR gradient'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_wet_canopy(
            self.pressure, self.temp_k, self.vpd, self.lai, self._fpar,
            self.rad_canopy)
        self.assertTrue(np.equal(evap.round(3), np.array([
            1.611, 1.615, 1.618, 1.621, 1.624 # kg m-2 hr-1
        ])).all())

    def test_wet_canopy_evaporation_by_canopy_radiation(self):
        'Accurately calculate wet canopy evaporation by canopy radiation'
        model = MOD16(self.params)
        evap = 60 * 60 * model.evaporation_wet_canopy(
            self.pressure, self.temp_k, self.vpd, self.lai, self.fpar,
            self._rad_canopy)
        self.assertTrue(np.equal(evap.round(3), np.array([
            0.974, 1.296, 1.618, 1.94 , 2.262 # kg m-2 hr-1
        ])).all())

    def test_psychrometric_constant(self):
        'Should accurately calculate the psychrometric constant'
        pressure = np.array((100e3, 80e3, 100e3, 80e3))
        temp_k = 273.15 + np.array((10, 10, 25, 25))
        answer = [65.74, 52.59, 66.69, 53.35]
        for i in range(0, 4):
            self.assertEqual(
                answer[i],
                psychrometric_constant(pressure[i], temp_k[i]).round(2))
        # Example from FAO:
        #   http://www.fao.org/3/X0490E/x0490e07.htm#psychrometric%20constant%20(g)
        self.assertEqual(
            54.55, np.round(psychrometric_constant(81.8e3, 25 + 273.15), 2))

    def test_radiation_net(self):
        'Should accurately calculate net radiation to the land surface'
        swrad = np.array((500, 5000, 500, 5000, 500, 5000, 500, 5000))
        albedo = np.array((0.4, 0.4, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8))
        temp_k = 273.15 + np.array((10, 10, 10, 10, 25, 25, 25, 25))
        answer = [223.3, 2923.3, 23.3, 923.3, 241.8, 2941.8, 41.8, 941.8]
        for i in range(0, 8):
            self.assertEqual(
                answer[i],
                radiation_net(swrad[i], albedo[i], temp_k[i]).round(1))

    def test_svp_slope(self):
        'Should accurately calculate slope of SVP curve'
        self.assertEqual( 82.3, svp_slope(273.15 + 10).round(1))
        self.assertEqual(144.8, svp_slope(273.15 + 20).round(1))
        self.assertEqual(188.8, svp_slope(273.15 + 25).round(1))


if __name__ == '__main__':
    unittest.main()
