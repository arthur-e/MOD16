r'''
The MODIS MOD16 terrestrial evapotranspiration (ET) algorithm. See the README
for full references.

**There are two types of interfaces in the MOD16 Python code.**

User-friendly methods of a `MOD16` instance, parameterized for a single
land-cover type:

- `MOD16.evapotranspiration()`
- `MOD16.transpiration()`
- `MOD16.evaporation_soil()`
- `MOD16.evaporation_wet_canopy()`
- And so on.

There is also a single, vectorized interface implemented as a static method
of the `MOD16` class, which can handle multiple land-cover types:

- `MOD16._evapotranspiration()`
- `MOD16._et()` (alias for the function above)

The user-friendly, instance methods have code blocks that are easy to read and
understand, but those methods might run slow for large spatial domains because
they incur a lot of Python overhead. The values returned by those functions
are in mass-flux units (kg [H2O] m-2 sec-1).

The vectorized interface is faster because it incurs less Python overhead; it
is useful, for example, when calibrating MOD16, because this may require
hundreds or thousands of function evaluations every second. It can also handle
multiple land-cover types. The vectorized interface returns values in energy
units (W m-2), for comparison to eddy covariance tower measurements.

NOTES:

1. Air pressure is an input field, which could be taken as given from a
    reanalysis dataset or calculated using the lapse-rate function given in
    the MOD16 C6.1 User's Guide (pp. 7-8), for estimating air pressure from
    elevation.
2. MOD16 C6.1 User's Guide has an outdated formula for net radiation
    (Equation 7, page 4); it is no longer based on surface emissivity and is
    instead based on the sum of short-wave and long-wave radiation; see
    `MOD16.radiation_soil()`.
3. MOD16 C6.1 User's Guide has an error in the calculation of boundary-layer
    resistance to bare soil evaporation (Equation 19, page 9); instead of what
    is written there: When VPD <= `VPD_open`, then `rbl_max` should be used
    and when VPD >= `VPD_close`, then `rbl_min` should be used.
4. MOD16 C6.1 User's Guide suggested that, in calculating the radiation
    received by the soil (Equation 8), only net radiation is modulated by
    bare soil area, \((1 - fPAR)\). In fact, the difference between net
    radiation and the ground heat flux is what is modulated; i.e., the
    correct equation is \(A_{soil} = (1 - fPAR)\times (A - G)\)
5. The MERRA2 longwave radiation field `LWGNT` is defined as the "surface net
    downward longwave flux", hence, it is usually negative because of the net
    outward flux of longwave radiation from the Earth's surface.
6. For numerical stability, the quantity `r_corr` used in this implementation
    is different from the term defined in the MOD16 C6.1 User's Guide (ca.
    Equation 13). Here, `r_corr` is a large number (greater than 1), equal to
    1/r, where r is the value provided in the User Guide; this avoids
    numerical instability associated with keeping track of a very small
    number.
7. Calculation of canopy conductance has changed since C6.1, see below.

In MOD16 C6.1, canopy conductance was calculated as:
$$
C_c = \frac{gl_{sh}(G_S + G_C)}{gl_{sh} + G_S + G_C} \text{LAI}(1 - F_{wet})
$$

However, in the new VIIRS VNP16 and updated MOD16 it is calculated as:
$$
C_c = \frac{G_0(G_S + G_C)}{G_0 + G_S + G_C}
$$

Where:
$$
G_0 = gl_{sh} \times \text{LAI}(1 - F_{wet})
$$

Based on [2], `MOD16.radiation_soil()` was updated. Based on [4],
`MOD16.soil_heat_flux()` was updated. Based on [7], `MOD16.transpiration()`
was updated.

TODO:

- `sw_rad_night` may not be be needed anywhere
'''

import warnings
import numpy as np
from collections import Counter
from typing import Iterable, Sequence, Tuple
from numbers import Number
from mod17 import linear_constraint

PFT_VALID = (1,2,3,4,5,6,7,8,9,10,12)
STEFAN_BOLTZMANN = 5.67e-8 # Stefan-Boltzmann constant, W m-2 K-4
SPECIFIC_HEAT_CAPACITY_AIR = 1013 # J kg-1 K-1, Monteith & Unsworth (2001)
# Ratio of molecular weight of water vapor to that of dry air (ibid.)
MOL_WEIGHT_WET_DRY_RATIO_AIR = 0.622
TEMP_LAPSE_RATE = 0.0065 # Standard temperature lapse rate [-(deg K) m-1]
GRAV_ACCEL = 9.80665 # Gravitational acceleration [m s-2]
GAS_LAW_CONST = 8.3143 # Ideal gas law constant [m3 Pa (mol)-1 K-1]
AIR_MOL_WEIGHT = 28.9644e-3 # Molecular weight of air [kg (mol)-1]
STD_TEMP_K = 288.15 # Standard temperature at sea level [deg K]
STD_PRESSURE_PASCALS = 101325.0 # Standard pressure at sea level [Pa]
# A pre-determined quantity, not physically meaningful, used in air_pressure()
AIR_PRESSURE_RATE = GRAV_ACCEL / (TEMP_LAPSE_RATE * (GAS_LAW_CONST / AIR_MOL_WEIGHT))

# Calculate the latent heat of vaporization (J kg-1)
latent_heat_vaporization = lambda temp_k: (2.501 - 0.002361 * (temp_k - 273.15)) * 1e6


class MOD16(object):
    '''
    The MODIS MxD16 Evapotranspiration model. The required model parameters are:

    - `tmin_close`: Temperature at which stomata are almost completely
        closed due to (minimum) temperature stress (deg C)
    - `tmin_open`: Temperature at which stomata are completely open, i.e.,
        there is no effect of temperature on transpiration (deg C)
    - `vpd_open`: The VPD at which stomata are completely open, i.e.,
        there is no effect of water stress on transpiration (Pa)
    - `vpd_close`: The VPD at which stomata are almost completely closed
        due to water stress (Pa)
    - `gl_sh`: Leaf conductance to sensible heat per unit LAI
        (m s-1 LAI-1);
    - `gl_wv`: Leaf conductance to evaporated water per unit LAI
        (m s-1 LAI-1);
    - `g_cuticular`: Leaf cuticular conductance (m s-1);
    - `csl`: Mean potential stomatal conductance per unit leaf area (m s-1);
    - `rbl_min`: Minimum atmospheric boundary layer resistance (s m-1);
    - `rbl_max`: Maximum atmospheric boundary layer resistance (s m-1);
    - `beta`: Factor in soil moisture constraint on potential soil
        evaporation, i.e., (VPD / beta); from Bouchet (1963)

    Parameters
    ----------
    params : dict
        Dictionary of model parameters
    '''
    required_parameters = [
        'tmin_close', 'tmin_open', 'vpd_open', 'vpd_close', 'gl_sh', 'gl_wv',
        'g_cuticular', 'csl', 'rbl_min', 'rbl_max', 'beta'
    ]

    def __init__(self, params: dict):
        self.params = params
        for key in self.required_parameters:
            setattr(self, key, params[key])

    @staticmethod
    def _et(
            params, lw_net_day, lw_net_night, sw_rad_day, sw_rad_night,
            sw_albedo, temp_day, temp_night, temp_annual, tmin, vpd_day,
            vpd_night, pressure, fpar, lai, f_wet = None, tiny = 1e-9,
            r_corr_list = None
        ) -> Number:
        '''
        Optimized ET code, intended for use in model calibration ONLY.
        Returns combined day and night ET.

        Parameters
        ----------
        params : list
            A list of arrays, each array representing a different parameter,
            in the order specified by `MOD16.required_parameters`. Each array
            should be a (1 x N) array, where N is the number of sites/pixels.
        *drivers
            Every subsequent argument is a separate (T x N) where T is the
            number of time steps and N is the number of sites/pixels.

        Returns
        -------
        numpy.ndarray
            The total latent heat flux [W m-2] for each site/pixel
        '''
        day, night = MOD16._evapotranspiration(
            params, lw_net_day, lw_net_night, sw_rad_day, sw_rad_night,
            sw_albedo, temp_day, temp_night, temp_annual, tmin, vpd_day,
            vpd_night, pressure, fpar, lai, f_wet = None, tiny = 1e-9,
            r_corr_list = r_corr_list)
        return np.add(day, night)

    @staticmethod
    def _evapotranspiration(
            params, lw_net_day, lw_net_night, sw_rad_day, sw_rad_night,
            sw_albedo, temp_day, temp_night, temp_annual, tmin, vpd_day,
            vpd_night, pressure, fpar, lai, f_wet = None, tiny = 1e-9,
            r_corr_list = None
        ) -> Iterable[Tuple[Sequence, Sequence]]:
        '''
        Optimized ET code, intended for use in model calibration ONLY. The
        `params` are expected to be given in the order specified by
        `MOD16.required_parameters`. NOTE: total ET values returned are in
        [W m-2], for comparison to tower ET values. Divide by the latent heat
        of vaporization (J kg-1) to obtain a mass flux (kg m-2 s-1).

        Parameters
        ----------
        params : list
            A list of arrays, each array representing a different parameter,
            in the order specified by `MOD16.required_parameters`. Each array
            should be a (1 x N) array, where N is the number of sites/pixels.
        *drivers
            Every subsequent argument is a separate (T x N) where T is the
            number of time steps and N is the number of sites/pixels.

        Returns
        -------
        list
            A 2-element list of (day, night) latent heat flux [W m-2] totals
        '''
        # Radiation intercepted by the soil surface --
        rad_net_day = sw_rad_day * (1 - sw_albedo) + lw_net_day
        rad_net_night = lw_net_night # At night, SW radiation should be zero

        # Soil heat flux
        g_soil = []
        condition = np.logical_and( # MOD16 UG Equation 9
            np.logical_and(
                temp_annual < (273.15 + 25),
                temp_annual > (273.15 + params[1])
            ), (temp_day - temp_night) >= 5)
        for rad_i, temp_i in zip(
                (rad_net_day, rad_net_night), (temp_day, temp_night)):
            g = np.where(condition, (4.73 * (temp_i - 273.15)) - 20.87, 0)
            # Modify soil heat flux under these conditions...
            g = np.where(np.abs(g) > (0.39 * np.abs(rad_i)), 0.39 * rad_i, g)
            g_soil.append(g)
        g_soil_day, g_soil_night = g_soil
        # -- End soil heat flux calculation

        # Radiation received by the soil, see Section 2.2 of User Guide
        g_soil_day = np.where(
            np.logical_and(rad_net_day - g_soil_day < 0, rad_net_day > 0),
            rad_net_day, g_soil_day)
        g_soil_night = np.where(
            np.logical_and(
                rad_net_day > 0,
                (rad_net_night - g_soil_night) < (-0.5 * rad_net_day)),
            rad_net_night + (0.5 * rad_net_day), g_soil_night)
        rad_soil_day = (1 - fpar) * (rad_net_day - g_soil_day)
        rad_soil_night = (1 - fpar) * (rad_net_night - g_soil_night)

        # Compute day and night components of each ET component
        grouped_drivers = zip(
            (temp_day, temp_night),
            (vpd_day, vpd_night),
            (sw_rad_day, sw_rad_night),
            (lw_net_day, lw_net_night),
            (rad_soil_day, rad_soil_night))
        e_canopy = list()
        e_soil = list()
        transpiration = list()
        et_total = list()
        for i, group in enumerate(grouped_drivers):
            daytime = (i == 0)
            temp_k, vpd, sw_rad, lw_net, rad_soil = group
            # Net radiation to surface, based on down-welling short-wave
            #   radiation and net long-wave radiation
            rad_net = sw_rad * (1 - sw_albedo) + lw_net
            # Radiation intercepted by the canopy
            rad_canopy = fpar * rad_net
            # Compute wet surface fraction and other quantities
            # -- Requiring relative humidity to be calculated from VPD:
            #   VPD = VPsat - VPactual; RH = VPactual / VPsat
            #     --> RH = (VPsat - VPD) / VPsat
            _svp = svp(temp_k)
            rhumidity = (_svp - vpd) / _svp
            if f_wet is None:
                f_wet = np.where(rhumidity < 0.7, 0, rhumidity**4)
            # Slope of saturation vapor pressure curve
            s = svp_slope(temp_k, _svp)
            # Latent heat of vaporization (J kg-1)
            lhv = latent_heat_vaporization(temp_k)
            # Psychrometric constant
            gamma = psychrometric_constant(pressure, temp_k)
            # Correction for atmospheric temperature and pressure
            #   (Equation 13, MOD16 C6.1 User's Guide)
            if r_corr_list is None:
                r_corr = (101300 / pressure) * (temp_k / 293.15)**1.75
            else:
                r_corr = r_corr_list[i]
            # Compute evaporation from wet canopy
            rho = MOD16.air_density(temp_k, pressure, rhumidity) # Air density
            # -- Resistance to radiative heat transfer through air ("rrc")
            r_r = (rho * SPECIFIC_HEAT_CAPACITY_AIR) / (
                4 * STEFAN_BOLTZMANN * temp_k**3)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # -- Wet canopy resistance to sensible heat ("rhc")
                r_h = 1 / (params[4] * lai * f_wet)
                # -- Wet canopy resistance to evaporated water on the surface
                #   ("rvc")
                r_e = 1 / (params[5] * lai * f_wet)
                # -- Aerodynamic resistance to evaporated water on the wet
                #   canopy surface ("rhrc")
                r_a_wet = np.divide(r_h * r_r, r_h + r_r) # (s m-1)

            # EVAPORATION FROM WET CANOPY
            e = np.divide(
                f_wet * ((s * rad_canopy) + (
                    rho * SPECIFIC_HEAT_CAPACITY_AIR * fpar * vpd * 1/r_a_wet
                )),
                s + ((pressure * SPECIFIC_HEAT_CAPACITY_AIR * r_e) *\
                    1/(lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR * r_a_wet)))
            e_canopy.append(np.where(lai * f_wet <= tiny, 0, e))

            # -- Surface conductance (zero at nighttime)
            g_surf = 0
            if daytime:
                m_tmin = linear_constraint(params[0], params[1])
                m_vpd = linear_constraint(params[2], params[3], 'reversed')
                g_surf = (params[7] * m_tmin(tmin - 273.15) * m_vpd(vpd))
            g_surf /= r_corr
            g_cuticular = params[6] / r_corr
            # -- Canopy conductance, should be zero when LAI or f_wet are zero;
            #   updated calculation for conductance to sensible heat, see User
            #   Guide's Equation 15
            gl_sh = params[4] * lai * (1 - f_wet)
            g = ((gl_sh * (g_surf + g_cuticular)) / (
                gl_sh + g_surf + g_cuticular))
            g_canopy = np.where(
                np.logical_and(lai > 0, (1 - f_wet) > 0), g, tiny)
            # -- Aerodynamic resistance to heat, water vapor from dry canopy
            #   surface into the air (Equation 16, MOD16 C6.1 User's Guide)
            r_a_dry = (1/params[4] * r_r) / (1/params[4] + r_r) # (s m-1)

            # PLANT TRANSPIRATION
            if np.any(g_surf > 0):
                t = (1 - f_wet) * ((s * rad_canopy) +\
                    (rho * SPECIFIC_HEAT_CAPACITY_AIR * fpar * (vpd / r_a_dry)))
                if daytime:
                    t /= (s + gamma * (1 + (1 / g_canopy) / r_a_dry))
                else:
                    # Simplify calculation because g_canopy := 0 at night
                    t /= (s + gamma)
            else:
                t = 0
            transpiration.append(t)

            # BARE SOIL EVAPORATION
            # -- Total aerodynamic resistance as a function of VPD and the
            #   atmospheric boundary layer resistance...
            # UPDATED 2024-02, as boundary-layer resistance should be at maximum
            #   when VPD is low and at minimum when VPD is high
            r_tot = np.where(vpd <= params[2], params[9], # rbl_max
                np.where(vpd >= params[3], params[8], # rbl_min
                params[9] - (
                    (params[9] - params[8]) * (params[3] - vpd))\
                        / (params[3] - params[2])))
            # ...CORRECTED for atmospheric temperature, pressure
            r_tot = r_tot / r_corr
            # -- Aerodynamic resistance at the soil surface
            r_as = (r_tot * r_r) / (r_tot + r_r)
            # -- Terms common to evaporation and potential evaporation
            numer = (s * rad_soil) +\
                (rho * SPECIFIC_HEAT_CAPACITY_AIR * (1 - fpar) * (vpd / r_as))
            denom = (s + gamma * (r_tot / r_as))
            # -- Evaporation from "wet" soil (saturated fraction)
            evap_sat = (numer * f_wet) / denom
            # -- (Potential) Evaporation from unsaturated fraction
            evap_unsat = (numer * (1 - f_wet)) / denom
            # -- Finally, apply the soil moisture constraint from Fisher et al.
            #   (2008); see MOD16 C6.1 User's Guide, pp. 9-10
            e = evap_sat + evap_unsat * rhumidity**(vpd / params[10])
            e_soil.append(e)

            # Result is the sum of the three components
            et_total.append((transpiration[i] + e_canopy[i] + e_soil[i]))
        return et_total

    @staticmethod
    def air_density(
            temp_k: Number, pressure: Number, rhumidity: Number
        ) -> Number:
        '''
        NIST simplified air density formula with buoyancy correction from:

            National Physical Laboratory (2021),
              "Buoyancy Correction and Air Density Measurement."
              http://resource.npl.co.uk/docs/science_technology/
                mass_force_pressure/clubs_groups/instmc_weighing_panel/
                buoycornote.pdf

        Parameters
        ----------
        temp_k : int or float or numpy.ndarray
            Air temperature (degrees K)
        pressure : int or float or numpy.ndarray
            Air pressure (Pa)
        rhumidity : int or float or numpy.ndarray
            Relative humidity, on the interval [0, 1]

        Returns
        -------
        int or float or numpy.ndarray
            Air density (kg m-3)
        '''
        return np.divide( # Convert Pa to mbar, RH to RH% (percentage)
            0.348444 * (pressure / 100) - (rhumidity * 100) *\
                (0.00252 * (temp_k - 273.15) - 0.020582),
            temp_k) # kg m-3

    @staticmethod
    def air_pressure(elevation_m: Number) -> Number:
        r'''
        Atmospheric pressure as a function of elevation. From the discussion on
        atmospheric statics (p. 168) in:

            Iribane, J.V., and W.L. Godson, 1981. Atmospheric Thermodynamics
                2nd Edition. D. Reidel Publishing Company, Dordrecht,
                The Netherlands.

        It is calculated:

        $$
        P_a = P_{\text{std}}\times \left(
            1 - \ell z T_{\text{std}}^{-1}
          \right)^{5.256}
        $$

        Where \(\ell\) is the standard temperature lapse rate (-0.0065 deg K
        per meter), \(z\) is the elevation in meters, and \(P_{\text{std}}\),
        \(T_{\text{std}}\) are the standard pressure (101325 Pa) and
        temperature (288.15 deg K) at sea level.

        Parameters
        ----------
        elevation_m : Number

        Returns
        -------
        Number
            Air pressure in Pascals
        '''
        temp_ratio = 1 - ((TEMP_LAPSE_RATE * elevation_m) / STD_TEMP_K)
        return STD_PRESSURE_PASCALS * np.pow(temp_ratio, AIR_PRESSURE_RATE)

    @staticmethod
    def vpd(qv10m: Number, pressure: Number, tmean: Number) -> Number:
        r'''
        Computes vapor pressure deficit (VPD) from surface meteorology:

        $$
        \text{VPD} = \left(610.7\times \text{exp}\left(
          \frac{17.38\times T}{239 + T}
        \right) - \text{AVP}\right)
        $$

        Where \(T\) is the temperature in deg K and the actual vapor pressure
        (AVP) is given:

        $$
        \text{AVP} = \frac{\text{QV10M}\times
            \text{P}}{0.622 + 0.379\times \text{QV10M}}
        $$

        Where P is the air pressure in Pascals and QV10M is the water vapor
        mixing ratio at 10-meter height.

        Parameters
        ----------
        qv10m : int or float or numpy.ndarray
            Water vapor mixing ratio at 10-meter height (Pa)
        pressure : int or float or numpy.ndarray
            Atmospheric pressure (Pa)
        tmean : int or float or numpy.ndarray
            Mean daytime temperature (degrees K)

        Returns
        -------
        int or float or numpy.ndarray
        '''
        temp_c = tmean - 273.15
        # Actual vapor pressure (Gates 1980, Biophysical Ecology, p.311)
        avp = (qv10m * pressure) / (0.622 + (0.379 * qv10m))
        # Saturation vapor pressure (similar to FAO formula)
        svp = 610.7 * np.exp((17.38 * temp_c) / (239 + temp_c))
        return svp - avp

    @staticmethod
    def rhumidity(temp_k: Number, vpd: Number) -> Number:
        r'''
        Calculates relative humidity as the difference between the saturation
        vapor pressure (SVP) and VPD, normalized by the SVP:

        $$
        \text{RH} = \frac{\text{SVP} - \text{VPD}}{\text{SVP}}
        $$

        Parameters
        ----------
        temp_k : int or float or numpy.ndarray
        vpd : int or float or numpy.ndarray

        Returns
        -------
        int or float or numpy.ndarray
        '''
        # Requiring relative humidity to be calculated from VPD:
        #   VPD = VPsat - VPactual; RH = VPactual / VPsat
        #     --> RH = (VPsat - VPD) / VPsat
        esat = svp(temp_k)
        avp = esat - vpd
        rh = avp / esat
        return np.where(avp < 0, 0, np.where(rh > 1, 1, rh))

    def evapotranspiration(
            self, lw_net_day: Number, lw_net_night: Number,
            sw_rad_day: Number, sw_rad_night: Number, sw_albedo: Number,
            temp_day: Number, temp_night: Number, temp_annual: Number,
            tmin: Number, vpd_day: Number, vpd_night: Number,
            pressure: Number, fpar: Number, lai: Number,
            f_wet: Number = None, separate: bool = False
        ) -> Iterable[Tuple[Sequence, Sequence]]:
        '''
        Instaneous evapotranspiration (ET) [kg m-2 s-1], the sum of bare soil
        evaporation, wet canopy evaporation, and canopy transpiration.

        Parameters
        ----------
        lw_net_day : Number
            Net downward long-wave radiation integrated during daylight hours
        lw_net_night : Number
            Net downward long-wave radiation integrated during nighttime hours
        sw_rad_day : Number
            Down-welling short-wave radiation integrated during daylight hours
        sw_rad_night : Number
            Down-welling short-wave radiation integrated during night-time hours
        sw_albedo : Number
            Down-welling short-wave albedo (under "black-sky" conditions)
        temp_day : Number
            Daytime prevailing air temperature at 10-meter height (deg K)
        temp_night : Number
            Nighttime prevailing air temperature at 10-meter height (deg K)
        temp_annual : Number
            Mean annual air temperature at 10-meter height (deg K)
        tmin : Number
            Minimum daily air temperature at 10-meter height (deg K)
        vpd_day : Number
            Daytime mean vapor pressure deficit (VPD) (Pa)
        vpd_night : Number
            Nighttime mean VPD (Pa)
        pressure : Number
            Air pressure (Pa)
        fpar : Number
            Fraction of photosynthetically active radiation (PAR) absorbed by
            the vegetation canopy
        lai : Number
            Leaf area index (LAI)
        f_wet : Number
            (Optional) Fraction of surface that is saturated with water
        separate : bool
            True to return the separate ET components; False to return their
            sum (Default: False)

        Returns
        -------
        tuple
            Sequence of (daytime, nighttime) ET flux(es)
        '''
        # Get radiation intercepted by the soil surface
        rad_soil_all = self.radiation_soil(
            lw_net_day, lw_net_night, sw_rad_day, sw_rad_night, sw_albedo,
            temp_day, temp_night, temp_annual, fpar)
        # Compute day and night components
        grouped_drivers = zip(
            (temp_day, temp_night),
            (vpd_day, vpd_night),
            (sw_rad_day, sw_rad_night),
            (lw_net_day, lw_net_night),
            rad_soil_all)
        e_canopy = list()
        e_soil = list()
        transpiration = list()
        et_total = list()
        for i, group in enumerate(grouped_drivers):
            daytime = (i == 0)
            temp_k, vpd, sw_rad, lw_net, rad_soil = group
            # Net radiation to surface, based on down-welling short-wave
            #   radiation and net long-wave radiation
            rad_net = sw_rad * (1 - sw_albedo) + lw_net
            # Radiation intercepted by the canopy
            rad_canopy = fpar * rad_net
            # Compute wet surface fraction and other quantities
            # -- Requiring relative humidity to be calculated from VPD:
            #   VPD = VPsat - VPactual; RH = VPactual / VPsat
            #     --> RH = (VPsat - VPD) / VPsat
            rhumidity = MOD16.rhumidity(temp_k, vpd)
            if f_wet is None:
                f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
            # Slope of saturation vapor pressure curve
            s = svp_slope(temp_k)
            # Latent heat of vaporization (J kg-1)
            lhv = latent_heat_vaporization(temp_k)
            # Correction for atmospheric temperature and pressure
            #   (Equation 13, MOD16 C6.1 User's Guide)
            r_corr = (101300 / pressure) * (temp_k / 293.15)**1.75
            # EVAPORATION FROM WET CANOPY
            e_canopy.append(self.evaporation_wet_canopy(
                pressure, temp_k, vpd, lai, fpar, rad_canopy, lhv, rhumidity,
                f_wet))
            # EVAPORATION FROM BARE SOIL
            e_soil.append(self.evaporation_soil(
                pressure, temp_k, vpd, fpar, rad_soil, r_corr, lhv, rhumidity,
                f_wet))
            # PLANT TRANSPIRATION
            transpiration.append(
                self.transpiration(
                    pressure, temp_k, vpd, lai, fpar, rad_canopy, tmin,
                    r_corr, lhv, rhumidity, f_wet, daytime = daytime))
            if separate:
                et_total.append((e_canopy[i], e_soil[i], transpiration[i]))
            else:
                et_total.append((e_canopy[i] + e_soil[i] + transpiration[i]))
        return et_total

    def evaporation_soil(
            self, pressure: Number, temp_k: Number, vpd: Number, fpar: Number,
            rad_soil: Number, r_corr: Number, lhv: Number = None,
            rhumidity: Number = None, f_wet: Number = None
        ) -> Number:
        r'''
        Evaporation from bare soil, calculated as the sum of evaporation from
        the saturated and unsaturated soil surfaces, as:

        $$
        \lambda E_{\mathrm{sat}} = F_{\text{wet}}
          \frac{s A_{\mathrm{soil}} + \rho C_p (1 - F_c) D r_a^{-1}}{s + \gamma(1 + r_t r_a^{-1})}
        $$
        $$
        \lambda E_{\mathrm{unsat}} = (1 - F_{\text{wet}})
          \frac{s A_{\mathrm{soil}} + \rho C_p (1 - F_c) D r_a^{-1}}{s + \gamma(1 + r_t r_a^{-1})}
        \phi^{D\beta^{-1}}
        $$

        NOTE: The `r_corr` argument to this function is different from that
        defined in Equation 13 in the MOD16 C61 User's Guide. For numerical
        stability, `r_corr` equals 1/r where r is the quantity defined in
        Equation 13. See `MOD16.evapotranspiration()` for how `r_corr` is
        calculated.

        Parameters
        ----------
        pressure : float or numpy.ndarray
            The air pressure in Pascals
        temp_k : float or numpy.ndarray
            The air temperature in degrees K
        vpd : float or numpy.ndarray
            The vapor pressure deficit in Pascals
        fpar : float or numpy.ndarray
            Fraction of photosynthetically active radiation (PAR) absorbed by
            the vegetation canopy
        rad_soil : float or numpy.ndarray
            Net radiation to the soil surface (J m-2 s-1)
        r_corr : float or numpy.ndarray or None
            The temperature and pressure correction factor for surface
            conductance
        lhv : float or numpy.ndarray or None
            (Optional) The latent heat of vaporization
        rhumidity : float or numpy.ndarray or None
            (Optional) The relative humidity
        f_wet : float or numpy.ndarray or None
            (Optional) The fraction of the surface that has standing water

        Returns
        -------
        float or numpy.ndarray
            Evaporation from bare soil (kg m-2 s-1)
        '''
        if lhv is None:
            lhv = latent_heat_vaporization(temp_k)
        if rhumidity is None:
            rhumidity = MOD16.rhumidity(temp_k, vpd)
        if f_wet is None:
            f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
        # Slope of saturation vapor pressure curve
        s = svp_slope(temp_k)
        # Air density
        rho = MOD16.air_density(temp_k, pressure, rhumidity)
        # Psychrometric constant
        gamma = psychrometric_constant(pressure, temp_k)
        # Resistance to radiative heat transfer through air ("rrc" or "rrs")
        r_r = (rho * SPECIFIC_HEAT_CAPACITY_AIR) / (
            4 * STEFAN_BOLTZMANN * temp_k**3)
        # Total aerodynamic resistance as a function of VPD and the
        #   atmospheric boundary layer resistance...
        # UPDATED 2024-02, as boundary-layer resistance should be at maximum
        #   when VPD is low and at minimum when VPD is high
        r_tot = np.where(vpd <= self.vpd_open, self.rbl_max,
            np.where(vpd >= self.vpd_close, self.rbl_min,
            self.rbl_max - (
                (self.rbl_max - self.rbl_min) * (self.vpd_close - vpd))\
                    / (self.vpd_close - self.vpd_open)))
        # ...CORRECTED for atmospheric temperature, pressure
        r_tot = r_tot / r_corr
        # -- Aerodynamic resistance at the soil surface
        r_as = (r_tot * r_r) / (r_tot + r_r)
        # -- Terms common to evaporation and potential evaporation
        numer = (s * rad_soil) +\
            (rho * SPECIFIC_HEAT_CAPACITY_AIR * (1 - fpar) * (vpd / r_as))
        denom = (s + gamma * (r_tot / r_as))
        # -- Evaporation from "wet" soil (saturated fraction)
        evap_sat = (numer * f_wet) / denom
        # -- (Potential) Evaporation from unsaturated fraction
        evap_unsat = (numer * (1 - f_wet)) / denom
        # BARE SOIL EVAPORATION
        # -- Finally, apply the soil moisture constraint from Fisher et al.
        #   (2008); see MOD16 C6.1 User's Guide, pp. 9-10
        e = np.where(evap_sat < 0, 0, evap_sat)
        e += np.where(
            evap_unsat < 0, 0,
            evap_unsat * np.power(rhumidity, vpd / self.beta))
        # Divide by the latent heat of vaporization (J kg-1) to obtain mass
        #   flux (kg m-2 s-1)
        return e / lhv

    def evaporation_wet_canopy(
            self, pressure: Number, temp_k: Number, vpd: Number, lai: Number,
            fpar: Number, rad_canopy: Number, lhv: Number = None,
            rhumidity: Number = None, f_wet: Number = None, tiny: float = 1e-9
        ) -> Number:
        r'''
        Wet canopy evaporation calculated according to the MODIS MOD16
        framework:

        $$
        \lambda E = F_{wet} \frac{
            s\, A_c\, F_c + \rho\, C_p\, D\, (F_c / r_w)
          }{s + (P_a\, C_p\, r_e)(\lambda\, \varepsilon\, r_w)^{-1}}
        $$

        Where:

        - \(s\) is the slope of the saturation vapor pressure curve (see
            `mod16.svp_slope()`)
        - \(A_c\) is the radiation intercepted by the canopy [J m-2 s-1]
        - \(F_c\) is the canopy fraction of ground cover (i.e., fPAR)
        - \(\rho\) is the density of air [kg m-3]
        - \(C_p\) is the specific heat capacity of air
        - \(D\) is the vapor pressure deficit
        - \(F_{wet}\) is the fraction of the land surface that is saturated
        - \(P_a\) is the air pressure (Pa)
        - \(\lambda\) is the latent heat of vaporization
        - \(\varepsilon\) is the ratio of molecular weight of water vapor to that
            of dry air

        Parameters
        ----------
        pressure : float or numpy.ndarray
            The air pressure in Pascals
        temp_k : float or numpy.ndarray
            The air temperature in degrees K
        vpd : float or numpy.ndarray
            The vapor pressure deficit in Pascals
        lai : float or numpy.ndarray
            The leaf area index (LAI)
        fpar : float or numpy.ndarray
            Fraction of photosynthetically active radiation (PAR) absorbed by
            the vegetation canopy
        rad_canopy : float or numpy.ndarray
            Net radiation to the canopy (J m-2 s-1)
        lhv : float or numpy.ndarray or None
            (Optional) The latent heat of vaporization
        rhumidity : float or numpy.ndarray or None
            (Optional) The relative humidity
        f_wet : float or numpy.ndarray or None
            (Optional) The fraction of the surface that has standing water
        tiny : float
            (Optional) A very small number, the numerical tolerance for zero
            (Default: 1e-9)

        Returns
        -------
        float or numpy.ndarray
            Evaporation from the wet canopy surface (kg m-2 s-1)
        '''
        if lhv is None:
            lhv = latent_heat_vaporization(temp_k)
        if rhumidity is None:
            rhumidity = MOD16.rhumidity(temp_k, vpd)
        if f_wet is None:
            f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
        # MODPR16_main.c L3893; zero LAI/ f_wet -> zero evaporation
        f_wet = np.where(f_wet == 0, f_wet + tiny, f_wet)
        lai = np.where(lai == 0, lai + tiny, lai)
        # Slope of saturation vapor pressure curve
        s = svp_slope(temp_k)
        # Air density
        rho = MOD16.air_density(temp_k, pressure, rhumidity)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # Wet canopy resistance to sensible heat ("rhc")
            r_h = 1 / (self.gl_sh * lai * f_wet)
            # Wet canopy resistance to evaporated water on the surface ("rvc")
            r_e = 1 / (self.gl_wv * lai * f_wet)
            # Resistance to radiative heat transfer through air ("rrc")
            r_r = (rho * SPECIFIC_HEAT_CAPACITY_AIR) / (
                4 * STEFAN_BOLTZMANN * temp_k**3)
        # Aerodynamic resistance to evaporated water on the wet canopy
        #   surface ("rhrc")
        r_a_wet = np.divide(r_h * r_r, r_h + r_r) # (s m-1)
        numer = f_wet * ((s * rad_canopy) +\
            (rho * SPECIFIC_HEAT_CAPACITY_AIR * fpar * (vpd * 1/r_a_wet)))
        denom = s + ((pressure * SPECIFIC_HEAT_CAPACITY_AIR * r_e) *\
            1/(lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR * r_a_wet))
        # Mu et al. (2011), Equation 17; PET (J m-2 s-1) is divided by the
        #   latent heat of vaporization (J kg-1) to obtain mass flux
        #   (kg m-2 s-1)
        evap = np.where(numer < 0, 0, (numer / denom) / lhv)
        # If f_wet or LAI are ~zero, then wet canopy evaporation is zero
        return np.where(np.logical_or(f_wet <= tiny, lai <= tiny), 0, evap)

    def radiation_soil(
            self, lw_net_day: Number, lw_net_night: Number,
            sw_rad_day: Number, sw_rad_night: Number, sw_albedo: Number,
            temp_day: Number, temp_night: Number, temp_annual: Number,
            fpar: Number
        ) -> Iterable[Tuple[Sequence, Sequence]]:
        r'''
        Net radiation received by the soil surface, calculated as the
        difference between the net radiation intercepted by the soil and the
        soil heat flux. Net incoming radiation to the land surface (soil and
        non-soil components) during the daytime is calculated:

        $$
        A = (1-\alpha)R_{S\downarrow} + R_L
        $$

        Where \(\alpha\) is the short-wave albedo (under "black-sky"
        conditions); \(R_{S\downarrow}\) is the (daytime) down-welling short-
        wave radidation; \(R_L\) is the (daytime) net downward long-wave
        radiation.

        At nighttime, net incoming radiation is simply the net downward long-
        wave radiation during nighttime hours:

        $$
        A = R_L
        $$

        And the net radiation received by the soil is modulated by the
        fractional area covered by vegetation, \(F_C\), such that areas with
        100% vegetation cover have no net radiation to the soil:

        $$
        A_{\text{soil}} = (1 - F_C)\times (A - G)
        $$

        Parameters
        ----------
        lw_net_day : int or float or numpy.ndarray
            Net downward long-wave radiation integrated during daylight hours
        lw_net_night : int or float or numpy.ndarray
            Net downward long-wave radiation integrated during nighttime hours
        sw_rad_day : int or float or numpy.ndarray
            Down-welling short-wave radiation integrated during daylight hours
        sw_rad_night : int or float or numpy.ndarray
            Down-welling short-wave radiation integrated during night-time hours
        sw_albedo : int or float or numpy.ndarray
            Down-welling short-wave albedo ("black-sky" albedo)
        temp_day : int or float or numpy.ndarray
            Average temperature (degrees K) during daylight hours
        temp_night : int or float or numpy.ndarray
            Average temperature (degrees K) during nighttime hours
        temp_annual : int or float or numpy.ndarray
            Annual average daily temperature (degrees K)
        fpar : float or numpy.ndarray
            Fraction of photosynthetically active radiation (PAR) absorbed by
            the vegetation canopy

        Returns
        -------
        tuple
            A 2-element tuple of (daytime, night-time) radiation received by the
            soil surface
        '''
        # At night, SW radiation should be zero (i.e., sw_rad_night = 0) which
        #   means that only LW radiation contributes (L3852 of MODPR16_main.c)
        rad_net_day = sw_rad_day * (1 - sw_albedo) + lw_net_day
        rad_net_night = lw_net_night
        g_soil_day, g_soil_night = self.soil_heat_flux(
            rad_net_day, rad_net_night, temp_day, temp_night, temp_annual)
        # Section 2.8 of MOD16 Collection 6.1 User's Guide (pp. 10-11);
        #   if soil heat flux is greater than net incoming radiation...
        g_soil_day = np.where(
            np.logical_and(rad_net_day - g_soil_day < 0, rad_net_day > 0),
            rad_net_day, g_soil_day)
        g_soil_night = np.where(
            np.logical_and(
                rad_net_day > 0,
                (rad_net_night - g_soil_night) < (-0.5 * rad_net_day)),
            rad_net_night + (0.5 * rad_net_day), g_soil_night)
        # "In the improved algorithm, there will be no soil heat flux
        #   interaction between the soil and atmosphere if the ground is 100%
        #   covered with vegetation" (Mu et al. 2011, RSE)
        # Equation 7, Mu et al. (2011)
        rad_soil_day = (1 - fpar) * (rad_net_day - g_soil_day)
        rad_soil_night = (1 - fpar) * (rad_net_night - g_soil_night)
        return (rad_soil_day, rad_soil_night)

    def soil_heat_flux(
            self, rad_net_day: Number, rad_net_night: Number,
            temp_day: Number, temp_night: Number, temp_annual: Number
        ) -> Iterable[Tuple[Sequence, Sequence]]:
        r'''
        Soil heat flux [MJ m-2], based on surface energy balance. If the mean
        annual temperature is between `Tmin_close` and 25 deg C and the
        contrast between daytime and nighttime air temperatures is greater
        than 5 deg C:

        $$
        G_{\text{soil}} = 4.73 T - 20.87
            \quad\text{iff}\quad T_{\text{min,close}}
            \le T_{\text{annual}} < 25 ,\,
                T_{\text{day}} - T_{\text{night}} \ge 5
        $$

        Otherwise, soil heat flux is zero.

        Finally, soil heat flux should be no greater than 39% of the net
        radiation to the land surface.

        $$
        G_{\text{soil}} = 0.39 A \quad\text{iff}\quad G_{soil} > 0.39 |A|
        $$

        Parameters
        ----------
        rad_net_day : int or float or numpy.ndarray
            Net radiation to the land surface during daylight hours; i.e.,
            integrated while the sun is up [MJ m-2]
        rad_net_night : int or float or numpy.ndarray
            Net radiation to the land surface during nighttime hours; i.e.,
            integrated while the sun is down [MJ m-2]
        temp_day : int or float or numpy.ndarray
            Average temperature (degrees K) during daylight hours
        temp_night : int or float or numpy.ndarray
            Average temperature (degrees K) during nighttime hours
        temp_annual : int or float or numpy.ndarray
            Annual average daily temperature (degrees K)

        Returns
        -------
        tuple
            A 2-element tuple of (daytime, nighttime) soil heat flux
        '''
        # See ca. Line 4355 in MODPR16_main.c
        g_soil = []
        condition = np.logical_and(
            np.logical_and(
                temp_annual < (273.15 + 25),
                temp_annual > (273.15 + self.tmin_close)
            ), (temp_day - temp_night) >= 5)
        for rad_i, temp_i in zip(
                (rad_net_day, rad_net_night), (temp_day, temp_night)):
            g = np.where(condition, (4.73 * (temp_i - 273.15)) - 20.87, 0)
            # Modify soil heat flux under these conditions...
            g = np.where(np.abs(g) > (0.39 * np.abs(rad_i)), 0.39 * rad_i, g)
            # g_soil is the soil heat flux when fPAR == 0
            # NOTE: We do not scale the result by (1 - fPAR) here, as in
            #   Mu et al. (2011), because this is accounted for when
            #   calculating net radidation received by the soil surface;
            #   see MOD16.radiation_soil()
            g_soil.append(g)
        return g_soil

    def surface_conductance(self, tmin: Number, vpd_day: Number) -> Number:
        r'''
        Surface conductance to transpiration (m s-1), during daylight hours,
        NOT corrected for atmospheric temperature and pressure. Surface
        conductance at night is assumed to be zero (as non-CAM plants close
        stomata at night to prevent water loss when there is no photosynthetic
        carbon gain).

        $$
        G_{\text{surf}} = C_L \times f(T_{\text{min}}) \times f(\text{VPD})
        $$

        Where \(C_L\) is the mean potential stomatal conductance per unit leaf
        area.

        Parameters
        ----------
        tmin : int or float or numpy.ndarray
            Daily minimum temperature (degrees K)
        vpd_day : int or float or numpy.ndarray
            Daytime VPD (Pa)

        Returns
        -------
        int or float or numpy.ndarray
            The daytime surface conductance to transpiration (m s-1)
        '''
        m_tmin = linear_constraint(self.tmin_close, self.tmin_open)
        m_vpd = linear_constraint(self.vpd_open, self.vpd_close, 'reversed')
        return (self.csl * m_tmin(tmin - 273.15) * m_vpd(vpd_day))

    def transpiration(
            self, pressure: Number, temp_k: Number, vpd: Number, lai: Number,
            fpar: Number, rad_canopy: Number, tmin: Number, r_corr: Number,
            lhv: Number = None, rhumidity: Number = None,
            f_wet: Number = None, daytime: bool = True, tiny = 1e-9
        ) -> Number:
        r'''
        Plant transpiration over daytime or night-time hours, in mass flux
        units (kg m-2 s-1), according to:

        $$
        \lambda E_T = \frac{s A_c + \rho C_p F_c D r_d^{-1}}
          {s + \gamma(1 + r_s r_d^{-1})}
        $$

        NOTE: The `r_corr` argument to this function is different from that
        defined in Equation 13 in the MOD16 C61 User's Guide. For numerical
        stability, `r_corr` equals 1/r where r is the quantity defined in
        Equaiton 13. See `MOD16.evapotranspiration()` for how `r_corr` is
        calculated.

        Parameters
        ----------
        pressure : float or numpy.ndarray
            The air pressure in Pascals
        temp_k : float or numpy.ndarray
            The air temperature in degrees K
        vpd : float or numpy.ndarray
            The vapor pressure deficit in Pascals
        lai : float or numpy.ndarray
            The leaf area index (LAI)
        fpar : float or numpy.ndarray
            Fraction of photosynthetically active radiation (PAR) absorbed by
            the vegetation canopy
        rad_canopy : float or numpy.ndarray
            Net radiation to the canopy (J m-2 s-1)
        tmin : float or numpy.ndarray
            Minimum daily temperature (degrees K)
        r_corr : float or numpy.ndarray or None
            The temperature and pressure correction factor for surface
            conductance
        lhv : float or numpy.ndarray or None
            (Optional) The latent heat of vaporization
        rhumidity : float or numpy.ndarray or None
            (Optional) The relative humidity
        f_wet : float or numpy.ndarray or None
            (Optional) The fraction of the surface that has standing water
        daytime : bool
            (Optional) True to calculate daytime (sun-up) transpiration, False
            to calculate night-time (sun-down) transpiration (Default: True)
        tiny : float
            (Optional) A very small number, the numerical tolerance for zero
            (Default: 1e-9)

        Returns
        -------
        float or numpy.ndarray
            Transpiration (kg m-2 s-1)
        '''
        if lhv is None:
            lhv = latent_heat_vaporization(temp_k)
        if rhumidity is None:
            rhumidity = MOD16.rhumidity(temp_k, vpd)
        if f_wet is None:
            f_wet = np.where(rhumidity < 0.7, 0, np.power(rhumidity, 4))
        # Slope of saturation vapor pressure curve
        s = svp_slope(temp_k)
        # Air density
        rho = MOD16.air_density(temp_k, pressure, rhumidity)
        # Psychrometric constant
        gamma = psychrometric_constant(pressure, temp_k)
        # Resistance to radiative heat transfer through air ("rrc")
        r_r = (rho * SPECIFIC_HEAT_CAPACITY_AIR) / (
            4 * STEFAN_BOLTZMANN * temp_k**3)
        # Surface conductance (zero at nighttime)
        g_surf = 0 # (Equation 13, MOD16 C6.1 User's Guide)
        if daytime:
            # NOTE: C_L (self.csl) is included in self.surface_conductance()
            g_surf = self.surface_conductance(tmin, vpd) / r_corr
        g_cuticular = self.g_cuticular / r_corr
        # -- Canopy conductance, should be zero when LAI or f_wet are zero;
        #   updated calculation for conductance to sensible heat, see User
        #   Guide's Equation 15
        gl_sh = self.gl_sh * lai * (1 - f_wet)
        g = ((gl_sh * (g_surf + g_cuticular)) / (
            gl_sh + g_surf + g_cuticular))
        g_canopy = np.where(np.logical_and(lai > 0, (1 - f_wet) > 0), g, tiny)
        # -- Aerodynamic resistance to heat, water vapor from dry canopy
        #   surface into the air (Equation 16, MOD16 C6.1 User's Guide)
        r_a_dry = (1/self.gl_sh * r_r) / (1/self.gl_sh + r_r) # (s m-1)
        # If canopy radiation is negative, drop (s * A_c) term in the
        #   transpriration calculation (L4046 in MODPR16_main.c)
        rad_canopy = np.where(rad_canopy < 0, 0, rad_canopy)
        # PLANT TRANSPIRATION
        t = (1 - f_wet) * ((s * rad_canopy) +\
            (rho * SPECIFIC_HEAT_CAPACITY_AIR * fpar * (vpd / r_a_dry)))
        if daytime:
            t /= (s + gamma * (1 + (1 / g_canopy) / r_a_dry))
        else:
            # At nighttime, g_canopy is zero, so simplify the calculation
            t /= (s + gamma) # (Equation 17, MOD16 C6.1 User's Guide)
        # Divide by the latent heat of vaporization (J kg-1) to obtain mass
        #   flux (kg m-2 s-1)
        return np.where(g_canopy <= tiny, 0, t / lhv)


def psychrometric_constant(pressure: Number, temp_k: Number) -> Number:
    r'''
    The psychrometric constant, which relates the vapor pressure to the air
    temperature. Calculation derives from the "Handbook of Hydrology" by D.R.
    Maidment (1993), Section 4.2.28.

    $$
    \gamma = \frac{C_p \times P}{\lambda\times 0.622}
    $$

    Where \(C_p\) is the specific heat capacity of air, \(P\) is air pressure,
    and \(\lambda\) is the latent heat of vaporization. The \(C_p\) of air
    varies with its saturation, so the value 1.013e-3 [MJ kg-1 (deg C)-1]
    reflects average atmospheric conditions.

    Parameters
    ----------
    pressure : float or numpy.ndarray
        The air pressure in Pascals
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
        The psychrometric constant at this pressure, temperature (Pa K-1)
    '''
    lhv = latent_heat_vaporization(temp_k) # Latent heat of vaporization (J kg-1)
    return (SPECIFIC_HEAT_CAPACITY_AIR * pressure) /\
        (lhv * MOL_WEIGHT_WET_DRY_RATIO_AIR)


def radiation_net(
        sw_rad: Number, sw_albedo: Number, temp_k: Number) -> Number:
    r'''
    DEPRECATED. Net incoming radiation to the land surface, calculated
    according to the MOD16 algorithm and Cleugh et al. (2007); see
    Equation 7 in the MODIS MOD16 Collection 6.1 User's Guide.

    - Cleugh, H. A., Leuning, R., Mu, Q., & Running, S. W. (2007).
      Regional evaporation estimates from flux tower and MODIS satellite data.
      *Remote Sensing of Environment*, 106(3), 285–304.

    $$
    R_{net} = (1 - \alpha)\times R_{S\downarrow} +
        (\varepsilon_a - \varepsilon_s) \times \sigma \times T^4
    \quad\mbox{where}\quad \varepsilon_s = 0.97
    $$

    Where \(\alpha\) is the MODIS albedo, `R_S` is down-welling short-wave
    radiation, \(\sigma\) is the Stefan-Boltzmann constant, and:

    $$
    \varepsilon_a = 1 - 0.26\,\mathrm{exp}\left(
      -7.77\times 10^{-4}\times (T - 273.15)^2
    \right)
    $$

    Parameters
    ----------
    sw_rad : float or numpy.ndarray
        Incoming short-wave radiation (W m-2)
    sw_albedo : float or numpy.ndarray
        Black-sky albedo, e.g., from MODIS MCD43A3
    temp_k : float or numpy.ndarray
        Air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
        Net incoming radiation to the land surface (W m-2)
    '''
    # Mu et al. (2011), Equation 5
    emis_surface = 0.97
    emis_atmos = 1 - 0.26 * np.exp(-7.77e-4 * np.power(temp_k - 273.15, 2))
    return sw_rad * (1 - sw_albedo) +\
        STEFAN_BOLTZMANN * (emis_atmos - emis_surface) * np.power(temp_k, 4)


def svp(temp_k: Number) -> Number:
    r'''
    The saturation vapor pressure, based on [
    the Food and Agriculture Organization's (FAO) formula, Equation 13
    ](http://www.fao.org/3/X0490E/x0490e07.htm).

    $$
    \mathrm{SVP} = 1\times 10^3\left(
    0.6108\,\mathrm{exp}\left(
      \frac{17.27 (T - 273.15)}{T - 273.15 + 237.3}
      \right)
    \right)
    $$

    This is also referred to as the August-Roche-Magnus equation.

    Parameters
    ----------
    temp_k : float or numpy.ndarray
        The air temperature in degrees K

    Returns
    -------
    float or numpy.ndarray
    '''
    temp_c = temp_k - 273.15
    # And convert from kPa to Pa
    return 1e3 * 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def svp_slope(temp_k: Number, s: Number = None) -> Number:
    r'''
    The slope of the saturation vapour pressure curve, which describes the
    relationship between saturation vapor pressure and temperature. This
    approximation is taken from the MOD16 C source code. An alternative is
    based on [the Food and Agriculture Organization's (FAO) formula,
    Equation 13 ](http://www.fao.org/3/X0490E/x0490e07.htm).

    $$
    \Delta = 4098\times [\mathrm{SVP}]\times (T - 273.15 + 237.3)^{-2}
    $$

    Parameters
    ----------
    temp_k : float or numpy.ndarray
        The air temperature in degrees K
    s : float or numpy.ndarray or None
        Saturation vapor pressure, if already known (Optional)

    Returns
    -------
    float or numpy.ndarray
        The slope of the saturation vapor pressure curve in Pascals per
        degree K (Pa degK-1)
    '''
    if s is None:
        s = svp(temp_k)
    return 17.38 * 239.0 * s / (239.0 + temp_k - 273.15)**2
