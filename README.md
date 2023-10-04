MOD16 Evapotranspiration Model
==============================

The MOD16 terrestrial evapotranspiration algorithm calculates three surface fluxes of latent heat (water vapor):

- Evaporation from bare soil surfaces
- Evaporation from wet canopy surfaces
- Transpiration from terrestrial vegetation

This Python implementation can be used to:

- Calibrate MOD16 on observed latent heat fluxes, such as from eddy covariance flux towers
- Run MOD16 for arbitrary spatial domains (arrays) over arbitrary time steps
- Calculate the sensitivity of the model to its parameters, based on observed latent heat fluxes

**The version of MOD16 in this Python implementation is a draft release of the algorithm that will be used in MODIS Collection 7 and in VIIRS Collection 2.** There are substantial changes from MODIS MOD16 Collection 6.1, [which are detailed in the module documentation](https://arthur-e.github.io/MOD16/) (under "NOTES").


Installation and Tests
----------------------

It's recommended that you install the package in "editable" mode using `pip`. From the root of the repository:

```sh
pip install -e .
```

If you want to install additional libraries needed for calibrating MOD16:

```sh
pip install -e .[calibration]
```

Tests can be run by:

```sh
python tests/tests.py
```


Documentation
-------------

[You can read the online module documentation here.](https://arthur-e.github.io/MOD16/) Below, an overview of the MOD16 algorithm is provided.


Algorithm Details
-----------------

*As described by K. Arthur Endsley, September 2023*

### Surface Energy Balance

Net radiation intercepted by the earth's surface, $R$, can be partitioned into fluxes of sensible heat ($H$), latent heat ($\lambda E$), ground heat ($G$), and the change in heat storage ($\Delta S$):

$$
R = H + \lambda E + G + \Delta S
$$

Latent heat, or heat that has been used to vaporize water, is the quantity of interest in the MOD16 algorithm and it can generally be described in terms:

$$
\lambda E = \frac{\rho\times C_p}{\gamma} \frac{e_{\text{sat}} - e}{r_A + r_S}
$$

The individual terms are described in the sections below, but the general idea is that the latent heat flux:

- Increases with the air's capacity to store water vapor, $e_{\text{sat}} - e$
- Increases with the heat storage capacity of air, $\rho\times C_p$
- Decreases with increasing aerodynamic and surface resistances, $r_A$ + $r_S$

The flux of latent heat is also termed **evapotranspiration (ET)** because it includes both *evaporated* water and *transpired* water vapor fluxes. Most discussions of modeling ET begin with a version of the equation above, which gets complicated quickly when we try to calculate ET over large areas using weather data. As such, our description of the MOD16 ET model is instead procedural, to aid implementation.


### The MOD16 Algorithm

> The evaporation of water is like a commercial transaction in which a wet surface sells water vapour to its environment in exchange for heat...The environment can supply heat by solar radiation, by turbulent transfer from the atmosphere, or by conduction from the soil.

- From *Monteith (1965).*

MOD16 is based on the Penman-Monteith (PM) approach to calculating evapotranspiration (ET). **The central idea of the PM approach is that it combines energy conservation and the saturation vapor pressure of the air to determine: 1) How much energy is available to vaporize water; and 2) When to stop vaporizing water (when the air is saturated).**

Evapotranspiration (ET) is the sum of three components (three sources of latent heat): transpiration, evaporation from wet canopy surfaces, and evaporation from bare soil surfaces. The MOD16 algorithm calculates separate daytime (i.e., sun in the sky) and nighttime quantities for each of these components. We can understand the MOD16 algorithm as a series of steps to calculate each component wherein the input drivers are partitioned into daytime and nighttime values, starting with the calculation of the net radiation received by a surface.

Throughout, unless otherwise specified: temperature is given in degrees Kelvin; pressure is given in Pascals (Pa); resistance is given in seconds per meter.


### Net Radiation Available

The net radiation available to vaporize water, $A$, is the sum of down-welling, short-wave radiation intercepted from the sun, $R_{S\downarrow}$, and net long-wave radiation, $R_L$, from the earth and atmosphere:

$$
A = (1-\alpha)R_{S\downarrow} + R_L
$$

Where $\alpha$ is the short-wave albedo (under "black-sky" conditions), i.e., the fraction of down-welling, short-wave radiation that is not absorbed by the earth's surface. MOD16 used to calculate the latter term using the emissivity of the earth's surface; as of MODIS Collection 6.1, we instead use the net long-wave radiation from a land model. Because the sun is the only source of short-wave radiation in this system, at nighttime, $R_{S\downarrow} = 0$ and the only source of energy is $R_L$.

The three components of ET imply there are two surface sources of water vapor: the vegetation canopy (leaf surfaces) and bare soil. Therefore, the next step is to calculate the net radiation received by each of these surfaces. The net radiation received by the soil is a balance between incoming radiation, $A$, and the ground heat flux, $G$, modulated by the fraction of vegetation cover, $F_C$; i.e., if the ground is completed covered by vegetation ($F_C = 1.0$), then there is no ground heat flux. In MOD16, we use the fraction of photosynthetically active radiation (fPAR), a measure of the fraction of ground area covered by photosynthetic vegetation, as our measure of $F_C$. Therefore, net radiation received by the soil surface is calculated:

$$
A_{\text{soil}} = (1 - F_C)\times (A - G)
$$

Ground heat flux is calculated based on the surface energy balance and a pre-determined (calibrated) minimum temperature constraint, $T_{\text{min,close}}$, which represents the temperature below which plant stomata are assumed to be completely closed. If the mean annual temperature is between $T_{\text{min,close}}$ and 25 degrees C, and if the contrast between daytime and nighttime air temperatures is greater than 5 degrees C, then ground heat flux is calculated based on an empirical relationship determined using field data and spectral vegetation indices (Jacobsen and Hansen 1999; Mu et al. 2011):

$$
G = 4.73 T - 20.87 \quad\mathrm{iff}\quad T_{\text{min,close}}
        \le T_{\text{annual}} < 25 \quad\text{and}\quad T_{\text{day}} - T_{\text{night}} \ge 5
$$

Otherwise, ground heat flux is zero. When non-zero (as above), we additionally constrain $G$ so that it is no greater than 39% of the net radiation to the land surface:

$$
G = 0.39 A \quad\mathrm{iff}\quad G > 0.39 |A|
$$

Again, $G$ and $A_{\text{soil}}$ are calculated separately for daytime and nighttime input values. When calculating daytime $G$, if the daytime $G$ would be greater than $A$, then $G = A$. When calculating nighttime $G$, if nighttime $A$ (equivalent to $R_L$) minus $G$ would be less than -0.5 times daytime $A$, then $G$ is taken to be equal to nighttime $A$ plus half of daytime $A$.


### Vapor Pressure, Humidity, and Saturated Fraction

There are a number of quantities used in calculating all three components that must be calculated separately for daytime and nighttime inputs.

The **saturation vapor pressure (SVP)** is calculated based on the August-Roche-Magnus equation, also used by [the Food and Agriculture Organization (FAO) (Equation 13)](http://www.fao.org/3/X0490E/x0490e07.htm):

$$
\text{SVP} = 1\times 10^3\left(
  0.6108\times \text{exp}\left(
    \frac{17.27 (T - 273.15)}{(T - 273.15) + 237.3}
    \right)
  \right)
$$

Where $T$ is the temperature in degrees K.

**Vapor pressure deficit (VPD)** can be calculated a variety of ways but is always defined as the difference between SVP and the actual vapor pressure (AVP); hence, MOD16 calculates VPD by first calculating the AVP, after Gates (1980):

$$
\text{AVP} = \frac{\text{QV10M}\times \text{P}}{0.622 + 0.379\times \text{QV10M}}
$$

Where QV10M is the water vapor mixing ratio at 10-meter height (units: Pa) and P is the surface pressure (units: Pa). Currently, MOD16 uses a slightly different formula for the calculation of SVP when calculating VPD (units: Pa):

$$
\text{VPD} = \left(610.7\times \text{exp}\left(
  \frac{17.38\times T}{239 + T}
\right) - \text{AVP}\right)
$$

**Relative humidity (RH)** can then be calculated as the difference between VPD and SVP, normalized by the SVP:

$$
\text{RH} = \frac{\text{SVP} - \text{VPD}}{\text{SVP}}
$$

After Fisher et al. (2008), we calculate **the fraction of the land surface that is saturated, $F_{\text{wet}}$,** based on the relative humidity:

- $F_{\text{wet}} = 0$ iff RH < 70%
- Otherwise, $F_{\text{wet}} = \text{RH}^4$

**The latent heat of vaporization, $\lambda$** (units: Joules per kilogram), is a key term that quantifies the amount of energy required to vaporize a kilogram of water, based on the current temperature of the water:

$$
\lambda = (2.501 - 0.002361\times (T - 273.15))\times 10^6
$$

**The psychrometric constant,** which relates the vapor pressure of air to its temperature, is calculated:

$$
\gamma = \frac{C_p \times P}{\lambda\times 0.622}
$$

This is the approach [used by the FAO](https://www.fao.org/3/X0490E/x0490e07.htm) but it is also described by Maidment (1969). $C_p = 1013$ (J per kilogram per Kelvin) is the specific heat capacity of air, P is the air pressure, and 0.622 is the ratio of molecular weights, water vapor to dry air. Values for $C_p$ and water vapor-dry air ratio are taken from Monteith & Unswoth (2001).

**The slope of the saturation vapor pressure curve,** $s$ (units: Pa per degree K), which describes the relationship between SVP and temperature, must also be determined:

$$
s = \frac{17.38\times 239\times \text{SVP}}{(239 + (T - 273.15))^2}
$$

Note that this formula comes from the MOD16 source code and cannot be further attributed. An alternative would be [the FAO formula (Equation 13).](http://www.fao.org/3/X0490E/x0490e07.htm)

**Air density** must also be calculated; we use the NIST simplified air density formula with buoyancy correction (NPL 2021):

$$
\rho = \frac{(0.34844\times P) - \text{RH}(0.00252\times (T - 273.15)) - 0.020582}{T}
$$

Where $P$ is air pressure in millibars; RH should be expressed in percentage terms.

In MODIS Collection 6.1, MOD16 used air pressure as given by the surface meteorology dataset used (e.g., as a field in the re-analysis dataset). Going forward, MOD16 will instead calculate air pressure as a function of elevation:

$$
P_a = P_{\text{std}}\times \left(
    1 - \ell z T_{\text{std}}^{-1}
  \right)^{5.256}
$$

Where $\ell$ is the standard temperature lapse rate (-0.0065 deg K per meter), $z$ is the elevation in meters, and $P_{\text{std}}$, $T_{\text{std}}$ are the standard pressure (101325 Pa) and temperature (288.15 deg K) at sea level. The exponent 5.256 is a pre-determined quantity based on physical constants and derived from the discussion on atmospheric statics in Iribane & Godson (1981).


### Evaporation from Wet Canopy

Evaporation from wet canopy occurs when a portion of the surface area under investigation is saturated with water; it consists of precipitated water intercepted by the leaves of trees and other plants. As with all the components of ET, it is calculated separately for daytime and nighttime inputs. First, net radiation to the canopy is calculated as:

$$
A_{\text{canopy}} = F_C\times A
$$

The movement of water vapor from the surface to the atmosphere is analogized to an electrical circuit (Zhang et al. 2016). Therefore, the next step involves calculating the *resistances* to the flow of water vapor. The **resistance to radiative heat transfer through the air,** $r_R$, is given as a function of temperature ($T$, degrees K) and air density ($\rho$):

$$
r_R = \frac{\rho C_p}{4 \sigma T^3}
$$

Where $\sigma$ is the Stefan-Boltzmann constant.

**The resistant of wet canopy to sensible heat** is given in terms of the leaf area index (LAI), $F_{\text{wet}}$, and the leaf conductance to sensible heat (per unit LAI), $g_{SH}$, which is a free parameter:

$$
r_{SH} = \frac{1}{g_{SH}\times \text{LAI}\times F_{\text{wet}}}
$$

Similarly, **the resistance of leaf surfaces in wet canopy to evaporated water,** $r_{WV}$, is given in terms of the leaf conductance to evaporated water (per unit LAI), $g_{WV}$, a free parameter:

$$
r_{WV} = \frac{1}{g_{WV}\times \text{LAI}\times F_{\text{wet}}}
$$

And, finally, the **aerodynamic resistance to evaporated water on the wet canopy surface,** $r_{\text{wet}}$, is given by combining the resistances $r_R$ and $r_{SH}$ in parallel, because heat (of the wet canopy surface) can be lost through either channel:

$$
r_{\text{wet}} = \frac{r_{SH}\times r_R}{r_{SH} + r_R}
$$

We now have all the quantities necessary to calculate the **wet canopy evaporation flux:**

$$
\lambda E_{\text{canopy}} = F_{\text{wet}} \frac{
        s  A_{\text{canopy}}  F_c + \rho  C_p  [\text{VPD}]  (F_c / r_{\text{wet}})
      }{s + (P_a  C_p  r_{WV})(0.622  \lambda  r_{\text{wet}})^{-1}}
$$

Again 0.622 is the ratio of molecular weights, water vapor to dry air. **Note that if $F_{\text{wet}}$ or LAI are zero, then $\lambda E_{\text{canopy}}$ is also zero.**


### Evaporation from Bare Soil Surfaces

Calculating evaporation from bare soil surfaces requires calculating both potential evaporation (PET) from the unsaturated soil surface and actual evaporation from the saturated soil surface. As with evaporation from wet canopy, we begin with calculating the resistances to water vapor fluxes.

**The total aerodynamic resistance to water vapor,** $r_{\text{total}}$, is given in terms of pre-determined (calibrated) quantities including:

- $r_{\text{BL,max}}$, the maximum boundary-layer resistance;
- $r_{\text{BL,min}}$, the minimum boundary-layer resistance;
- $\text{VPD}^{\text{close}}$, the vapor pressure deficit (VPD) at which stomata are almost completely closed due to water stress;
- $\text{VPD}^{\text{open}}$, the VPD at which stomata are almost completely open, i.e., experiencing no water stress.

$r_{\text{total}}$ strongly depends on the atmospheric demand for water vapor (i.e., VPD or $D$):

- iff VPD $\le \text{VPD}^{\text{open}}$:
  - $r_{\text{total}} = r_{\text{corr}} r_{\text{BL,max}}$
- iff VPD $\ge \text{VPD}^{\text{close}}$:
  - $r_{\text{total}} = r_{\text{corr}} r_{\text{BL,min}}$
- And if and only if VPD is between these values:

$$
r_{\text{total}} = r_{\text{corr}} r_{\text{BL,max}} - \frac{(r_{\text{BL,max}} - r_{\text{BL,min}})(\text{VPD}^{\text{close}} -
   \text{VPD})}{
     \text{VPD}^{\text{close}} - \text{VPD}^{\text{open}}}
$$

Essentially, when VPD is low, the boundary-layer resistance is at its maximum ($r_{\text{BL,max}}$); the atmosphere's demand for water is very low, so there is greater resistance to accepting more water vapor from the surface. When VPD is high, (greater than or equal to $\text{VPD}^{\text{close}}$), atmospheric water vapor is relatively scarce and the boundary-layer resistance is at a minimum. In between these two extremes, we linearly interpolate the boundary layer resistance.

**As the conductance of water vapor through the air varies with the air's temperature and pressure, and because prescribed values are assumed to be representative of standard temperature (293.15 deg K) and pressure (101300 Pa) conditions, a correction factor, $r_{\text{corr}}$, is applied; this is used elsewhere as well:**

$$
r_{\text{corr}} = \left(
  \frac{101300}{P_a}\left(\frac{T}{293.15}\right)^{1.75}
  \right)^{-1}
$$

**Aerodynamic resistance at the soil surface,** $r_{\text{AS}}$, is calculated as the parallel resistance of $r_R$ (from our wet-canopy evaporation calculations, above) and $r_{\text{total}}$:

$$
r_{\text{AS}} = \frac{r_R\times r_{\text{total}}}{r_R + r_{\text{total}}}
$$

Evaporation from the soil consists of the same, core PM equation:

$$
\lambda E_{\text{soil}}^* = \frac{s A_{\text{soil}} + \rho C_p (1 - F_C) [\text{VPD}] r_{\text{AS}}^{-1}}{s + \gamma r_{\text{total}}r_{\text{AS}}^{-1}}
$$

But actual evaporation from the saturated soil surface is calculated:

$$
\lambda E_{\text{soil,sat}} = F_{\text{wet}} \lambda E_{\text{soil}}^*
$$

While the actual evaporation of the unsaturated soil surface is calculated by scaling the potential evaporation by an empirical soil moisture constraint:

$$
\lambda E_{\text{unsat}} = (1 - F_{\text{wet}}) \lambda E_{\text{soil}}^* \left(
  \frac{\text{RH}}{100}
    \right)^{\text{VPD}/\beta}
$$

Where VPD is the vapor pressure deficit and $\beta$ is a free parameter, though in Collection 6.1 and earlier versions its value was fixed at 250 (Pa). Because good, global soil moisture datasets were not available when MOD16 was first developed, this "soil moisture" constraint is actually based on the relative humidity (RH).

**Finally, actual evaporation from bare soil surfaces is given as the sum of the evaporation from saturated and unsaturated fractions:**

$$
\lambda E_{\text{soil}} = \lambda E_{\text{soil,sat}} + \lambda E_{\text{unsat}}
$$


### Canopy Transpiration

Transpiration from plants depends on a key parameter, the canopy conductance, which is derived from the mean stomatal and cuticular conductances of the various leaf elements that make up the canopy. Specifically, canopy conductance to transpired water vapor per unit LAI ($C_C$), is derived from the parallel conductance of cuticular ($g_C$) and stomatal (or surface) conductance, in series with the leaf boundary layer conductance ($g_{\text{BL}}$):

$$
C_C = \frac{g_{\text{BL}}(g_S + g_C)}{g_{\text{BL}} + g_S + g_C} \quad\mbox{iff}\quad \text{LAI} > 0 \quad\text{and}\quad (1 - F_{\text{wet}}) > 0
$$

Where $g_S$ is the surface conductance, described below. If LAI or $(1 - F_{\text{wet}})$ are equal to zero, then canopy conductance is also zero.

**The surface conductance** is a proxy for the bulk conductance of water vapor from plant stomata, throughout the canopy. It is approximated using linear functions of VPD and daily minimum temperature ($T_{\text{min}}$):

$$
g_S = C_L\times f(T_{\text{min}})\times f(\text{VPD})\times r_{\text{corr}}
$$

Where $C_L$ is the mean potential stomatal conductance per unit LAI and $f()$ represents linear functions that transform VPD ($D$) or $T_{\text{min}}$ into dimensionless scalars on $[0,1]$:

- iff $T_{\text{min}} \ge T_{\text{min,open}}$: $f(T_{\text{min}}) = 1$
- iff $T_{\text{min}} \le T_{\text{min,close}}$: $f(T_{\text{min}}) = 0$
- And if and only if $T_{\text{min}}$ is in between these values:

$$
f(T_{\text{min}}) = \frac{T_{\text{min}} - T_{\text{min,close}}}{T_{\text{min,open}} - T_{\text{min,close}}}
$$

- iff $\text{VPD} \le \text{VPD}^{\text{open}}$: $f(\text{VPD}) = 1$
- iff $\text{VPD} \ge \text{VPD}^{\text{close}}$: $f(\text{VPD}) = 0$
- And if and only if VPD is in between these values:

$$
f(\text{VPD}) = \frac{\text{VPD}^{\text{close}} -
   \text{VPD}}{
     \text{VPD}^{\text{close}} - \text{VPD}^{\text{open}}}
$$

**Although the stomata of many plant species do not entirely close at night, in MOD16, it is assumed that $g_S = 0$ at nighttime,** as this optimizes the intrinsic trade-off between water loss and carbon gain during a photoperiod (night) in which carbon gain typically isn't possible due to the lack of photosynthetically active radiation.

Leaf boundary layer conductance ($g_{\text{BL}}$) is calculated:

$$
g_{\text{BL}} = g_{SH}\times \text{LAI}\times (1 - F_{\text{wet}})
$$

Where $g_{SH}$ is the leaf conductance to sensible heat per unit LAI, a free parameter.

Cuticular conductance is calculated based on the free parameter $g_{\text{cuticular}}$, which is the expected average leaf cuticular conductance:

$$
g_C = g_{\text{cuticular}}\times r_{\text{corr}}
$$

The last term in our plant transpiration calculation is **the aerodynamic resistance of the dry canopy,** $r_{\text{dry}}$, calculated as a parallel resistance to radiative ($r_R$, see wet canopy evaporation calculations) and convective heat transfer, where the latter is the inverse of leaf conductance to sensible heat ($g_{\text{SH}}$):

$$
r_{\text{dry}} = \frac{g_{\text{SH}}^{-1}\times r_R}{g_{\text{SH}}^{-1} + r_R}
$$

**Finally, we compute plant transpiration as:**

$$
\lambda E_{\text{trans}} = (1 - F_{\text{wet}})
  \frac{s A_C + \rho C_p F_C [\text{VPD}] r_{\text{dry}}^{-1}}{s + \gamma(1 + C_C^{-1} r_{\text{dry}}^{-1})}
$$

Note that, at nighttime, the denominator of $\lambda E_{\text{trans}}$ simplifies to $(s + \gamma)$, as canopy conductance is assumed to be zero. Also, if $F_{\text{wet}} = 1$, then $\lambda E_{\text{trans}} = 0$.


### Total Daily ET and Potential ET

Total daily evapotranspiration (ET) is the sum of the three components, canopy evaporation ($\lambda E_{\text{canopy}}$), bare soil evaporation ($\lambda E_{\text{soil}}$), and transpiration ($\lambda E_{\text{trans}}$):

$$
\lambda E = \lambda E_{\text{canopy}} + \lambda E_{\text{soil}} + \lambda E_{\text{trans}}
$$

Daily potential ET (PET) is also calculated in MOD16; it is the sum of wet canopy evaporation, evaporation from saturated soil ($\lambda E_{\text{soil,sat}}$), evaporation from unsaturated soil *without* the soil moisture correction ($F_{\text{wet}}) \lambda E_{\text{soil}}^*$), and potential transpiration, ($\lambda E_{\text{trans,potential}}$):

$$
\lambda E_{\text{potential}} = \lambda E_{\text{canopy}} + \lambda E_{\text{soil,sat}} + (1 - F_{\text{wet}}) \lambda E_{\text{soil}}^* + \lambda E_{\text{trans,potential}}
$$

Where potential transpiration is given by the Priestly-Taylor equation:

$$
\lambda E_{\text{trans,potential}} = \frac{\alpha s A_C (1 - F_{\text{wet}})}{s + \gamma}
$$

Where $\alpha = 1.26$.

Note that all of the ET values are given in energy units, [W m-2]. If you wish to obtain a water vapor mass flux (units: kg per square meter per second), then divide by the latent heat of vaporization ($\lambda$, units: J per kilogram). Because 1 kg of pure water covers a square meter to 1 mm thickness, the mass flux is also equivalent to millimeters per second.


### Free Parameters

| Parameter                  | Description                                                 |
|:---------------------------|:------------------------------------------------------------|
| $T_{\text{min,close}}$     | Temperature at which stomata almost completely closed (C)   |
| $T_{\text{min,open}}$      | Temperature at which stomata almost fully opened (C)        |
| $\text{VPD}^{\text{close}}$| VPD at which stomata are almost completely closed (Pa)      |
| $\text{VPD}^{\text{open}}$ | VPD at which stomata are almost completely opened (Pa)      |
| $g_{SH}$                   | Leaf conductance to sensible heat per unit LAI (m s-1 LAI-1)|
| $g_{WV}$                   | Leaf cond. to evaporated water per unit LAI (m s-1 LAI-1)   |
| $g_{\text{cuticular}}$     | Leaf cuticular conductance (m s-1)                          |
| $C_L$                      | Mean potential stomatal cond. per unit leaf area (m s-1)    |
| $r_{\text{BL,min}}$        | Minimum leaf boundary layer resistance (s m-1)              |
| $r_{\text{BL,max}}$        | Maximum leaf boundary layer resistance (s m-1)              |
| $\beta$                    | Soil moisture constraint on potential soil evaporation      |



Acknowledgments
---------------

This software was developed under a grant from NASA (80NSSC22K0198).


References
--------------

- Fisher, J. B., Tu, K., and Baldocchi, D. D. 2008. Global estimates of the land atmosphere water flux based on monthly AVHRR and ISLSCP-II data, validated at FLUXNET sites. *Remote Sensing of Environment.* 112(3):901−919.
- Gates, D. M. 1980. Biophysical Ecology. Springer Advanced Texts in Life Sciences. Springer New York, NY.
- Iribane, J.V., and W.L. Godson, 1981. Atmospheric Thermodynamics. 2nd Edition. D. Reidel Publishing Company, Dordrecht, The Netherlands.
- Jacobsen, A. and B. U. Hansen. 1999. Estimation of the soil heat flux/net radiation ratio based on spectral vegetation indexes in high-latitude Arctic areas. *International Journal of Remote Sensing.* 20(2):445-461.
- Maidment, D. 1969. Handbook of Hydrology.
- Monteith, J. L., and M. Unsworth. 2001. Principles of Environmental Physics. Second Ed.
- Mu, Q., Heinsch, F. A., Zhao, M., & Running, S. W. 2007. Development of a global evapotranspiration algorithm based on MODIS and global meteorology data. *Remote Sensing of Environment,* 111(4), 519–536.
- Mu, Q., M. Zhao, and S. W. Running. 2011. Improvements to a MODIS global terrestrial evapotranspiration algorithm. *Remote Sensing of Environment* 115 (8):1781–1800.
- National Physical Laboratory. 2021. "Buoyancy Correction and Air Density Measurement." http://resource.npl.co.uk/docs/science_technology/mass_force_pressure/clubs_groups/instmc_weighing_panel/buoycornote.pdf Accessed: September 3, 2023.
- Zhang, K., J. S. Kimball, and S. W. Running. 2016. A review of remote sensing based actual evapotranspiration estimation. *Wiley Interdisciplinary Reviews: Water* 3 (6):834–853.
