/*
This file IS NOT INCLUDED in the MIT license of the MOD16 library.
The contents of this file are NOT LICENSED FOR ANY USE. The source
code herein is only used to verify the Python implementation against
the operational C implementation.

To compile:

  gcc -Wall canopy.c -lm -o canopy

 */

#include <math.h>
#include <stdio.h>

#define Cp 1013.0   /* (J/(kg K)) specific heat at constant pressure */
#define SBC 5.67e-8 /* (W/(m2 K4)) Stefan-Boltzmann constant */
#define epsl                                                                   \
  0.622 /* ratio of the molecular weight of water vapor to that for dry air */
#define alfa                                                                   \
  1.26 /* the constant for calculating potential evapotranspiration */

typedef struct FINAL_OUTPUT {
  /* daily and annual outputs */
  int col;
  int row;
  double Tmin_reduction;     /* 0.0 to 1.0, derived scalor */
  double VPD_reduction;      /* 0.0 to 1.0, derived scalor */
  double ET_daily;           /* KgC/m^2/d */
  double LE_daily;           /* KgC/m^2/d */
  double PET_daily;          /* KgC/m^2/d */
  double PLE_daily;          /* KgC/m^2/d */
  double CanopyEvap_daily;   /* KgC/m^2/d */
  double CanopyLEvap_daily;  /* KgC/m^2/d */
  double TransET_daily;      /* KgC/m^2/d */
  double TransLE_daily;      /* KgC/m^2/d */
  double annual_ET;          /* KgC/m^2/yr */
  double annual_LE;          /* KgC/m^2/yr */
  double annual_PET;         /* KgC/m^2/yr */
  double annual_PLE;         /* KgC/m^2/yr */
  double annual_CanopyEvap;  /* KgC/m^2/yr */
  double annual_CanopyLEvap; /* KgC/m^2/yr */
  double annual_TransET;     /* KgC/m^2/yr */
  double annual_TransLE;     /* KgC/m^2/yr */
} FINAL_OUTPUT;

typedef struct VIIRS_ARRAY {
  /* daily remotely sensed FPAR and LAI inputs */
  double FPAR; /* fraction, from 0.0 to 1.0 */
  double LAI;  /* unitless, single sided, from 0.0 to 10.0 */
  /* daily remotely sensed ALBEDO input */
  double ALBEDO;
} VIIRS_ARRAY;

typedef struct MET_ARRAY {
  /* number of days in the running year */
  int REAL_NUM_DAYS; /* 365 or 366 dependent on leap or normal year */

  int NUM_COM_DAYS; /* added Sudipta 8 or 5(6 for leap year) */

  /* Index derived from VIIRS UMD land cover classification system*/
  int ZERO_BASED_UMD_VEG_LC; /* must be force it within 0 to 10 corresponding to
                                1 to 10 and 12 in the UMD LC VNP12Q1 system */
  /* global 1km elevation data input */
  double ELEV;
  double pressure; /* atmospheric pressure at surface */

  /* daily meteorological inputs */
  double Tavg;              /* Celsius */
  double Tavg_ann;          /* Celsius. The annual average temperature */
  double Tmin;              /* Celsius */
  double VPD_day;           /* Pascal */
  double SWrad;             /* MJ/m^2/d */
  double Tday;              /* Celsius */
  double Tnight;            /* Celsius */
  double RH_day;            /* no unit */
  double SVP_day;           /* Pascal */
  double VPD_night;         /* Pascal */
  double RH_night;          /* no unit */
  double SVP_night;         /* Pascal */
  double AIR_DENSITY_day;   /* kg/m3 */
  double AIR_DENSITY_night; /* kg/m3 */
  float day_length;         /* s */
  float night_length;       /* s */

  double AVPday;
  double AVPnight;
  double SWGDN;
  double LWrad_day;
  double LWrad_night;
} MET_ARRAY;

typedef struct BPLUT {
  /* Key parameters for VIIRS ET algorithm */
  double Tmin_min; /* Celsius */
  double Tmin_max; /* Celsius */
  double VPD_min;  /* Pascal */
  double VPD_max;  /* Pascal */
  double gl_sh; /* Leaf conductance to sensible heat, per unit all-sided LAI */
  double gl_e_wv;     /* Leaf conductance to evaporated water vapor, per unit
                         projected LAI */
  double g_cuticular; /* cuticular conductance (projected area basis) */
  double Cl;   /* put in the LUT, different values for different vegetation types. version 1.0, 4/6/2009. */    /* the mean surface conductance per unit leaf aream */
  double RBL_MIN; /* boundary layer resistance */
  double RBL_MAX; /* boundary layer resistance */
} VNP16_BPLUT;

int canopy_evaporation_day(double *ET, double *LE, double *PET, double *PLE,
                           double lamda, VIIRS_ARRAY *modis_array,
                           MET_ARRAY *met_array, double A, double Fc,
                           double Fwet, VNP16_BPLUT *bplut);
int canopy_evaporation_night(double *ET, double *LE, double *PET, double *PLE,
                             double lamda, VIIRS_ARRAY *modis_array,
                             MET_ARRAY *met_array, double A, double Fc,
                             double Fwet, VNP16_BPLUT *bplut);

struct FINAL_OUTPUT final_output;
struct VIIRS_ARRAY modis_array;
struct MET_ARRAY met_array;
struct BPLUT VNP16BPLUT = {
    /* PFT 7 (Open Shrubland)*/
    -8, 8.8, 650, 4400, 0.02, 0.02, 1E-05, 0.0055, 60, 95};

int main() {
  int i;
  double A = 419 * (1 - 0.116) + -117;
  double lamda[3] = {2470184.48771, 2455251.68213, 2441553.3254};
  double a_Tday[3] = {286.20189, 292.52667, 298.3286};
  double a_pressure[3] = {92753.47, 92753.47, 92753.47};
  double a_VPD_day[3] = {710.9, 1249.4, 1979.};
  double a_AIR_DENSITY_day[3] = {1.12699, 1.10054, 1.07786};
  double a_SVP_day[3] = {1502.86, 2249.56, 3201.63};
  double a_RH_day[3] = {0.52697, 0.4446, 0.38188};
  double SWrad = 419;  /* Short-wave radiation */
  double Fc = 0.35839; /* fPAR */
  double Fwet_day[3] = {0, 0.4, 0.8};

  /* Prescribed for VIIRS_ARRAY */
  double LAI[3] = {0.3, 0.6, 1.0};

  /* Prescribed for FINAL_OUTPUT */
  double Tmin_reduction[3] = {0.8196, 1.0, 1.0};
  double VPD_reduction[3] = {0.98376, 0.84016, 0.6456};

  /* Outputs */
  double PLE[1] = {0};
  double LE[1] = {0};
  double ET[1] = {0};
  double PET[1] = {0};

  /* Met array defaults */
  met_array.day_length = 1;
  met_array.night_length = 1;

  printf("Parameter Sweep\n");
  for (i = 0; i < 3; i++) {
    /*printf("----- Parameter sweep %d of 3\n", i + 1);*/
    met_array.AIR_DENSITY_day = a_AIR_DENSITY_day[i];
    met_array.AIR_DENSITY_night = a_AIR_DENSITY_day[i]; /* i.e., same as day */
    met_array.pressure = a_pressure[i];
    met_array.VPD_day = a_VPD_day[i];
    met_array.VPD_night = a_VPD_day[i]; /* i.e., same as day */
    met_array.SWrad = SWrad;
    met_array.Tday = a_Tday[i] - 273.15;
    met_array.Tnight = a_Tday[i] - 273.15; /* i.e., same as day */
    met_array.RH_day = a_RH_day[i] * 100;
    met_array.RH_night = a_RH_day[i] * 100;
    met_array.SVP_day = a_SVP_day[i];   /* i.e., same as day */
    met_array.SVP_night = a_SVP_day[i]; /* i.e., same as day */
    modis_array.LAI = LAI[i];
    final_output.Tmin_reduction = Tmin_reduction[i];
    final_output.VPD_reduction = VPD_reduction[i];

    canopy_evaporation_day(ET, LE, PET, PLE, lamda[i], &modis_array,
      &met_array, A, Fc, Fwet_day[i], &VNP16BPLUT);
    printf("--- Daytime canopy evap.: %f\n", (float)*ET);
    canopy_evaporation_night(ET, LE, PET, PLE, lamda[i], &modis_array,
      &met_array, A, Fc, Fwet_day[i], &VNP16BPLUT);
    printf("--- Nighttime canopy evap.: %f\n", (float)*ET);
  }


  return 0;
};

int canopy_evaporation_day(double *ET, double *LE, double *PET, double *PLE,
                           double lamda, VIIRS_ARRAY *modis_array,
                           MET_ARRAY *met_array, double A, double Fc,
                           double Fwet, VNP16_BPLUT *bplut) {
  double Taa;
  double VPD;
  /* KAE NOTE that rcorr is not needed in this function
  double rcorr; correction factor for temp and pressure */
  double rh, rho, rhr, rr, rv;
  double ta, tk;
  double s;

  /* assign ta (Celsius) and tk (Kelvins) */
  ta = (double)(met_array->Tday);
  tk = ta + 273.15;

  /* KAE NOTE that rcorr is not needed in this function
  correct conductances for temperature and pressure based on Jones (1992)
  with standard conditions assumed to be 20 deg C, 101300 Pa
  rcorr = 1.0 / (pow((met_array->Tday + 273.15) / 293.15, 1.75) * 101300 /
                 met_array->pressure);
  */

  /*	if(Fwet ==0.0 || modis_array->LAI[k] ==0.) */
  if (fabs(Fwet * modis_array->LAI) <= 1.0e-7)
    *PLE = 0.0;
  else {
    /* Canopy conductane to sensible heat */
    rh = 1.0 / (bplut->gl_sh * modis_array->LAI * Fwet);
    /* Canopy conductance to evaporated water vapor */
    rv = 1.0 / (bplut->gl_e_wv * modis_array->LAI * Fwet);

    /* calculate density of air (rho) as a function of air temperature */
    rho = met_array->AIR_DENSITY_day;
    /* calculate resistance to radiative heat transfer through air, rr */
    rr = rho * Cp / (4.0 * SBC * (tk * tk * tk));
    /* calculate combined resistance to convective and radiative heat transfer,
    parallel resistances : rhr = (rh * rr) / (rh + rr) */
    rhr = (rh * rr) / (rh + rr);

    /* calculate saturated vapor pressure, the slope of the curve relating
     * saturation water vapor pressure to temperature */
    Taa = 239.0 + (double)met_array->Tday;
    s = 17.38 * 239.0 * (double)met_array->SVP_day / (Taa * Taa); /* (Pa/K) */

    /* for humidity deficit  */
    /* method 1 for VPD: use SVP and VP */
    VPD = (double)met_array->VPD_day; /* VP is provided in auclimate data. use
                                         the definition for VPD. */
    if (VPD < 0.0)
      VPD = 0.0;

    /* calculate the potential latent heat flux */
    /* I fixed a bug befor. The second term in the denominator should be
     * mutiplied with the area of the plant covered by water. version 1.0,
     * 3/10/2009 */
    /* this is calculated as potential evaporation on the open water.
     * version 1.0, 3/12/2009 */
    *PLE = (s * A * Fc + (rho * Cp * VPD * met_array->day_length * Fc / rhr)) *
           Fwet /
           (((met_array->pressure * Cp * rv) / (lamda * epsl * rhr)) + s);
    /* January 12, 2013, Qiaozhen Mu */
    if (met_array->RH_day < 70. && *PLE < 0.0)
      *PLE = 0.0;
  }

  /* calculate the latent heat flux */
  *LE = *PLE;

  /* calculate the evapotranspiration */
  *ET = *LE / lamda;
  /*		printf("LE_day=%f, lamda_day=%f, ET_day=%f ", *LE, lamda, *ET);
   */
  /* calculate the potential evapotranspiration */
  *PET = *PLE / lamda;

  return (0);
}

int canopy_evaporation_night(double *ET, double *LE, double *PET, double *PLE,
                             double lamda, VIIRS_ARRAY *modis_array,
                             MET_ARRAY *met_array, double A, double Fc,
                             double Fwet, VNP16_BPLUT *bplut) {
  double Taa;
  double VPD;
  /* KAE NOTE that rcorr is not needed in this function
  double rcorr; correction factor for temp and pressure */
  double rh, rho, rhr, rr, rv;
  double ta, tk;
  double s;

  /* assign ta (Celsius) and tk (Kelvins) */
  ta = (double)(met_array->Tnight);
  tk = ta + 273.15;

  /* KAE NOTE that rcorr is not needed in this function
  correct conductances for temperature and pressure based on Jones (1992)
  with standard conditions assumed to be 20 deg C, 101300 Pa
  rcorr = 1.0 / (pow((met_array->Tday + 273.15) / 293.15, 1.75) * 101300 /
                 met_array->pressure);
  */

  /*	if(Fwet == 0.0 || modis_array->LAI[k] == 0.) */
  if (fabs(Fwet * modis_array->LAI) <= 1.0e-7)
    *PLE = 0.0;
  else {
    /* Canopy conductane to sensible heat */
    rh = 1.0 / (bplut->gl_sh * modis_array->LAI * Fwet);
    /* Canopy conductance to evaporated water vapor */
    rv = 1.0 / (bplut->gl_e_wv * modis_array->LAI * Fwet);

    /* calculate density of air (rho) as a function of air temperature */
    rho = met_array->AIR_DENSITY_night;
    /* calculate resistance to radiative heat transfer through air, rr */
    rr = rho * Cp / (4.0 * SBC * (tk * tk * tk));
    /* calculate combined resistance to convective and radiative heat transfer,
    parallel resistances : rhr = (rh * rr) / (rh + rr) */
    rhr = (rh * rr) / (rh + rr);

    /* calculate saturated vapor pressure, the slope of the curve relating
     * saturation water vapor pressure to temperature */
    /* emails with Helen */
    Taa = 239.0 + (double)met_array->Tnight;
    s = 17.38 * 239.0 * (double)met_array->SVP_night / (Taa * Taa); /* (Pa/K) */

    /* for humidity deficit  */
    /* method 1 for VPD: use SVP and VP */
    VPD = (double)met_array->VPD_night; /* VP is provided in auclimate data. use
                                           the definition for VPD. */
    if (VPD < 0.0)
      VPD = 0.0;

    /* calculate the potential latent heat flux */
    /* I fixed a bug befor. The second term in the denominator should be
     * mutiplied with the area of the plant covered by water. version 1.0,
     * 3/10/2009 */
    /* this is calculated as potential evaporation on the open water.
     * version 1.0, 3/12/2009 */
    *PLE =
        (s * A * Fc + (rho * Cp * VPD * met_array->night_length * Fc / rhr)) *
        Fwet / (((met_array->pressure * Cp * rv) / (lamda * epsl * rhr)) + s);
    /* January 12, 2013, Qiaozhen Mu */
    if (met_array->RH_night < 70. && *PLE < 0.0)
      *PLE = 0.0;
  }
  /* calculate the latent heat flux */
  *LE = *PLE;

  /* calculate the evapotranspiration */
  *ET = *LE / lamda;
  /*		printf("LE_night=%f, lamda_night=%f, ET_night=%f ", *LE, lamda,
   * *ET); */
  /* calculate the potential evapotranspiration */
  *PET = *PLE / lamda;

  return (0);
}
