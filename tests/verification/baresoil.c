/*
This file IS NOT INCLUDED in the MIT license of the MOD16 library.
The contents of this file are NOT LICENSED FOR ANY USE. The source
code herein is only used to verify the Python implementation against
the operational C implementation.

To compile:

  gcc -Wall baresoil.c -lm

 */

#include <math.h>
#include <stdio.h>

#define Cp 1013.0   /* (J/(kg K)) specific heat at constant pressure */
#define SBC 5.67e-8 /* (W/(m2 K4)) Stefan-Boltzmann constant */
#define epsl                                                                   \
  0.622 /* ratio of the molecular weight of water vapor to that for dry air */

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

struct MET_ARRAY met_array;

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

struct BPLUT VNP16BPLUT = {
  /* PFT 7 (Open Shrubland)*/
  -8, 8.8, 650, 4400, 0.02, 0.02, 1E-05, 0.0055, 60, 95
};

int baresoil_et_day(double *ET, double *LE, double *PET, double *PLE,
                    double lamda, MET_ARRAY *met_array, VNP16_BPLUT *bplut,
                    double A, double Fc, double G, double Fwet_day);
int baresoil_et_night(double *ET, double *LE, double *PET, double *PLE,
                      double lamda, MET_ARRAY *met_array, VNP16_BPLUT *bplut,
                      double A, double Fc, double G, double Fwet_night);

int main() {
  int i;
  double A = 255.27261;
  double lamda[3] = {2470184.48771, 2455251.68213, 2441553.3254};
  double a_Tday[3] = {286.20189, 292.52667, 298.3286};
  double a_pressure[3] = {92753.47, 92753.47, 92753.47};
  double a_VPD_day[3] = {710.9, 1249.4, 1979.};
  double a_AIR_DENSITY_day[3] = {1.12791642, 1.10072136, 1.07518633};
  double a_SVP_day[3] = {1502.86379395, 2249.56542969, 3201.63623047};
  double a_RH_day[3] = {0.52696977, 0.44460384, 0.38187856};
  double SWrad = 419; /* Short-wave radiation */
  double G = 0.0;           /* Soil heat flux */
  double Fc = 0.35839;      /* fPAR */
  double Fwet_day[3] = {0, 0.4, 0.8};

  /* Outputs */
  double PLE[1] = {0};
  double LE[1] = {0};
  double ET[1] = {0};
  double PET[1] = {0};

  /* Met array defaults */
  met_array.day_length = 1;

  for (i = 0; i < 3; i++) {
    printf("----- Iteration %d of 3\n", i + 1);
    met_array.AIR_DENSITY_day = a_AIR_DENSITY_day[i];
    met_array.pressure = a_pressure[i];
	  met_array.VPD_day = a_VPD_day[i];
	  met_array.SWrad = SWrad;
	  met_array.Tday = a_Tday[i];
	  met_array.RH_day = a_RH_day[i];
	  met_array.SVP_day = a_SVP_day[i];

	  baresoil_et_day(
	      ET, LE, PET, PLE, lamda[i], &met_array, &VNP16BPLUT, A, Fc, G, Fwet_day[i]);

    printf("Soil LE: %f\n", (float)*LE);
    printf("Soil PLE: %f\n", (float)*PLE);
    /*
    printf("PLE_soil: %f\n", (float)PLE_soil);
    printf("PLE_water: %f\n", (float)PLE_water);
    */
    printf("Soil ET: %f\n", (float)*ET);


	}

  return 0;
};

int baresoil_et_day(double *ET, double *LE, double *PET, double *PLE,
                    double lamda, MET_ARRAY *met_array, VNP16_BPLUT *bplut,
                    double A, double Fc, double G, double Fwet_day) {
  double Taa;
  double VPD;
  double rcorr; /* correction factor for temp and pressure */
  double rbl, rho, rhr, rr;
  double ta, tk;
  double s;
  double Fsm;

  double PLE_soil, PLE_water; /* changed the function to calculate saturate
                                 water on soil, version 1.0, 3/12/2009 */

  /* assign ta (Celsius) and tk (Kelvins) */
  ta = (double)(met_array->Tday);
  tk = ta + 273.15;

  /* correct conductances for temperature and pressure based on Jones (1992)
  with standard conditions assumed to be 20 deg C, 101300 Pa */
  rcorr = 1.0 / (pow((met_array->Tday + 273.15) / 293.15, 1.75) * 101300 /
                 met_array->pressure);

  /* January 12, 2013, Qiaozhen Mu */
  if (met_array->VPD_day < bplut->VPD_min)
    rbl = bplut->RBL_MIN * rcorr;
  else if (met_array->VPD_day >= bplut->VPD_min &&
           met_array->VPD_day <= bplut->VPD_max)
    rbl = (bplut->RBL_MAX - (bplut->RBL_MAX - bplut->RBL_MIN) *
                                (bplut->VPD_max - met_array->VPD_day) /
                                (bplut->VPD_max - bplut->VPD_min)) *
          rcorr;
  else
    rbl = bplut->RBL_MAX * rcorr;

  /* calculate density of air (rho) as a function of air temperature */
  rho = met_array->AIR_DENSITY_day;
  /* calculate resistance to radiative heat transfer through air, rr */
  rr = rho * Cp / (4.0 * SBC * (tk * tk * tk));
  /* calculate combined resistance to convective and radiative heat transfer,
  parallel resistances : rhr = (rh * rr) / (rh + rr) */
  rhr = (rbl * rr) / (rbl + rr);

  /* calculate saturated vapor pressure, the slope of the curve relating
   * saturation water vapor pressure to temperature */
  Taa = 239.0 + (double)met_array->Tday;
  s = 17.38 * 239.0 * (double)met_array->SVP_day / (Taa * Taa); /* (Pa/K) */

  /* for humidity deficit  */
  /* method 1 for VPD: use SVP and VP */
  VPD = (double)met_array->VPD_day; /* VP is provided in auclimate data. use the
                                       definition for VPD. */
  if (VPD < 0.0)
    VPD = 0.0;

  /* changed the function to calculate saturate water on soil, version 1.0,
   * 3/12/2009 */
  PLE_soil = (s * (A - G) * (1.0 - Fc) +
              (rho * Cp * VPD * met_array->day_length * (1.0 - Fc) / rhr)) *
             (1.0 - Fwet_day) /
             (((met_array->pressure * Cp * rbl) / (lamda * epsl * rhr)) + s);

  /* January 12, 2013, Qiaozhen Mu */
  if (met_array->RH_day < 70. && PLE_soil < 0.0)
    PLE_soil = 0.0;
  PLE_water = (s * (A - G) * (1.0 - Fc) +
               (rho * Cp * VPD * met_array->day_length * (1.0 - Fc) / rhr)) *
              Fwet_day /
              (((met_array->pressure * Cp * rbl) / (lamda * epsl * rhr)) + s);
  /* January 12, 2013, Qiaozhen Mu */
  if (met_array->RH_day < 70. && PLE_water < 0.0)
    PLE_water = 0.0;
  *PLE = PLE_soil + PLE_water;

  /* calculate the latent heat flux */
  if (met_array->RH_day == 0.)
    *LE = PLE_water;
  else {
    Fsm = pow((double)(met_array->RH_day / 100.0), (double)VPD / 250.0);
    if (Fsm > 1.0)
      Fsm = 1.0;
    *LE = PLE_water + Fsm * PLE_soil;
  }

  /* calculate the evapotranspiration */
  *ET = *LE / lamda;
  /*	printf("LE_day=%f, lamda_day=%f, ET_day=%f ", *LE, lamda, *ET); */
  /* calculate the potential evapotranspiration */
  *PET = *PLE / lamda;
  /*	printf("Fsm=%f, PLEsoil=%f, LEsoil=%f\n",Fsm, *PLE, *LE);  */

  return (0);
}

int baresoil_et_night(double *ET, double *LE, double *PET, double *PLE,
                      double lamda, MET_ARRAY *met_array, VNP16_BPLUT *bplut,
                      double A, double Fc, double G, double Fwet_night) {
  double Taa;
  double VPD;
  double rcorr; /* correction factor for temp and pressure */
  double rbl, rho, rhr, rr;
  double ta, tk;
  double s;
  double Fsm;

  double PLE_soil, PLE_water; /* changed the function to calculate saturate
                                 water on soil, version 1.0, 3/12/2009 */

  /* assign ta (Celsius) and tk (Kelvins) */
  ta = (double)(met_array->Tnight);
  tk = ta + 273.15;

  /* correct conductances for temperature and pressure based on Jones (1992)
  with standard conditions assumed to be 20 deg C, 101300 Pa */
  rcorr = 1.0 / (pow((met_array->Tnight + 273.15) / 293.15, 1.75) * 101300 /
                 met_array->pressure);

  /* January 12, 2013, Qiaozhen Mu */
  if (met_array->VPD_night < bplut->VPD_min)
    rbl = bplut->RBL_MIN * rcorr;
  else if (met_array->VPD_night >= bplut->VPD_min &&
           met_array->VPD_night <= bplut->VPD_max)
    rbl = (bplut->RBL_MAX - (bplut->RBL_MAX - bplut->RBL_MIN) *
                                (bplut->VPD_max - met_array->VPD_night) /
                                (bplut->VPD_max - bplut->VPD_min)) *
          rcorr;
  else
    rbl = bplut->RBL_MAX * rcorr;

  /* calculate density of air (rho) as a function of air temperature */
  rho = met_array->AIR_DENSITY_night;
  /* calculate resistance to radiative heat transfer through air, rr */
  rr = rho * Cp / (4.0 * SBC * (tk * tk * tk));
  /* calculate combined resistance to convective and radiative heat transfer,
  parallel resistances : rhr = (rh * rr) / (rh + rr) */
  rhr = (rbl * rr) / (rbl + rr);

  /* calculate saturated vapor pressure, the slope of the curve relating
   * saturation water vapor pressure to temperature */
  Taa = 239.0 + (double)met_array->Tnight;
  s = 17.38 * 239.0 * (double)met_array->SVP_night / (Taa * Taa); /* (Pa/K) */

  /* for humidity deficit  */
  /* method 1 for VPD: use SVP and VP */
  VPD = (double)met_array->VPD_night; /* VP is provided in auclimate data. use
                                         the definition for VPD. */
  if (VPD < 0.0)
    VPD = 0.0;

  /* changed the function to calculate saturate water on soil, version 1.0,
   * 3/12/2009 */
  PLE_soil = (s * (A - G) * (1.0 - Fc) +
              (rho * Cp * VPD * met_array->night_length * (1.0 - Fc) / rhr)) *
             (1.0 - Fwet_night) /
             (((met_array->pressure * Cp * rbl) / (lamda * epsl * rhr)) + s);
  /* January 12, 2013, Qiaozhen Mu */
  if (met_array->RH_night < 70. && PLE_soil < 0.0)
    PLE_soil = 0.0;
  PLE_water = (s * (A - G) * (1.0 - Fc) +
               (rho * Cp * VPD * met_array->night_length * (1.0 - Fc) / rhr)) *
              Fwet_night /
              (((met_array->pressure * Cp * rbl) / (lamda * epsl * rhr)) + s);
  /* January 12, 2013, Qiaozhen Mu */
  if (met_array->RH_night < 70. && PLE_water < 0.0)
    PLE_water = 0.0;
  *PLE = PLE_soil + PLE_water;

  /* calculate the latent heat flux */
  if (met_array->RH_night == 0.)
    *LE = PLE_water;
  else {
    Fsm = pow((double)(met_array->RH_night / 100.0), (double)VPD / 250.0);
    if (Fsm > 1.0)
      Fsm = 1.0;
    *LE = PLE_water + Fsm * PLE_soil;
  }

  /* calculate the evapotranspiration */
  *ET = *LE / lamda;
  /*		printf("LE_night=%f, lamda_night=%f, ET_night=%f ", *LE, lamda,
   * *ET); */
  /* calculate the potential evapotranspiration */
  *PET = *PLE / lamda;
  /*		printf("Fsm=%f, PLEsoil=%f, LEsoil=%f\n",Fsm, *PLE, *LE);  */

  return (0);
}
