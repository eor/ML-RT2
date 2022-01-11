import math as m
import numpy as np
import numba as nb
from numba import jit, jit_module

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
c = 299792458.0
c_cgi = 29979245800.0
h_eV = 4.135667662e-15
k_BeV = 8.617333262e-05
sigmaSB = 5.670374419e-08   # here: 5.670367e-08 W m^-2 K^-4

# -----------------------------------------------------------------
# Cosmological parameters
# -----------------------------------------------------------------
cosmoOmegaM = 0.315
cosmoOmegaB = 0.0491

# -----------------------------------------------------------------
# Properties of Pop III ZAMS stars.
# -----------------------------------------------------------------
pop3M = np.zeros(13)
pop3L = np.zeros(13)
pop3T = np.zeros(13)

# Taken from Schaerer, (2002) A&A 382, 28–42, table 3.
# Units are: Mass/M_sol, log_10(L/L_sol), log_10(T/K)

pop3M[0],  pop3L[0],  pop3T[0] = 1000., 7.444,   5.026
pop3M[1],  pop3L[1],  pop3T[1] = 500.,  7.106,   5.029
pop3M[2],  pop3L[2],  pop3T[2] = 400.,  6.984,   5.028
pop3M[3],  pop3L[3],  pop3T[3] = 300.,  6.819,   5.007
pop3M[4],  pop3L[4],  pop3T[4] = 200.,  6.574,   4.999
pop3M[5],  pop3L[5],  pop3T[5] = 120.,  6.243,   4.981
pop3M[6],  pop3L[6],  pop3T[6] = 80.,   5.947,   4.970
pop3M[7],  pop3L[7],  pop3T[7] = 60.,   5.715,   4.943
pop3M[8],  pop3L[8],  pop3T[8] = 40.,   5.420,   4.900
pop3M[9],  pop3L[9],  pop3T[9] = 25.,   4.890,   4.850
pop3M[10], pop3L[10], pop3T[10] = 15.,  4.324,   4.759
pop3M[11], pop3L[11], pop3T[11] = 9.,   3.709,   4.622
pop3M[12], pop3L[12], pop3T[12] = 5.,   2.870,   4.440


# -----------------------------------------------------------------
# Compute stellar mass
# -----------------------------------------------------------------
def compute_stellar_mass(halo_mass, redshift, verbose=False):
    z = redshift
    a = 1./(1. + z)

    # equation parameters
    M_10 = 11.514
    M_1a = -1.793
    M_1z = -0.251

    epsilon_0 = -1.777
    epsilon_a = -0.006
    epsilon_a2 = -0.119
    epsilon_z = -0.000

    alpha_0 = -1.412
    alpha_a = 0.731

    delta_0 = 3.508
    delta_a = 2.608
    delta_z = -0.043

    gamma_0 = 0.316
    gamma_a = 1.319
    gamma_z = 0.279

    # equations (4) from [1]
    nu = m.exp(-4.*a*a)
    M_1 = 10**(M_10 + (M_1a*(a-1.) + M_1z*z)*nu)
    epsilon = 10**(epsilon_0 + (epsilon_a*(a-1.) + epsilon_z*z)*nu + epsilon_a2*(a-1.))
    alpha = alpha_0 + (alpha_a*(a-1.))*nu
    delta = delta_0 + (delta_a*(a-1.) + delta_z*z)*nu
    gamma = gamma_0 + (gamma_a*(a-1.) + gamma_z*z)*nu

    # equations (3) from [1]
    x = m.log10(halo_mass / M_1)
    if x > -2.849:
        div = 1. + m.exp(10**(-x))
    else:
        div = 1e307

    frac = ((m.log10(1.+m.exp(x)))**gamma) / div
    f_x = (-1)*m.log10(10**(alpha*x) + 1.0) + delta * frac
    x = 0
    frac = ((m.log10(1.+m.exp(x)))**gamma) / (1. + m.exp(10**(-x)))
    f_0 = (-1)*m.log10(10**(alpha*x) + 1.0) + delta * frac

    tmp_m = m.log10(epsilon*M_1) + f_x - f_0
    stellar_mass = 10**tmp_m

    return stellar_mass


def SED_black_body(E, T):
    expo = E / (k_BeV * T)
    try:
        return (2.0 * m.pi / (c_cgi * c_cgi * h_eV * h_eV)) * (E * E * E) / (m.exp(expo) - 1.)
    except:
        return 1e-300


# -----------------------------------------------------------------
# Simple SEDs in functional form
# -----------------------------------------------------------------
def SED_power_law(E, alpha):
    return m.pow(E, -alpha)


# -----------------------------------------------------------------
# Simplified version of simpsons integration
# -----------------------------------------------------------------
def simpson(y, x):
    # source: https://masonstoecker.com/2021/04/03/Simpson-and-Numba.html
    n = len(y)-1
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i+1]-x[i]
        if h[i] == 0:
            np.delete(h, i)
            np.delete(y, i)
    n = len(h)-1
    s = 0
    for i in range(1, n, 2):
        a = h[i]*h[i]
        b = h[i]*h[i-1]
        c = h[i-1]*h[i-1]
        d = h[i] + h[i-1]
        alpha = (2*a+b-c)/h[i]
        beta = d*d*d/b
        gamma = (-a+b+2*c)/h[i-1]
        s += alpha*y[i+1]+beta*y[i]+gamma*y[i-1]

    if (n+1) % 2 == 0:
        alpha = h[n-1]*(3-h[n-1]/(h[n-1]+h[n-2]))
        beta = h[n-1]*(3+h[n-1]/h[n-2])
        gamma = -h[n-1]*h[n-1]*h[n-1]/(h[n-2]*(h[n-1]+h[n-2]))
        return (s+alpha*y[n]+beta*y[n-1]+gamma*y[n-2])/6
    else:
        return s/6


def generate_SED_single_pop3(starMass=100, eHigh=1.e4, eLow=10.4, N=1000, logGrid=False, fEsc=0.1):
    starM = starMass
    starT = 10**(np.interp(starM, pop3M[::-1], pop3T[::-1]))
    starL = 10**(np.interp(starM, pop3M[::-1], pop3L[::-1]))

    # energies should be in increasing order
    if eHigh < eLow:
        eLow, eHigh = eHigh, eLow

    # 1. SET UP GRIDS
    energies = np.array([eHigh])
    intensities = np.array([SED_black_body(energies[0], starT)])
    # fill energies array
    if logGrid:
        # create energies along a logarithmic grid, so that eHigh *C**(N-1) = eLow
        C = m.pow(eLow/eHigh, 1./(N-1))

        for i in range(1, N):
            eTmp = energies[i-1]*C
            energies = np.append(energies, eTmp)
    else:
        # create a grid of energies, so that eHigh + C*(N-1) = eLow
        C = (eLow-eHigh)/(N-1.)
        for i in range(1, N):
            eTmp = energies[i-1]+C
            energies = np.append(energies, eTmp)

    # let's compute some intensities
    for i in range(1, N):
        tmpE = energies[i]
        tmpI = SED_black_body(tmpE, starT)
        intensities = np.append(intensities, tmpI)

    # 2. NORMALIZATION
    # integrate to obtain total energy (normalize)
    energiesLog = np.array([m.log(energies[0])])
    sed4Norm = np.array([intensities[0]*energies[0]])

    for i in range(1, N):
        eTmp = energies[i]
        energiesLog = np.append(energiesLog, m.log(eTmp))
        sed4Norm = np.append(sed4Norm, intensities[i]*eTmp)    # log integration trick

    # energies should be increasing in value, or else the result can be negative
    integral = simpson(sed4Norm[::-1], energiesLog[::-1])
    G = (starL*3.828e26/1.6022e-19)/integral

    intensities *= G
    intensities *= fEsc

    # sort energies if they are descending
    if energies[0] > energies[-1]:
        energies = energies[::-1]
        intensities = intensities[::-1]

    return energies, intensities


def generate_SED_stars_IMF(halo_mass, redshift, eLow=10.4, eHigh=1.e4, N=1000,  logGrid=False,
                           starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35,
                           targetSourceAge=10., fEsc=0.1, redux=True):

    # 0. set up computing grids just like in the other functions
    energies = np.array([0.0])
    intensities = np.array([0.0])

    # 1. get stellar mass
    total_stellar_mass = compute_stellar_mass(halo_mass, redshift, verbose=False)

    N0 = (2-imfIndex)*total_stellar_mass/(starMassMax**(2-imfIndex) - starMassMin**(2-imfIndex))

    # bin width
    dBin = float(starMassMax-starMassMin)/imfBins

    # array (list) of masses in bin center
    cBin = []

    # array (list) of total mass in each bin
    MBin = []

    for i in range(0, imfBins):
        cBin.append(starMassMin + (i+0.5)*dBin)
        MBin.append(N0/(2-imfIndex)*(((i+1)*dBin+starMassMin)**(2-imfIndex)-(i*dBin+starMassMin)**(2-imfIndex)))

    t_ms_list = []   # time a star spends on the main sequence, in Myr
    t_sol = 1.e4     # time our sun spends on the main sequence, in Myr

    for i in range(0, imfBins):
        t_ms_list.append(t_sol * m.pow(cBin[i], -2.5))

    for i in range(0, imfBins):

        energies_tmp, intensities_tmp = generate_SED_single_pop3(starMass=cBin[i], eHigh=eHigh, eLow=eLow,
                                                                N=N, logGrid=logGrid, fEsc=fEsc)

        if i == 0:
            energies = energies_tmp
            if redux and t_ms_list[i] < targetSourceAge:
                redux_factor = targetSourceAge / t_ms_list[i]
                intensities = ((MBin[i]/cBin[i]) / redux_factor) * intensities_tmp
            else:
                intensities = MBin[i]/cBin[i] * intensities_tmp
        else:

            if redux and t_ms_list[i] < targetSourceAge:
                redux_factor = targetSourceAge / t_ms_list[i]
                intensities += ((MBin[i]/cBin[i]) / redux_factor) * intensities_tmp

            else:
                intensities += MBin[i]/cBin[i] * intensities_tmp

    # sort energies if they are descending
    if energies[0] > energies[-1]:
        energies = energies[::-1]
        intensities = intensities[::-1]

    return energies, intensities


# -----------------------------------------------------------------
# SED for a pure power-law (QSO) source
# -----------------------------------------------------------------
def generate_SED_PL(halo_mass, eHigh=1.e4, eLow=10.4, alpha=1.0, N=1.e4, logGrid=False, qsoEfficiency=1.0):

    # energies should be in increasing order
    if eHigh < eLow:
        eLow, eHigh = eHigh, eLow

    # 1. SET UP GRIDS
    energies = np.array([eLow])
    intensities = np.array([SED_power_law(energies[0], alpha)])

    # fill energies array
    if logGrid:
        # create energies along a logarithmic grid, so that eHigh *C**(N-1) = eLow
        C = m.pow(eHigh/eLow, 1./(N-1))
        for i in range(1, N):
            eTmp = energies[i-1]*C
            energies = np.append(energies, eTmp)
            tmpI = SED_power_law(eTmp, alpha)
            intensities = np.append(intensities, tmpI)
    else:
        # create a grid of energies, so that eHigh + C*(N-1) = eLow
        C = (eHigh-eLow)/(N-1.)
        for i in range(1, N):
            eTmp = energies[i-1]+C
            energies = np.append(energies, eTmp)
            tmpI = SED_power_law(eTmp, alpha)
            intensities = np.append(intensities, tmpI)

    # 2. NORMALIZATION
    # integrate to obtain total energy (normalize)
    bh_mass = 1e-4*(cosmoOmegaB/cosmoOmegaM)*halo_mass        # halo_mass should be given in [M_sun]
    eddLum = 1.26*1e31 * bh_mass * 6.241e18                    # conversion from [Joule/s] to [eV/s]

    energies_log = np.array([m.log(energies[0])])
    sed4Norm = np.array([intensities[0]*energies[0]])

    for i in range(1, int(N)):
        eTmp = energies[i]
        energies_log = np.append(energies_log, m.log(eTmp))
        sed4Norm = np.append(sed4Norm, intensities[i]*eTmp)    # log integration trick

    # energies should be increasing in value, or else the result of the integral can be negative
    integral = simpson(sed4Norm, energies_log)
    A = (qsoEfficiency*eddLum)/integral

    intensities *= A

    # sort energies if they are descending
    if energies[0] > energies[-1]:
        energies = energies[::-1]
        intensities = intensities[::-1]

    return energies, intensities


# -----------------------------------------------------------------
# SED for a combined power-law (QSO) and IMF source
# -----------------------------------------------------------------
def generate_SED_IMF_PL(halo_mass, redshift,
                        eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                        starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                        alpha=1.0, qsoEfficiency=0.1,
                        targetSourceAge=10.0):

    energies_IMF, intensities_IMF = generate_SED_stars_IMF(halo_mass,
                                                           redshift,
                                                           eLow=eLow,
                                                           eHigh=eHigh,
                                                           N=N,
                                                           logGrid=logGrid,
                                                           starMassMin=starMassMin,
                                                           starMassMax=starMassMax,
                                                           targetSourceAge=targetSourceAge,
                                                           fEsc=fEsc,
                                                           imfBins=imfBins,
                                                           imfIndex=imfIndex)

    energies_PL, intensities_PL = generate_SED_PL(halo_mass,
                                                  eHigh=eHigh,
                                                  eLow=eLow,
                                                  alpha=alpha,
                                                  N=N,
                                                  logGrid=logGrid,
                                                  qsoEfficiency=qsoEfficiency)

    # sort energies if they are descending
    if energies_IMF[0] > energies_IMF[-1]:
        energies_IMF = energies_IMF[::-1]
        intensities_IMF = intensities_IMF[::-1]
    if energies_PL[0] > energies_PL[-1]:
        energies_PL = energies_PL[::-1]
        intensities_PL = intensities_PL[::-1]

    # check if the 'energies_*' are identical, then add the two intensity arrays
    if np.sum(energies_IMF - energies_PL) < 1E-5:  # there might be small deviations
        # add data
        intensities = intensities_IMF + intensities_PL
        energies = energies_IMF

        return energies, intensities
    else:
        raise ValueError


jit_module(nopython=True, cache=True, error_model="numpy")
