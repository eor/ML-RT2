#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------
# About
#-----------------------------------------------------------------

# This Python module produces input SEDs for STARDUST.
# The units in the output files are the folloing:
#  Photon Energy [eV]  LOG total energy [eV/s/eV]

# Functions for the following SEDs are currently available:
#   (1) Single pop III star (black body-like) for a given stellar mass
#   (2) QSO-like (power law) for a given host halo mass,
#   (3) A halo of given mass populated with pop III stars following an IMF
#   (4) A combined SED from (2) and (3)

#-----------------------------------------------------------------
# Libs
#-----------------------------------------------------------------
# check for required libraries
# import imp, sys
# packages = ['numpy']
# for p in packages:
#     try:
#         imp.find_module(p)
#     except ImportError:
#         print('Package %s not found. Exiting.'%p)
#         sys.exit(1)


import math as m
import numpy as np
from numba import jit, jit_module

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
c = 299792458.0
c_cgi = 29979245800.0
h_eV = 4.135667662e-15
k_BeV = 4.135667662e-15
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
# Units are:    Mass/M_sol, log_10(L/L_sol), log_10(T/K)

pop3M[0],  pop3L[0],  pop3T[0]   =    1000.,  7.444,   5.026
pop3M[1],  pop3L[1],  pop3T[1]   =    500.,   7.106,   5.029
pop3M[2],  pop3L[2],  pop3T[2]   =    400.,   6.984,   5.028
pop3M[3],  pop3L[3],  pop3T[3]   =    300.,   6.819,   5.007
pop3M[4],  pop3L[4],  pop3T[4]   =    200.,   6.574,   4.999
pop3M[5],  pop3L[5],  pop3T[5]   =    120.,   6.243,   4.981
pop3M[6],  pop3L[6],  pop3T[6]   =     80.,   5.947,   4.970
pop3M[7],  pop3L[7],  pop3T[7]   =     60.,   5.715,   4.943
pop3M[8],  pop3L[8],  pop3T[8]   =     40.,   5.420,   4.900
pop3M[9],  pop3L[9],  pop3T[9]   =     25.,   4.890,   4.850
pop3M[10], pop3L[10], pop3T[10]  =     15.,   4.324,   4.759
pop3M[11], pop3L[11], pop3T[11]  =      9.,   3.709,   4.622
pop3M[12], pop3L[12], pop3T[12]  =      5.,   2.870,   4.440



def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = np.sum(y[slice0] + 4*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0]
        h1 = h[sl1]
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = h0 / h1
        tmp = hsum/6.0 * (y[slice0] * (2 - 1.0/h0divh1) +
                          y[slice1] * (hsum * hsum / hprod) +
                          y[slice2] * (2 - h0divh1))
        result = np.sum(tmp, axis=axis)
    return result

def simpson(y, x=None, dx=1.0, axis=-1, even='avg'):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.
    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.
    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.
    even : str {'avg', 'first', 'last'}, optional
        'avg' : Average two results:1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.
    See Also
    --------
    quad: adaptive quadrature using QUADPACK
    romberg: adaptive Romberg quadrature
    quadrature: adaptive Gaussian quadrature
    fixed_quad: fixed-order Gaussian quadrature
    dblquad: double integrals
    tplquad: triple integrals
    romb: integrators for sampled data
    cumulative_trapezoid: cumulative integration for sampled data
    ode: ODE integrators
    odeint: ODE integrators
    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.
    Examples
    --------
    >>> from scipy import integrate
    >>> x = np.arange(0, 10)
    >>> y = np.arange(0, 10)
    >>> integrate.simpson(y, x)
    40.5
    >>> y = np.power(x, 3)
    >>> integrate.simpson(y, x)
    1642.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25
    >>> integrate.simpson(y, x, even='first')
    1644.5
    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    first_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice1 = (slice(None),)*nd
        slice2 = (slice(None),)*nd
        if even not in ['avg', 'last', 'first']:
            raise ValueError("Parameter 'even' must be "
                             "'avg', 'last', or 'first'.")
        # Compute using Simpson's rule on first intervals
        if even in ['avg', 'first']:
            slice1 = tupleset(slice1, axis, -1)
            slice2 = tupleset(slice2, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5*last_dx*(y[slice1]+y[slice2])
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
        # Compute using Simpson's rule on last set of intervals
        if even in ['avg', 'last']:
            slice1 = tupleset(slice1, axis, 0)
            slice2 = tupleset(slice2, axis, 1)
            if x is not None:
                first_dx = x[tuple(slice2)] - x[tuple(slice1)]
            val += 0.5*first_dx*(y[slice2]+y[slice1])
            result += _basic_simpson(y, 1, N-2, x, dx, axis)
        if even == 'avg':
            val /= 2.0
            result /= 2.0
        result = result + val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


# -----------------------------------------------------------------
# SED for a single star (black body)
# -----------------------------------------------------------------
def generate_SED_single_pop3(starMass=100, eHigh=10000, eLow=10.4, fileName=None, N=1000, logGrid=False, fEsc=0.1):

    starM = starMass
    starT = 10**(np.interp(starM, pop3M[::-1], pop3T[::-1]))
    starL = 10**(np.interp(starM, pop3M[::-1], pop3L[::-1]))

    #print "Generating SED for one black-body-like source: M=%e\tL=%e\tT=%e"%(starM, starL, starT)

    # 0. SANITY CHECKS
    # we assume that energies will be given in eV
    if(eHigh<eLow):
        tmp = eLow
        eLow = eHigh
        eHigh = tmp

    # 1. SET UP GRIDS
    energies    = np.array([])
    intensities = np.array([])

    # fill energies array
    if(logGrid):
        # create energies along a logarithmic grid, so that eHigh *C**(N-1) = eLow
        C = m.pow(eLow/eHigh,1./(N-1))
        energies = np.append(energies, eHigh)
        for i in range(1,N):
            eTmp = energies[i-1]*C
            energies = np.append(energies, eTmp)

    else:
        # create a grid of energies, so that eHigh + C*(N-1) = eLow
        C = (eLow-eHigh)/(N-1.)
        energies = np.append(energies, eHigh)
        for i in range(1,N):
            eTmp = energies[i-1]+C
            energies = np.append(energies, eTmp)

    # let's compute some intensities
    for i in range (0, N):
        tmpE = energies[i]
        tmpI = SED_black_body(tmpE, starT)
        intensities = np.append( intensities, tmpI )

    # 2. NORMALIZATION
    # integrate to obtain total energy (normalize)
    sed4Norm    = np.array([])
    energiesLog = np.array([])

    for i in range(0,N):
        eTmp = energies[i]
        energiesLog = np.append(energiesLog, m.log(eTmp) )
        sed4Norm    = np.append(sed4Norm, intensities[i]*eTmp)    # log integration trick

    integral = simpson(sed4Norm[::-1], energiesLog[::-1], even='avg') # energies should be increasing in value, or else the result can be negative

    Gk = starL*3.828e26/1.6022e-19
    G = float(Gk)/float(integral)

    intensities *= G
    intensities *= fEsc

    # sort energies if they are descending
    if energies[0]>energies[-1]:
        energies = energies[::-1]
        intensities = intensities[::-1]

    # 3. WRITE DATA
    if(fileName):
        write_data(fileName, energies, intensities)

    return energies, intensities


# -----------------------------------------------------------------
# SED for a number of stars (black body) of the same mass
# -----------------------------------------------------------------
def generate_SED_stars_BB(haloMass, redshift, starMass=100, eHigh=10000, eLow=10.4, fileName=None, N=1000, logGrid=False, fEsc=0.1):


    # 0. set up computing grids
    energies = np.array([])
    intensities = np.array([])

    # 1. get stellar mass and number of stars of mass starMass
    totalStellarMass = compute_stellar_mass( haloMass, redshift, verbose=False )
    #totalStellarMass = sm.compute_stellar_mass_simple( haloMass, fStar=0.1)

    numStars = totalStellarMass/float(starMass)

    print("------------------------------------------------------------------------------------")
    print("Generating SED for a halo containing black-body-like sources:")
    print("\tRedshift \t\t= %.3f"%(redshift))
    print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
    print("\tTotal stellar mass \t= %e M_sol"%(totalStellarMass))
    print("\tNumber of stars\t\t= %e"%(numStars))

    energies, intensitiesTmp = generate_SED_single_pop3(starMass=starMass, eHigh=eHigh, eLow=eLow, fileName=None, N=N, logGrid=logGrid, fEsc=fEsc)
    intensities = numStars * intensitiesTmp


    # sort energies if they are descending
    if energies[0] > energies[-1]:
        energies    = energies[::-1]
        intensities = intensities[::-1]

    # 3. write data
    if(fileName):
        write_data(fileName, energies, intensities)


    return energies, intensities

# -----------------------------------------------------------------
# SED produced by stars in a halo with a given stellar mass
# -----------------------------------------------------------------
def generate_SED_stars_IMF(haloMass, redshift, eLow=10.4, eHigh=10000, N=1000,  logGrid=False,
                           starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, targetSourceAge=10.,fEsc = 0.1,
                           fileName=None, redux=True, silent=False):


    # Here we make the following assumptions:
    # - a halo of haloMass contains a certain total stellar mass (totalStellarMass)
    # - the stars follow an IMF that is characterized by a spectral index imfIndex
    #   and exists in the limits of [starMassMin, starMassMax].
    # - the stars are pop 3 stars and their luminosities and effective temperatures
    #   are interpolated from table 3 given in Schaerer (2002) (see above).


    # 0. set up computing grids just like in the other functions
    energies    = np.array([])
    intensities = np.array([])

    # 1. get stellar mass

    totalStellarMass = compute_stellar_mass( haloMass, redshift, verbose=False )
    #totalStellarMass = sm.compute_stellar_mass_simple( haloMass, fStar=fStar)

    # if(not silent):
    #     print("------------------------------------------------------------------------------------")
    #     print("Generating SED for a halo featuring black-body-like sources, which follow an IMF:")
    #     print("\tRedshift \t\t= %.3f"%(redshift))
    #     print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
    #     print("\tTotal stellar mass \t= %e M_sol"%(totalStellarMass))
    #     print("\tMinimum star mass  \t= %.3f M_sol"%(starMassMin))
    #     print("\tMaximum star mass  \t= %.3f M_sol"%(starMassMax))



    # 2. construct IMF from starMassMin to starMassMax with imfBins and imfIndex

    # IMF normalization constant
    N0   = (2-imfIndex)*totalStellarMass/(starMassMax**(2-imfIndex)- starMassMin**(2-imfIndex))
    # bin width
    dBin = float(starMassMax-starMassMin)/imfBins
    # array (list) of masses in bin center
    cBin = [starMassMin + (i+0.5)*dBin for i in range(0,imfBins)]
    # array (list) of total mass in each bin
    MBin = [N0/(2-imfIndex)*( ((i+1)*dBin+starMassMin)**(2-imfIndex)-(i*dBin+starMassMin)**(2-imfIndex) ) for i in range(0,imfBins)]

    if (redux):
        tSol = 10000   # time our sun spends on the main sequence, in Myr

        tMSList = [  tSol*(cBin[i])**(-2.5) for i in range(0,imfBins)]

    # for each bin, compute the SED for a single star in that bin.
    # E.g. take the mass of the bin center as the starMass and use it to
    # generate a pure black body SED with the function generate_SED_single_pop3
    # We add up the intensities arrays we get from every bin --> our SED

    for i in range(0, imfBins):

        energiesTmp, intensitiesTmp = generate_SED_single_pop3(starMass=cBin[i], eHigh=eHigh, eLow=eLow, fileName=None, N=N, logGrid=logGrid, fEsc=fEsc)

        if(i==0):
            energies    = energiesTmp

            if (redux and tMSList[i]<targetSourceAge):
                reduxFactor = targetSourceAge / tMSList[i]
                intensities = ((MBin[i]/cBin[i])/ reduxFactor ) * intensitiesTmp
            else:
                intensities = MBin[i]/cBin[i] * intensitiesTmp
        else:

            if (redux and tMSList[i]<targetSourceAge):
                reduxFactor = targetSourceAge / tMSList[i]
                intensities += ((MBin[i]/cBin[i])/ reduxFactor ) * intensitiesTmp

            else:
                intensities += MBin[i]/cBin[i] * intensitiesTmp

    # sort energies if they are descending
    if energies[0]>energies[-1]:
        energies    = energies[::-1]
        intensities = intensities[::-1]



    # 3. write data
    if(fileName):
        write_data(fileName, energies, intensities)



    return energies, intensities


# -----------------------------------------------------------------
# SED for a pure power-law (QSO) source
# -----------------------------------------------------------------
def generate_SED_PL(haloMass, eHigh=10000, eLow=10.4, fileName=None, alpha=1.0, N=10000, logGrid=False, qsoEfficiency=1.0, silent=False):

    # 0. SANITY CHECKS
    # we assume that energies will be given in eV
    if eHigh < eLow:
        tmp = eLow
        eLow = eHigh
        eHigh = tmp

    # 1.  SET UP ARRAYS
    energies = np.array([])
    intensities = np.array([])       # basically SED without normalization

    # fill energies array
    if logGrid :
        # create energies along a logarithmic grid, so that eLow*C**(N-1) = eHigh
        C = m.pow(eHigh/eLow,1./(N-1))
        energies = np.append(energies,eLow)
        for i in range(1,N):
            eTmp = energies[i-1]*C
            energies = np.append(energies, eTmp)
    else:
        # create a linear wavelength grid, so that eLow + C*(N-1) = eHigh
        C = (eHigh-eLow)/(N-1.)
        energies = np.append(energies, eLow)
        for i in range(1,N):
            eTmp = energies[i-1]+C
            energies = np.append(energies, eTmp)

    # compute energies
    for i in range(0, N):
        tmp = SED_power_law(energies[i], alpha)
        intensities = np.append(intensities, tmp)


    # 2. NORMALIZATION
    # Our SED is I(E) = A E**(-alpha), we therefore need to find A.
    # A(M_BH) = (efficiency*EddingtonLuminosity) / int_{E range} E**{-alpha}dE

    bhMass = 1e-4*(cosmoOmegaB/cosmoOmegaM)*haloMass        # haloMass should be given in [M_sun]
    eddLum = 1.26*1e31*(bhMass)*6.241e18                    # conversion from [Joule/s] to [eV/s]

    # if not silent:
    #     print("------------------------------------------------------------------------------------")
    #     print("Generating SED for a halo featuring a PL-type source:")
    #     print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
    #     print("\tPL index \t= %.3f "%(alpha))
    #     print("\tblack hole mass    \t= %.3f M_sol"%(bhMass))
    #     print("\tEddington luminosity\t= %e eV/s"%(eddLum))
    #
    # integrate to obtain total energy (normalize)
    sed4Norm    = np.array([])
    energiesLog = np.array([])

    for i in range(0,N):
        eTmp = energies[i]
        energiesLog = np.append(energiesLog, m.log(eTmp) )
        sed4Norm    = np.append(sed4Norm, intensities[i]*eTmp)      # log integration trick

    integral = simpson(sed4Norm, energiesLog, even='avg')   # this scipy function wants the arguments (y,x, even=...)

    #print "Normalizing SED: %e"%(integral/(eHigh-eLow))
    A = (qsoEfficiency*eddLum)/integral

    # normalize the intensities:
    intensities *= A

    # sort energies if they are descending
    if energies[0]>energies[-1]:
        energies    = energies[::-1]
        intensities = intensities[::-1]

    # 3 WRITE DATA
    if(fileName):
        write_data(fileName, energies, intensities)

    return energies, intensities


# -----------------------------------------------------------------
# SED for a power-law (QSO) source and stars (IMF)
# -----------------------------------------------------------------
def generate_SED_IMF_PL_NUMBA(haloMass, redshift, eLow=10.4, eHigh=10000, N=2000,  logGrid=True,
                        starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                        alpha=1.0, qsoEfficiency=0.1,
                        targetSourceAge=10.0,
                        fileName=None):


    # print("------------------------------------------------------------------------------------")
    # print("Generating SED for a halo featuring IMF and PL-type sources:")
    # print("\tRedshift \t\t= %.3f"%(redshift))
    # print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
    # print("\tPL index \t= %.3f "%(alpha))
    # print("\tMinimum star mass  \t= %.3f M_sol"%(starMassMin))
    # print("\tMaximum star mass  \t= %.3f M_sol"%(starMassMax))

    energies_IMF, intensities_IMF = generate_SED_stars_IMF(haloMass,
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
                                                           imfIndex=imfIndex,
                                                           fileName=None,
                                                           silent=True
                                                           )


    energies_PL, intensities_PL = generate_SED_PL(haloMass,
                                                  eHigh=eHigh,
                                                  eLow=eLow,
                                                  fileName=None,
                                                  alpha=alpha,
                                                  N=N,
                                                  logGrid=logGrid,
                                                  qsoEfficiency=qsoEfficiency,
                                                  silent=True
                                                  )

    # sort energies if they are descending
    if energies_IMF[0]>energies_IMF[-1]:
        energies_IMF    = energies_IMF[::-1]
        intensities_IMF = intensities_IMF[::-1]
    if energies_PL[0]>energies_PL[-1]:
        energies_PL    = energies_PL[::-1]
        intensities_PL = intensities_PL[::-1]

    # check if the 'energies_*' are identical, then add the two intensity arrays
    if np.sum(energies_IMF-energies_PL)<1E-5:  # there might be small deviations

        # add data
        intensities = intensities_IMF + intensities_PL
        energies    = energies_IMF

        # write data
        if(fileName):
            write_data(fileName, energies, intensities)

        return energies, intensities

    else:
        print('WARNING: cannot add intensities of PL and IMF, because energy arrays don\'t match!')


# -----------------------------------------------------------------
# Simple SEDs in functional form
# -----------------------------------------------------------------
def SED_power_law(E, alpha):

    return m.pow(E, -alpha)


def SED_black_body(E, T):

    expo = E / (k_BeV * T)
    # try:
    return (2.0 * m.pi / (c_cgi * c_cgi * h_eV * h_eV        )) * (E * E * E) / (m.exp(expo) - 1.)
    # except OverflowError:
        # return 1e-300
        # for expo > 709, result of m.exp() is too large for python float (= C double)


# -----------------------------------------------------------------
# write data
# -----------------------------------------------------------------
def write_data(fileName, energies, intensities):

    with open(fileName, 'w') as f:
        f.write('# Energy [eV]  TOTAL [eV/s/eV]\n')
        N = len(energies)
        for i in range(0, N):
            f.write('%e\t%e\n'%(energies[i], intensities[i]) )


def tick_function(X):
    v = c/(X*1e-10) 	# Hz
    E = v*h_eV
    return ["%.3f" % i for i in E]


# -----------------------------------------------------------------
# Conversion routines
# -----------------------------------------------------------------
def eV2ang(E):
    v = E/h_eV
    return c/(v*1e-10)

def ang2eV(X):
    return h_eV*c/(X*1e-10) 	# Hz

# -----------------------------------------------------------------
# Compute stellar mass - the cheap way
# -----------------------------------------------------------------
def compute_stellar_mass_simple( haloMass, fStar=0.1 ):

    # This is how stellar masses were computed in the
    # Thomas & Zaroubi version of STARDUST


    stellarMass = fStar * (cosmoOmegaB/cosmoOmegaM) * haloMass

    return stellarMass


# -----------------------------------------------------------------
# Compute stellar mass
# -----------------------------------------------------------------
def compute_stellar_mass( haloMass, redshift, verbose=False ):

    # This function computes the stellar mass for a given
    # redshift and mass of a dark matter halo

    # References:
    # [1]:  Behroozi, Wechsler, & Conroy, 2013, ApJ 770:57 (for 0<z<8)

    # TODO:
    #       - find a better solution for higher redshifts

    z = redshift
    a = 1./(1.+ z)

    # equation parameters
    M_10        = 11.514
    M_1a        = -1.793
    M_1z        = -0.251

    epsilon_0   = -1.777
    epsilon_a   = -0.006
    epsilon_a2  = -0.119
    epsilon_z   = -0.000

    alpha_0     = -1.412
    alpha_a     = 0.731

    delta_0     = 3.508
    delta_a     = 2.608
    delta_z     = -0.043

    gamma_0     = 0.316
    gamma_a     = 1.319
    gamma_z     = 0.279

    # equations (4) from [1]
    nu      = m.exp(-4.*a*a)
    M_1     = 10**( M_10 + ( M_1a*(a-1.) + M_1z*(z) )*nu )
    epsilon = 10**( epsilon_0 + ( epsilon_a*(a-1.) + epsilon_z*(z) )*nu + epsilon_a2*(a-1.) )
    alpha   = alpha_0 + ( alpha_a*(a-1.) )*nu
    delta   = delta_0 + ( delta_a*(a-1.) + delta_z*(z) )*nu
    gamma   = gamma_0 + ( gamma_a*(a-1.) + gamma_z*(z) )*nu


    # equations (3) from [1]
    x = m.log10( haloMass / M_1 )

    # try:
    divisor =   1.+ m.exp( 10**(-x) )
    # except OverflowError:
    #     # if x < -2.849, m.exp( 10**(-x) ) becomes too large (>1e307)
    #     divisor = 1e307

    frac    = ( ( m.log10( 1.+m.exp(x) ) )**gamma )/ divisor
    f_x     = (-1)*m.log10( 10**(alpha*x) + 1.0 ) + delta*( frac )

    x       = 0
    frac    = ( ( m.log10( 1.+m.exp(x) ) )**gamma )/ ( 1.+ m.exp( 10**(-x) ) )
    f_0     = (-1)*m.log10( 10**(alpha*x) + 1.0 ) + delta*( frac )


    tmpM = m.log10( epsilon*M_1 ) + f_x - f_0

    stellarMass = 10**(tmpM)

    # if(verbose):
    #     print("z = %.4f\t M_halo = %e \t M_stellar = %e \t M_stellar/M_halo= %f"%(z, m.log10(haloMass), m.log10(stellarMass), m.log10(stellarMass/haloMass) ))

    return stellarMass

jit_module(nopython=True, error_model="numpy")
