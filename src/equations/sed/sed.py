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
# packages = ['numpy', 'scipy', 'matplotlib']
# for p in packages:
#     try:
#         imp.find_module(p)
#     except ImportError:
#         print('Package %s not found. Exiting.'%p)
#         sys.exit(1)


import math as m
import numpy as np
import scipy
import scipy.constants
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from scipy import integrate


from . import sed_stellar_mass as sm
from . import sed_user_settings as us


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
c = scipy.constants.value("speed of light in vacuum")
c_cgi = c*100
h_eV = scipy.constants.value("Planck constant in eV s")
k_BeV = scipy.constants.value("Boltzmann constant in eV/K")
sigmaSB = scipy.constants.value("Stefan-Boltzmann constant")   # here: 5.670367e-08 W m^-2 K^-4

# -----------------------------------------------------------------
# Cosmological parameters
# -----------------------------------------------------------------
cosmoOmegaM = us.cosmoOmegaM
cosmoOmegaB = us.cosmoOmegaB

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


# -----------------------------------------------------------------
# SED for a pure power-law (QSO) source
# -----------------------------------------------------------------
def generate_SED_PL(haloMass, eHigh=1.e4, eLow=10.4, fileName=None, alpha=1.0, N=10000, logGrid=False, qsoEfficiency=1.0, silent=False):

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

    if not silent:
        print("------------------------------------------------------------------------------------")
        print("Generating SED for a halo featuring a PL-type source:")
        print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
        print("\tPL index \t= %.3f "%(alpha))
        print("\tblack hole mass    \t= %.3f M_sol"%(bhMass))
        print("\tEddington luminosity\t= %e eV/s"%(eddLum))

    # integrate to obtain total energy (normalize)
    sed4Norm    = np.array([])
    energiesLog = np.array([])

    for i in range(0,N):
        eTmp = energies[i]
        energiesLog = np.append(energiesLog, m.log(eTmp) )
        sed4Norm    = np.append(sed4Norm, intensities[i]*eTmp)      # log integration trick

    integral = integrate.simps(sed4Norm, energiesLog, even='avg')   # this scipy function wants the arguments (y,x, even=...)

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
# SED for a single star (black body)
# -----------------------------------------------------------------
def generate_SED_single_pop3(starMass=100, eHigh=1.e4, eLow=10.4, fileName=None, N=1000, logGrid=False, fEsc=0.1):

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

    integral = integrate.simps(sed4Norm[::-1], energiesLog[::-1], even='avg') # energies should be increasing in value, or else the result can be negative

    G = (starL*3.828e26/1.6022e-19)/(integral)

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
def generate_SED_stars_BB(haloMass, redshift, starMass=100, eHigh=1.e4, eLow=10.4, fileName=None, N=1000, logGrid=False, fEsc=0.1):


    # 0. set up computing grids
    energies = np.array([])
    intensities = np.array([])

    # 1. get stellar mass and number of stars of mass starMass
    totalStellarMass = sm.compute_stellar_mass( haloMass, redshift, verbose=False )
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
def generate_SED_stars_IMF(haloMass, redshift, eLow=10.4, eHigh=1.e4, N=1000,  logGrid=False,
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

    totalStellarMass = sm.compute_stellar_mass( haloMass, redshift, verbose=False )
    #totalStellarMass = sm.compute_stellar_mass_simple( haloMass, fStar=fStar)

    if(not silent):
        print("------------------------------------------------------------------------------------")
        print("Generating SED for a halo featuring black-body-like sources, which follow an IMF:")
        print("\tRedshift \t\t= %.3f"%(redshift))
        print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
        print("\tTotal stellar mass \t= %e M_sol"%(totalStellarMass))
        print("\tMinimum star mass  \t= %.3f M_sol"%(starMassMin))
        print("\tMaximum star mass  \t= %.3f M_sol"%(starMassMax))



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
        tSol   = 1e4   # time our sun spends on the main sequence, in Myr

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
# SED for a power-law (QSO) source and stars (IMF)
# -----------------------------------------------------------------
def generate_SED_IMF_PL(haloMass, redshift, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                        starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                        alpha=1.0, qsoEfficiency=0.1,
                        targetSourceAge=10.0,
                        fileName=None):


    print("------------------------------------------------------------------------------------")
    print("Generating SED for a halo featuring IMF and PL-type sources:")
    print("\tRedshift \t\t= %.3f"%(redshift))
    print("\tHost halo mass \t\t= %e M_sol"%(haloMass))
    print("\tPL index \t= %.3f "%(alpha))
    print("\tMinimum star mass  \t= %.3f M_sol"%(starMassMin))
    print("\tMaximum star mass  \t= %.3f M_sol"%(starMassMax))

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
        exit(1)


# -----------------------------------------------------------------
# Simple SEDs in functional form
# -----------------------------------------------------------------
def SED_power_law(E, alpha):

    return m.pow(E, -alpha)


def SED_black_body(E, T):

    expo = E / (k_BeV * T)
    try:
        return (2.0 * m.pi / (c_cgi * c_cgi * h_eV * h_eV        )) * (E * E * E) / (m.exp(expo) - 1.)
    except OverflowError:
        return 1e-300
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


# -----------------------------------------------------------------
# Plotting routine
# -----------------------------------------------------------------
def plot_SED(fileName, eHigh=1e4, eLow=10.4, logX=True, logY=False, xLimit=None, yLimit=None, altFileName=None, legend=None, legendLoc=1, colors=None, lineStyles=None):

    # fileName : path to file name as string

    if not colors:
        colors = ['black', 'red', 'blue', 'green']

    if not lineStyles:
        lineStyles = ['-', '-', '-', '-']

    if not type(fileName) is list:
        fileName = [fileName]   # turn fileName into list
        if legend:
          legend = [legend]

    energies = []
    intensities = []

    for i in range(0,len(fileName)):
        energies.append(np.array([]))
        intensities.append(np.array([]))
        with open(fileName[i], 'r') as f:
            # skip first line (header)
            next(f)
            # read file and extract desired columns
            for line in f:
                columns        = [float(x) for x in line.split()]
                tmpE, tmpI     = columns[0], columns[1]
                energies[i]    = np.append(energies[i],tmpE)
                intensities[i] = np.append(intensities[i],tmpI)

    # define plot and axes
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1,1,1)
    ax1.minorticks_on()

    if (xLimit):
        ax1.set_xlim(xLimit[0], xLimit[1])
    else:
        ax1.set_xlim(eLow, eHigh)

    if( logX ):
        ax1.set_xscale('log')

    if( logY ):
        ax1.set_yscale('log')

    if(yLimit):
        ax1.set_ylim(yLimit[0], yLimit[1])


    mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')
    params = {'legend.fontsize':15}
    plt.rcParams.update(params)

    #rc('font',**{'family':'sans'})
    #rc('text', usetex=True)
    ##rc('lines', linewidth=2)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.xaxis.set_tick_params(labelsize=16)

    ax1.set_xlabel(r'$\mathrm{Photon}\ \mathrm{energy}\ [\mathrm{eV}]$',fontsize=16 )
    ax1.set_ylabel(r'$\mathrm{Total}\ \mathrm{intensity}\ [\mathrm{eV/s/eV}]$', fontsize=16)

    for i in range(0,len(fileName)):
        if legend:
            ax1.plot(energies[i], intensities[i],  ls=lineStyles[i], lw=1.5, color=colors[i],label=legend[i])
        else:
            ax1.plot(energies[i], intensities[i],  ls=lineStyles[i], lw=1.0, label=r'Foo', color=colors[i])

    if legend:
        ax1.legend(loc=legendLoc, ncol=1)
        #print legend.get_frame_on()
        #plt.legend(loc=legendLoc, ncol=2)
        #ax1.legend.get_frame().set_linewidth(1)
    #if( logX ):
        #ax2 = ax1.twiny()
        #ax2.set_xscale('log')

        #ax1Ticks = ax1.get_xticks()
        #ax2Ticks = ax1Ticks

        #ax2.set_xticks(ax2Ticks)
        #ax2.set_xbound(ax1.get_xbound())
        #ax2.set_xticklabels(tick_function(ax2Ticks))

        ##ax1.set_xlabel("Frequency (GHz)")
        #ax2.set_xlabel(r'Energy [$\mathrm{eV}$]')

    # if it exists, trim the .dat suffix
    if altFileName:
        plotFileName = altFileName
    else:
        if len(fileName) == 1:
            plotFileName = fileName[0].split(".dat")[0]+".pdf"
        else:
            plotFileName = 'SED_comparison.pdf'

    plt.tight_layout()
    fig.savefig(plotFileName)

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
# BEGIN OF SANDBOX
# -----------------------------------------------------------------

# # if __name__ == "__main__":
#
#     # case = 2
#
#
#     # -----------------------------------------------------------------
#     # SED comparison for BEARS pipeline paper
#     # -----------------------------------------------------------------
#     #if (case==0):
#
#
#         #dir_BB_8  = 'BB_8'
#
#         ##dir_IMF_10 = 'IMF_redux'
#         #dir_IMF_10 = 'IMF_redux_fEsc0.10_tmp'
#         #dir_IMF_05 = 'IMF_redux_fEsc0.05'
#
#         #dir_PL_1_5 = 'PL_alpha1.5'
#         #dir_PL_1_0 = 'PL_alpha1.0'
#
#         #dir_CE_1_0 = 'PL_alpha1.0_below_z8.5+IMF_redux'
#         #dir_CE_1_5 = 'PL_alpha1.5+IMF_redux_fEsc0.05'
#
#         #M           = 10
#         #z           = 8.0
#
#         #sedFile    ='sed_%s_M%.3f_z%.3f.dat'%('BB',M,z)
#
#         #fileList =  [dir_BB_8   +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('BB',M,z) ,
#                      #dir_IMF_10 +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('IMF',M,z),
#                      #dir_IMF_05 +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('IMF',M,z),
#                      #dir_PL_1_5 +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('PL',M,z),
#                      #dir_PL_1_0 +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('PL',M,z),
#                      #dir_CE_1_5 +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('IMF+PL',M,z),
#                      #dir_CE_1_0 +'/'+ 'sed_%s_M%.3f_z%.3f.dat'%('IMF+PL',M,z),
#                     #]
#
#         #legend=['BB 8', 'IMF 10',  'IMF 05',  'PL 1.5', 'PL 1.0', 'CE 1.5', 'CE 1.0']
#
#         #colors = ['black','blue','blue','green','green','red','red', 'gray']
#         #ls     = ['-'    ,  '-', '--', '-', '--', '-', '--', '-', '--',]
#
#         #outFile =  'SED_comparison_M%.3f_z%.3f.pdf'%(M,z)
#
#
#         #plot_SED(fileName=fileList, eHigh=1e4, eLow=10.4, logX=True, logY=True, xLimit=[12,1e4], yLimit=[2e48,8e53], altFileName=outFile, legend=legend, legendLoc=1, colors=colors, lineStyles=ls)
#
#
#
#     # -----------------------------------------------------------------
#     # BB comparison
#     # -----------------------------------------------------------------
#     if (case==1):
#
#         # variable star mass, z = 9, log M_halo = 11
#         z       = 9.
#         logM    = 11.0
#
#         starM   = 10.0
#         f1 = 'sed_BB_Mhalo%.3f_z%.3f_Mstar%.3f.dat'%( logM, z, starM )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=starM, eHigh=1.e4, eLow=10.4, fileName=f1, N=1000, logGrid=True)
#
#
#         starM   = 200.0
#         f2 = 'sed_BB_Mhalo%.3f_z%.3f_Mstar%.3f.dat'%( logM, z, starM )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=starM, eHigh=1.e4, eLow=10.4, fileName=f2, N=1000, logGrid=True)
#
#
#         starM = 700
#         f3 = 'sed_BB_Mhalo%.3f_z%.3f_Mstar%.3f.dat'%( logM, z, starM )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=starM, eHigh=1.e4, eLow=10.4, fileName=f3, N=1000, logGrid=True)
#
#
#         plot_SED(fileName=[f1,f2,f3], logX=True, logY=True, yLimit=[2e51, 1e57], xLimit=[1e1,1e4], legend=['BB 10','BB 200','BB 700'],legendLoc=1, altFileName="SED_comparison_BB_1.pdf")
#
#
#
#         # variable host halo mass
#         z       = 7.
#         logM    = 8.0
#
#         f1 = 'sed_BB_M%.3f_z%.3f.dat'%( logM, z )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=200, eHigh=1.e4, eLow=10.4, fileName=f1, N=1000, logGrid=True)
#
#         logM    = 10.0
#         f2 = 'sed_BB_M%.3f_z%.3f.dat'%( logM, z )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=200, eHigh=1.e4, eLow=10.4, fileName=f2, N=1000, logGrid=True)
#
#         logM    = 12.0
#         f3 = 'sed_BB_M%.3f_z%.3f.dat'%( logM, z )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=200, eHigh=1.e4, eLow=10.4, fileName=f3, N=1000, logGrid=True)
#
#         logM    = 14.0
#         f4 = 'sed_BB_M%.3f_z%.3f.dat'%( logM, z )
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=200, eHigh=1.e4, eLow=10.4, fileName=f4, N=1000, logGrid=True)
#
#
#
#
#         plot_SED(fileName=[f1,f2,f3,f4], logX=True, logY=True, yLimit=[2e47, 5e60], xLimit=[1e1,1e4], legend=['BB M08','BB M10','BB M12','BB M14'],legendLoc=1, altFileName="SED_comparison_BB_2.pdf")
#
#
#
#
#     # -----------------------------------------------------------------
#     # Comparison of PL, BB, IMF, IMF+PL, const z, const halo mass
#     # -----------------------------------------------------------------
#     if (case==2):
#
#         z       = 7.
#         logM    = 10.0
#
#         f1 = 'sed_PL_M%.3f_z%.3f.dat'%( logM, z )
#         f2 = 'sed_IMF_M%.3f_z%.3f.dat'%( logM, z )
#         f3 = 'sed_BB_M%.3f_z%.3f.dat'%( logM, z )
#         f4 = 'sed_IMF+PL_M%.3f_z%.3f.dat'%( logM, z )
#
#
#         generate_SED_PL(haloMass=10**logM, eHigh=1.e4, eLow=10.4, fileName=f1, alpha=1.5, N=1000, logGrid=False, qsoEfficiency=0.1)
#
#
#         generate_SED_stars_IMF(haloMass=10**logM, redshift=z, eLow=10.4, eHigh=1.e4, N=1000,  logGrid=True,
#                                 starMassMin=5, starMassMax=100, imfBins=99, imfIndex=2.35, fileName=f2)
#
#         generate_SED_stars_BB(haloMass=10**logM, redshift=z, starMass=150, eHigh=1.e4, eLow=10.4, fileName=f3, N=1000, logGrid=True)
#
#
#         generate_SED_IMF_PL(haloMass=10**logM, redshift=z, eLow=10.4, eHigh=1.e4, N=1000,  logGrid=True,
#                             starMassMin=30, starMassMax=100, imfBins=99, imfIndex=2.35,
#                             alpha=1.0, qsoEfficiency=0.1,
#                             fileName=f4)
#
#         plot_SED(fileName=[f1,f2,f3,f4], logX=True, logY=True, yLimit=[2e47, 5e54], xLimit=[1e1,1e4], legend=['PL','IMF','BB','PL + IMF'],legendLoc=1, altFileName="SED_comparison_4_types.pdf")
#
#
#
#     # -----------------------------------------------------------------
#     # Test of the IMF SED
#     # -----------------------------------------------------------------
#     if (case==3):
#         z       = 7.
#         logM    = 8.0
#
#         a = 'sed_IMF_M%.3f_z%.3f'%( logM, z )
#
#         generate_SED_stars_IMF(haloMass=10**logM, redshift=z, eLow=10.4, eHigh=1.e4, N=1000,  logGrid=False,
#                                 starMassMin=5, starMassMax=100, imfBins=99, imfIndex=2.35,
#                                 fileName=a)
#         plot_SED(fileName=a, logX=True, logY=True, yLimit=[1e25, 1e55], legend='IMF',legendLoc=1, altFileName="SED_IMF_test.pdf")
#
#
