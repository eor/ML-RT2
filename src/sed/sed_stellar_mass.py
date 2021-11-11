#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------
# About
# -----------------------------------------------------------------
# This file contains functions to compute the stellar mass for a given 
# dark matter halo of mass M

# -----------------------------------------------------------------
# Libs
# -----------------------------------------------------------------
import math as m
from . import sed_user_settings as us 

# -----------------------------------------------------------------
# Cosmological parameters
# -----------------------------------------------------------------
# TODO: get these from the pipeline later


cosmo_omega_M = us.cosmo_omega_M
cosmo_omega_B = us.cosmo_omega_B


# -----------------------------------------------------------------
# Compute stellar mass - the cheap way
# -----------------------------------------------------------------
def compute_stellar_mass_simple(halo_mass, f_star=0.1):
        
    # This is how stellar masses were computed in the 
    # Thomas & Zaroubi version of STARDUST
        
    stellar_mass = f_star * (cosmo_omega_B/cosmo_omega_M) * halo_mass
    
    return stellar_mass


# -----------------------------------------------------------------
# Compute stellar mass 
# -----------------------------------------------------------------
def compute_stellar_mass(halo_mass, redshift, verbose=False):
    
    # This function computes the stellar mass for a given 
    # redshift and mass of a dark matter halo

    # References: 
    # [1]:  Behroozi, Wechsler, & Conroy, 2013, ApJ 770:57 (for 0<z<8)

    # TODO: find a better solution for higher redshifts

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
    nu = m.exp(-4. * a * a)
    M_1 = 10**(M_10 + (M_1a * (a-1.) + M_1z * z) * nu)
    epsilon = 10**(epsilon_0 + (epsilon_a * (a-1.) + epsilon_z * z) * nu + epsilon_a2 * (a-1.))
    alpha = alpha_0 + (alpha_a * (a-1.)) * nu
    delta = delta_0 + (delta_a * (a-1.) + delta_z * z) * nu
    gamma = gamma_0 + (gamma_a * (a-1.) + gamma_z * z) * nu

    # equations (3) from [1]
    x = m.log10(halo_mass / M_1)

    try:        
        divisor = 1. + m.exp(10**(-x))        
    except OverflowError:
        # if x < -2.849, m.exp(10**(-x)) becomes too large (>1e307)
        divisor = 1e307
        
    frac = ((m.log10(1.+m.exp(x)))**gamma) / divisor    
    f_x = -1 * m.log10(10**(alpha * x) + 1.0) + delta * frac
  
    x = 0
    frac = ((m.log10(1.+m.exp(x)))**gamma) / (1. + m.exp(10**(-x)))
    f_0 = (-1)*m.log10(10**(alpha*x) + 1.0) + delta * frac

    tmp_m = m.log10(epsilon * M_1) + f_x - f_0

    stellar_mass = 10**(tmp_m)
        
    if verbose:
        print("z = %.4f\t M_halo = %e \t M_stellar = %e \t M_stellar/M_halo= %f" % 
              (z, m.log10(halo_mass), m.log10(stellar_mass), m.log10(stellar_mass/halo_mass)))
     
    return stellar_mass


# -----------------------------------------------------------------
# Testing 
# -----------------------------------------------------------------
if __name__ == "__main__":
  
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------
    # Test run: Compare difference for fixed z (change M_halo)
    # -----------------------------------------------------------------
    mStartLog = 8.0
    f_star = 0.1
    z = 16.0
    mHaloLog = np.arange(8, 15, 0.05)

    mStarLog1 = np.zeros(len(mHaloLog))
    mStarLog2 = np.zeros(len(mHaloLog))
    
    for i in range(0, len(mHaloLog)):
        
        mStarLog1[i] = m.log10(compute_stellar_mass_simple(10**mHaloLog[i], f_star))
        mStarLog2[i] = m.log10(compute_stellar_mass(10**mHaloLog[i], z))
    
    fig = plt.figure()
    fig.clear()  
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('log halo mass')
    ax.set_ylabel('log stellar mass')
    # ax.set_ylim(1, 1e6)
    # ax.set_yscale('log')
    ax.minorticks_on()
    ax.plot(mHaloLog, mStarLog1, lw=2.0, color="blue", label='Simple')
    ax.plot(mHaloLog, mStarLog2, lw=2.0, color="red", label='Behroozi++')
    ax.legend(loc=4)
    fig.suptitle('Redshift = %.3f'%z)  
    fig.savefig('compare_Mstar_z%.3f.png'%z)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Test run: Compare difference for fixed M (z-evolution)
    # -----------------------------------------------------------------

    zList = np.arange(6, 15, 0.05)
    mHaloLogFix = 13
    
    mStarLog1 = np.zeros(len(zList))
    mStarLog2 = np.zeros(len(zList))
    for i in range(0, len(zList)):

        mStarLog1[i] = m.log10(compute_stellar_mass_simple(10**mHaloLogFix, f_star))
        mStarLog2[i] = m.log10(compute_stellar_mass(10**mHaloLogFix, zList[i]))
    
    fig = plt.figure()
    fig.clear()  
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('log Stellar mass')
    ax.minorticks_on()
    ax.plot(zList, mStarLog1, lw=2.0, color="blue", label='Simple')
    ax.plot(zList, mStarLog2, lw=2.0, color="red", label='Behroozi++')
    ax.legend(loc=5)
    fig.suptitle('log halo mass = %.3f'%mHaloLogFix)
    fig.savefig('compare_Mstar_M%.3f.png'%mHaloLogFix)
    plt.close(fig)

    
    







