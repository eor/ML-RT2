import os
import equations.sed.sed as sed
from timeit import default_timer as timer
from numba import jit
import equations.sed.sed_numba as sed_nb
import random
import numpy as np

iterations = 50
print_after_epochs = 10
print('-----------------sampling over %d iterations-----------------\n'%(iterations))

print('\n-----------------Executing NON-NUMBA version-----------------\n')
start = timer()
for _ in range(iterations):
    E,I = sed.generate_SED_IMF_PL(12.0, 12.0, eLow=10.4, eHigh=1.0e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
end = timer()
print("total time:", (end - start))
print("elapsed avg. time:", (end - start)/iterations)

print('\n-----------------Compiling NUMBA version-----------------\n')

start = timer()
E, I = sed_nb.generate_SED_IMF_PL(12.0, 12.0, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                alpha=1.0, qsoEfficiency=0.1,
                targetSourceAge=10.0)
end = timer()
print("NUMBA: compilation_time + execution time:", (end - start))

print('\n-----------------Executing NUMBA version-----------------\n')
start = timer()
for i in range(iterations):
    a = timer()
    haloMass = random.uniform(8.0, 15.0)
    redshift = random.uniform(6.0, 13.0)
    sed_nb.generate_SED_IMF_PL(haloMass, redshift, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0)
    b = timer()
    if i % print_after_epochs == 0:
        print('executing iteration %d....%f'%(i,b-a))
end = timer()
print("NUMBA: execution time:", (end - start)/iterations)

print('\n-----------------Testing the implementations-----------------')
haloMasses = [8.0, 12.0, 15.0]
redShifts = [6.0, 8.5, 13.0]

for mass in haloMasses:
    for rs in redShifts:
        E1,I1 = sed.generate_SED_IMF_PL(mass, rs, eLow=10.4, eHigh=1.0e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
        E2, I2 = sed_nb.generate_SED_IMF_PL(mass, rs, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0)
        print(I1)
        print(I2)
        print('For haloMass=%f redShift=%f, RMSE energy array %f and RMSE intensity array %f'%(mass, rs, np.mean(E1!=E2), np.sum(energies_IMF-energies_PL)))
