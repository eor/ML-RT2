import os
import equations.sed.sed as sed
from timeit import default_timer as timer
from numba import jit
import equations.sed.sed_numba as sed_nb
import random

iterations = 30
print_after_epochs = 5
print('-----------------sampling over %d iterations-----------------'%(iterations))

print('-----------------Executing NON-NUMBA version-----------------')
start = timer()
for _ in range(iterations):
    sed.generate_SED_IMF_PL(12.0, 12.0, eLow=10.4, eHigh=1.0e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
end = timer()
print("total time:", (end - start))
print("elapsed avg. time:", (end - start)/iterations)

print('-----------------Compiling NUMBA version-----------------')

start = timer()
sed_nb.generate_SED_IMF_PL(12.0, 12.0, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                alpha=1.0, qsoEfficiency=0.1,
                targetSourceAge=10.0,
                fileName=None)
end = timer()
print("NUMBA: compilation_time + execution time:", (end - start))

print('-----------------Executing NUMBA version-----------------')
start = timer()
for i in range(iterations):
    a = timer()
    haloMass = random.uniform(8.0, 15.0)
    redshift = random.uniform(6.0, 13.0)
    sed_nb.generate_SED_IMF_PL(haloMass, redshift, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
    b = timer()
    if i % print_after_epochs == 0:
        print('executing iteration %d....%f'%(i,b-a))
end = timer()
print("NUMBA: execution time:", (end - start)/iterations)
