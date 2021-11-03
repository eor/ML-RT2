import os
import equations.sed.sed as sed
from timeit import default_timer as timer
from numba import jit
import equations.sed.sed_numba as sed_nb
start = timer()
sed.generate_SED_IMF_PL(12.0, 12.0, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
end = timer()
print("elapsed time:", end - start)

start = timer()
sed_nb.generate_SED_IMF_PL_NUMBA(12.0, 12.0, eLow=10.4, eHigh=10000, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
end = timer()
print("NUMBA: compilation_time + execution time:", end - start)

start = timer()
sed_nb.generate_SED_IMF_PL_NUMBA(12.0, 12.0, eLow=10.4, eHigh=1.e4, N=2000,  logGrid=True,
                    starMassMin=5, starMassMax=500, imfBins=100, imfIndex=2.35, fEsc=0.1,
                    alpha=1.0, qsoEfficiency=0.1,
                    targetSourceAge=10.0,
                    fileName=None)
end = timer()
print("NUMBA: execution time:", end - start)
