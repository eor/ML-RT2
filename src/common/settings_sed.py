# -----------------------------------------------------------------
#  Settings for the input SEDs
# -----------------------------------------------------------------

SED_ENERGY_MIN = 1.0    # eV
SED_ENERGY_MAX = 1e4    # eV
SED_ENERGY_DELTA = 0.1  # eV (resolution)

# -----------------------------------------------------------------
#  Settings for the SED sampling of the p-space?
# -----------------------------------------------------------------
# ...

# -----------------------------------------------------------------
#  Settings for SED generation with 8 parameters, they are
#  same as in the first ML-RT project
# -----------------------------------------------------------------
#        1. haloMassLog         interval=[8.0, 15.0]
#        2. redshift            interval=[6.0, 13.0]
#        3. sourceAge           interval=[0.1, 20.0]
#        4. qsoAlpha            interval=[0.0, 2.0]
#        5. qsoEfficiency       interval=[0.0, 1.0]
#        6. starsEscFrac        interval=[0.0, 1.0]
#        7. starsIMFSlope       interval=[0.0, 2.5]
#        8. starsIMFMassMin  interval=[5.0, 500.0]
# (old)  8. starsIMFMassMinLog  interval=[0.6989700043360189, 2.6989700043360187]

p8_limits = list()
p8_limits.append([8.0, 15.0])
p8_limits.append([6.0, 13.0])
p8_limits.append([0.1, 20.0])
p8_limits.append([0.0, 2.0])
p8_limits.append([0.0, 1.0])
p8_limits.append([0.0, 1.0])
p8_limits.append([0.0, 2.5])
#p8_limits.append([0.6989700043360189, 2.6989700043360187])  # 5 -> 500
p8_limits.append([5.0, 500.0])

p8_names_latex = list()
p8_names_latex.append('\log_{10}{\mathrm{M}_\mathrm{halo}}')
p8_names_latex.append('z')
p8_names_latex.append('t_{\mathrm{source}}')
p8_names_latex.append('-\\alpha_{\mathrm{QSO}}')
p8_names_latex.append('\\epsilon_{\mathrm{QSO}}')
p8_names_latex.append('f_{\mathrm{esc,\\ast}}')
p8_names_latex.append('\\alpha_{\mathrm{IMF,\\ast}}')
#p8_names_latex.append('\log_{10}{\mathrm{M}_{\mathrm{min}, \\ast}}')
p8_names_latex.append('{\mathrm{M}_{\mathrm{min}, \\ast}}')


# -----------------------------------------------------------------
#  Settings for SED generation with 5 parameters, they are
#  same as in the first ML-RT project
# -----------------------------------------------------------------
# Parameter names and their respective intervals
# 1. haloMassLog        interval=[8.0, 15.0]
# 2. redshift           interval=[6.0, 13.0]
# 3. sourceAge          interval=[0.1, 20.0]
# 4. qsoAlpha           interval=[1.0, 2.0]
# 5. starsEscFrac       interval=[0.0, 1.0]


p5_limits = list()
p5_limits.append([8.0, 15.0])
p5_limits.append([6.0, 13.0])
p5_limits.append([0.1, 20.0])
p5_limits.append([1.0, 2.0])
p5_limits.append([0.0, 1.0])


p5_names_latex = list()
p5_names_latex.append('\log_{10}{\mathrm{M}_\mathrm{halo}}')
p5_names_latex.append('z')
p5_names_latex.append('t_{\mathrm{source}}')
p5_names_latex.append('-\\alpha_{\mathrm{QSO}}')
p5_names_latex.append('f_{\mathrm{esc,\\ast}}')


# -----------------------------------------------------------------
#  Setting/Parameter limits for sampling of density vection,
#  that will be used for computation of tau
# -----------------------------------------------------------------
# 1. r -> interval=[0.0, 1500.0] kpc
# 2. redshift (z) -> interval=same as redshift in p8_limits
# 3. avergage number density H_II -> interval=[0,1000]
# 4. avergage number density He_II -> interval=[0,1000]
# 5. avergage number density He_III -> interval=[0,1000]

density_vector_limits = list()
density_vector_limits.append([0, 1500])
density_vector_limits.append(p8_limits[1])
density_vector_limits.append([0, 1000])
density_vector_limits.append([0, 1000])
density_vector_limits.append([0, 1000])

density_vector_names = list()
density_vector_names.append('r(kpc)')
density_vector_names.append('z')
density_vector_names.append('n_{H_{II}}')
density_vector_names.append('n_{He_{II}}')
density_vector_names.append('n_{He_{III}}')

