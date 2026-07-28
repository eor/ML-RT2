
# -----------------------------------------------------------------
#  Setting/Parameter limits for sampling of tau input vector,
#  that will be used for computation of tau
# -----------------------------------------------------------------
# 1. r -> interval=[0.0, 1500.0] kpc
# 2. redshift (z) -> interval=same as redshift in p8_limits
# 3. average number density H_II -> interval=[0.0, 1.0]
# 4. average ionised He_II -> interval=[0.0, 1.0]
# 5. average ionised He_III -> interval=[0.0 ,1.0]

# 2, 3, 4, 5 not used anymore.

tau_input_vector_limits = list()
tau_input_vector_limits.append([0.0, 1500.0])
tau_input_vector_limits.append([6.0, 13.0])
tau_input_vector_limits.append([0.0, 1.0])
tau_input_vector_limits.append([0.0, 1.0])
tau_input_vector_limits.append([0.0, 1.0])

tau_input_vector_names = list()
tau_input_vector_names.append('r(kpc)')
tau_input_vector_names.append('z')
tau_input_vector_names.append('x_{H_{II}}')
tau_input_vector_names.append('x_{He_{II}}')
tau_input_vector_names.append('x_{He_{III}}')
