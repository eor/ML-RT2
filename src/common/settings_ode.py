# -----------------------------------------------------------------
#  Settings for the ODE-like NN input
# -----------------------------------------------------------------

# We need ranges (min, max, lin or log scaling) for


# x_H_II   (0.0, 1.0) in linear scale
# x_He_II   (0.0, 1.0) in linear scale
# x_He_III   (0.0, 1.0) in linear scale
# Temperature   (-0.3, 13.1) in log in the data set, should maybe be clipped for the lower bound to 0
# tau           TBD
# time          (0.01 -  20.0) based on the previous data set
ode_parameter_limits = list()
# ode_parameter_limits.append([-6.0, 0.0])
# ode_parameter_limits.append([-6.0, 0.0])
# ode_parameter_limits.append([-6.0, 0.0])
ode_parameter_limits.append([0.0, 1.0])  # linear scale
ode_parameter_limits.append([0.0, 1.0])  # linear scale
ode_parameter_limits.append([0.0, 1.0])  # linear scale
ode_parameter_limits.append([-0.3, 13.1])
ode_parameter_limits.append([0.0, 6000.0])  # Could be done in log maybe?
ode_parameter_limits.append([0.01, 20.0])

ode_parameter_names = list()
ode_parameter_names.append('\log_{10}(x_{H_{II}})')
ode_parameter_names.append('\log_{10}(x_{He_{II}})')
ode_parameter_names.append('\log_{10}(x_{He_{III}})')
ode_parameter_names.append('\log_{10}(T_{\mathrm{kin}}/\mathrm{K})')
ode_parameter_names.append('\tau')
ode_parameter_names.append('t_{\mathrm{source}}')
