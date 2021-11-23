
class ODE_BC:
    def __init__(self, conf):
        pass

    def compute_ode_residual(self, flux_vector, state_vector, time_vector, u_approximation):
        u_0, u_1, u_2, u_3 = u_approximation(flux_vector, state_vector, time_vector)

        x_H_II_prediction = u_0
        x_He_II_prediction = u_1
        x_He_III_prediction = u_2
        T_prediction = u_3

    def boundary_condition_x_H_II(self):
        # [TODO]: complete this
        pass
