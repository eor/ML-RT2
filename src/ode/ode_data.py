import os
import torch
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import Pool

import sys; sys.path.append('..')

import common.sed_numba as sed_numba
from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA
from common.physics import *

from pretraining.settings import tau_input_vector_limits


class ODEData:
    """
    This class contains a set of functions to sample random data for ode model
    training and returns the data as an object of pytorch Dataloader class.
    """

    def __init__(self, batch_size, device=torch.device('cpu')):
        """
        Args:
        device: (torch.device) the desired device for the required pytorch dataloader.
        """
        self.device = device
        self.batch_size = batch_size

    def generate_data(self, n_samples, mode='train'):
        """
        Args:
        n_samples: number of samples to generate.
        mode: whether data that is being generated will be used for training, testing
              or validation. There is no change in computation of data in either of the cases.
              However, some verbrose is avoided in case mode == 'train'.

        This fuction leverage multi-processing to do the following tasks:
        1. generates n_samples of training data,
        2. convert all the samples into tensors and copy them to required device.
        3. divide the data into batches
        4. returns the data as an object of pytorch of dataloader class. so that it
           can be used for training, testing or validation.

        Returns an object of torch.utils.data.DataLoader class which has inputs in
        the following order: [flux_vectors, state_vectors, time_vectors,
                            parameter_vectors, energies_vectors, target_residuals].
        """

        if mode != 'train':
            print('\nGenerating %d samples for %s set......'%(n_samples, mode))

        # sample parameter vectors for sed generators.
        parameter_vectors = self.sample_parameter_vectors(n_samples)

        # sample state vectors to be used as input for main model.
        state_vectors = self.sample_state_vectors(n_samples)

        # use multi-processing for generating flux vectors.
        # get number of usable cpu cores using os.cpu_count()
        n_workers_count = os.cpu_count()

        # split parameter_vectors into list of smaller arrays such that all
        # workers get equal number of parameters to process.
        parameter_vectors = np.array_split(parameter_vectors, n_workers_count, axis=0)

        # record time at the start of the process
        start = timer()

        # create maximum processes we can create to sample flux vectors and
        # store the output.
        with Pool(n_workers_count) as pool:
            # pool.map returns a list of tuples. so, we need to convert output into
            # two independent lists using zip operation.
            flux_vectors, energies_vectors = zip(*pool.map(self.sample_flux_vectors, parameter_vectors))

        # record time at the end of the process
        end = timer()

        # If we are generating data for testing or validation, print the total
        # time take by the generation.
        if mode != 'train':
            print('Time taken to generate %d samples for %s set: %f s'%(n_samples, mode, end - start))

        # combine all inputs and outputs back into original shape.
        parameter_vectors = np.concatenate(parameter_vectors, axis=0)
        flux_vectors = np.concatenate(flux_vectors, axis=0)
        energies_vectors = np.concatenate(energies_vectors, axis=0)

        # sample time vectors for the main model.
        # Lower limit for the time vector will be from the start of the EoR
        # and upper limit will be the current age of source that was used to model
        # the source in sed generator.
        lower_limit = ps_ode[5][0] * np.ones(shape=(n_samples, 1))
        upper_limit = parameter_vectors[:, 2].reshape(n_samples, 1).copy()
        time_vectors = np.random.uniform(lower_limit, upper_limit, size=(n_samples, 1))

        # sample target residual labels
        target_residuals = np.zeros((n_samples))

        data = [flux_vectors, state_vectors, time_vectors, parameter_vectors,
                                            energies_vectors, target_residuals]

        # convert numpy data into pytorch dataloader
        dataloader = self.convert_numpy_data_to_tensor(data)

        return dataloader


    def convert_numpy_data_to_tensor(self, data):
        """
        This function converts numpy data into pytorch dataloader.
        Args: numpy array or list of numpy arrays.
        """

        # convert numpy data to pytorch tenosor
        data = [torch.from_numpy(d).float().to(self.device) for d in data]
        # convert tensor data into pytorch dataset.
        dataset = TensorDataset(*data)
        # convert tensor dataset to tensor dataloader.
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        return dataloader


    def sample_tau_vectors(self, parameters, intensities_vectors, energy_vector, n_samples):
        """
        Args:
        parameters: parameters which were used to generate intensities
                    using sed generator. These parameters define a uninque source in
                    radiative transfer system.
        intensities_vectors:
        energy_vector: energies at which intenisities were computed.
        n_samples: number of samples to be sampled.

        Returns randomly sampled parameters that were used to compute tau vectors
        and tau vectors themselves. These vectors will later be used for the
        computation for flux vectors.

        return shape: numpy array -> shape((n_samples, 2000))
        """

        # obtain sigmas for the energy_vector. As we have same energy vector for
        # every sample in train_set, it can be computed just once and can be used again
        # and again.
        # shape: (2000)
        physics = Physics.getInstance()
        physics.set_energy_vector(energy_vector)
        sigmas_H_I = physics.get_photo_ionisation_cross_section_hydrogen()
        sigmas_He_I = physics.get_photo_ionisation_cross_section_helium1()
        sigmas_He_II = physics.get_photo_ionisation_cross_section_helium2()

        # sample parameters for computing tau
        r = np.random.randint(tau_input_vector_limits[0][0],
                            tau_input_vector_limits[0][1], size=(n_samples, 1))
        redshift = parameters[:, 1].reshape((n_samples, -1))

        # compute total initial densities before ionisation using redhsift values
        n_H_0 = CONSTANT_n_H_0 * np.power(1 + redshift, 3)
        n_He_0 = CONSTANT_n_He_0 * np.power(1 + redshift, 3)

        # sample random ionisation fractions between 0 and 1 for neutral hydrogen,
        # helium and singly ionised helium.
        ionisation_fraction_H_I = np.random.random(size=(n_samples, 1))
        ionisation_fraction_He_I = np.random.random(size=(n_samples, 1))
        ionisation_fraction_He_II = 1 - ionisation_fraction_He_I

        # use the sampled ionisation fractions and initial number densities to compute
        # number densities of neutral hydrogen, helium and singly ionised helium.
        num_density_H_I = n_H_0 * ionisation_fraction_H_I
        num_density_He_I = n_He_0 * ionisation_fraction_He_I
        num_density_He_II = n_He_0 * ionisation_fraction_He_II

        # concatenate individual parameters to tau_input_vector
        tau_input_vector = np.concatenate((r, redshift, num_density_H_I,
                                  num_density_He_I, num_density_He_II), axis=1)

        # generate tau from tau_input_vector
        tau = sigmas_H_I[np.newaxis, :] * num_density_H_I
        tau += sigmas_He_I[np.newaxis, :] * num_density_He_I
        tau += sigmas_He_II[np.newaxis, :] * num_density_He_II
        tau *= r * KPC_to_CM

        return tau_input_vector, tau

    def sample_flux_vectors(self, parameters):
        """
        Args:
        parameters: set of 8 parameters that are used to model a unique source
                    in radiative-transfer model. shape: (n_samples, 8)

        Returns flux vectors and energies at which they were computed.
        """

        n_samples = parameters.shape[0]
        intensities_vectors = []
        energies_vectors = []

        # generate intensities for the training.
        for i in range(n_samples):
            # generate sed from parameters
            energies, intensities = sed_numba.generate_SED_IMF_PL(halo_mass=parameters[i][0],
                                                                  redshift=parameters[i][1],
                                                                  eLow=SED_ENERGY_MIN,
                                                                  eHigh=SED_ENERGY_MAX,
                                                                  N=2000,  logGrid=True,
                                                                  starMassMin=parameters[i][7],
                                                                  starMassMax=500,
                                                                  imfBins=50,
                                                                  imfIndex=parameters[i][6],
                                                                  fEsc=parameters[i][5],
                                                                  alpha=parameters[i][3],
                                                                  qsoEfficiency=parameters[i][4],
                                                                  targetSourceAge=parameters[i][2])

            intensities_vectors.append(intensities)
            energies_vectors.append(energies)

        # convert lists to numpy arrays (batch_size)
        intensities_vectors = np.asarray(intensities_vectors)
        energies_vectors = np.asarray(energies_vectors)

        # obtain tau using the sampled parameters, intensities vector
        # and energy vector. Tau will further be used to compute flux vextors.
        tau_input_vector, tau = self.sample_tau_vectors(parameters,
                             intensities_vectors, energies_vectors[0], n_samples)

        # obtain r vector from tau input vector
        r = tau_input_vector[:, 0].reshape((n_samples, 1))

        # obtain flux_vector from intensities_vector by multiplying with tau
        # add a small number to r to avoid division by zero
        flux_vectors = intensities_vectors * np.exp(-1 * tau) / (4 * np.pi * np.power(r + 1e-5, 2))

        # convert flux vectors to log scale.
        flux_vectors = np.log10(flux_vectors + 1.0e-6)

        return flux_vectors, energies_vectors


    def sample_parameter_vectors(self, n_samples):
        """
        Args:
        n_samples: number of samples to be generated.
        Returns randomly sampled parameter vectors for sed generator.
        return shape: numpy array -> shape((n_samples, 8))
        """

        # generate all parameters needed for sed generator in shape (n_samples, 1).
        haloMassLog = np.random.uniform(ps_sed[0][0], ps_sed[0][1], size=(n_samples, 1))
        redshift = np.random.uniform(ps_sed[1][0], ps_sed[1][1], size=(n_samples, 1))
        sourceAge = np.random.uniform(ps_sed[2][0], ps_sed[2][1], size=(n_samples, 1))
        qsoAlpha = np.random.uniform(ps_sed[3][0], ps_sed[3][1], size=(n_samples, 1))
        qsoEfficiency = np.random.uniform(ps_sed[4][0], ps_sed[4][1], size=(n_samples, 1))
        starsEscFrac = np.random.uniform(ps_sed[5][0], ps_sed[5][1], size=(n_samples, 1))
        starsIMFSlope = np.random.uniform(ps_sed[6][0], ps_sed[6][1], size=(n_samples, 1))
        starsIMFMassMin = np.random.uniform(ps_sed[7][0], ps_sed[7][1], size=(n_samples, 1))

        # concaenate the parameters to a single vector.
        parameter_vectors = np.concatenate((haloMassLog, redshift, sourceAge, qsoAlpha,
                                           qsoEfficiency, starsEscFrac, starsIMFSlope, starsIMFMassMin), axis=1)
        return parameter_vectors

    def sample_state_vectors(self, n_samples):
        """
        Args: n_samples: number of samples to be sampled.
        Returns randomly sampled state vectors to be used as input to main model.
        return shape: numpy array -> shape((n_samples, 4))
        """

        # sample all the state parameters individually.
        x_H_II = np.random.uniform(ps_ode[0][0], ps_ode[0][1], size=(n_samples, 1))
        x_He_II = np.random.uniform(ps_ode[1][0], ps_ode[1][1], size=(n_samples, 1))
        x_He_III = np.random.uniform(ps_ode[2][0], ps_ode[2][1], size=(n_samples, 1))
        T = np.random.uniform(ps_ode[3][0], ps_ode[3][1], size=(n_samples, 1))

        # concatenate the parameters to a single vector.
        state_vectors = np.concatenate((x_H_II, x_He_II, x_He_III, T), axis=1)

        return state_vectors
