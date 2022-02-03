import numpy as np
import os.path as osp
import heapq
import sys; sys.path.append('..')

from common.utils import *
from common.settings import *
from common.settings_sed import *
from common.physics import *

from pretraining.plot import *


# -----------------------------------------------------------------
#  auto plot flux vector comparison
# -----------------------------------------------------------------
def analysis_auto_plot_flux_vectors(config, k=10, base_path=None, prefix='best', epoch=None):
    """
    Automatically plot true and regenerated flux_vectors vs photon energies.

    Args:
        config:  config object of the training run
        k: int, number of best and worst samples to plot (based on MSE comparison)
        base_path: string, output path of training run
        prefix: 'test' or 'best'
        epoch: int

    Returns:

    """

    if base_path is not None:
        data_dir_path = osp.join(base_path, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(base_path, PLOT_DIR)
    else:
        data_dir_path = osp.join(config.out_dir, DATA_PRODUCTS_DIR)
        plot_dir_path = osp.join(config.out_dir, PLOT_DIR)

    if prefix == 'test':
        epoch = epoch
    elif prefix == 'best':
        epoch = config.best_epoch

    parameter_true_file = prefix + ('_parameters_%d_epochs.npy' % (epoch))
    flux_vector_true_file_name = prefix + ('_flux_vectors_true_%d_epochs.npy' % (epoch))
    flux_vector_gen_file_name = prefix + ('_flux_vectors_gen_%d_epochs.npy' % (epoch))

    parameters = np.load(osp.join(data_dir_path, parameter_true_file))
    flux_vectors_true = np.load(osp.join(data_dir_path, flux_vector_true_file_name))
    flux_vectors_gen = np.load(osp.join(data_dir_path, flux_vector_gen_file_name))

    # 2. compute MSE
    mse_array = (np.square((flux_vectors_true) - (flux_vectors_gen))).mean(axis=1)
    log_mse = np.log10(mse_array + 1e-11)

    # 3. find k lowest and largest MSE values and their respective indexes
    k_large_list = heapq.nlargest(k, range(len(mse_array)), mse_array.take)
    k_small_list = heapq.nsmallest(k, range(len(mse_array)), mse_array.take)

    # 4. sample an energy_vector to plot on x-axis.
    energy_vector = np.logspace(np.log10(SED_ENERGY_MIN), np.log10(SED_ENERGY_MAX), num=len(flux_vectors_true[0]))

    # 5.  plot profiles for largest MSE
    print('Producing profile plot(s) for profiles with %d highest MSE' % k)
    for i in range(len(k_large_list)):
        index = k_large_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        plot_flux_vector_comparison(flux_vector_true=flux_vectors_true[index],
                                    flux_vector_gen=flux_vectors_gen[index],
                                    parameters=parameters[index],
                                    energy_vector=energy_vector,
                                    n_epoch=epoch,
                                    output_dir=plot_dir_path,
                                    prefix=prefix,
                                    mse=mse_array[index]
                                    )

    # 6.  plot profiles for smallest MSE
    print('Producing profile plot(s) for profiles with %d lowest MSE' % k)
    for i in range(len(k_small_list)):
        index = k_small_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        tmp_parameters = parameters[index]
        tmp_profile_true = flux_vectors_true[index]
        tmp_profile_gen = flux_vectors_gen[index]

        plot_flux_vector_comparison(flux_vector_true=flux_vectors_true[index],
                                    flux_vector_gen=flux_vectors_gen[index],
                                    parameters=parameters[index],
                                    energy_vector=energy_vector,
                                    n_epoch=epoch,
                                    output_dir=plot_dir_path,
                                    prefix=prefix,
                                    mse=mse_array[index]
                                    )


# -----------------------------------------------------------------
# Plot k random profiles from dataset.
# -----------------------------------------------------------------
def analysis_pretraining_dataset(data_dir, base_path, mode='train', prefix='data', k=10):
    """
    Plot a number of examples from the pre-training data set, i.e. N(E), I(E), tau versus photon energy.

    Args:
        data_dir:
        base_path:
        mode:
        prefix:
        k:
    """

    # output directory for results
    data_analysis_dir_path = osp.join(base_path, DATA_ANALYSIS)

    # create directories
    utils_create_output_dirs([data_analysis_dir_path])

    if mode == 'train':
        # load the main dataset when mode is train
        (parameters,
         energies,
         intensities,
         tau_input_vector,
         tau,
         flux_vectors) = utils_load_pretraining_data(data_dir)
    else:
        # load the development dataset when mode is dev
        (parameters,
         energies,
         intensities,
         tau_input_vector,
         tau,
         flux_vectors) = utils_load_pretraining_data(data_dir, file_name='data_pretraining_dev_set.npy.npz')

    # compute mean, min, max in dataset
    print('\nGenerating dataset summary.....')
    print('Average of flux_vectors in dataset: ', np.mean(flux_vectors))
    minimum, maximum = np.min(flux_vectors), np.max(flux_vectors)
    print('Maximum and minimum value of flux_vectors in dataset: ', minimum, maximum)

    print('Average of tau in dataset: ', np.mean(tau))
    minimum, maximum = np.min(tau), np.max(tau)
    print('Maximum and minimum value of tau in dataset: ', minimum, maximum)

    # plot histogram for data_set_distribution
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(flux_vectors.flatten(), bins=10, log=True, density=True)
    plt.savefig(osp.join(data_analysis_dir_path, 'data_set_distribution.png'))

    print('Successfully saved histogram for dataset to:', osp.join(data_analysis_dir_path, 'data_set_distribution.png'))

    # obtain ionisation cross-section using energy vector.
    p = Physics.getInstance()
    p.set_energy_vector(energies[0])
    cross_section = p.get_photo_ionisation_cross_section_hydrogen()

    # generate k random numbers between 0 and len(dataset)
    np.random.seed(PRETRAINING_SEED)
    random_indices = np.random.randint(0, high=flux_vectors.shape[0], size=(k))
    # use the random numbers as indices to pick random profiles from dataset
    flux_vectors = flux_vectors[random_indices, :]
    intensities = intensities[random_indices, :]
    tau = tau[random_indices, :]
    parameters = parameters[random_indices, :]

    # broadcast single profile of cross-section to k profiles.
    cross_section = np.broadcast_to(cross_section[None, :], (k, len(cross_section)))

    # stack the data to plot
    plot_profile_data = np.stack((flux_vectors, intensities, tau, cross_section), axis=1)

    print('Producing analysis plot(s) for %d random profiles' % k)
    for i in range(k):
        plot_profiles_dataset(plot_profile_data[i],
                              energies[0],
                              parameters[i],
                              data_analysis_dir_path,
                              prefix,
                              random_indices[i]
                             )


# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there! Let\'s analyse some results\n')

    path = './output_pretraining/'
    data_dir = '../../data/pretraining/'
    # config = utils_load_config(path)
    analysis_pretraining_dataset(data_dir, path, prefix='data', k=10)
#     analysis_auto_plot_flux_vectors(config, k=50, base_path=path, prefix='best')

    print('\n Completed! \n')
