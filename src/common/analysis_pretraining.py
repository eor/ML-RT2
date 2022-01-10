import numpy as np
import os.path as osp
import heapq
try:
    from common.utils import *
    from common.plot_pretraining import *
    from common.settings import *
except ImportError:
    from plot_pretraining import *
    from utils import *
    from settings import *

# -----------------------------------------------------------------
# set of functions to operate on saved results from training 
# and generate meaningful plots. All results are saved in default PLOT_DIR.
# ------------------------------------------------------------------


# -----------------------------------------------------------------
# Automatically plot true and regenerated flux_vectors
# -----------------------------------------------------------------
def analysis_auto_plot_flux_vectors(config, k=20, base_path=None, prefix='best', epoch=None):

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

    # 4.  plot profiles for largest MSE
    print('Producing profile plot(s) for profiles with %d highest MSE' % k)
    for i in range(len(k_large_list)):
        index = k_large_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        plot_flux_vector_comparison(
            flux_vector_true=flux_vectors_true[index],
            flux_vector_gen=flux_vectors_gen[index],
            parameters=parameters[index],
            n_epoch=epoch,
            output_dir=plot_dir_path,
            prefix=prefix,
            mse=mse_array[index]
        )

    # 5.  plot profiles for smallest MSE
    print('Producing profile plot(s) for profiles with %d lowest MSE' % k)
    for i in range(len(k_small_list)):
        index = k_small_list[i]
        print('{:3d} \t MSE = {:.4e} \t parameters: {}'.format(i, mse_array[index], parameters[index]))

        tmp_parameters = parameters[index]
        tmp_profile_true = flux_vectors_true[index]
        tmp_profile_gen = flux_vectors_gen[index]

        plot_flux_vector_comparison(
            flux_vector_true=flux_vectors_true[index],
            flux_vector_gen=flux_vectors_gen[index],
            parameters=parameters[index],
            n_epoch=epoch,
            output_dir=plot_dir_path,
            prefix=prefix,
            mse=mse_array[index]
        )

        
# -----------------------------------------------------------------
#  run the following if this file is called directly
# -----------------------------------------------------------------
if __name__ == '__main__':

    print('Hello there! Let\'s analyse some results\n')
    
    path = '../output_pretraining/run_2022_01_09__06_49_54/'
    config = utils_load_config(path)
    analysis_auto_plot_flux_vectors(config, k=20, base_path=path, prefix='best')

    print('\n Completed! \n')

