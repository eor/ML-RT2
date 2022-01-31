import argparse
import os
import sys; sys.path.append('..')
import signal
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.autograd import Variable
from random import random
from matplotlib import pyplot as plt

import common.sed_numba as sed_numba
from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA

from common.utils import *
from common.physics import *
from common.settings_crt import *
from common.settings import *
from common.data_log import *
from pretraining.analysis import *
from pretraining.models import *

# check for CUDA
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    cuda = False
    device = torch.device("cpu")
    torch.set_default_tensor_type(torch.FloatTensor)


# -----------------------------------------------------------------
#  evaluate network with test or validation data set
# -----------------------------------------------------------------
def pre_training_evaluation(current_epoch, data_loader, model, path, config,
                            print_results=False, save_results=False, best_model=False):
    """
    function runs the given dataset through the Autoencoder, returns mse_loss,
    and saves the results as well as ground truth to numpy file, if save_results=True.

    Args:
        current_epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
        save_results: whether to save actual and generated profiles locally (default: False)
        best_model: flag for testing on best model
        print_results: print results to screen
    """

    if save_results:
        print("\033[94m\033[1mTesting the Autoencoder at epoch %d \033[0m" % current_epoch)

    if cuda:
        model.cuda()

    if save_results:
        input_flux_vectors = torch.tensor([], device=device)
        regen_flux_vectors = torch.tensor([], device=device)
        parameter_vectors = torch.tensor([], device=device)

    model.eval()

    loss_mse = 0.0

    with torch.no_grad():
        for i, (parameters, flux_vectors) in enumerate(data_loader):

            # pass through the model
            out_flux_vector = model(flux_vectors)

            # compute loss via MSE:
            loss = F.mse_loss(input=out_flux_vector, target=flux_vectors, reduction='mean')

            loss_mse += loss.item()

            if save_results:
                # collate data
                input_flux_vectors = torch.cat((input_flux_vectors, flux_vectors), 0)
                regen_flux_vectors = torch.cat((regen_flux_vectors, out_flux_vector), 0)
                parameter_vectors = torch.cat((parameter_vectors, parameters), 0)

    # mean of computed losses
    loss_mse = loss_mse / len(data_loader)

    if print_results:
        print("Results: AVERAGE MSE: %e" % (loss_mse))

    if save_results:
        # move data to CPU, re-scale parameters, and write everything to file
        input_flux_vectors = input_flux_vectors.cpu().numpy()
        regen_flux_vectors = regen_flux_vectors.cpu().numpy()
        parameter_vectors = parameter_vectors.cpu().numpy()

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        utils_save_pretraining_test_data(
            flux_vectors_true=input_flux_vectors,
            flux_vectors_gen=regen_flux_vectors,
            parameters=parameter_vectors,
            path=path,
            epoch=current_epoch,
            prefix=prefix
        )

    return loss_mse


def force_stop_signal_handler(sig, frame):
    global FORCE_STOP
    FORCE_STOP = True
    print("\033[96m\033[1m\nTraining will stop after this epoch. Please wait.\033[0m\n")


# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def pre_training_main(config):

    # -----------------------------------------------------------------
    # create unique output path and run directories, save config
    # -----------------------------------------------------------------
    if config.run == '':
        run_id = 'run_' + utils_get_current_timestamp()
    else:
        run_id = 'run_' + config.run
    config.out_dir = os.path.join(config.out_dir, run_id)
    setattr(config, 'run', run_id)

    utils_create_run_directories(config.out_dir, DATA_PRODUCTS_DIR, PLOT_DIR)
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # path to store the data and plots after training
    data_products_path = os.path.join(config.out_dir, DATA_PRODUCTS_DIR)
    plot_path = os.path.join(config.out_dir, PLOT_DIR)

    # -----------------------------------------------------------------
    # run dataset analysis
    # -----------------------------------------------------------------
    if config.dataset_analysis:
        analysis_pretraining_dataset(config.data_dir, config.out_dir, mode=config.mode, prefix='data', k=50)

    # -----------------------------------------------------------------
    # load the data and update config with the dataset configuration,
    # -----------------------------------------------------------------
    if config.mode == 'train':
        # load the main dataset when mode is train
        parameters, _, _, _, _, flux_vectors = utils_load_pretraining_data(config.data_dir,
                                                                           file_name='data_pretraining.npy.npz')
    else:
        # load the development dataset when mode is dev
        parameters, _, _, _, _, flux_vectors = utils_load_pretraining_data(config.data_dir,
                                                                           file_name='data_pretraining_dev_set.npy.npz')

    setattr(config, 'len_SED_input', flux_vectors.shape[1])
    setattr(config, 'n_samples', flux_vectors.shape[0])

    # -----------------------------------------------------------------
    # shuffle/log space
    # -----------------------------------------------------------------
    if PRETRAINING_SHUFFLE:
        np.random.seed(PRETRAINING_SEED)
        indices = np.arange(config.n_samples, dtype=np.int32)
        indices = np.random.permutation(indices)
        flux_vectors = flux_vectors[indices]
        parameters = parameters[indices]

    if PRETRAINING_LOG_PROFILES:
        # add a small number to avoid trouble
        flux_vectors = np.log10(flux_vectors + 1.0e-6)

    # -----------------------------------------------------------------
    # convert data into tensors and split it into required lengths
    # -----------------------------------------------------------------
    flux_vectors = torch.Tensor(flux_vectors)
    parameters = torch.Tensor(parameters)

    # calculate length for train. val and test dataset from fractions
    train_length = int(PRETRAINING_SPLIT_FRACTION[0] * config.n_samples)
    validation_length = int(PRETRAINING_SPLIT_FRACTION[1] * config.n_samples)
    test_length = config.n_samples - train_length - validation_length

    # split the dataset
    dataset = torch.utils.data.TensorDataset(parameters, flux_vectors)

    train_dataset, validation_dataset, test_dataset = \
        torch.utils.data.random_split(dataset,
                                      (train_length, validation_length, test_length),
                                      generator=torch.Generator(device).manual_seed(PRETRAINING_SEED)
                                      )

    # -----------------------------------------------------------------
    # data loaders from dataset
    # -----------------------------------------------------------------
    train_loader = torch_data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = torch_data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------------------------------------------
    # tensorboard (to check results, visit localhost:6006)
    # -----------------------------------------------------------------
    data_log = DataLog.getInstance(config.out_dir)
    data_log.start_server()

    # -----------------------------------------------------------------
    # initialise model
    # -----------------------------------------------------------------
    model = AE1(config)
    print('\n\tUsing model AE1 on device: %s\n' % device)

    if cuda:
        model.cuda()

    # -----------------------------------------------------------------
    # Optimizers
    # -----------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )

    # -----------------------------------------------------------------
    # book keeping arrays
    # -----------------------------------------------------------------
    avg_train_loss_array = np.empty(0)
    avg_val_loss_array = np.empty(0)

    # -----------------------------------------------------------------
    # keep the model with min validation loss
    # -----------------------------------------------------------------
    best_model = copy.deepcopy(model)
    best_loss = np.inf
    best_epoch = 0
    n_epoch_without_improvement = 0

    # -----------------------------------------------------------------
    # FORCE_STOP
    # -----------------------------------------------------------------
    global FORCE_STOP
    FORCE_STOP = False
    if FORCE_STOP_ENABLED:
        signal.signal(signal.SIGINT, force_stop_signal_handler)
        print('\n Press Ctrl + C to stop the training anytime and exit while saving the results.\n')

    # -----------------------------------------------------------------
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):
        epoch_loss = 0
        # set model mode
        model.train()
        for i, (_, flux_vectors) in enumerate(train_loader):

            # zero the gradients on each iteration
            optimizer.zero_grad()
            regen_flux_vectors = model(flux_vectors)

            # compute loss
            loss = F.mse_loss(input=regen_flux_vectors, target=flux_vectors, reduction='mean')
            loss.backward()
            optimizer.step()

            # sum the loss values
            epoch_loss += loss.item()

        # end-of-epoch book keeping
        train_loss = epoch_loss / len(train_loader)
        avg_train_loss_array = np.append(avg_train_loss_array, train_loss)

        val_loss = pre_training_evaluation(
            current_epoch=epoch,
            data_loader=val_loader,
            model=model,
            path=data_products_path,
            config=config,
            print_results=False,
            save_results=False,
            best_model=False
        )

        avg_val_loss_array = np.append(avg_val_loss_array, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            n_epoch_without_improvement = 0
        else:
            n_epoch_without_improvement += 1

        # log losses in tensorboard
        data_log.log_losses(train_loss, val_loss)
        data_log.update_data()

        print("[Epoch %d/%d] [Train loss: %e] [Validation loss: %e][Best_epoch: %d]"
              % (epoch, config.n_epochs, train_loss, val_loss, best_epoch))

        # early stopping check
        if FORCE_STOP:
            print("\033[96m\033[1m\nStopping Early\033[0m\n")
            stopped_early = True
            epochs_trained = epoch
            break

        if epoch % config.testing_interval == 0:
            pre_training_evaluation(best_epoch, test_loader, best_model, data_products_path, config,
                                    print_results=True, save_results=True)

    # -----------------------------------------------------------------
    # Evaluate the best model by using the test set
    # -----------------------------------------------------------------
    test_loss = pre_training_evaluation(
        current_epoch=best_epoch,
        data_loader=test_loader,
        model=best_model,
        path=data_products_path,
        config=config,
        print_results=True,
        save_results=True,
        best_model=True
    )

    # -----------------------------------------------------------------
    # Save the loss functions
    # -----------------------------------------------------------------
    utils_save_loss(avg_train_loss_array, data_products_path, config.n_epochs, prefix='train')
    utils_save_loss(avg_val_loss_array, data_products_path, config.n_epochs, prefix='val')

    # -----------------------------------------------------------------
    # Save the best model and the final model
    # -----------------------------------------------------------------
    utils_save_pretraining_model(best_model.state_dict(),
                                 data_products_path, best_epoch, best_model=True)
    utils_save_pretraining_model(model.state_dict(),
                                 data_products_path, config.n_epochs, best_model=False)

    # -----------------------------------------------------------------
    # Save some results to config object for later use
    # -----------------------------------------------------------------
    setattr(config, 'best_epoch', best_epoch)

    setattr(config, 'best_val_mse', best_loss)
    setattr(config, 'best_test_mse', test_loss)

    # -----------------------------------------------------------------
    # Overwrite config object
    # -----------------------------------------------------------------
    utils_save_config_to_log(config)
    utils_save_config_to_file(config)

    # -----------------------------------------------------------------
    # shutdown tensorboard server
    # -----------------------------------------------------------------
    data_log.close()

    # [TODO] do analysis here......
    # 1. plot loss functions.
    # 2. try some ways to represent and compare the true and regen flux vectors.

    if config.analysis:
        print("\n\033[96m\033[1m\nRunning analysis...\033[0m\n")
        analysis_auto_plot_flux_vectors(config, k=20, base_path=config.out_dir, prefix='best')
        print("\n\033[96m\033[1m\nDone with analysis.\033[0m\n")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT2 - pre-training module for SED\'s')

    parser.add_argument('--out_dir', type=str, default='output_pretraining', metavar='(string)',
                        help='Path to pre-training output directory (plots and data), default: ./output_pretraining/')

    parser.add_argument('--data_dir', type=str, default='../../data/pretraining', metavar='(string)',
                        help='Path to teh pre-training data directory, default: ../../data/pretraining')
    parser.add_argument('--run', type=str, default='', metavar='(string)',
                        help='Specific run name for the experiment, default: timestamp')

    parser.add_argument("--testing_interval", type=int,
                        default=50, help="epoch interval between testing runs")

    # network optimisation
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches (default=32)")

    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate, default=0.0001")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: beta1 - decay of first order momentum of gradient, default=0.9")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: beta2 - decay of first order momentum of gradient, default=0.999")
    parser.add_argument("--mode", type=str, default='train',
                        help="Dataset to be used for training (dev/train), default=train")

    # model configuration
    parser.add_argument("--model", type=str, default='AE1', help="Model to use")

    # batch norm
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true',
                        help="use batch normalisation in network (default)")
    parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false',
                        help="use batch normalisation in network")
    parser.set_defaults(batch_norm=False)

    # drop out
    parser.add_argument("--dropout", dest='dropout', action='store_true',
                        help="use dropout regularisation in network (default)")
    parser.add_argument("--no-dropout", dest='dropout', action='store_false',
                        help="do not use dropout regularisation in network")
    parser.set_defaults(dropout=False)
    parser.add_argument("--dropout_value", type=float, default=0.25, help="dropout probability, default=0.25 ")

    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")

    # analysis
    parser.add_argument("--analysis", dest='analysis', action='store_true',
                        help="automatically generate some plots (default)")
    parser.add_argument("--no-analysis", dest='analysis', action='store_false', help="do not run analysis")
    parser.set_defaults(analysis=True)

    parser.add_argument("--dataset_analysis", dest='dataset_analysis', action='store_true',
                        help="Compute mean, min, max on dataset and generate relevant plots")
    parser.add_argument("--no_dataset_analysis", dest='dataset_analysis', action='store_false',
                        help="Do not generate data summary")
    parser.set_defaults(dataset_analysis=False)

    my_config = parser.parse_args()

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.device = device

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    # run main program
    pre_training_main(my_config)
