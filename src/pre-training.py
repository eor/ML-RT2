import argparse
import os
import signal

import copy
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as torch_data

from common.settings_ode import ode_parameter_limits as ps_ode
from common.settings_sed import p8_limits as ps_sed
from common.settings_sed import SED_ENERGY_MIN, SED_ENERGY_MAX, SED_ENERGY_DELTA
from common.utils import *
from common.physics import *
from common.settings_crt import *
from common.settings import *
from common.data_log import *
from sed import sed_numba
from models_pretraining import *
from random import random

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
#   use AE with test or val set
# -----------------------------------------------------------------
def pretraining_evaluation(current_epoch, data_loader, model, path, config, print_results=False, save_results=False, best_model=False):
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
    """

    if save_results:
        print("\033[94m\033[1mTesting the Autoencoder at epoch %d \033[0m" % current_epoch)

    if cuda:
        model.cuda()

    if save_results:
        input_flux_vectors = torch.tensor([], device=device)
        regen_flux_vectors = torch.tensor([], device=device)

    model.eval()

    loss_mse = 0.0

    with torch.no_grad():
        for i, flux_vectors in enumerate(data_loader):

            # pass through the model
            out_flux_vector = model(flux_vectors)

            # compute loss via MSE:
            loss = F.mse_loss(input=out_flux_vector, target=flux_vectors, reduction='mean')

            loss_mse += loss.item()

            if save_results:
                # collate data
                input_flux_vectors = torch.cat((input_flux_vectors, flux_vectors), 0)
                regen_flux_vectors = torch.cat((regen_flux_vectors, out_flux_vector), 0)

    # mean of computed losses
    loss_mse = loss_mse / len(data_loader)

    if print_results:
        print("Results: AVERAGE MSE: %e" % (loss_mse))

    if save_results:
        # move data to CPU, re-scale parameters, and write everything to file
        input_flux_vectors = input_flux_vectors.cpu().numpy()
        regen_flux_vectors = regen_flux_vectors.cpu().numpy()

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        utils_save_pretraining_test_data(
            flux_vectors_true=input_flux_vectors,
            flux_vectors_gen=regen_flux_vectors,
            path=path,
            epoch=current_epoch,
            prefix=prefix
        )

    return loss_mse

# -----------------------------------------------------------------
#  Main
# -----------------------------------------------------------------
def main(config):

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
    # load the data and update config with the dataset conifguration
    # -----------------------------------------------------------------
    parameters, energies, intensities, density_vector, tau,\
     flux_vectors = utils_load_pretraining_data(config.data_dir)
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

    if PRETRAINING_LOG_PROFILES:
        # add a small number to avoid trouble
        flux_vectors = np.log10(flux_vectors + 1.0e-6)

    # -----------------------------------------------------------------
    # convert data into tensors and split it into requried legths
    # -----------------------------------------------------------------
    # numpy array to tensors
    flux_vectors = torch.Tensor(flux_vectors)

    # calculate length for train. val and test dataset from fractions
    train_length = int(PRETRAINING_SPLIT_FRACTION[0] * config.n_samples)
    validation_length = int(PRETRAINING_SPLIT_FRACTION[1] * config.n_samples)
    test_length = config.n_samples - train_length - validation_length

    # split the dataset
    train_dataset, validation_dataset, test_dataset = \
     torch.utils.data.random_split(flux_vectors,
                    (train_length, validation_length,test_length),
                    generator=torch.Generator(device).manual_seed(PRETRAINING_SEED))

    # -----------------------------------------------------------------
    # dataloaders from dataset
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
    print('\n\tusing model AE1 on device: %s\n'%(device))

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
    #  Main training loop
    # -----------------------------------------------------------------
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):
        epoch_loss = 0
        # set model mode
        model.train()
        for i, flux_vectors in enumerate(train_loader):
            # train the model here
            
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

        val_loss = pretraining_evaluation(
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

        if epoch % config.testing_interval == 0:
            pretraining_evaluation(best_epoch, test_loader, best_model, data_products_path, config, print_results=True, save_results=True)


    # -----------------------------------------------------------------
    # Evaluate the best model by using the test set
    # -----------------------------------------------------------------
    test_loss = pretraining_evaluation(
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


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='ML-RT2 - pre-training module for SED\'s')

    parser.add_argument('--out_dir', type=str, default='output_pretraining', metavar='(string)',
                        help='Path to output directory, used for all plots and data products, default: ./output_pretraining/')
    parser.add_argument('--data_dir', type=str, default='../data/sed_samples', metavar='(string)',
                        help='Path of the data directory from which data is to be read for training the model, default: ../data/sed_samples')
    parser.add_argument('--run', type=str, default='', metavar='(string)',
                        help='Specific run name for the experiment, default: ./output/timestamp')

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

    # model configuration
    parser.add_argument("--model", type=str, default='AE1', help="Model to use")
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true',
                        help="use batch normalisation in network (default)")
    parser.add_argument('--no-batch_norm', dest='batch_norm', action='store_false',
                        help="use batch normalisation in network")
    parser.set_defaults(batch_norm=False)

    parser.add_argument("--dropout", dest='dropout', action='store_true',
                        help="use dropout regularisation in network (default)")
    parser.add_argument("--no-dropout", dest='dropout', action='store_false',
                        help="do not use dropout regularisation in network")
    parser.set_defaults(dropout=False)
    parser.add_argument("--dropout_value", type=float, default=0.25, help="dropout probability, default=0.25 ")

    parser.add_argument("--len_latent_vector", type=int, default=8,
                        help="length of reduced SED vector")

    my_config = parser.parse_args()

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.device = device

    # print summary
    print("\nUsed parameters:\n")
    for arg in vars(my_config):
        print("\t", arg, getattr(my_config, arg))

    # run main program
    main(my_config)
