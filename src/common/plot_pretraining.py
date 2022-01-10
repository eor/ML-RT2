import os
import os.path as osp
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import rc
import seaborn as sns
try:
    from common.settings_sed import p8_limits, p8_names_latex
except ImportError:
    from settings_sed import p8_limits, p8_names_latex

# -----------------------------------------------------------------
# Plot flux vectoes (True and regenerated)
# -----------------------------------------------------------------
def plot_flux_vector_comparison(flux_vector_true, flux_vector_gen, n_epoch, output_dir,
                        prefix, mse, file_type='pdf', parameters=None, add_errors=True):

    # -----------------------------------------------------------------
    # font settings 
    # -----------------------------------------------------------------
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)
    font_size_title = 26
    font_size_ticks = 26
    font_size_legends = 22
    font_size_x_y = 30

    # -----------------------------------------------------------------
    # figure setup
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(11, 10))
    plt.subplots_adjust(top=0.82)
        
    # -----------------------------------------------------------------
    # grid setup
    # -----------------------------------------------------------------
    outer = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.3)
    # create two subplots inside the main subplot
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.0, height_ratios=[3, 1])
    # references to two inner subplots
    ax0 = fig.add_subplot(inner[0])
    ax1 = fig.add_subplot(inner[1], sharex=ax0)

    # -----------------------------------------------------------------
    # first plot (true and regenerated flux_vectors)
    # -----------------------------------------------------------------
    ax0.plot(flux_vector_true, c='red', label='Truth')
    ax0.plot(flux_vector_gen, c='blue', label='Regenerated')

    if add_errors:
        ax0.plot([], [], alpha=.0, label='MSE: %e' % (mse))

    ax0.legend(loc=0, frameon=False, prop={'size': font_size_legends})
    # set y-label
    ax0.set_ylabel(r'$ \log_{10}(N(E)) $', fontsize=font_size_x_y)
    ax0.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax0.tick_params(axis='y', which='both', right=True, top=True, labelsize=font_size_ticks)
    fig.add_subplot(ax0)

    # -----------------------------------------------------------------
    # second plot (diff / relative error)
    # -----------------------------------------------------------------
    absolute_error = flux_vector_true - flux_vector_gen
    ax1.plot(absolute_error, c='black', label='Absolute error', linewidth=0.6)
    ax1.grid(which='major', color='#999999', linestyle='-', linewidth='0.4', alpha=0.4)
    ax1.set_ylabel(r'Abs error', fontsize=font_size_x_y)
    ax1.set_xlabel(r'Energy $(eV)$', fontsize=font_size_x_y)
    ax1.set_xticks(np.arange(0, len(flux_vector_true), step=50), minor=True)
    ax1.tick_params(axis='both', which='both', right=True, top=True, labelsize=font_size_ticks)
    
    # -----------------------------------------------------------------
    # add parameters as title
    # -----------------------------------------------------------------
    param_names = p8_names_latex
    if parameters is not None and len(parameters) > 0:
        a = ''
        for j in range(len(param_names)):
            # add line break after every 3 parameters are added
            if j != 0 and j % 3 == 0:
                a += '\n'
            # append the parameter with its name and value to title string
            value = parameters[j]
            name = '$' + param_names[j]
            a = a + name + ' = ' + str(value) + '$'
            if j == 2:
                a += '$\mathrm{Myr}$'
            a += '\, \, \, '

    fig.suptitle(a, fontsize=font_size_title, y=0.98)

    # -----------------------------------------------------------------
    # get MSE and construct file name
    # -----------------------------------------------------------------
    log_mse = np.log10(mse + 1.0e-11)
    file_name = '{:s}_profile_epoch_{:d}_logMSE_{:.4e}'.format(prefix, n_epoch, log_mse)
    file_name = file_name.replace('.', '_')
    file_name += '.' + file_type

    plt.savefig(os.path.join(output_dir, file_name))
    plt.close('all')

