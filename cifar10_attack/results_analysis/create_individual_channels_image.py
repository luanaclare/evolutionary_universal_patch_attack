import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ast import literal_eval
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For colorbar control
import os

# Best runs in order for scenarios 3%, 5%, 10%
modelNames = {'VGG16': ['VGG16__run__2024_06_26__06_36_09_480__112681534716641280', 'VGG16__run__2024_07_09__10_05_07_686__112755966455709696', 'VGG16__run__2024_07_01__20_38_08_586__112713157088772096'],
              'ResNet50' : ['ResNet50__run__2024_06_21__22_37_27_519__112657003152605184', 'ResNet50__run__2024_07_28__10_27_02_652__112863636530921472', 'ResNet50__run__2024_07_25__12_01_58_461__112847022880260096']}
scenarios = ['f1','f2', 'f3']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 20})

# Loop over each scenario and model name
for scenario_idx, scenario in enumerate(scenarios):
    for modelName, run_list in modelNames.items():
        
        # Retrieve the best run folder for the current scenario (based on index)
        if scenario_idx < len(run_list):
            main_folder = run_list[scenario_idx]
            
        ficheiro = f'../{main_folder}/best_info.csv'
        data = pd.read_csv(ficheiro)

        # Extract perturbation data as a string and convert it to a list
        perturbations_str = data['perturbation'].iloc[0]
        perturbations = literal_eval(perturbations_str)  # Convert string to list
        perturbations = np.array(perturbations)  # Convert list to NumPy array

        # perturbation random
        ficheiro_random = f'../{main_folder}/logs/generations/gen00050.csv'
        ix = 0 # best individual
        data_random = pd.read_csv(ficheiro_random)
        perturbations_str_random = data_random[' perturbation'].iloc[ix]
        perturbations_random = literal_eval(perturbations_str_random)  # Convert string to list
        perturbations_random = np.array(perturbations_random)

        # Separate perturbations into red, green, and blue channels
        red_channel = perturbations[:, :, 0]
        green_channel = perturbations[:, :, 1]
        blue_channel = perturbations[:, :, 2]

        red_channel_random = perturbations_random[:, :, 0]
        green_channel_random = perturbations_random[:, :, 1]
        blue_channel_random = perturbations_random[:, :, 2]

        # Define a Diverging colormap
        diverging_colormap = plt.get_cmap('coolwarm')  # Or you can use any other diverging colormap

        # Plotting each channel
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Function to add colorbars with adjustable size
        def add_colorbar(ax, im, size="5%", pad=0.5):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=size, pad=pad)
            fig.colorbar(im, cax=cax, orientation='vertical')

        # Plot Red Channel
        im1 = axs[0].imshow(red_channel, cmap=diverging_colormap, vmin=-0.1*255, vmax=0.1*255)
      #  axs[0].set_title('Red Channel')
        axs[0].axis('off')
        add_colorbar(axs[0], im1)

        # Plot Green Channel
        im2 = axs[1].imshow(green_channel, cmap=diverging_colormap, vmin=-0.1*255, vmax=0.1*255)
     #   axs[1].set_title('Green Channel')
        axs[1].axis('off')
        add_colorbar(axs[1], im2)

        # Plot Blue Channel
        im3 = axs[2].imshow(blue_channel, cmap=diverging_colormap, vmin=-0.1*255, vmax=0.1*255)
     #   axs[2].set_title('Blue Channel')
        axs[2].axis('off')
        add_colorbar(axs[2], im3)

        plt.tight_layout()
        plt.savefig(f'ind_images/{scenario}_{modelName}_perturbation_channels.png', bbox_inches='tight')

        # Plotting each channel
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot Red Channel
        im1 = axs[0].imshow(red_channel_random, cmap=diverging_colormap, vmin=-0.1*255, vmax=0.1*255)
      #  axs[0].set_title('Red Channel')
        axs[0].axis('off')
        add_colorbar(axs[0], im1)

        # Plot Green Channel
        im2 = axs[1].imshow(green_channel_random, cmap=diverging_colormap, vmin=-0.1*255, vmax=0.1*255)
     #   axs[1].set_title('Green Channel')
        axs[1].axis('off')
        add_colorbar(axs[1], im2)

        # Plot Blue Channel
        im3 = axs[2].imshow(blue_channel_random, cmap=diverging_colormap, vmin=-0.1*255, vmax=0.1*255)
     #   axs[2].set_title('Blue Channel')
        axs[2].axis('off')
        add_colorbar(axs[2], im3)

        plt.tight_layout()
        if not os.path.exists('channel_images'):
            os.mkdir('channel_images')
        plt.savefig(f'channel_images/{scenario}_{modelName}_perturbation_channels_{ix}.png', bbox_inches='tight')
    mamama = mamama+1