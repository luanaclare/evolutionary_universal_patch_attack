import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from ast import literal_eval
import numpy as np

matplotlib.rcParams.update({'font.size': 40})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

modelNames = ['VGG16', 'ResNet50']
gens_heatmap = [0, 300, 600]
scenarios = ['f1', 'f2', 'f3']
save_folder = 'graphs'

for scenario in scenarios:
    main_folder = f'../runs_final/{scenario}'
    for modelName in modelNames:
        for gen in gens_heatmap:
            # Initialize accumulators for the heatmaps and a counter for the number of runs
            heatmap_accumulator = None
            run_count = 0
            
            # Loop over all folders for the scenario-model
            for folder_name in os.listdir(main_folder):
                if folder_name.startswith(modelName):
                    run_folder = os.path.join(main_folder, folder_name)
                    ficheiro = f'{run_folder}/logs/generations/gen{str(gen).zfill(5)}.csv'
                    if os.path.exists(ficheiro):
                        data = pd.read_csv(ficheiro)
                        perturbations = []
                        best_perturbations = []
                        for i in range(200):
                            perturbation_str = data[' perturbation'].iloc[i]
                            perturbation_list = literal_eval(perturbation_str)
                            perturbation = np.array(perturbation_list)
                            perturbation[perturbation != 0] = 1
                            perturbations.append(perturbation)

                        # Stack perturbations into a single array
                        perturbations_stack = np.stack(perturbations)

                        # Collapse the color channels (32x32x3 -> 32x32) by checking if any channel is non-zero
                        heatmap_data = np.any(perturbations_stack, axis=3).astype(int)

                        # Sum across all perturbations (200 perturbations -> single heatmap)
                        heatmap_data_sum = np.sum(heatmap_data, axis=0)

                        # Accumulate the heatmap data
                        if heatmap_accumulator is None:
                            heatmap_accumulator = heatmap_data_sum
                        else:
                            heatmap_accumulator += heatmap_data_sum
                        
                        # Increment the run counter
                        run_count += 1
            
            # After looping over all runs, calculate the average heatmap if we processed any runs
            if run_count > 0:
                heatmap_avg = heatmap_accumulator / run_count

                # Plot the average heatmap
                plt.figure(figsize=(10, 8))
                plt.xlim(0, 31)
                plt.ylim(0, 31)
                plt.imshow(heatmap_avg, cmap='viridis', interpolation='nearest', vmin=0, vmax=200)
                if gen == 600: plt.colorbar(label='Number of Individuals')

                # Rotate and force x and y labels to appear
                plt.xticks(ticks=[0, 5, 10, 15, 20, 25, 31], rotation=45, fontsize=30)  # Rotated x-axis labels
                plt.yticks(ticks=[0, 5, 10, 15, 20, 25, 31], rotation=45, fontsize=30)   # Regular y-axis labels

                # Save the averaged heatmap as an image
                heatmap_filename = f'{save_folder}/coverage_{scenario}_{modelName}_avg_{modelName}_gen{gen}.png'
                plt.savefig(heatmap_filename, bbox_inches='tight')
                
                # Close the plot to avoid memory issues
                plt.close()


for scenario in scenarios:
    main_folder = f'runs_final/{scenario}'
    for modelName in modelNames:
        for gen in gens_heatmap:
            # Initialize accumulators for the heatmaps and a counter for the number of runs
            heatmap_accumulator = None
            run_count = 0
            
            # Loop over all folders for the scenario-model
            for folder_name in os.listdir(main_folder):
                if folder_name.startswith(modelName):
                    run_folder = os.path.join(main_folder, folder_name)
                    ficheiro = f'{run_folder}/logs/generations/gen{str(gen).zfill(5)}.csv'
                    
                    if os.path.exists(ficheiro):
                        data = pd.read_csv(ficheiro)
                        
                        # Get the first perturbation
                        perturbation_str = data[' perturbation'].iloc[0]
                        perturbation_list = literal_eval(perturbation_str)
                        perturbation = np.array(perturbation_list)
                        perturbation[perturbation != 0] = 1

                        heatmap_data = np.any(perturbation, axis=2).astype(int)

                        # Accumulate the heatmap data
                        if heatmap_accumulator is None:
                            heatmap_accumulator = heatmap_data
                        else:
                            heatmap_accumulator += heatmap_data
                        
                        # Increment the run counter
                        run_count += 1
            
            # After looping over all runs, calculate the average heatmap if we processed any runs
            if run_count > 0:
                heatmap_avg = heatmap_accumulator

                # Plot the average heatmap
                plt.figure(figsize=(12, 10))
                plt.xlim(0, 31)
                plt.ylim(0, 31)
                plt.imshow(heatmap_avg, cmap='viridis', interpolation='nearest', vmin=0, vmax=30)
                if gen == 600: plt.colorbar(label='Number of Best Individuals')

                # Rotate and force x and y labels to appear
                plt.xticks(ticks=[0, 5, 10, 15, 20, 25, 31], rotation=45, fontsize=30)  # Rotated x-axis labels
                plt.yticks(ticks=[0, 5, 10, 15, 20, 25, 31], rotation=45, fontsize=30)   # Regular y-axis labels

                # Save the averaged heatmap as an image
                heatmap_filename = f'{save_folder}/coverage_{scenario}_{modelName}_best_{modelName}_gen{gen}.png'
                plt.savefig(heatmap_filename, bbox_inches='tight')
                
                # Close the plot to avoid memory issues
                plt.close()